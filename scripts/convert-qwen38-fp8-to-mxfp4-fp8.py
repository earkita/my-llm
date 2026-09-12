#!/usr/bin/env python3
"""Stream Qwen3.8 Flash-Next FP8 into the reference MXFP4/FP8 layout.

The reference checkpoint is a format oracle only: this program reads its JSON
files and safetensors headers, but never reads reference tensor payloads.  All
output tensor values and quantization scales are copied from, or calculated
from, the source checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import struct
import sys
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path("/mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-FP8")
DEFAULT_REFERENCE = Path("/mnt/ai/models/qwen/Qwen3.8-Flash-Next-MXFP4-FP8")
DEFAULT_OUTPUT = Path("/mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-MXFP4-FP8")
DEFAULT_REPORT_DIR = Path("logs/conversion/qwen38-uncensored-mxfp4-fp8")
KV_SCALE_SHARD = "model-kvscales.safetensors"
FP8_MAX = 448.0
FP8_BLOCK = 128
MXFP4_GROUP = 32

DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
}
FLOAT_DTYPES = {
    "F8_E4M3",
    "F8_E4M3FN",
    "F8_E4M3FNUZ",
    "F8_E5M2",
    "F8_E5M2FNUZ",
    "F16",
    "BF16",
    "F32",
    "F64",
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    offsets: tuple[int, int]
    data_start: int

    @property
    def nbytes(self) -> int:
        return self.offsets[1] - self.offsets[0]


@dataclass
class Inventory:
    root: Path
    tensors: dict[str, TensorInfo]
    shard_metadata: dict[str, dict[str, str] | None]
    index_metadata: dict[str, Any]
    errors: list[str]


@dataclass(frozen=True)
class PlanEntry:
    target_name: str
    target_file: str
    target_dtype: str
    target_shape: tuple[int, ...]
    action: str
    source_names: tuple[str, ...]
    detail: str


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length <= 1 or header_length > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header length in {path}: {header_length}")
        raw = handle.read(header_length)
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not an object: {path}")
    return header, 8 + header_length


def load_inventory(root: Path) -> Inventory:
    root = root.resolve()
    index_path = root / "model.safetensors.index.json"
    if not root.is_dir():
        raise FileNotFoundError(root)
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid weight_map: {index_path}")

    errors: list[str] = []
    tensors: dict[str, TensorInfo] = {}
    shard_metadata: dict[str, dict[str, str] | None] = {}
    indexed_shards = sorted({str(value) for value in weight_map.values()})
    actual_shards = sorted(path.name for path in root.glob("*.safetensors"))
    unindexed = sorted(set(actual_shards) - set(indexed_shards))
    if unindexed:
        errors.append(f"Unindexed shards: {unindexed}")

    for shard_name in indexed_shards:
        shard_path = root / shard_name
        if not shard_path.is_file():
            errors.append(f"Missing indexed shard: {shard_name}")
            continue
        try:
            header, data_start = _read_safetensors_header(shard_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
            continue
        shard_metadata[shard_name] = header.pop("__metadata__", None)
        file_size = shard_path.stat().st_size
        for name, raw_info in header.items():
            if name in tensors:
                errors.append(f"Duplicate tensor: {name}")
                continue
            try:
                dtype = str(raw_info["dtype"])
                shape = tuple(int(dim) for dim in raw_info["shape"])
                offsets = tuple(int(offset) for offset in raw_info["data_offsets"])
                if len(offsets) != 2:
                    raise ValueError("data_offsets length is not two")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"Invalid header for {name}: {exc}")
                continue
            if dtype not in DTYPE_BITS:
                errors.append(f"Unsupported dtype {dtype}: {name}")
            elif math.prod(shape) * DTYPE_BITS[dtype] != (offsets[1] - offsets[0]) * 8:
                errors.append(f"Tensor byte-size mismatch: {name}")
            if offsets[0] < 0 or offsets[1] < offsets[0] or data_start + offsets[1] > file_size:
                errors.append(f"Out-of-range offsets: {name}")
            tensors[name] = TensorInfo(name, shard_name, dtype, shape, offsets, data_start)

    header_names = set(tensors)
    index_names = set(weight_map)
    for name in sorted(index_names - header_names):
        errors.append(f"Index entry absent from headers: {name}")
    for name in sorted(header_names - index_names):
        errors.append(f"Header tensor absent from index: {name}")
    for name in sorted(index_names & header_names):
        if str(weight_map[name]) != tensors[name].file:
            errors.append(f"Index/header shard mismatch: {name}")
    return Inventory(root, tensors, shard_metadata, index.get("metadata", {}), errors)


def _without_quantization_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)

    def remove(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("quantization_config", None)
            for child in node.values():
                remove(child)

    remove(value)
    return value


def _expert_sources(prefix: str, projection: str, count: int) -> tuple[str, ...]:
    result: list[str] = []
    projections = ("gate_proj", "up_proj") if projection == "gate_up_proj" else ("down_proj",)
    for expert in range(count):
        for item in projections:
            weight = f"{prefix}.{expert}.{item}.weight"
            result.extend((weight, weight + "_scale_inv"))
    return tuple(result)


def _source_tuple_text(names: tuple[str, ...], limit: int = 8) -> str:
    if len(names) <= limit:
        return json.dumps(names, separators=(",", ":"))
    return json.dumps((*names[:limit], f"... {len(names) - limit} more"), separators=(",", ":"))


def build_plan(
    source: Inventory,
    reference: Inventory,
    *,
    include_kv_scales: bool = False,
) -> tuple[list[PlanEntry], dict[str, Any]]:
    source_config = read_json(source.root / "config.json")
    expert_count = int(source_config["text_config"]["num_experts"])
    entries: list[PlanEntry] = []
    skipped: list[str] = []

    for name, target in sorted(reference.tensors.items()):
        if target.file == KV_SCALE_SHARD:
            if include_kv_scales:
                raise RuntimeError(
                    "KV activation scales require calibration prompts and cannot be derived from checkpoint weights"
                )
            skipped.append(name)
            continue

        direct = source.tensors.get(name)
        if direct is not None and direct.dtype == target.dtype and direct.shape == target.shape:
            entries.append(
                PlanEntry(name, target.file, target.dtype, target.shape, "COPY_SOURCE", (name,), "byte-identical")
            )
            continue

        fused = re.fullmatch(
            r"(model\.language_model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)_(packed|scale)",
            name,
        )
        if fused:
            prefix, projection, kind = fused.groups()
            sources = _expert_sources(prefix, projection, expert_count)
            action = "FUSE_MXFP4_WEIGHT" if kind == "packed" else "FUSE_MXFP4_SCALE"
            entries.append(
                PlanEntry(
                    name,
                    target.file,
                    target.dtype,
                    target.shape,
                    action,
                    sources,
                    f"{expert_count} routed experts; FP8 dequant then MXFP4 group {MXFP4_GROUP}",
                )
            )
            continue

        if name.endswith(".weight_packed"):
            source_name = name.removesuffix("_packed")
            if source_name in source.tensors:
                entries.append(
                    PlanEntry(
                        name,
                        target.file,
                        target.dtype,
                        target.shape,
                        "MXFP4_WEIGHT",
                        (source_name,),
                        f"SOURCE BF16 to MXFP4 group {MXFP4_GROUP}",
                    )
                )
                continue

        if name.endswith(".weight_scale"):
            source_name = name.removesuffix("_scale")
            if source_name in source.tensors:
                entries.append(
                    PlanEntry(
                        name,
                        target.file,
                        target.dtype,
                        target.shape,
                        "MXFP4_SCALE",
                        (source_name,),
                        f"generated E8M0 scale, group {MXFP4_GROUP}",
                    )
                )
                continue

        if name.endswith(".weight_scale_inv"):
            source_name = name.removesuffix("_scale_inv")
            source_weight = source.tensors.get(source_name)
            if source_weight is not None and source_weight.dtype == "BF16":
                entries.append(
                    PlanEntry(
                        name,
                        target.file,
                        target.dtype,
                        target.shape,
                        "FP8_SCALE",
                        (source_name,),
                        f"generated SOURCE block-FP8 scale, {FP8_BLOCK}x{FP8_BLOCK}",
                    )
                )
                continue

        source_weight = source.tensors.get(name)
        if (
            target.dtype == "F8_E4M3"
            and source_weight is not None
            and source_weight.dtype == "BF16"
            and source_weight.shape == target.shape
        ):
            entries.append(
                PlanEntry(
                    name,
                    target.file,
                    target.dtype,
                    target.shape,
                    "FP8_WEIGHT",
                    (name,),
                    f"SOURCE BF16 to block-FP8, {FP8_BLOCK}x{FP8_BLOCK}",
                )
            )
            continue

        entries.append(
            PlanEntry(name, target.file, target.dtype, target.shape, "ERROR", (), "no source-derived rule")
        )

    missing_sources: list[str] = []
    for entry in entries:
        for source_name in entry.source_names:
            if source_name not in source.tensors:
                missing_sources.append(source_name)
    consumed = {name for entry in entries for name in entry.source_names}
    unconsumed = sorted(set(source.tensors) - consumed)
    counts = Counter(entry.action for entry in entries)
    by_file: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        by_file[entry.target_file][entry.action] += 1
    summary = {
        "source": str(source.root),
        "reference_format_oracle": str(reference.root),
        "reference_tensor_payloads_read": False,
        "source_tensors": len(source.tensors),
        "reference_tensors": len(reference.tensors),
        "planned_output_tensors": len(entries),
        "actions": dict(sorted(counts.items())),
        "target_shards": len(by_file),
        "target_shard_actions": {name: dict(sorted(value.items())) for name, value in sorted(by_file.items())},
        "skipped_reference_kv_activation_scales": skipped,
        "kv_scale_policy": (
            "Omitted: reference file contains calibrated activation values, not checkpoint-weight transforms; "
            "RDNA4 validation uses BF16/auto KV cache and reference config declares kv_cache_scheme=null."
        ),
        "missing_source_tensors": sorted(set(missing_sources)),
        "unconsumed_source_tensors": unconsumed,
        "source_inventory_errors": source.errors,
        "reference_inventory_errors": reference.errors,
    }
    return entries, summary


def assert_safe_plan(source: Inventory, reference: Inventory, entries: list[PlanEntry], summary: dict[str, Any]) -> None:
    failures: list[str] = []
    if source.errors:
        failures.append(f"SOURCE inventory errors={len(source.errors)}")
    if reference.errors:
        failures.append(f"REFERENCE inventory errors={len(reference.errors)}")
    if summary["actions"].get("ERROR"):
        failures.append(f"unsupported target tensors={summary['actions']['ERROR']}")
    if summary["missing_source_tensors"]:
        failures.append(f"missing source tensors={len(summary['missing_source_tensors'])}")
    if summary["unconsumed_source_tensors"]:
        failures.append(f"unconsumed source tensors={len(summary['unconsumed_source_tensors'])}")
    source_config = read_json(source.root / "config.json")
    reference_config = read_json(reference.root / "config.json")
    if _without_quantization_config(source_config) != _without_quantization_config(reference_config):
        failures.append("model semantics differ outside quantization_config")
    if failures:
        raise RuntimeError("Unsafe conversion plan: " + "; ".join(failures))


def write_plan(entries: list[PlanEntry], summary: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "plan-summary.json", summary)
    with (report_dir / "tensor-plan.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["target_name", "target_file", "target_dtype", "target_shape", "action", "source_names", "detail"]
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for entry in entries:
            row = asdict(entry)
            row["target_shape"] = json.dumps(entry.target_shape, separators=(",", ":"))
            row["source_names"] = _source_tuple_text(entry.source_names)
            writer.writerow(row)


def _auxiliary_files(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix == ".safetensors" or path.name == "model.safetensors.index.json":
            continue
        result[path.name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def run_dry_run(args: argparse.Namespace) -> int:
    source = load_inventory(args.source)
    reference = load_inventory(args.reference)
    entries, summary = build_plan(source, reference)
    write_plan(entries, summary, args.report_dir)
    write_json(
        args.report_dir / "config-comparison.json",
        {
            "semantics_equal_without_quantization_config": _without_quantization_config(
                read_json(source.root / "config.json")
            )
            == _without_quantization_config(read_json(reference.root / "config.json")),
            "source_quantization_config": read_json(source.root / "config.json").get("quantization_config"),
            "reference_quantization_config": read_json(reference.root / "config.json").get("quantization_config"),
            "source_auxiliary_files": _auxiliary_files(source.root),
        },
    )
    assert_safe_plan(source, reference, entries, summary)
    print(json.dumps({key: summary[key] for key in ("source_tensors", "reference_tensors", "planned_output_tensors", "actions", "target_shards", "kv_scale_policy")}, indent=2))
    print(f"Plan: {args.report_dir / 'tensor-plan.tsv'}")
    return 0


class TensorLoader:
    def __init__(self, inventory: Inventory, stack: ExitStack) -> None:
        self.inventory = inventory
        self.stack = stack
        self.handles: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        from safetensors import safe_open

        info = self.inventory.tensors[name]
        if info.file not in self.handles:
            self.handles[info.file] = self.stack.enter_context(
                safe_open(self.inventory.root / info.file, framework="pt", device="cpu")
            )
        return self.handles[info.file].get_tensor(name)


def _quantize_fp8_block(weight: Any, device: str) -> tuple[Any, Any]:
    import torch

    if weight.ndim != 2:
        raise ValueError(f"FP8 block quantization requires a matrix, got {tuple(weight.shape)}")
    rows, columns = weight.shape
    padded_rows = math.ceil(rows / FP8_BLOCK) * FP8_BLOCK
    padded_columns = math.ceil(columns / FP8_BLOCK) * FP8_BLOCK
    value = weight.to(device=device, dtype=torch.bfloat16)
    if (padded_rows, padded_columns) != (rows, columns):
        padded = torch.zeros((padded_rows, padded_columns), device=device, dtype=torch.bfloat16)
        padded[:rows, :columns] = value
        value = padded
    blocks = value.reshape(
        padded_rows // FP8_BLOCK,
        FP8_BLOCK,
        padded_columns // FP8_BLOCK,
        FP8_BLOCK,
    ).permute(0, 2, 1, 3)
    max_abs = blocks.float().abs().amax(dim=(2, 3))
    scale = (max_abs / FP8_MAX).to(torch.bfloat16)
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized_blocks = (blocks.float() / safe_scale.float()[:, :, None, None]).to(torch.float8_e4m3fn)
    quantized_blocks = torch.where(
        (scale != 0)[:, :, None, None], quantized_blocks, torch.zeros_like(quantized_blocks)
    )
    quantized = quantized_blocks.permute(0, 2, 1, 3).reshape(padded_rows, padded_columns)[:rows, :columns]
    return quantized.contiguous().cpu(), scale.contiguous().cpu()


def _quantize_mxfp4(weight: Any, device: str) -> tuple[Any, Any]:
    import torch
    from quark.torch.kernel.mx.triton import DequantScaleRoundingMode, downcast_to_mxfp

    if weight.shape[-1] % MXFP4_GROUP:
        raise ValueError(f"MXFP4 input K is not divisible by {MXFP4_GROUP}: {tuple(weight.shape)}")
    original_shape = tuple(weight.shape)
    matrix = weight.reshape(-1, original_shape[-1]).to(device=device, dtype=torch.bfloat16).contiguous()
    packed, scale, _ = downcast_to_mxfp(
        matrix,
        torch.uint8,
        axis=-1,
        DEQUANT_SCALE_ROUNDING_MODE=DequantScaleRoundingMode.EVEN,
    )
    packed_shape = (*original_shape[:-1], original_shape[-1] // 2)
    scale_shape = (*original_shape[:-1], original_shape[-1] // MXFP4_GROUP)
    return packed.reshape(packed_shape).contiguous().cpu(), scale.reshape(scale_shape).contiguous().cpu()


def _dequantize_source_fp8(weight: Any, scale: Any, device: str) -> Any:
    import torch
    from quark.torch.quantization.file2file_quantization import _weight_dequant_fp8

    return _weight_dequant_fp8(
        weight.to(device).contiguous(),
        scale.to(device).contiguous(),
        model_dtype=torch.bfloat16,
    )


def _fused_mxfp4(
    entry: PlanEntry,
    loader: TensorLoader,
    device: str,
    expert_chunk: int,
) -> tuple[Any, Any]:
    import torch

    match = re.fullmatch(
        r"(model\.language_model\.layers\.\d+\.mlp\.experts)\.(gate_up_proj|down_proj)_packed",
        entry.target_name,
    )
    if not match:
        raise ValueError(f"Invalid fused target: {entry.target_name}")
    prefix, projection = match.groups()
    expert_count = entry.target_shape[0]
    packed_result = torch.empty(entry.target_shape, dtype=torch.uint8, device="cpu")
    scale_name = entry.target_name.removesuffix("_packed") + "_scale"
    scale_columns = entry.target_shape[-1] * 2 // MXFP4_GROUP
    scale_result = torch.empty((*entry.target_shape[:-1], scale_columns), dtype=torch.uint8, device="cpu")
    projections = ("gate_proj", "up_proj") if projection == "gate_up_proj" else ("down_proj",)

    for start in range(0, expert_count, expert_chunk):
        stop = min(start + expert_chunk, expert_count)
        real_experts: list[Any] = []
        for expert in range(start, stop):
            pieces: list[Any] = []
            for item in projections:
                source_name = f"{prefix}.{expert}.{item}.weight"
                pieces.append(
                    _dequantize_source_fp8(
                        loader.get(source_name),
                        loader.get(source_name + "_scale_inv"),
                        device,
                    )
                )
            real_experts.append(torch.cat(pieces, dim=0) if len(pieces) == 2 else pieces[0])
        real = torch.stack(real_experts, dim=0)
        packed, scale = _quantize_mxfp4(real, device)
        packed_result[start:stop].copy_(packed)
        scale_result[start:stop].copy_(scale)
        del pieces, real_experts, real, packed, scale
        torch.cuda.empty_cache()
    expected_scale_name = entry.target_name.removesuffix("_packed") + "_scale"
    if expected_scale_name != scale_name:
        raise AssertionError(expected_scale_name)
    return packed_result, scale_result


def _copy_auxiliary(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        if not item.is_file() or item.suffix == ".safetensors" or item.name in {
            "config.json",
            "model.safetensors.index.json",
        }:
            continue
        shutil.copy2(item, destination / item.name)


def _write_output_config(source: Path, reference: Path, destination: Path) -> None:
    source_config = read_json(source / "config.json")
    reference_config = read_json(reference / "config.json")
    source_config["quantization_config"] = copy.deepcopy(reference_config["quantization_config"])
    write_json(destination / "config.json", source_config)


def _planned_index(entries: list[PlanEntry]) -> dict[str, Any]:
    weight_map = {entry.target_name: entry.target_file for entry in entries}
    total_size = sum(math.prod(entry.target_shape) * DTYPE_BITS[entry.target_dtype] // 8 for entry in entries)
    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def _conversion_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in ("amd-quark", "compressed-tensors", "safetensors", "torch"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


def _prepare_partial(output: Path, resume: bool) -> Path:
    partial = output.parent / f".{output.name}.partial"
    if output.exists():
        raise FileExistsError(f"OUTPUT already exists: {output}")
    if partial.exists() and not resume:
        raise FileExistsError(f"Partial output exists; inspect it or pass --resume: {partial}")
    partial.mkdir(parents=False, exist_ok=resume)
    return partial


def run_convert(args: argparse.Namespace) -> int:
    source = load_inventory(args.source)
    reference = load_inventory(args.reference)
    entries, summary = build_plan(source, reference)
    write_plan(entries, summary, args.report_dir)
    assert_safe_plan(source, reference, entries, summary)
    source_path, reference_path, output_path = source.root, reference.root, args.output.resolve()
    if len({source_path, reference_path, output_path}) != 3:
        raise RuntimeError("SOURCE, REFERENCE and OUTPUT must be distinct")
    partial = _prepare_partial(output_path, args.resume)
    _copy_auxiliary(source_path, partial)
    _write_output_config(source_path, reference_path, partial)
    index = _planned_index(entries)
    write_json(partial / "model.safetensors.index.json", index)

    from safetensors.torch import save_file

    entries_by_file: dict[str, list[PlanEntry]] = defaultdict(list)
    for entry in entries:
        entries_by_file[entry.target_file].append(entry)
    shard_names = sorted(entries_by_file)
    for shard_number, shard_name in enumerate(shard_names, 1):
        destination = partial / shard_name
        if args.resume and destination.is_file():
            print(f"[{shard_number:03d}/{len(shard_names):03d}] resume {shard_name}", flush=True)
            continue
        tensors: dict[str, Any] = {}
        entry_by_name = {entry.target_name: entry for entry in entries_by_file[shard_name]}
        with ExitStack() as stack:
            loader = TensorLoader(source, stack)
            for entry in sorted(entries_by_file[shard_name], key=lambda item: item.target_name):
                if entry.target_name in tensors:
                    continue
                if entry.action == "COPY_SOURCE":
                    tensors[entry.target_name] = loader.get(entry.source_names[0])
                elif entry.action == "FP8_WEIGHT":
                    weight, scale = _quantize_fp8_block(loader.get(entry.source_names[0]), args.device)
                    scale_name = entry.target_name + "_scale_inv"
                    if scale_name not in entry_by_name:
                        raise RuntimeError(f"Missing planned FP8 scale: {scale_name}")
                    tensors[entry.target_name] = weight
                    tensors[scale_name] = scale
                elif entry.action == "MXFP4_WEIGHT":
                    packed, scale = _quantize_mxfp4(loader.get(entry.source_names[0]), args.device)
                    scale_name = entry.target_name.removesuffix("_packed") + "_scale"
                    if scale_name not in entry_by_name:
                        raise RuntimeError(f"Missing planned MXFP4 scale: {scale_name}")
                    tensors[entry.target_name] = packed
                    tensors[scale_name] = scale
                elif entry.action == "FUSE_MXFP4_WEIGHT":
                    packed, scale = _fused_mxfp4(entry, loader, args.device, args.expert_chunk)
                    scale_name = entry.target_name.removesuffix("_packed") + "_scale"
                    if scale_name not in entry_by_name:
                        raise RuntimeError(f"Missing planned fused scale: {scale_name}")
                    tensors[entry.target_name] = packed
                    tensors[scale_name] = scale
                elif entry.action in {"FP8_SCALE", "MXFP4_SCALE", "FUSE_MXFP4_SCALE"}:
                    if entry.target_name not in tensors:
                        raise RuntimeError(f"Generated scale appeared without its weight: {entry.target_name}")
                else:
                    raise RuntimeError(f"Unsupported action {entry.action}: {entry.target_name}")

        expected_names = set(entry_by_name)
        if set(tensors) != expected_names:
            raise RuntimeError(
                f"Generated tensor set mismatch in {shard_name}: missing={sorted(expected_names - set(tensors))} "
                f"unexpected={sorted(set(tensors) - expected_names)}"
            )
        for name, tensor in tensors.items():
            expected = reference.tensors[name]
            if tuple(tensor.shape) != expected.shape:
                raise RuntimeError(f"Shape mismatch before save: {name}: {tuple(tensor.shape)} != {expected.shape}")
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_file(
            {name: tensor.contiguous() for name, tensor in tensors.items()},
            temporary,
            metadata=reference.shard_metadata.get(shard_name) or {"format": "pt"},
        )
        os.replace(temporary, destination)
        del tensors
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
        print(f"[{shard_number:03d}/{len(shard_names):03d}] wrote {shard_name}", flush=True)

    provenance = {
        "source": str(source_path),
        "reference_format_oracle": str(reference_path),
        "output": str(output_path),
        "reference_tensor_payloads_read": False,
        "reference_weights_or_scales_copied": False,
        "conversion": {
            "mxfp4": "AMD Quark RTN, E2M1 packed low nibble first, E8M0 scale, group_size=32",
            "mxfp4_scale_rounding": "even",
            "fp8": "E4M3, symmetric static weights, 128x128 block, BF16 scale_inv",
            "source_fp8_dequantization": "weight * source BF16 weight_scale_inv per 128x128 block",
            "expert_chunk": args.expert_chunk,
            "device": args.device,
        },
        "versions": _conversion_versions(),
        "source_config_sha256": sha256_file(source_path / "config.json"),
        "source_index_sha256": sha256_file(source_path / "model.safetensors.index.json"),
        "reference_config_sha256": sha256_file(reference_path / "config.json"),
        "reference_index_sha256": sha256_file(reference_path / "model.safetensors.index.json"),
        "output_index_sha256": sha256_file(partial / "model.safetensors.index.json"),
        "summary": summary,
    }
    write_json(partial / "conversion-provenance.json", provenance)
    os.replace(partial, output_path)
    print(f"Conversion complete: {output_path}")
    return 0


def _hash_tensor_data(root: Path, info: TensorInfo, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    remaining = info.nbytes
    with (root / info.file).open("rb") as handle:
        handle.seek(info.data_start + info.offsets[0])
        while remaining:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                raise IOError(f"Unexpected EOF: {info.name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _validate_auxiliary(source: Path, output: Path) -> dict[str, Any]:
    ignored = {"config.json", "model.safetensors.index.json", "conversion-provenance.json"}
    source_files = _auxiliary_files(source)
    output_files = _auxiliary_files(output)
    expected = sorted(set(source_files) - ignored)
    return {
        "checked": len(expected),
        "missing": [name for name in expected if name not in output_files],
        "sha256_mismatch": [
            name
            for name in expected
            if name in output_files and source_files[name]["sha256"] != output_files[name]["sha256"]
        ],
    }


def run_validate(args: argparse.Namespace) -> int:
    import torch
    from safetensors import safe_open

    source = load_inventory(args.source)
    reference = load_inventory(args.reference)
    output = load_inventory(args.output)
    entries, plan_summary = build_plan(source, reference)
    assert_safe_plan(source, reference, entries, plan_summary)
    expected = {entry.target_name: entry for entry in entries}
    missing = sorted(set(expected) - set(output.tensors))
    unexpected = sorted(set(output.tensors) - set(expected))
    shape_mismatch = sorted(
        name for name in set(expected) & set(output.tensors) if expected[name].target_shape != output.tensors[name].shape
    )
    dtype_mismatch = sorted(
        name for name in set(expected) & set(output.tensors) if expected[name].target_dtype != output.tensors[name].dtype
    )

    nonfinite: list[str] = []
    checked_finite = 0
    for shard_name in sorted(output.shard_metadata):
        with safe_open(output.root / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if output.tensors[name].dtype not in FLOAT_DTYPES:
                    continue
                tensor = handle.get_tensor(name)
                checked_finite += tensor.numel()
                if not bool(torch.isfinite(tensor.float()).all()):
                    nonfinite.append(name)

    copied_hash_mismatch: list[str] = []
    copied_hashes_checked = 0
    if not args.skip_hashes:
        for entry in entries:
            if entry.action != "COPY_SOURCE":
                continue
            copied_hashes_checked += 1
            if _hash_tensor_data(source.root, source.tensors[entry.source_names[0]]) != _hash_tensor_data(
                output.root, output.tensors[entry.target_name]
            ):
                copied_hash_mismatch.append(entry.target_name)

    source_config = read_json(source.root / "config.json")
    reference_config = read_json(reference.root / "config.json")
    output_config = read_json(output.root / "config.json")
    auxiliary = _validate_auxiliary(source.root, output.root)
    actual_total_size = sum(info.nbytes for info in output.tensors.values())
    validation = {
        "source": str(source.root),
        "reference_format_oracle": str(reference.root),
        "output": str(output.root),
        "source_tensors": len(source.tensors),
        "reference_tensors": len(reference.tensors),
        "output_tensors": len(output.tensors),
        "expected_output_tensors": len(expected),
        "output_shards": len(output.shard_metadata),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "nonfinite_tensors": nonfinite,
        "finite_elements_checked": checked_finite,
        "copied_tensor_hashes_checked": copied_hashes_checked,
        "copied_tensor_hash_mismatch": copied_hash_mismatch,
        "source_semantics_preserved": _without_quantization_config(source_config)
        == _without_quantization_config(output_config),
        "quantization_config_matches_reference": output_config.get("quantization_config")
        == reference_config.get("quantization_config"),
        "output_index_total_size": output.index_metadata.get("total_size"),
        "actual_tensor_bytes": actual_total_size,
        "auxiliary_files": auxiliary,
        "source_inventory_errors": source.errors,
        "output_inventory_errors": output.errors,
        "reference_payloads_read": False,
        "kv_scale_policy": plan_summary["kv_scale_policy"],
    }
    write_json(args.report_dir / "checkpoint-validation.json", validation)
    failures = {
        key: value
        for key, value in {
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatch": shape_mismatch,
            "dtype_mismatch": dtype_mismatch,
            "nonfinite": nonfinite,
            "copied_hash_mismatch": copied_hash_mismatch,
            "source_inventory_errors": source.errors,
            "output_inventory_errors": output.errors,
            "auxiliary_missing": auxiliary["missing"],
            "auxiliary_hash_mismatch": auxiliary["sha256_mismatch"],
        }.items()
        if value
    }
    if validation["output_index_total_size"] != actual_total_size:
        failures["index_total_size"] = [validation["output_index_total_size"], actual_total_size]
    if not validation["source_semantics_preserved"]:
        failures["config"] = ["source semantics not preserved"]
    if not validation["quantization_config_matches_reference"]:
        failures.setdefault("config", []).append("quantization_config differs from format oracle")
    print(json.dumps(validation, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"Validation failed: {json.dumps(failures, sort_keys=True)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    common.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    common.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser("dry-run", parents=[common])
    dry_run.set_defaults(func=run_dry_run)
    convert = subparsers.add_parser("convert", parents=[common])
    convert.add_argument("--device", default="cuda:0")
    convert.add_argument("--expert-chunk", type=int, default=8)
    convert.add_argument("--resume", action="store_true")
    convert.set_defaults(func=run_convert)
    validate = subparsers.add_parser("validate", parents=[common])
    validate.add_argument("--skip-hashes", action="store_true")
    validate.set_defaults(func=run_validate)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "expert_chunk", 1) < 1:
        raise SystemExit("--expert-chunk must be positive")
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
