#!/usr/bin/env python3
"""Convert a block-FP8 GLM-5.3 checkpoint to a reference Quark MXFP4 layout.

The source checkpoint is the sole source of model tensors.  The reference is
read only and is used to classify the target representation and to obtain the
Quark configuration.  Tensor inventories are read directly from safetensors
headers, so the dry-run does not load model data.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SOURCE = Path("/mnt/ai/models/glm/GLM-5.3-Flash-UNCENSORED-FP8")
DEFAULT_REFERENCE = Path("/mnt/ai/models/glm/GLM-5.3-Flash-Quark-MXFP4")
DEFAULT_OUTPUT = Path("/mnt/ai/models/glm/GLM-5.3-Flash-UNCENSORED-Quark-MXFP4")
DEFAULT_REPORT_DIR = Path("logs/conversion/glm53-uncensored-quark-mxfp4")

SCALE_INV_SUFFIX = "_scale_inv"
QUARK_SCALE_SUFFIX = "_scale"
FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"}
E8M0_DTYPES = {"U8", "F8_E8M0"}
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
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
            raise ValueError(f"Invalid safetensors header length {header_length}: {path}")
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid safetensors JSON header: {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not an object: {path}")
    return header, 8 + header_length


def load_inventory(root: Path) -> Inventory:
    root = root.resolve()
    index_path = root / "model.safetensors.index.json"
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing index: {index_path}")

    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid weight_map in {index_path}")

    errors: list[str] = []
    tensors: dict[str, TensorInfo] = {}
    shard_metadata: dict[str, dict[str, str] | None] = {}
    shard_names = sorted({str(name) for name in weight_map.values()})
    actual_shards = sorted(path.name for path in root.glob("*.safetensors"))
    unindexed_shards = sorted(set(actual_shards) - set(shard_names))
    if unindexed_shards:
        errors.append(f"Unindexed safetensors shards: {unindexed_shards}")

    for shard_name in shard_names:
        shard_path = root / shard_name
        if not shard_path.is_file():
            errors.append(f"Index points to missing shard: {shard_name}")
            continue
        try:
            header, data_start = _read_safetensors_header(shard_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        metadata = header.pop("__metadata__", None)
        shard_metadata[shard_name] = metadata
        file_size = shard_path.stat().st_size
        for tensor_name, raw_info in header.items():
            if tensor_name in tensors:
                errors.append(f"Duplicate tensor across shards: {tensor_name}")
                continue
            if not isinstance(raw_info, dict):
                errors.append(f"Invalid tensor header for {tensor_name} in {shard_name}")
                continue
            try:
                dtype = str(raw_info["dtype"])
                shape = tuple(int(dim) for dim in raw_info["shape"])
                offsets = tuple(int(offset) for offset in raw_info["data_offsets"])
                if len(offsets) != 2:
                    raise ValueError("data_offsets must contain two integers")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"Invalid tensor header for {tensor_name} in {shard_name}: {exc}")
                continue
            if offsets[0] < 0 or offsets[1] < offsets[0] or data_start + offsets[1] > file_size:
                errors.append(f"Out-of-range data offsets for {tensor_name} in {shard_name}: {offsets}")
            dtype_bits = DTYPE_BITS.get(dtype)
            if dtype_bits is None:
                errors.append(f"Unknown safetensors dtype {dtype} for {tensor_name}")
            else:
                expected_bits = math.prod(shape) * dtype_bits
                if expected_bits != (offsets[1] - offsets[0]) * 8:
                    errors.append(
                        f"Byte-size mismatch for {tensor_name}: shape={shape} dtype={dtype} "
                        f"bytes={offsets[1] - offsets[0]}"
                    )
            tensors[tensor_name] = TensorInfo(
                name=tensor_name,
                file=shard_name,
                dtype=dtype,
                shape=shape,
                offsets=(offsets[0], offsets[1]),
                data_start=data_start,
            )

    header_names = set(tensors)
    index_names = set(weight_map)
    for name in sorted(index_names - header_names):
        errors.append(f"Index entry missing from shard headers: {name}")
    for name in sorted(header_names - index_names):
        errors.append(f"Tensor missing from index: {name}")
    for name in sorted(index_names & header_names):
        if str(weight_map[name]) != tensors[name].file:
            errors.append(
                f"Index shard mismatch for {name}: index={weight_map[name]} header={tensors[name].file}"
            )

    return Inventory(
        root=root,
        tensors=tensors,
        shard_metadata=shard_metadata,
        index_metadata=index.get("metadata", {}),
        errors=errors,
    )


def component_for(name: str, mtp_layer_indices: frozenset[int] = frozenset()) -> str:
    lowered = name.lower()
    if "visual" in lowered or ".vision" in lowered:
        return "VISION"
    layer_match = re.search(r"\.language_model\.layers\.(\d+)\.", lowered)
    if (
        "mtp" in lowered
        or "nextn_predict" in lowered
        or "next_n" in lowered
        or (layer_match is not None and int(layer_match.group(1)) in mtp_layer_indices)
    ):
        return "MTP"
    if "lm_head" in lowered:
        return "LM_HEAD"
    if "embed_tokens" in lowered or ".embedding" in lowered or ".embeddings" in lowered:
        return "EMBEDDING"
    if ".shared_experts." in lowered or ".shared_expert." in lowered:
        return "MOE_SHARED"
    if ".experts." in lowered:
        return "MOE_ROUTED"
    if ".mlp.gate." in lowered or lowered.endswith(".mlp.gate.weight") or ".router" in lowered:
        return "ROUTER"
    if ".self_attn." in lowered or ".attention." in lowered:
        return "ATTENTION"
    if ".mlp." in lowered:
        return "DENSE_MLP"
    return "OTHER"


def _shape_text(shape: tuple[int, ...] | None) -> str:
    return "" if shape is None else json.dumps(shape, separators=(",", ":"))


def _scale_names(inventory: Inventory, weight_name: str) -> list[str]:
    candidates = [
        weight_name + SCALE_INV_SUFFIX,
        weight_name + QUARK_SCALE_SUFFIX,
        weight_name.removesuffix(".weight") + ".scale",
    ]
    return [candidate for candidate in candidates if candidate in inventory.tensors]


def _target_name(source_name: str, reference: Inventory) -> str | None:
    if source_name in reference.tensors:
        return source_name
    if source_name.endswith(SCALE_INV_SUFFIX):
        candidate = source_name[: -len(SCALE_INV_SUFFIX)] + QUARK_SCALE_SUFFIX
        if candidate in reference.tensors:
            return candidate
    return None


def _is_linear_weight(name: str, info: TensorInfo) -> bool:
    if not name.endswith(".weight") or len(info.shape) < 2:
        return False
    module_name = name.removesuffix(".weight")
    return not module_name.endswith("norm") and "embed" not in module_name


def _is_mxfp4_target(reference: Inventory, weight_name: str) -> bool:
    weight = reference.tensors.get(weight_name)
    scale = reference.tensors.get(weight_name + QUARK_SCALE_SUFFIX)
    if weight is None or scale is None:
        return False
    return weight.dtype == "U8" and scale.dtype in E8M0_DTYPES


def build_plan(source: Inventory, reference: Inventory) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    primary_actions: dict[str, str] = {}
    text_config = read_json(source.root / "config.json").get("text_config", {})
    hidden_layers = int(text_config.get("num_hidden_layers", 0))
    nextn_layers = int(text_config.get("num_nextn_predict_layers", 0))
    mtp_layer_indices = frozenset(range(hidden_layers, hidden_layers + nextn_layers))

    for name in sorted(source.tensors):
        source_info = source.tensors[name]
        target_name = _target_name(name, reference)
        target_info = reference.tensors.get(target_name) if target_name is not None else None
        source_scales = _scale_names(source, name) if name.endswith(".weight") else []
        reference_scales = _scale_names(reference, name) if name.endswith(".weight") else []
        action = "ERROR"
        target_representation = "MISSING"
        detail = "No logical counterpart in REFERENCE"

        if name.endswith(SCALE_INV_SUFFIX):
            base_name = name[: -len(SCALE_INV_SUFFIX)]
            if target_info is not None:
                action = "SPECIAL_HANDLING"
                target_representation = "MXFP4_SCALE_GENERATED" if _is_mxfp4_target(reference, base_name) else "FP8_SCALE_RENAMED"
                detail = (
                    "Consumed during FP8 dequantization; Quark generates e8m0 scale"
                    if _is_mxfp4_target(reference, base_name)
                    else "Rename to Quark weight_scale and preserve source bytes"
                )
        elif name.endswith(".weight") and target_info is not None:
            if _is_mxfp4_target(reference, name):
                scale = reference.tensors[name + QUARK_SCALE_SUFFIX]
                valid_shape = (
                    len(source_info.shape) == len(target_info.shape) == len(scale.shape) == 2
                    and source_info.shape[0] == target_info.shape[0] == scale.shape[0]
                    and source_info.shape[1] == target_info.shape[1] * 2
                    and source_info.shape[1] == scale.shape[1] * 32
                )
                if source_info.dtype in FP8_DTYPES and source_scales and valid_shape:
                    action = "CONVERT_FP8_TO_MXFP4"
                    target_representation = f"U8_PACKED_FP4{target_info.shape}+{scale.dtype}_SCALE{scale.shape}"
                    detail = "Dequantize SOURCE block-FP8, then Quark OCP MXFP4 quantize and pack"
                else:
                    detail = "REFERENCE requests MXFP4, but SOURCE dtype/scale or target shapes are incompatible"
            elif source_info.shape == target_info.shape and source_info.dtype == target_info.dtype:
                if source_info.dtype in FP8_DTYPES:
                    source_scale_infos = [source.tensors[item] for item in source_scales]
                    reference_scale_infos = [reference.tensors[item] for item in reference_scales]
                    scale_compatible = (
                        len(source_scale_infos) == len(reference_scale_infos) == 1
                        and source_scale_infos[0].shape == reference_scale_infos[0].shape
                        and source_scale_infos[0].dtype == reference_scale_infos[0].dtype
                    )
                    if scale_compatible:
                        action = "PRESERVE_SOURCE_FP8"
                        target_representation = f"{target_info.dtype}{target_info.shape}+SOURCE_BLOCK_SCALE"
                        detail = "Keep SOURCE FP8 weight and scale bytes; rename scale for Quark"
                    else:
                        detail = "FP8 weight matches but its SOURCE/REFERENCE scale representation differs"
                elif source_info.dtype == "BF16":
                    action = "PRESERVE_SOURCE_BF16"
                    target_representation = f"BF16{target_info.shape}"
                    detail = "Copy SOURCE tensor unchanged"
                else:
                    action = "PRESERVE_SOURCE_OTHER"
                    target_representation = f"{target_info.dtype}{target_info.shape}"
                    detail = "Copy SOURCE tensor unchanged"
            else:
                detail = "SOURCE and REFERENCE weight representations do not match a supported rule"
            primary_actions[name] = action
        elif target_info is not None and source_info.shape == target_info.shape and source_info.dtype == target_info.dtype:
            if source_info.dtype == "BF16":
                action = "PRESERVE_SOURCE_BF16"
            elif source_info.dtype in FP8_DTYPES:
                action = "PRESERVE_SOURCE_FP8"
            else:
                action = "PRESERVE_SOURCE_OTHER"
            target_representation = f"{target_info.dtype}{target_info.shape}"
            detail = "Copy SOURCE tensor unchanged"
        elif target_info is not None:
            detail = "SOURCE and REFERENCE shape/dtype mismatch"

        rows.append(
            {
                "tensor_name": name,
                "source_file": source_info.file,
                "source_shape": _shape_text(source_info.shape),
                "source_dtype": source_info.dtype,
                "source_scale_tensors": json.dumps(source_scales, separators=(",", ":")),
                "reference_name": target_name or "",
                "reference_file": target_info.file if target_info else "",
                "reference_shape": _shape_text(target_info.shape if target_info else None),
                "reference_dtype": target_info.dtype if target_info else "",
                "reference_scale_tensors": json.dumps(reference_scales, separators=(",", ":")),
                "classification": component_for(name, mtp_layer_indices),
                "action": action,
                "target_representation": target_representation,
                "detail": detail,
            }
        )

    mapped_reference_names = [row["reference_name"] for row in rows if row["reference_name"]]
    duplicate_targets = sorted(name for name, count in Counter(mapped_reference_names).items() if count > 1)
    missing_reference_tensors = sorted(set(reference.tensors) - set(mapped_reference_names))
    action_counts = Counter(row["action"] for row in rows)
    component_actions: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        component_actions[row["classification"]][row["action"]] += 1

    summary = {
        "source": str(source.root),
        "reference": str(reference.root),
        "total_source_tensors": len(source.tensors),
        "total_reference_tensors": len(reference.tensors),
        "converted_to_mxfp4": sum(action == "CONVERT_FP8_TO_MXFP4" for action in primary_actions.values()),
        "preserved_fp8": sum(action == "PRESERVE_SOURCE_FP8" for action in primary_actions.values()),
        "preserved_bf16": action_counts["PRESERVE_SOURCE_BF16"],
        "preserved_other": action_counts["PRESERVE_SOURCE_OTHER"],
        "special": action_counts["SPECIAL_HANDLING"],
        "additional_mxfp4_scale_tensors": sum(
            action == "CONVERT_FP8_TO_MXFP4" for action in primary_actions.values()
        ),
        "unknown_or_error": action_counts["ERROR"],
        "missing_reference_tensors": missing_reference_tensors,
        "duplicate_target_mappings": duplicate_targets,
        "source_inventory_errors": source.errors,
        "reference_inventory_errors": reference.errors,
        "actions": dict(sorted(action_counts.items())),
        "components": {
            component: dict(sorted(counts.items())) for component, counts in sorted(component_actions.items())
        },
    }
    return rows, summary


def _without_quantization_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)

    def remove(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node.pop("quantization_config", None)
        for child in node.values():
            remove(child)

    remove(value)
    return value


def _auxiliary_inventory(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix == ".safetensors" or path.name == "model.safetensors.index.json":
            continue
        result[path.name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def config_comparison(source: Path, reference: Path) -> dict[str, Any]:
    source_config = read_json(source / "config.json")
    reference_config = read_json(reference / "config.json")
    source_semantics = _without_quantization_config(source_config)
    reference_semantics = _without_quantization_config(reference_config)
    source_aux = _auxiliary_inventory(source)
    reference_aux = _auxiliary_inventory(reference)
    shared_aux = sorted(set(source_aux) & set(reference_aux))
    return {
        "source_model_type": source_config.get("model_type"),
        "reference_model_type": reference_config.get("model_type"),
        "source_architectures": source_config.get("architectures"),
        "reference_architectures": reference_config.get("architectures"),
        "model_semantics_equal_without_quantization_config": source_semantics == reference_semantics,
        "source_quantization_config": source_config.get("quantization_config"),
        "reference_quantization_summary": _quantization_summary(reference_config.get("quantization_config", {})),
        "source_auxiliary_files": source_aux,
        "reference_auxiliary_files": reference_aux,
        "shared_auxiliary_file_hashes_equal": {
            name: source_aux[name]["sha256"] == reference_aux[name]["sha256"] for name in shared_aux
        },
        "source_only_auxiliary_files": sorted(set(source_aux) - set(reference_aux)),
        "reference_only_auxiliary_files": sorted(set(reference_aux) - set(source_aux)),
    }


def _quantization_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "quant_method": config.get("quant_method"),
        "quant_mode": config.get("quant_mode"),
        "version": config.get("version"),
        "global_quant_config": config.get("global_quant_config"),
        "exclude_count": len(config.get("exclude") or []),
        "layer_quant_config_count": len(config.get("layer_quant_config") or {}),
        "layer_quant_config_weight_formats": dict(
            sorted(
                Counter(
                    json.dumps(
                        {
                            "dtype": entry.get("weight", {}).get("dtype"),
                            "qscheme": entry.get("weight", {}).get("qscheme"),
                            "block_size": entry.get("weight", {}).get("block_size"),
                            "scale_type": entry.get("weight", {}).get("scale_type"),
                        },
                        sort_keys=True,
                    )
                    for entry in (config.get("layer_quant_config") or {}).values()
                ).items()
            )
        ),
        "kv_cache_quant_config": config.get("kv_cache_quant_config"),
        "kv_cache_post_rope": config.get("kv_cache_post_rope"),
        "export": config.get("export"),
    }


def write_dry_run(
    source: Inventory,
    reference: Inventory,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = report_dir / "dry-run.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(report_dir / "dry-run-summary.json", summary)
    write_json(report_dir / "config-comparison.json", config_comparison(source.root, reference.root))
    write_json(
        report_dir / "safetensors-metadata.json",
        {
            "source": source.shard_metadata,
            "reference": reference.shard_metadata,
            "source_index_metadata": source.index_metadata,
            "reference_index_metadata": reference.index_metadata,
        },
    )


def assert_plan_is_safe(summary: dict[str, Any]) -> None:
    failures: list[str] = []
    if summary["unknown_or_error"]:
        failures.append(f"unknown/error tensors: {summary['unknown_or_error']}")
    if summary["missing_reference_tensors"]:
        failures.append(f"unmapped REFERENCE tensors: {len(summary['missing_reference_tensors'])}")
    if summary["duplicate_target_mappings"]:
        failures.append(f"duplicate target mappings: {len(summary['duplicate_target_mappings'])}")
    if summary["source_inventory_errors"]:
        failures.append(f"SOURCE inventory errors: {len(summary['source_inventory_errors'])}")
    if summary["reference_inventory_errors"]:
        failures.append(f"REFERENCE inventory errors: {len(summary['reference_inventory_errors'])}")
    if failures:
        raise RuntimeError("Dry-run is not safe: " + "; ".join(failures))


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Total source tensors: {summary['total_source_tensors']}")
    print(f"Converted to MXFP4: {summary['converted_to_mxfp4']}")
    print(f"Preserved FP8: {summary['preserved_fp8']}")
    print(f"Preserved BF16: {summary['preserved_bf16']}")
    print(f"Additional MXFP4 scale tensors: {summary['additional_mxfp4_scale_tensors']}")
    print(f"Special: {summary['special']}")
    print(f"Unknown/error: {summary['unknown_or_error']}")
    print(f"Missing REFERENCE tensors: {len(summary['missing_reference_tensors'])}")


def run_dry_run(args: argparse.Namespace) -> int:
    source = load_inventory(args.source)
    reference = load_inventory(args.reference)
    rows, summary = build_plan(source, reference)
    write_dry_run(source, reference, rows, summary, args.report_dir)
    print_summary(summary)
    print(f"Detailed mapping: {args.report_dir / 'dry-run.tsv'}")
    assert_plan_is_safe(summary)
    comparison = config_comparison(source.root, reference.root)
    if not comparison["model_semantics_equal_without_quantization_config"]:
        raise RuntimeError("SOURCE and REFERENCE model semantics differ outside quantization_config")
    return 0


def _build_conversion_qconfig(
    source: Inventory,
    reference: Inventory,
    rows: list[dict[str, Any]],
) -> Any:
    from quark.torch.quantization.config.config import QConfig
    from quark.torch.quantization.file2file_quantization import _build_exclude_aware_quant_config

    reference_config = read_json(reference.root / "config.json")
    reference_quant_config = copy.deepcopy(reference_config["quantization_config"])
    reference_quant_config.pop("export", None)

    converted = {row["tensor_name"] for row in rows if row["action"] == "CONVERT_FP8_TO_MXFP4"}
    preserve_modules = sorted(
        info.name.removesuffix(".weight")
        for info in source.tensors.values()
        if _is_linear_weight(info.name, info) and info.name not in converted
    )
    reference_quant_config["exclude"] = preserve_modules
    reference_quant_config["layer_quant_config"] = {}
    qconfig = QConfig.from_dict(reference_quant_config)

    source_hf_config = read_json(source.root / "config.json")
    expected_export_config = _build_exclude_aware_quant_config(
        str(source.root), qconfig, source_hf_config, True
    ).to_dict()
    # Quark's final config exporter canonicalizes this list after deriving the
    # FP8 layer overrides from an internal set.
    expected_export_config["exclude"] = sorted(expected_export_config["exclude"])
    actual_reference_config = copy.deepcopy(reference_config["quantization_config"])
    actual_reference_config.pop("export", None)
    if expected_export_config != actual_reference_config:
        expected_path = DEFAULT_REPORT_DIR / "expected-derived-quantization-config.json"
        actual_path = DEFAULT_REPORT_DIR / "reference-quantization-config.json"
        write_json(expected_path, expected_export_config)
        write_json(actual_path, actual_reference_config)
        raise RuntimeError(
            "Derived Quark metadata does not exactly match REFERENCE; inspect "
            f"{expected_path} and {actual_path}"
        )
    return qconfig


def run_convert(args: argparse.Namespace) -> int:
    source_path = args.source.resolve()
    reference_path = args.reference.resolve()
    output_path = args.output.resolve()
    if len({source_path, reference_path, output_path}) != 3:
        raise RuntimeError("SOURCE, REFERENCE and OUTPUT must be three distinct directories")
    if output_path.exists():
        entries = sorted(path.name for path in output_path.iterdir()) if output_path.is_dir() else []
        raise FileExistsError(f"OUTPUT already exists ({len(entries)} entries): {output_path}")

    source = load_inventory(source_path)
    reference = load_inventory(reference_path)
    rows, summary = build_plan(source, reference)
    write_dry_run(source, reference, rows, summary, args.report_dir)
    assert_plan_is_safe(summary)
    comparison = config_comparison(source.root, reference.root)
    if not comparison["model_semantics_equal_without_quantization_config"]:
        raise RuntimeError("SOURCE and REFERENCE model semantics differ outside quantization_config")

    from quark.torch import ModelQuantizer

    qconfig = _build_conversion_qconfig(source, reference, rows)
    quantizer = ModelQuantizer(qconfig)
    quantizer.direct_quantize_checkpoint(
        pretrained_model_path=str(source_path),
        save_path=str(output_path),
        keep_excluded_layers_as_original_model_state=True,
        device=args.device,
    )

    # Quark copies every source subdirectory, including Hugging Face's local
    # download cache.  It is transient state rather than checkpoint metadata
    # and may contain many GiB of stale ``*.incomplete`` files.
    output_cache = output_path / ".cache"
    if output_cache.is_dir():
        shutil.rmtree(output_cache)

    for source_file in source_path.iterdir():
        if source_file.is_file() and source_file.name.lower().startswith("readme"):
            shutil.copy2(source_file, output_path / source_file.name)

    provenance = {
        "source": str(source_path),
        "reference_format_oracle": str(reference_path),
        "output": str(output_path),
        "reference_weights_copied": False,
        "conversion_api": "quark.torch.ModelQuantizer.direct_quantize_checkpoint",
        "device": args.device,
        "source_config_sha256": sha256_file(source_path / "config.json"),
        "source_index_sha256": sha256_file(source_path / "model.safetensors.index.json"),
        "reference_config_sha256": sha256_file(reference_path / "config.json"),
        "reference_index_sha256": sha256_file(reference_path / "model.safetensors.index.json"),
        "summary": summary,
    }
    write_json(output_path / "conversion-provenance.json", provenance)
    print(f"Conversion completed: {output_path}")
    return 0


def _hash_tensor_data(root: Path, info: TensorInfo, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    remaining = info.nbytes
    with (root / info.file).open("rb") as handle:
        handle.seek(info.data_start + info.offsets[0])
        while remaining:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                raise IOError(f"Unexpected EOF while hashing {info.name} in {info.file}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _validate_auxiliary_files(source: Path, output: Path) -> dict[str, Any]:
    ignored = {"config.json", "model.safetensors.index.json", "conversion-provenance.json"}
    source_aux = _auxiliary_inventory(source)
    output_aux = _auxiliary_inventory(output)
    expected_names = sorted(set(source_aux) - ignored)
    missing = [name for name in expected_names if name not in output_aux]
    mismatch = [
        name
        for name in expected_names
        if name in output_aux and source_aux[name]["sha256"] != output_aux[name]["sha256"]
    ]
    return {"checked": len(expected_names), "missing": missing, "sha256_mismatch": mismatch}


def run_validate(args: argparse.Namespace) -> int:
    source = load_inventory(args.source)
    reference = load_inventory(args.reference)
    output = load_inventory(args.output)
    rows, dry_summary = build_plan(source, reference)
    assert_plan_is_safe(dry_summary)

    missing = sorted(set(reference.tensors) - set(output.tensors))
    unexpected = sorted(set(output.tensors) - set(reference.tensors))
    shape_mismatch: list[str] = []
    dtype_mismatch: list[str] = []
    for name in sorted(set(reference.tensors) & set(output.tensors)):
        if reference.tensors[name].shape != output.tensors[name].shape:
            shape_mismatch.append(name)
        if reference.tensors[name].dtype != output.tensors[name].dtype:
            dtype_mismatch.append(name)

    source_config = read_json(source.root / "config.json")
    reference_config = read_json(reference.root / "config.json")
    output_config = read_json(output.root / "config.json")
    source_semantics_preserved = _without_quantization_config(source_config) == _without_quantization_config(
        output_config
    )
    quark_config_matches_reference = output_config.get("quantization_config") == reference_config.get(
        "quantization_config"
    )

    hash_rows: list[dict[str, str]] = []
    hash_mismatch: list[str] = []
    if not args.skip_hashes:
        for row in rows:
            if row["action"] not in {
                "PRESERVE_SOURCE_FP8",
                "PRESERVE_SOURCE_BF16",
                "PRESERVE_SOURCE_OTHER",
            }:
                continue
            source_info = source.tensors[row["tensor_name"]]
            output_info = output.tensors.get(row["reference_name"])
            if output_info is None:
                continue
            source_hash = _hash_tensor_data(source.root, source_info)
            output_hash = _hash_tensor_data(output.root, output_info)
            match = source_hash == output_hash
            hash_rows.append(
                {
                    "source_name": source_info.name,
                    "output_name": output_info.name,
                    "source_sha256": source_hash,
                    "output_sha256": output_hash,
                    "match": str(match),
                }
            )
            if not match:
                hash_mismatch.append(source_info.name)

        # Preserved FP8 scale tensors have SPECIAL_HANDLING because their key is renamed.
        for row in rows:
            if row["action"] != "SPECIAL_HANDLING" or row["target_representation"] != "FP8_SCALE_RENAMED":
                continue
            source_info = source.tensors[row["tensor_name"]]
            output_info = output.tensors.get(row["reference_name"])
            if output_info is None:
                continue
            source_hash = _hash_tensor_data(source.root, source_info)
            output_hash = _hash_tensor_data(output.root, output_info)
            match = source_hash == output_hash
            hash_rows.append(
                {
                    "source_name": source_info.name,
                    "output_name": output_info.name,
                    "source_sha256": source_hash,
                    "output_sha256": output_hash,
                    "match": str(match),
                }
            )
            if not match:
                hash_mismatch.append(source_info.name)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    if hash_rows:
        with (args.report_dir / "preserved-tensor-hashes.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(hash_rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(hash_rows)

    auxiliary = _validate_auxiliary_files(source.root, output.root)
    validation = {
        "source": str(source.root),
        "reference": str(reference.root),
        "output": str(output.root),
        "total_source_tensors": len(source.tensors),
        "total_reference_tensors": len(reference.tensors),
        "total_output_tensors": len(output.tensors),
        "converted_to_mxfp4": dry_summary["converted_to_mxfp4"],
        "preserved_fp8": dry_summary["preserved_fp8"],
        "preserved_bf16": dry_summary["preserved_bf16"],
        "additional_mxfp4_scale_tensors": dry_summary["additional_mxfp4_scale_tensors"],
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "source_inventory_errors": source.errors,
        "reference_inventory_errors": reference.errors,
        "output_inventory_errors": output.errors,
        "source_semantics_preserved": source_semantics_preserved,
        "quark_config_matches_reference": quark_config_matches_reference,
        "preserved_hashes_checked": len(hash_rows),
        "preserved_hash_mismatch": hash_mismatch,
        "auxiliary_files": auxiliary,
        "output_shard_metadata_matches_reference": output.shard_metadata == reference.shard_metadata,
    }
    write_json(args.report_dir / "validation.json", validation)
    failures = {
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "output_inventory_errors": output.errors,
        "preserved_hash_mismatch": hash_mismatch,
        "auxiliary_missing": auxiliary["missing"],
        "auxiliary_hash_mismatch": auxiliary["sha256_mismatch"],
    }
    if not source_semantics_preserved:
        failures["config"] = ["SOURCE semantics were not preserved"]
    if not quark_config_matches_reference:
        failures.setdefault("config", []).append("Quark config differs from REFERENCE")
    nonempty_failures = {key: value for key, value in failures.items() if value}
    print(json.dumps(validation, indent=2, sort_keys=True))
    if nonempty_failures:
        raise RuntimeError(f"Validation failed: {json.dumps(nonempty_failures, sort_keys=True)}")
    return 0


def _load_tensor(checkpoint: Inventory, name: str, device: str) -> Any:
    from safetensors import safe_open

    info = checkpoint.tensors[name]
    with safe_open(checkpoint.root / info.file, framework="pt", device=device) as handle:
        return handle.get_tensor(name)


def _quality_sample_names(rows: list[dict[str, Any]], count: int) -> list[str]:
    converted_by_component: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["action"] == "CONVERT_FP8_TO_MXFP4":
            converted_by_component[row["classification"]].append(row["tensor_name"])
    samples: list[str] = []
    for component in ("MOE_ROUTED", "MOE_SHARED"):
        names = converted_by_component.get(component, [])
        if names:
            samples.append(names[0])
    all_names = sorted(
        row["tensor_name"] for row in rows if row["action"] == "CONVERT_FP8_TO_MXFP4"
    )
    for index in (len(all_names) // 2, len(all_names) - 1):
        if all_names and all_names[index] not in samples:
            samples.append(all_names[index])
    return samples[:count]


def run_quality(args: argparse.Namespace) -> int:
    import torch
    from quark.torch.kernel import mx as quark_mx
    from quark.torch.quantization.file2file_quantization import _weight_dequant_fp8

    source = load_inventory(args.source)
    reference = load_inventory(args.reference)
    output = load_inventory(args.output)
    rows, summary = build_plan(source, reference)
    assert_plan_is_safe(summary)
    names = args.tensor or _quality_sample_names(rows, args.samples)
    classifications = {row["tensor_name"]: row["classification"] for row in rows}
    results: list[dict[str, Any]] = []

    for name in names:
        source_scale_name = name + SCALE_INV_SUFFIX
        output_scale_name = name + QUARK_SCALE_SUFFIX
        if name not in source.tensors or source_scale_name not in source.tensors:
            raise KeyError(f"Missing SOURCE FP8 weight/scale pair for {name}")
        if name not in output.tensors or output_scale_name not in output.tensors:
            raise KeyError(f"Missing OUTPUT MXFP4 weight/scale pair for {name}")
        source_weight = _load_tensor(source, name, args.device)
        source_scale = _load_tensor(source, source_scale_name, args.device)
        output_weight = _load_tensor(output, name, args.device)
        output_scale = _load_tensor(output, output_scale_name, args.device)
        source_dequant = _weight_dequant_fp8(
            source_weight.contiguous(), source_scale.contiguous(), model_dtype=torch.bfloat16
        )
        output_dequant = quark_mx.dq_mxfp4(
            output_weight.contiguous(), output_scale.contiguous(), torch.bfloat16
        )
        if source_dequant.shape != output_dequant.shape:
            raise RuntimeError(f"Dequantized shape mismatch for {name}")

        max_abs = 0.0
        abs_sum = 0.0
        sq_sum = 0.0
        source_sq_sum = 0.0
        output_sq_sum = 0.0
        dot_sum = 0.0
        element_count = source_dequant.numel()
        row_chunk = max(1, min(source_dequant.shape[0], 256))
        for start in range(0, source_dequant.shape[0], row_chunk):
            source_chunk = source_dequant[start : start + row_chunk].float()
            output_chunk = output_dequant[start : start + row_chunk].float()
            delta = source_chunk - output_chunk
            max_abs = max(max_abs, delta.abs().max().item())
            abs_sum += delta.abs().sum().item()
            sq_sum += delta.square().sum().item()
            source_sq_sum += source_chunk.square().sum().item()
            output_sq_sum += output_chunk.square().sum().item()
            dot_sum += (source_chunk * output_chunk).sum().item()
        results.append(
            {
                "tensor_name": name,
                "classification": classifications[name],
                "shape": list(source_dequant.shape),
                "max_abs_error": max_abs,
                "mean_abs_error": abs_sum / element_count,
                "rmse": math.sqrt(sq_sum / element_count),
                "relative_l2_error": math.sqrt(sq_sum / source_sq_sum),
                "cosine_similarity": dot_sum / math.sqrt(source_sq_sum * output_sq_sum),
            }
        )
        del source_weight, source_scale, output_weight, output_scale, source_dequant, output_dequant
        torch.cuda.empty_cache()

    payload = {"device": args.device, "samples": results}
    write_json(args.report_dir / "quality-metrics.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(source=DEFAULT_SOURCE, reference=DEFAULT_REFERENCE, output=DEFAULT_OUTPUT)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    common.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    common.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry_run = subparsers.add_parser("dry-run", parents=[common], help="Classify all tensors without loading data")
    dry_run.set_defaults(func=run_dry_run)

    convert = subparsers.add_parser("convert", parents=[common], help="Run streaming Quark conversion")
    convert.add_argument("--device", default="cuda:0")
    convert.set_defaults(func=run_convert)

    validate = subparsers.add_parser("validate", parents=[common], help="Validate OUTPUT against SOURCE and REFERENCE")
    validate.add_argument("--skip-hashes", action="store_true")
    validate.set_defaults(func=run_validate)

    quality = subparsers.add_parser("quality", parents=[common], help="Compare dequantized converted tensors")
    quality.add_argument("--device", default="cuda:0")
    quality.add_argument("--samples", type=int, default=4)
    quality.add_argument("--tensor", action="append", default=[])
    quality.set_defaults(func=run_quality)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
