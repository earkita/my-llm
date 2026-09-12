#!/usr/bin/env python3
"""Validate an indexed or single-file safetensors checkpoint tensor by tensor."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import struct
from typing import Any


FLOAT_DTYPES = {"BF16", "F16", "F32", "F64", "F8_E4M3", "F8_E5M2"}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _header(path: Path) -> tuple[dict[str, Any], int]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"invalid safetensors prefix: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > size - 8:
            raise RuntimeError(f"invalid safetensors header size: {path}")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise RuntimeError(f"invalid safetensors header object: {path}")
    return header, size - 8 - header_size


def validate(model_dir: Path, output: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    model_dir = model_dir.resolve()
    started = datetime.now().astimezone()
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("checkpoint index has no weight_map")
        shard_names = sorted(set(weight_map.values()))
    else:
        index = None
        weight_map = None
        shard_names = sorted(path.name for path in model_dir.glob("*.safetensors"))
        if not shard_names:
            raise RuntimeError("checkpoint has no safetensors files")
    missing_shards = [name for name in shard_names if not (model_dir / name).is_file()]
    unexpected_index_entries: list[str] = []
    missing_index_entries: list[str] = []
    duplicate_tensors: list[str] = []
    invalid_headers: list[str] = []
    nonfinite_tensors: list[str] = []
    seen: set[str] = set()
    dtype_counts: dict[str, int] = {}
    tensor_specs: dict[str, dict[str, Any]] = {}
    finite_elements = 0
    tensor_payload_bytes = 0
    file_bytes = 0

    for shard_name in shard_names:
        path = model_dir / shard_name
        if not path.is_file():
            continue
        file_bytes += path.stat().st_size
        try:
            header, data_bytes = _header(path)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            invalid_headers.append(f"{shard_name}: {exc}")
            continue
        tensor_payload_bytes += data_bytes
        names = [name for name in header if name != "__metadata__"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in names:
                if name in seen:
                    duplicate_tensors.append(name)
                seen.add(name)
                if weight_map is not None and weight_map.get(name) != shard_name:
                    unexpected_index_entries.append(name)
                metadata = header[name]
                dtype = metadata.get("dtype") if isinstance(metadata, dict) else None
                dtype_name = str(dtype)
                dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
                tensor_specs[name] = {
                    "dtype": dtype_name,
                    "shape": metadata.get("shape") if isinstance(metadata, dict) else None,
                }
                if dtype_name == "I64" and name.endswith(".weight_shape"):
                    tensor_specs[name]["values"] = handle.get_tensor(name).tolist()
                if dtype_name not in FLOAT_DTYPES:
                    continue
                tensor = handle.get_tensor(name)
                finite_elements += tensor.numel()
                if not bool(torch.isfinite(tensor.float()).all()):
                    nonfinite_tensors.append(name)

    if weight_map is not None:
        missing_index_entries = sorted(set(weight_map) - seen)
    index_total_size = index.get("metadata", {}).get("total_size") if index else None
    if index is None:
        total_size_semantics = "index_not_present"
    elif index_total_size == tensor_payload_bytes:
        total_size_semantics = "tensor_payload_bytes"
    elif index_total_size == file_bytes:
        total_size_semantics = "file_bytes"
    else:
        total_size_semantics = "unrecognized"
    quantization = config.get("quantization_config")
    compressed_mxfp4_ok = (
        isinstance(quantization, dict)
        and quantization.get("quant_method") == "compressed-tensors"
        and quantization.get("format") == "mxfp4-pack-quantized"
        and quantization.get("quantization_status") == "compressed"
    )
    compressed_group = (
        quantization.get("config_groups", {}).get("group_0", {})
        if isinstance(quantization, dict)
        else {}
    )
    compressed_weight = (
        compressed_group.get("weights", {})
        if isinstance(compressed_group, dict)
        else {}
    )
    compressed_w4a16_ok = (
        isinstance(quantization, dict)
        and quantization.get("quant_method") == "compressed-tensors"
        and quantization.get("format") == "pack-quantized"
        and quantization.get("quantization_status") == "compressed"
        and compressed_group.get("input_activations") is None
        and compressed_group.get("output_activations") is None
        and compressed_weight.get("type") == "int"
        and compressed_weight.get("num_bits") == 4
        and compressed_weight.get("group_size") == 128
        and compressed_weight.get("strategy") == "group"
        and compressed_weight.get("symmetric") is True
    )
    compressed_tensors_ok = compressed_mxfp4_ok or compressed_w4a16_ok
    quark_ok = (
        isinstance(quantization, dict)
        and quantization.get("quant_method") == "quark"
        and quantization.get("export", {}).get("weight_format")
        == "real_quantized"
    )
    quark_int4_modules = 0
    quark_int4_shape_errors: list[str] = []
    quark_exclusion_errors: list[str] = []
    if quark_ok:
        weight_config = quantization.get("global_quant_config", {}).get("weight", {})
        group_size = weight_config.get("group_size")
        pack_factor = 8
        quantized_weights = sorted(
            name
            for name, spec in tensor_specs.items()
            if spec["dtype"] == "I32" and name.endswith(".weight")
        )
        quark_int4_modules = len(quantized_weights)
        for weight_name in quantized_weights:
            prefix = weight_name.removesuffix(".weight")
            zero_name = prefix + ".weight_zero_point"
            scale_name = prefix + ".weight_scale"
            weight_shape = tensor_specs[weight_name]["shape"]
            zero = tensor_specs.get(zero_name)
            scale = tensor_specs.get(scale_name)
            if not (
                isinstance(group_size, int)
                and isinstance(weight_shape, list)
                and len(weight_shape) == 2
                and zero is not None
                and zero["dtype"] == "I32"
                and zero["shape"]
                == [weight_shape[0] // group_size, weight_shape[1]]
                and scale is not None
                and scale["dtype"] in FLOAT_DTYPES
                and scale["shape"]
                == [weight_shape[0] // group_size, weight_shape[1] * pack_factor]
            ):
                quark_int4_shape_errors.append(weight_name)
        for name, spec in tensor_specs.items():
            if spec["dtype"] != "I32":
                continue
            if name.startswith(("model.visual.", "mtp.")) or name.startswith(
                "lm_head."
            ):
                quark_exclusion_errors.append(name)
            if not name.endswith((".weight", ".weight_zero_point")):
                quark_int4_shape_errors.append(name)
    compressed_w4a16_modules = 0
    compressed_w4a16_shape_errors: list[str] = []
    compressed_w4a16_exclusion_errors: list[str] = []
    if compressed_w4a16_ok:
        group_size = int(compressed_weight["group_size"])
        pack_factor = 8
        packed_weights = sorted(
            name
            for name, spec in tensor_specs.items()
            if spec["dtype"] == "I32" and name.endswith(".weight_packed")
        )
        compressed_w4a16_modules = len(packed_weights)
        ignored_patterns = [
            value.removeprefix("re:")
            for value in quantization.get("ignore", [])
            if isinstance(value, str) and value.startswith("re:")
        ]
        for packed_name in packed_weights:
            prefix = packed_name.removesuffix(".weight_packed")
            packed_shape = tensor_specs[packed_name]["shape"]
            scale = tensor_specs.get(prefix + ".weight_scale")
            weight_shape = tensor_specs.get(prefix + ".weight_shape")
            if not (
                isinstance(packed_shape, list)
                and len(packed_shape) == 2
                and all(isinstance(value, int) and value > 0 for value in packed_shape)
            ):
                compressed_w4a16_shape_errors.append(packed_name)
                continue
            output_size = packed_shape[0]
            input_size = packed_shape[1] * pack_factor
            if not (
                input_size % group_size == 0
                and scale is not None
                and scale["dtype"] in FLOAT_DTYPES
                and scale["shape"] == [output_size, input_size // group_size]
                and weight_shape is not None
                and weight_shape["dtype"] == "I64"
                and weight_shape["shape"] == [2]
                and weight_shape.get("values") == [output_size, input_size]
            ):
                compressed_w4a16_shape_errors.append(packed_name)
            if any(re.fullmatch(pattern, prefix) for pattern in ignored_patterns):
                compressed_w4a16_exclusion_errors.append(packed_name)
        if any(name.endswith(".weight_zero_point") for name in tensor_specs):
            compressed_w4a16_shape_errors.append("unexpected asymmetric zero points")
    checks = {
        "all_indexed_shards_present": not missing_shards,
        "headers_valid": not invalid_headers,
        "index_matches_headers": not unexpected_index_entries and not missing_index_entries,
        "tensor_names_unique": not duplicate_tensors,
        "all_float_tensors_finite": not nonfinite_tensors,
        "index_total_size_recognized": total_size_semantics != "unrecognized",
        "quantization_config_recognized": compressed_tensors_ok or quark_ok,
        "compressed_w4a16_structure": not compressed_w4a16_ok
        or (
            compressed_w4a16_modules > 0
            and not compressed_w4a16_shape_errors
            and not compressed_w4a16_exclusion_errors
        ),
        "quark_int4_structure": not quark_ok
        or (
            quark_int4_modules > 0
            and not quark_int4_shape_errors
            and not quark_exclusion_errors
        ),
    }
    payload = {
        "schema_version": 1,
        "started_at": started.isoformat(),
        "model_directory": str(model_dir),
        "index": index_path.name if index is not None else None,
        "shards": len(shard_names),
        "tensors": len(seen),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "finite_elements_checked": finite_elements,
        "tensor_payload_bytes": tensor_payload_bytes,
        "file_bytes": file_bytes,
        "index_total_size": index_total_size,
        "index_total_size_semantics": total_size_semantics,
        "missing_shards": missing_shards,
        "invalid_headers": invalid_headers,
        "unexpected_index_entries": sorted(unexpected_index_entries),
        "missing_index_entries": missing_index_entries,
        "duplicate_tensors": sorted(duplicate_tensors),
        "nonfinite_tensors": nonfinite_tensors,
        "quark_int4_modules": quark_int4_modules,
        "quark_int4_shape_errors": sorted(set(quark_int4_shape_errors)),
        "quark_exclusion_errors": sorted(quark_exclusion_errors),
        "compressed_w4a16_modules": compressed_w4a16_modules,
        "compressed_w4a16_shape_errors": sorted(
            set(compressed_w4a16_shape_errors)
        ),
        "compressed_w4a16_exclusion_errors": sorted(
            compressed_w4a16_exclusion_errors
        ),
        "checks": checks,
        "passed": all(checks.values()),
        "finished_at": datetime.now().astimezone().isoformat(),
    }
    _write_json(output, payload)
    print(json.dumps({key: payload[key] for key in ("passed", "shards", "tensors", "finite_elements_checked", "checks")}, indent=2))
    if not payload["passed"]:
        raise RuntimeError("checkpoint integrity validation failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.model_dir, args.output)


if __name__ == "__main__":
    main()
