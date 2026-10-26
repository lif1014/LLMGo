"""Repair the parameter names a transformers-5.x ``verl.model_merger`` writes.

Round-7 evidence (jobs j000019 and j000020):
`verl.model_merger` merged `global_step_4/actor` correctly - the shards hold the
training model's own Hugging Face names (`model.language_model.layers.N.*`,
`model.visual.*`) - but it then rebuilt the directory with
`AutoModelForVision2Seq.from_config(...)` and `save_pretrained(state_dict=...)`.
Transformers 5.4 nests the text tower one level deeper and moves the vision tower
under it, so the *export* came out as
`model.language_model.language_model.language_model.layers.N.*` and
`model.language_model.visual.*`. vLLM 0.17 resolves the architecture, then finds no
parameter in that naming and aborts with
`ValueError: Following weights were not initialized from checkpoint`.

Nothing about the numbers is wrong: only the names. This module maps the merged
names back onto the base checkpoint naming (the one the pinned evaluator is known to
load), streams the tensor bytes into a freshly keyed safetensors file so the 9 GB
weight payload is never held in memory, and refuses to finish unless the mapped key
set matches the base checkpoint exactly, apart from documented omissions.

No torch is required: only the safetensors header is parsed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import struct

from llmgo.rsi.contracts import read_json

HEADER_ALIGNMENT = 8
COPY_BLOCK = 64 * 2**20
# The base checkpoint carries a 15-tensor multi-token-prediction module
# (`mtp.*`, `mtp_num_hidden_layers: 1`). The training export declares
# `mtp_num_hidden_layers: 0` and never trains or emits those tensors, so their
# absence is consistent, not a broken name map. Everything else must match.
EXPECTED_MISSING_PREFIXES = ("mtp.",)


def remap_weight_key(name):
    """Map a merged-export parameter name onto the base checkpoint naming.

    Two 5.4-era prefixes have to be unwound, and both rules are idempotent, so an
    export that already uses base naming is left untouched:

    * `model.language_model.language_model.language_model.layers.0...` - the text
      tower's own prefix is repeated; adjacent duplicates collapse to one.
    * `model.language_model.visual.blocks.0...` - the vision tower was nested under
      the text tower; it belongs at the top level (`model.visual.blocks.0...`).
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Weight names must be non-empty strings")
    parts = []
    for part in name.split("."):
        if part == "language_model" and parts and parts[-1] == "language_model":
            continue
        parts.append(part)
    if "visual" in parts:
        index = parts.index("visual")
        if index and parts[index-1] == "language_model":
            del parts[index-1]
    mapped = ".".join(parts)
    if not mapped or mapped.startswith(".") or mapped.endswith(".") or ".." in mapped:
        raise ValueError(f"Cannot remap weight name {name!r}")
    return mapped


def read_safetensors_header(path):
    """Return (header, data_start) without reading the tensor payload."""
    path = Path(path)
    with path.open("rb") as handle:
        prefix = handle.read(HEADER_ALIGNMENT)
        if len(prefix) != HEADER_ALIGNMENT:
            raise ValueError(f"{path} is too short to be a safetensors file")
        (length,) = struct.unpack("<Q", prefix)
        if length <= 0 or length > 100 * 2**20:
            raise ValueError(f"{path} has an implausible header length {length}")
        raw = handle.read(length)
    if len(raw) != length:
        raise ValueError(f"{path} header is truncated")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{path} does not contain a safetensors header: {error}") from None
    if not isinstance(header, dict):
        raise ValueError(f"{path} header is not a JSON object")
    return header, HEADER_ALIGNMENT+length


def shard_index(directory):
    """Parameter name -> (dtype, shape, shard) for every safetensors shard in a directory."""
    directory = Path(directory)
    index = {}
    shards = sorted(path for path in directory.glob("*.safetensors") if path.is_file())
    if not shards:
        raise ValueError(f"No *.safetensors files in {directory}")
    for shard in shards:
        header, _ = read_safetensors_header(shard)
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if not isinstance(value, dict) or "shape" not in value or "dtype" not in value:
                raise ValueError(f"{shard} has an invalid entry for {key!r}")
            index[key] = {"dtype": value["dtype"], "shape": list(value["shape"]),
                          "shard": shard.name}
    return index


def rewrite_safetensors_keys(source, target=None, remap=remap_weight_key):
    """Write a copy of `source` whose header uses remapped names; data bytes are copied.

    Offsets are untouched because only names change, so the payload stays bit-identical
    while the 9 GB blob is streamed in fixed-size blocks.
    """
    source = Path(source)
    header, data_start = read_safetensors_header(source)
    mapped = {}
    changed = 0
    for key, value in header.items():
        if key == "__metadata__":
            mapped[key] = value
            continue
        name = remap(key)
        if name in mapped:
            raise ValueError(f"Remap collides on {name!r} ({key!r}) in {source}")
        changed += name != key
        mapped[name] = value
    if not changed:
        return {"file": source.name, "written": False, "renamed": 0, "parameters": len(header)-1}
    size = source.stat().st_size
    payload_end = max((value["data_offsets"][1] for key, value in mapped.items()
                       if key != "__metadata__"), default=0)
    if data_start+payload_end > size:
        raise ValueError(f"{source} is truncated: payload needs {data_start+payload_end}, has {size}")
    blob = json.dumps(mapped, separators=(",", ":")).encode("utf-8")
    padding = (-len(blob)) % HEADER_ALIGNMENT
    blob += b" "*padding
    destination = Path(target) if target else source.with_name(source.name+".remapping")
    temporary = destination.with_name(destination.name+".tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            writer.write(struct.pack("<Q", len(blob)))
            writer.write(blob)
            reader.seek(data_start)
            remaining = size-data_start
            while remaining:
                block = reader.read(min(COPY_BLOCK, remaining))
                if not block:
                    raise ValueError(f"{source} ended {remaining} bytes early")
                writer.write(block)
                remaining -= len(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"file": destination.name, "written": True, "renamed": changed,
            "parameters": len(header)-1}


def _unexpected_names(index, base_index):
    return sorted(set(index)-set(base_index))


def _missing_names(index, base_index):
    return sorted(set(base_index)-set(index))


def compare_weight_index(export_index, base_index, allowed_missing=EXPECTED_MISSING_PREFIXES):
    """Report names that would make vLLM skip or reject a checkpoint."""
    unexpected = _unexpected_names(export_index, base_index)
    missing = _missing_names(export_index, base_index)
    allowed = [name for name in missing if any(name.startswith(p) for p in allowed_missing)]
    refused_missing = [name for name in missing if name not in set(allowed)]
    shape_mismatch, dtype_deviation = [], []
    for name in sorted(set(export_index)&set(base_index)):
        export_dtype, export_shape = export_index[name]["dtype"], export_index[name]["shape"]
        base_dtype, base_shape = base_index[name]["dtype"], base_index[name]["shape"]
        if export_shape != base_shape:
            shape_mismatch.append({"name": name, "export": export_shape, "base": base_shape})
        elif export_dtype != base_dtype:
            dtype_deviation.append({"name": name, "export": export_dtype, "base": base_dtype})
    report = {"export_parameters": len(export_index), "base_parameters": len(base_index),
        "unexpected": unexpected, "missing_allowed": allowed, "missing_unexpected": refused_missing,
        "shape_mismatch": shape_mismatch, "dtype_deviations": dtype_deviation,
        # A dtype difference is recorded, not fatal: vLLM allocates these parameters
        # itself (the A_log/dt_bias/GDN norms are float32 in the engine) and
        # `default_weight_loader` casts the checkpoint tensor on copy, exactly as it
        # already does for the base model's float32 tensors. A shape difference would
        # corrupt a tensor, so that stays fatal.
        "ok": not unexpected and not refused_missing and not shape_mismatch}
    return report


def verify_weight_names(export, base, allowed_missing=EXPECTED_MISSING_PREFIXES):
    """Compare an export's parameter names with the base checkpoint's, without writing."""
    report = compare_weight_index(shard_index(export), shard_index(base), allowed_missing)
    report.update(export=str(export), base=str(base))
    return report


def _rewrite_index(directory, remap):
    """Rewrite model.safetensors.index.json if the merger wrote one."""
    index_path = Path(directory)/"model.safetensors.index.json"
    if not index_path.is_file():
        return None
    index = read_json(index_path)
    weight_map = index.get("weight_map") or {}
    mapped = {remap(name): shard for name, shard in weight_map.items()}
    if mapped == weight_map:
        return {"file": index_path.name, "written": False}
    if len(mapped) != len(weight_map):
        raise ValueError(f"Remap collides in {index_path}")
    index["weight_map"] = mapped
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return {"file": index_path.name, "written": True, "renamed": len(mapped)}


def normalize_export_weights(export, base, allowed_missing=EXPECTED_MISSING_PREFIXES, remap=remap_weight_key):
    """Remap a merged export onto the base naming, then prove the key sets agree.

    Raises before writing anything if the mapping would produce a name the base
    checkpoint does not have, or if a shared tensor's dtype/shape differs - that is
    the failure mode that silently cost two GPU jobs, so it must fail here instead.
    """
    export, base = Path(export), Path(base)
    base_index = shard_index(base)
    before = shard_index(export)
    projected = {}
    for name, value in before.items():
        mapped_name = remap(name)
        if mapped_name in projected:
            raise ValueError(f"Remap collides on {mapped_name!r} in {export}")
        projected[mapped_name] = value
    verdict = compare_weight_index(projected, base_index, allowed_missing)
    if not verdict["ok"]:
        raise ValueError("Exported weights do not match the base checkpoint naming: "
            f"unexpected={verdict['unexpected'][:8]} missing={verdict['missing_unexpected'][:8]} "
            f"shape_mismatch={verdict['shape_mismatch'][:4]} "
            f"(unexpected={len(verdict['unexpected'])}, "
            f"missing={len(verdict['missing_unexpected'])}, "
            f"shape_mismatch={len(verdict['shape_mismatch'])})")
    # target=shard: the rewrite goes through a temporary file and os.replace, so the
    # export directory ends up with the remapped shard while any hardlink another
    # directory holds (scripts/repair_export.py links rather than copies) keeps the
    # original bytes.
    written = [rewrite_safetensors_keys(shard, target=shard, remap=remap)
               for shard in sorted(Path(export).glob("*.safetensors")) if shard.is_file()]
    index = _rewrite_index(export, remap)
    after = shard_index(export)
    final = compare_weight_index(after, base_index, allowed_missing)
    if not final["ok"]:
        raise ValueError(f"Key remap did not converge for {export}: {final}")
    return {"export": str(export), "base": str(base), "renamed_parameters":
            sum(row["renamed"] for row in written), "shards": written, "index": index,
            "allowed_missing_prefixes": list(allowed_missing),
            "before": {"unexpected": _unexpected_names(before, base_index),
                       "missing": _missing_names(before, base_index)},
            "verification": final}
