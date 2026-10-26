"""CPU contracts for repairing the transformers-5.x parameter names in an export.

Round-7 evidence: jobs j000019 and j000020 both started vLLM on a merged export and
died with `ValueError: Following weights were not initialized from checkpoint`,
because `verl.model_merger` (transformers 5.4) writes
`model.language_model.language_model.language_model.layers.N.*` and
`model.language_model.visual.*` instead of the base checkpoint's
`model.language_model.layers.N.*` and `model.visual.*`. These tests use hand-built
safetensors headers, so they need no torch and never touch a real 9 GB shard.
"""
import json
from pathlib import Path
import struct

import pytest

from llmgo.training.export_weights import (remap_weight_key, normalize_export_weights,
    verify_weight_names, read_safetensors_header, shard_index)

BLOCK = 8


def write_safetensors(path, tensors):
    """tensors: name -> (dtype, shape, payload bytes). Returns the payload bytes."""
    header, body, offset = {}, b"", 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset+len(payload)]}
        body += payload
        offset += len(payload)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" "*((-len(blob)) % BLOCK)
    Path(path).write_bytes(struct.pack("<Q", len(blob))+blob+body)
    return body


def payload_of(path):
    path = Path(path)
    _, start = read_safetensors_header(path)
    return path.read_bytes()[start:]


BASE_KEYS = {
    "model.language_model.embed_tokens.weight": ("BF16", [4, 2]),
    "model.language_model.layers.0.input_layernorm.weight": ("BF16", [4]),
    "model.visual.pos_embed.weight": ("BF16", [2, 2]),
    "mtp.norm.weight": ("BF16", [4]),
}


def build_pair(tmp_path, *, export_keys=None, shapes=None):
    base, export = tmp_path/"base", tmp_path/"export"
    base.mkdir()
    export.mkdir()
    (base/"config.json").write_text("{}")
    (export/"config.json").write_text("{}")
    shards = {}
    for name, (dtype, shape) in BASE_KEYS.items():
        shards[name] = (dtype, shape, bytes(range(2*len(shape))))
    write_safetensors(base/"model.safetensors", shards)
    mangled = {
        "model.language_model.language_model.language_model.embed_tokens.weight":
            "model.language_model.embed_tokens.weight",
        "model.language_model.language_model.language_model.layers.0.input_layernorm.weight":
            "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.visual.pos_embed.weight": "model.visual.pos_embed.weight",
    }
    tensors = {}
    for key, (dtype, shape) in BASE_KEYS.items():
        if key.startswith("mtp."):
            continue
        mapped = {value: name for name, value in mangled.items()}.get(key, key)
        tensors[mapped] = (dtype, shapes.get(key, shape) if shapes else shape,
                           bytes(range(2*len(shape))))
    for name in (export_keys or []):
        tensors[name] = ("BF16", [1], b"\x00\x00")
    write_safetensors(export/"model.safetensors", tensors)
    return base, export


def test_remap_rule_undoes_the_5x_nesting_and_is_idempotent():
    assert remap_weight_key(
        "model.language_model.language_model.language_model.layers.6.input_layernorm.weight"
        ) == "model.language_model.layers.6.input_layernorm.weight"
    assert remap_weight_key("model.language_model.visual.blocks.0.attn.qkv.bias") \
        == "model.visual.blocks.0.attn.qkv.bias"
    assert remap_weight_key("model.language_model.embed_tokens.weight") \
        == "model.language_model.embed_tokens.weight"
    for name in ("model.language_model.layers.0.mlp.down_proj.weight",
                 "model.visual.patch_embed.proj.weight", "mtp.fc.weight"):
        assert remap_weight_key(remap_weight_key(name)) == name


def test_normalize_export_weights_remaps_names_and_preserves_the_payload(tmp_path):
    base, export = build_pair(tmp_path)
    before = payload_of(export/"model.safetensors")
    report = normalize_export_weights(export, base)
    assert report["verification"]["ok"] is True
    assert report["renamed_parameters"] == 3
    assert report["verification"]["missing_allowed"] == ["mtp.norm.weight"]
    assert set(shard_index(export)) == set(BASE_KEYS)-{"mtp.norm.weight"}
    # Only the names changed: the tensor bytes came across untouched.
    assert payload_of(export/"model.safetensors") == before
    # A second pass is a no-op, so a repaired export can be re-checked safely.
    again = normalize_export_weights(export, base)
    assert again["renamed_parameters"] == 0 and again["shards"][0]["written"] is False
    assert verify_weight_names(export, base)["ok"] is True


def test_normalize_export_weights_refuses_a_name_the_base_does_not_have(tmp_path):
    base, export = build_pair(tmp_path, export_keys=["model.language_model.layers.7.gate.weight"])
    header, _ = read_safetensors_header(export/"model.safetensors")
    with pytest.raises(ValueError, match="unexpected"):
        normalize_export_weights(export, base)
    # The refusal happens before anything is written.
    after, _ = read_safetensors_header(export/"model.safetensors")
    assert after == header
    assert verify_weight_names(export, base)["ok"] is False


def test_normalize_export_weights_refuses_a_shape_mismatch(tmp_path):
    base, export = build_pair(tmp_path, shapes={"model.visual.pos_embed.weight": [3, 2]})
    with pytest.raises(ValueError, match="shape_mismatch"):
        normalize_export_weights(export, base)


def test_dtype_deviations_are_reported_but_not_fatal(tmp_path):
    # The real export stores the base model's 48 float32 GDN tensors as bfloat16:
    # training keeps fp32 parameters and the merger writes the config dtype. vLLM
    # allocates those parameters itself and casts on copy, so this is recorded in the
    # report instead of failing the export - unlike a shape mismatch, which would
    # silently corrupt a tensor.
    base = tmp_path/"base"; base.mkdir()
    export = tmp_path/"export"; export.mkdir()
    write_safetensors(base/"model.safetensors",
        {"model.language_model.layers.0.linear_attn.A_log": ("F32", [32], bytes(128))})
    write_safetensors(export/"model.safetensors",
        {"model.language_model.language_model.language_model.layers.0.linear_attn.A_log":
            ("BF16", [32], bytes(64))})
    report = normalize_export_weights(export, base)
    assert report["verification"]["ok"] is True
    assert report["verification"]["dtype_deviations"] == [
        {"name": "model.language_model.layers.0.linear_attn.A_log",
         "export": "BF16", "base": "F32"}]


def test_verify_weight_names_accepts_only_the_documented_omission(tmp_path):
    base, export = build_pair(tmp_path)
    normalize_export_weights(export, base)
    report = verify_weight_names(export, base)
    assert report["ok"] is True and report["unexpected"] == []
    assert report["missing_unexpected"] == [] and report["missing_allowed"] == ["mtp.norm.weight"]
    # Without the documented allowance the same directory is refused.
    strict = verify_weight_names(export, base, allowed_missing=())
    assert strict["ok"] is False and strict["missing_unexpected"] == ["mtp.norm.weight"]
