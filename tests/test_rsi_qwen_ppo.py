"""Opt-in tiny Qwen3.5 architecture integration; no downloaded student weights/GPU.

LLMGO_QWEN_TINY_TEST=1 PYTHONPATH=src python -m pytest -q tests/test_rsi_qwen_ppo.py
Requires a CPU PyTorch + Transformers/PEFT/tokenizers environment. This proves the
real multimodal model shell, LoRA, generation, PPO, export and batch-boundary
resume wiring on a randomly initialized miniature model, not 4B compatibility,
BT4 scoring accuracy or chess strength.
"""
import os
from pathlib import Path
import pytest


@pytest.mark.skipif(os.environ.get("LLMGO_QWEN_TINY_TEST") != "1", reason="opt-in tiny Qwen3.5 PPO integration")
def test_tiny_qwen_ppo_export_reload_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import (Qwen3_5Config, Qwen3_5TextConfig,
        Qwen3_5ForConditionalGeneration, Qwen3_5Tokenizer)
    from llmgo.training.ppo import _train, load_hf_model
    from llmgo.rsi.contracts import read_json
    from llmgo.training.rollout import seed_curriculum

    torch.set_num_threads(1)
    torch.manual_seed(4170)
    vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "<think>": 3, "</think>": 4}
    for symbol in sorted(ByteLevel.alphabet()):
        vocab.setdefault(symbol, len(vocab))
    text = Qwen3_5TextConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layer_types=["linear_attention", "full_attention"], linear_conv_kernel_dim=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=1, linear_num_value_heads=2,
        pad_token_id=0, eos_token_id=1)
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
        "out_hidden_size": 16, "patch_size": 2, "temporal_patch_size": 1,
        "spatial_merge_size": 1, "num_position_embeddings": 16}
    config = Qwen3_5Config(text_config=text, vision_config=vision,
        image_token_id=12, video_token_id=13, vision_start_token_id=14, vision_end_token_id=15)
    parent = tmp_path/"parent"
    initial = Qwen3_5ForConditionalGeneration(config)
    # Parent serving preferences must not silently warp on-policy training samples.
    initial.generation_config.do_sample = True
    initial.generation_config.min_p = .8
    initial.generation_config.forced_eos_token_id = 1
    initial.save_pretrained(parent)
    tokenizer = Qwen3_5Tokenizer(vocab=vocab, merges=[], pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    tokenizer.chat_template = "{% for m in messages %}{{m['content']}} {% endfor %}<think> "
    tokenizer.save_pretrained(parent)
    (parent/"processor_config.json").write_text('{"integration_test":true}\n')

    class Critic:
        identity = {"backend": "test-fixed", "score": "expectation"}
        def value(self, board, actor):
            return .5, {"mock": True}
        def close(self):
            pass

    cfg = {"batch_size": 2, "minibatch_size": 1, "steps": 2, "ppo_epochs": 2,
           "lora_rank": 2, "max_response_tokens": 16, "learning_rate": .001, "checkpoint_every": 1}
    progress = []
    records = seed_curriculum(2)
    export = _train(parent, records, [], Critic(), {}, cfg, tmp_path/"first", device="cpu",
                    progress=lambda stage, message, **kw: progress.append(stage))
    assert {"sampling", "labeling", "updating", "updated"} <= set(progress)
    result = read_json(export.parent/"training-result.json")
    assert result["backend"] == "hf_ppo" and result["updates"] == 8 and not result["sft_enabled"]
    original, trained = load_hf_model(parent), load_hf_model(export)
    assert original.config.model_type == trained.config.model_type == "qwen3_5"
    assert any(not torch.equal(v, trained.state_dict()[k]) for k, v in original.state_dict().items())
    assert (export/"processor_config.json").read_bytes() == (parent/"processor_config.json").read_bytes()
    sample = read_json(next((export.parent/"rollouts").glob("*.json")))["samples"][0]
    assert len(sample["behavior_logprobs"]) == len(sample["generated_token_ids"]) > 0
    assert "value_before" in sample and "failure_reason" in sample
    assert sample["behavior_recompute_max_abs_delta"] < .2
    assert trained.generation_config.forced_eos_token_id == original.generation_config.forced_eos_token_id == 1
    resumed_export = _train(parent, records, [], Critic(), {}, cfg, tmp_path/"resumed", device="cpu",
                            resume=tmp_path/"first/checkpoint-000001")
    resumed = load_hf_model(resumed_export)
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in trained.state_dict().items())
    # It really reloads through the same generation interface used to sample.
    ids = tokenizer.apply_chat_template([{"role":"user","content":"chess"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True, return_tensors="pt", return_dict=False)
    with torch.no_grad():
        generated = resumed.generate(input_ids=ids, max_new_tokens=2, do_sample=False)
    assert generated.shape[1] > ids.shape[1]
