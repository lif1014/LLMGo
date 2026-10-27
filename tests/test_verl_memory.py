"""Round-9: the OOM in vLLM's wake_up and the coexistence arithmetic behind it.

j000023 (24 steps, lr 5e-6) trained two steps and then died in
``vllm/device_allocator/cumem.py:241 allocator.wake_up`` while ``ServerAdapter.resume`` was
re-mapping the rollout weights, because the training process still held the blocks its
caching allocator had reserved. These are CPU contracts for the fix and the guard.
"""
import asyncio
import sys
from types import ModuleType

import pytest

from llmgo.training import verl_memory
from llmgo.training.verl_ppo import (MEASURED_ACTOR_RESERVED_BYTES_AT_16K, coexistence_preflight,
                                     training_config)

GIB = 2**30


def _fake_adapter_module(adapter):
    name = "verl.workers.rollout.vllm_rollout.vllm_rollout"
    module = ModuleType(name)
    module.ServerAdapter = adapter
    return name, module


def test_wake_cache_flush_flushes_before_the_remote_wake_up_and_is_idempotent(monkeypatch):
    calls = []

    class Adapter:
        async def resume(self, tags):
            calls.append(("resume", tuple(tags)))

    name, module = _fake_adapter_module(Adapter)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(verl_memory, "_flush_training_cache", lambda: calls.append(("flush",)))
    assert verl_memory.install_wake_cache_flush() is True
    installed = Adapter.resume
    # A second install must not stack another wrapper (that would flush N times per step).
    assert verl_memory.install_wake_cache_flush() is True
    assert Adapter.resume is installed
    asyncio.run(Adapter().resume(["weights"]))
    assert calls == [("flush",), ("resume", ("weights",))]


def test_wake_cache_flush_reports_false_instead_of_breaking_an_unknown_build(monkeypatch):
    name = "verl.workers.rollout.vllm_rollout.vllm_rollout"
    monkeypatch.setitem(sys.modules, name, ModuleType(name))  # no ServerAdapter at all
    assert verl_memory.install_wake_cache_flush() is False
    # A class that exists but has no resume() must be tolerated too: registered module
    # state must never be able to abort a training run at import time.
    class Renamed:
        pass
    monkeypatch.setitem(sys.modules, name, _fake_adapter_module(Renamed)[1])
    assert verl_memory.install_wake_cache_flush() is False
    # Explicitly simulate an unavailable import even on Workers with real verl.
    monkeypatch.setitem(sys.modules, name, None)
    assert verl_memory.install_wake_cache_flush() is False


def test_flush_without_cuda_is_a_no_op():
    # The devspace has no GPU; the helper must report "nothing flushed" rather than raise.
    assert verl_memory._flush_training_cache() is False


def test_round8_layout_needed_the_flush_and_the_round9_layout_still_fits():
    round8 = training_config({"max_response_tokens":16384, "rollout_memory_utilization":.5}, 8)
    before = coexistence_preflight(round8, 8)
    assert before["vllm_pool_bytes"] == int(79*GIB*0.5)
    assert before["actor_reserved_hint_bytes"] == MEASURED_ACTOR_RESERVED_BYTES_AT_16K
    # Reproduces j000023: without releasing the cached blocks the two do not fit...
    assert before["fits_without_flush"] is False
    # ...but what is *live* after an empty_cache does fit, so the flush is the fix rather
    # than a smaller pool.
    assert before["fits_with_flush"] is True
    assert before["needs_cache_flush"] is True

    round9 = training_config({"max_response_tokens":16384, "rollout_memory_utilization":.45}, 8)
    after = coexistence_preflight(round9, 8)
    assert after["rollout_wake_cache_flush"] is True
    assert after["fits"] is True
    assert after["margin_with_flush_bytes"] > 8*GIB
    # The pool stays large enough to keep the KV capacity round 8 measured at 220,704 tokens
    # (~27 GiB at 0.5); 0.45 removes only ~4 GiB of it.
    assert 0.40 <= round9["rollout_memory_utilization"] <= 0.50


def test_coexistence_guard_demands_the_flush_or_a_smaller_pool():
    no_flush = training_config({"rollout_memory_utilization":0.5, "max_response_tokens":16384,
                                "rollout_wake_cache_flush":False}, 8)
    assert coexistence_preflight(no_flush, 8)["fits"] is False
    small_pool = training_config({"rollout_memory_utilization":0.18, "max_response_tokens":16384,
                                  "rollout_wake_cache_flush":False}, 8)
    assert coexistence_preflight(small_pool, 8)["fits_without_flush"] is True
    with pytest.raises(ValueError, match="rollout_wake_cache_flush"):
        training_config({"rollout_wake_cache_flush":"yes"}, 8)
    short = training_config({"max_response_tokens":4096}, 8)
    assert coexistence_preflight(short, 8)["actor_reserved_hint_bytes"] == MEASURED_ACTOR_RESERVED_BYTES_AT_16K//4
