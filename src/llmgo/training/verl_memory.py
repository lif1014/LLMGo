"""Release the trainer's cached-but-unused blocks before vLLM re-maps its weights.

j000023 (round 8, the 24-step conservative run) trained two steps cleanly and then died
inside the *second* ``update_weights`` with

    ray::vLLMHttpServer.wake_up() ... vllm/device_allocator/cumem.py:241 allocator.wake_up(tags)
    RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:137

verl's colocated flow is: rollout generates -> ``rollout.sleep()`` (vLLM unmaps weights and,
with full weights, the KV cache too) -> reference log-probs -> actor update -> ``update_weights``
which calls ``resume(tags=["weights"])`` -> vLLM re-creates the physical pages of its weights
(8 GiB bf16, measured "took 7.99 GiB memory" in that job's log).

``resume`` is called before ``verl.utils.memory_utils.aggressive_empty_cache`` runs, so at the
moment vLLM asks the driver for those pages the trainer still holds every block its caching
allocator ever reserved (that job peaked at ``max_memory_allocated_gb 39.05`` with
``max_memory_reserved_gb 47.83``). The vLLM engine runs in a *separate* process on the same
device, so it cannot see or reuse that memory; the request fails even though the pages are
free-but-cached. j000016 survived the same settings four times and j000023 died on its third
remap, i.e. the layout was marginal and the response-length-dependent KV demand decided it.

The training process keeps the stock PyTorch allocator (the CuMem allocator is installed in
the vLLM engine process), so ``torch.cuda.empty_cache()`` there really does hand the unused
blocks back to the driver. This module wraps the *client-side* ``resume`` - the one that runs
in the training process, immediately before the remote ``wake_up`` - instead of editing
installed packages. ``install_wake_cache_flush`` is idempotent and never raises: an
unrecognised verl build is reported as ``False`` so the run continues with the old behaviour.
"""
from functools import wraps


def _flush_training_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
    except Exception:
        return False
    return False


def install_wake_cache_flush():
    """Free cached blocks before vLLM re-maps rollout weights in ``ServerAdapter.resume``."""
    try:
        from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter
        original = ServerAdapter.resume
        if getattr(original, "_llmgo_wake_cache_flush", False):
            return True

        @wraps(original)
        async def resume(self, *args, **kwargs):
            _flush_training_cache()
            return await original(self, *args, **kwargs)

        resume._llmgo_wake_cache_flush = True
        ServerAdapter.resume = resume
        return True
    except Exception:
        # A renamed class, a different rollout backend or a missing async client all mean
        # "this build has no remap path to guard": report it and keep the run unchanged.
        return False
