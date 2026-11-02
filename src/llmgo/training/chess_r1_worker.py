"""Native verl worker with compatibility installed in the GPU worker process."""
import json
import os
from pathlib import Path

from verl.workers.engine_workers import ActorRolloutRefWorker


class ChessR1Worker(ActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        # HFModelConfig is constructed on the coordinator then pickled into Ray
        # workers. Its external_lib import does not rerun on deserialization.
        from .vllm_compat import install_rope_compat
        from .verl_memory import install_wake_cache_flush
        from .chess_r1_attention import install_3d_position_compat, install_infer_position_compat
        from .chess_r1_vocab import install_streamed_vocab_statistics
        import transformers.modeling_flash_attention_utils as fa
        proof = {
            'pid': os.getpid(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'rope_compat': bool(install_rope_compat()),
            'weight_remap_cache_flush': bool(install_wake_cache_flush()),
            'mrope_guard_installed': bool(install_3d_position_compat()),
            'inference_jagged_positions_repaired': bool(install_infer_position_compat()),
            'streamed_full_vocabulary_statistics': bool(install_streamed_vocab_statistics()),
        }
        assert getattr(fa._is_packed_sequence, '_chess_r1_mrope_guard', False)
        super().__init__(*args, **kwargs)
        folder = Path(os.environ['CHESS_R1_OUTPUT'])/'worker-compatibility'
        folder.mkdir(parents=True, exist_ok=True)
        (folder/f'{os.getpid()}.json').write_text(json.dumps(proof))
