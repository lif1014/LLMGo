"""Process-local guard for Qwen3.5 MRoPE positions with pinned HF FlashAttention.

Upstream report: Dao-AILab/flash-attention#2381. MRoPE axes are not packed
sequences; flattening them into cu_seqlens invents tokens absent from Q/K/V.
"""
from functools import wraps


def use_native_text_positions(batch):
    """Use HF's equivalent 2D position input for this text-only experiment.

    Qwen3.5TextModel expands 2D positions to its four identical text/MRoPE
    axes internally. Keeping them 2D across Ray avoids VLM jagged metadata.
    """
    import torch
    positions = batch.batch['position_ids']
    if positions.ndim == 2:
        return
    if positions.ndim != 3 or positions.shape[1] != 4:
        raise ValueError('Unexpected Qwen3.5 position axes')
    mask = batch.batch['attention_mask'].bool()
    for axis in range(1, 4):
        if not torch.equal(positions[:, 0][mask], positions[:, axis][mask]):
            raise ValueError('Nonidentical MRoPE axes cannot be collapsed to text positions')
    batch.batch['position_ids'] = positions[:, 0].clone()


def install_infer_position_compat():
    """Apply verl's existing jagged-position repair to inference as well.

    train_mini_batch already calls this repair; infer_batch omits it in this
    pinned version. TensorDict consolidation/pickle can lose _ragged_idx=2.
    """
    from verl.workers.engine_workers import TrainingWorker
    from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
    original = TrainingWorker.infer_batch
    if getattr(original, '_chess_r1_jagged_position_guard', False):
        return True

    @wraps(original)
    def infer_batch(self, data, *args, **kwargs):
        maybe_fix_3d_position_ids(data)
        return original(self, data, *args, **kwargs)

    infer_batch._chess_r1_jagged_position_guard = True
    TrainingWorker.infer_batch = infer_batch
    return True


def install_3d_position_compat():
    import transformers.modeling_flash_attention_utils as fa
    original = fa._is_packed_sequence
    if getattr(original, '_chess_r1_mrope_guard', False):
        return True

    @wraps(original)
    def is_packed(position_ids, batch_size):
        result = False if position_ids is not None and position_ids.ndim > 2 else original(position_ids, batch_size)
        # A bounded routing receipt establishes what the actual actor/ref path
        # sees. It contains only shapes and counts, never generated tokens.
        import os
        output = os.environ.get('CHESS_R1_OUTPUT')
        if output and len(seen_shapes) < 4:
            shape = tuple(position_ids.shape) if position_ids is not None else None
            if shape not in seen_shapes:
                import json
                from pathlib import Path
                seen_shapes.add(shape)
                folder = Path(output)/'attention-routing'
                folder.mkdir(parents=True, exist_ok=True)
                row = {'pid': os.getpid(), 'position_shape': shape, 'batch_size': batch_size,
                       'packed': bool(result), 'guard_installed': True}
                (folder/f'{os.getpid()}-{len(seen_shapes)}.json').write_text(json.dumps(row))
        return result

    seen_shapes = set()
    is_packed._chess_r1_mrope_guard = True
    fa._is_packed_sequence = is_packed
    return True


def gpu_acceptance():
    """Compare real FA2 forward/backward against independent float32 SDPA math."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    install_3d_position_compat()
    torch.manual_seed(41)
    q = torch.randn(1,128,16,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    k = torch.randn(1,128,4,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    v = torch.randn_like(k,requires_grad=True)
    positions = torch.arange(128,device='cuda').view(1,1,128).expand(3,1,128)
    output = _flash_attention_forward(q,k,v,attention_mask=None,query_length=128,is_causal=True,
                                     position_ids=positions,dropout=0.,attn_implementation='flash_attention_2')
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        expected = F.scaled_dot_product_attention(q.float().transpose(1,2),
            k.float().repeat_interleave(4,dim=2).transpose(1,2),
            v.float().repeat_interleave(4,dim=2).transpose(1,2),is_causal=True).transpose(1,2)
    error = (output.float()-expected).abs()
    if float(error.max()) > .04 or float(error.mean()) > .003:
        raise RuntimeError('MRoPE guard failed independent attention parity')
    output.float().square().mean().backward()
    for item in [q,k,v]:
        if not torch.isfinite(item.grad).all() or float(item.grad.norm()) <= 0:
            raise RuntimeError('MRoPE guard has invalid attention gradients')
    return {'status':'passed','max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),
            'reference':'float32 PyTorch SDPA math','fa2_backward_finite_nonzero':True,
            'mrope_axes':3,'query_heads':16,'kv_heads':4,'head_dim':256}
