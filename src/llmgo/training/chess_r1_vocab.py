"""Exact, token-chunked log-probability and entropy for the padded verl path.

The pinned engine materializes full-vocabulary softmax and concatenates padded
logits. Both are avoidable. Stream token rows while retaining the entire vocabulary;
recompute probabilities in backward and allocate one shared logits gradient.
"""
from functools import wraps

import torch


class _VocabularyStatistics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, temperature, calculate_entropy, chunk_size):
        batch, length, vocab = logits.shape
        flat = logits.reshape(-1, vocab)
        target = labels.reshape(-1)
        logp = torch.empty(flat.shape[0], device=logits.device, dtype=torch.float32)
        norm = torch.empty_like(logp)
        mean = torch.zeros_like(logp)
        for start in range(0, len(flat), chunk_size):
            end = min(start+chunk_size, len(flat))
            rows = torch.arange(start, end, device=logits.device)//length
            z = flat[start:end].float()/temperature[rows, None]
            lse = torch.logsumexp(z, dim=-1)
            norm[start:end] = lse
            logp[start:end] = z.gather(1, target[start:end, None]).squeeze(1)-lse
            if calculate_entropy:
                mean[start:end] = (torch.softmax(z, dim=-1)*z).sum(-1)
        ctx.save_for_backward(logits, labels, temperature, norm, mean)
        ctx.calculate_entropy, ctx.chunk_size = calculate_entropy, chunk_size
        entropy = (norm-mean) if calculate_entropy else torch.zeros_like(norm)
        return logp.view(batch, length), entropy.view(batch, length)

    @staticmethod
    def backward(ctx, logp_gradient, entropy_gradient):
        logits, labels, temperature, norm, mean = ctx.saved_tensors
        batch, length, vocab = logits.shape
        flat = logits.reshape(-1, vocab)
        target = labels.reshape(-1)
        gradient = torch.empty_like(logits)
        flat_gradient = gradient.view(-1, vocab)
        gp = logp_gradient.reshape(-1) if logp_gradient is not None else torch.zeros_like(norm)
        gh = entropy_gradient.reshape(-1) if entropy_gradient is not None else torch.zeros_like(norm)
        for start in range(0, len(flat), ctx.chunk_size):
            end = min(start+ctx.chunk_size, len(flat))
            rows = torch.arange(start, end, device=logits.device)//length
            temp = temperature[rows, None]
            z = flat[start:end].float()/temp
            probability = torch.exp(z-norm[start:end, None])
            factor = -gp[start:end, None]
            if ctx.calculate_entropy:
                factor = factor+gh[start:end, None]*(mean[start:end, None]-z)
            probability.mul_(factor)
            probability.scatter_add_(1, target[start:end, None], gp[start:end, None])
            probability.div_(temp)
            flat_gradient[start:end].copy_(probability)
        return gradient, None, None, None, None


def vocabulary_statistics(logits, labels, temperature, calculate_entropy=True, chunk_size=128):
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError('Vocabulary statistics need [batch, tokens, vocab] logits and aligned labels')
    temperature = temperature.to(device=logits.device, dtype=torch.float32)
    if temperature.shape != (logits.shape[0],) or not bool((temperature > 0).all()):
        raise ValueError('Invalid per-sequence temperature')
    return _VocabularyStatistics.apply(logits, labels, temperature, bool(calculate_entropy), int(chunk_size))


def install_streamed_vocab_statistics():
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    original = FSDPEngineWithLMHead.prepare_model_outputs
    if getattr(original, '_chess_r1_streamed_vocabulary', False):
        return True

    @wraps(original)
    def prepare(self, output, output_args, micro_batch, logits_processor_func):
        padded = not tu.get_non_tensor_data(micro_batch, 'use_remove_padding', True)
        fused = tu.get_non_tensor_data(micro_batch, 'use_fused_kernels', False)
        if not padded or fused:
            return original(self, output, output_args, micro_batch, logits_processor_func)
        if tu.get_non_tensor_data(micro_batch, 'pad_mode', DatasetPadMode.NO_PADDING) != DatasetPadMode.NO_PADDING:
            raise ValueError('Chess-R1 vocabulary adapter only supports native NO_PADDING batches')
        if tu.get_non_tensor_data(micro_batch, 'calculate_sum_pi_squared', False):
            raise ValueError('Chess-R1 does not enable sum_pi_squared')
        input_ids = micro_batch['input_ids']
        offsets = input_ids.offsets()
        lengths = offsets.diff()
        rolled = output_args['input_ids_rmpad_rolled']
        # Only token-label arrays are padded; full-vocabulary logits stay in place.
        labels = torch.nn.utils.rnn.pad_sequence(
            torch.split(rolled, lengths.tolist()), batch_first=True, padding_value=0)
        calculate_entropy = tu.get_non_tensor_data(micro_batch, 'calculate_entropy', False)
        logp, entropy = vocabulary_statistics(output.logits, labels, output_args['temperature'],
                                              calculate_entropy=calculate_entropy)
        valid = torch.arange(logp.shape[1], device=logp.device)[None, :] < lengths[:, None]
        result = {'log_probs': torch.nested.nested_tensor_from_jagged(logp[valid], offsets)}
        if calculate_entropy:
            result['entropy'] = torch.nested.nested_tensor_from_jagged(entropy[valid], offsets)
        return result

    prepare._chess_r1_streamed_vocabulary = True
    FSDPEngineWithLMHead.prepare_model_outputs = prepare
    return True


def numerical_acceptance(device='cpu'):
    """Compare full values and shared gradients with independent torch formulas."""
    torch.manual_seed(17)
    evidence = []
    for dtype in [torch.float32, torch.bfloat16]:
        x = torch.randn(2, 7, 97, device=device, dtype=dtype, requires_grad=True)
        reference = x.detach().float().requires_grad_(True)
        labels = torch.randint(97, (2, 7), device=device)
        temperature = torch.tensor([1., .7], device=device)
        weights = torch.randn(2, 7, device=device)
        p, h = vocabulary_statistics(x, labels, temperature, chunk_size=3)
        z = reference/temperature[:, None, None]
        log_all = z.log_softmax(-1)
        expected_p = log_all.gather(-1, labels[..., None]).squeeze(-1)
        expected_h = -(log_all.exp()*log_all).sum(-1)
        ((p+.001*h)*weights).sum().backward()
        ((expected_p+.001*expected_h)*weights).sum().backward()
        assert torch.allclose(p, expected_p, atol=2e-6, rtol=2e-6)
        assert torch.allclose(h, expected_h, atol=2e-6, rtol=2e-6)
        tolerance = .008 if dtype == torch.bfloat16 else 2e-6
        assert torch.allclose(x.grad.float(), reference.grad, atol=tolerance, rtol=tolerance)
        evidence.append({'dtype': str(dtype), 'logprob_max_abs_delta': float((p-expected_p).abs().max()),
            'entropy_max_abs_delta': float((h-expected_h).abs().max()),
            'joint_gradient_max_abs_delta': float((x.grad.float()-reference.grad).abs().max())})
    return {'status': 'passed', 'device': device, 'full_vocabulary': True,
            'independent_logsoftmax_reference': True, 'cases': evidence}
