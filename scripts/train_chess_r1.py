#!/usr/bin/env python3
"""Independent native verl GRPO launcher, executed only in a submitted GPU job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recipe', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    r = json.loads(a.recipe.read_text())
    assert r['framework'] == 'verl' and r['algorithm'] == 'grpo'
    assert not any(r[k] for k in ('sft', 'harness', 'skills'))
    assert r['answer_protocol'] == 'chess-r1-boxed-san-v2'
    assert r.get('enable_thinking', False) is False
    assert r['temperature'] == 1 and r['max_response_tokens'] == 16384
    assert type(r['checkpoint_every']) is int and r['checkpoint_every'] > 0
    a.output.mkdir(parents=True, exist_ok=True)
    run_lock = (a.output/'run.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.output/'result.json').exists():
        previous = json.loads((a.output/'identity.json').read_text())
        if previous['recipe'] != r:
            raise RuntimeError('Completed output belongs to another recipe')
        return
    from llmgo.training.chess_r1_resume import resume_plan
    source_release = Path(__file__).resolve().parents[1]
    resumed = resume_plan(a.output, r, source_release)
    os.environ.update(CHESS_R1_OUTPUT=str(a.output), WANDB_RUN_ID=r['run_id'],
                      TOKENIZERS_PARALLELISM='false', LLMGO_WAKE_CACHE_FLUSH='1')
    os.environ.update(CHESS_R1_RESUME_ATTEMPT=str(resumed['attempt']),
                      CHESS_R1_RESUME_STEP=str(resumed['step']))
    os.environ['CHESS_R1_MEMORY_ONLY'] = '1' if r.get('mode') == 'long_context_acceptance' else '0'
    segment_target = int(os.environ.get('CHESS_R1_SEGMENT_TARGET', r['steps']))
    if not 1 <= segment_target <= r['steps']:
        raise ValueError('Segment target outside the frozen training plan')
    os.environ['CHESS_R1_VERIFY_CHECKPOINT_RESTORE'] = str(int(
        r['steps'] == 2 and r.get('mode') != 'long_context_acceptance'))
    from llmgo.training.chess_r1_verl import atomic, progress, ChessR1TaskRunner
    from llmgo.training.chess_r1_config import overrides
    import ray
    import torch
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    gpus = int(r.get('allocated_gpus',8))
    if not 1 <= gpus <= 8 or torch.cuda.device_count() != gpus:
        raise RuntimeError('Visible GPUs do not match this submitted job recipe')
    identity = {'recipe': r, 'gpu_names': [torch.cuda.get_device_name(i) for i in range(gpus)],
                                    'python': sys.executable, 'verl': verl.__version__,
                                    'source_release': str(Path(__file__).resolve().parents[1]),
                                    'compile_environment': {k:os.environ.get(k) for k in
                                        ('MAX_JOBS','FLASHINFER_NVCC_THREADS','TORCHINDUCTOR_COMPILE_THREADS',
                                         'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS','FLASHINFER_WORKSPACE_BASE')}}
    if not (a.output/'identity.json').exists():
        atomic(a.output/'identity.json', identity)
    atomic(a.output/'resume.json', resumed)
    if r.get('mode') == 'long_context_acceptance':
        # Ray's TaskRunner is CPU-only. Exercise CUDA before creating it, inside
        # this submitted GPU job, then release temporary test allocations.
        from llmgo.training.chess_r1_attention import gpu_acceptance
        progress('attention_gpu_parity')
        atomic(a.output/'attention-gpu-acceptance.json', gpu_acceptance())
        from llmgo.training.chess_r1_vocab import numerical_acceptance
        atomic(a.output/'vocabulary-gpu-acceptance.json', numerical_acceptance('cuda'))
        torch.cuda.empty_cache()
    progress('starting_original_270m')
    source = str(Path(__file__).resolve().parents[1]/'src')
    oracle_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', PYTHONPATH=source,
                      OMP_NUM_THREADS='8', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    child = None
    log = (a.output/'oracle.log').open('a')
    try:
        child = subprocess.Popen([r['teacher_python'], '-m', 'llmgo.training.chess_r1_oracle',
                                  '--upstream', r['teacher_upstream'], '--checkpoint', r['teacher_checkpoint'],
                                  '--output', str(a.output/'oracle')], env=oracle_env,
                                 stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+360
        while not (a.output/'oracle/ready.json').is_file():
            if child.poll() is not None:
                raise RuntimeError('Original270M service failed; see own oracle.log')
            if time.monotonic() > deadline:
                raise TimeoutError('Original270M readiness timed out')
            time.sleep(2)
        os.environ['CHESS_R1_ORACLE_URL'] = json.loads((a.output/'oracle/ready.json').read_text())['url']
        namespace = 'c1-'+uuid.uuid4().hex[:8]
        env = {k: os.environ[k] for k in ('CHESS_R1_ORACLE_URL', 'CHESS_R1_OUTPUT', 'WANDB_RUN_ID',
                                          'TOKENIZERS_PARALLELISM', 'LLMGO_WAKE_CACHE_FLUSH', 'CHESS_R1_MEMORY_ONLY',
                                          'CHESS_R1_VERIFY_CHECKPOINT_RESTORE')}
        env.update({k: os.environ[k] for k in ('CHESS_R1_RESUME_ATTEMPT', 'CHESS_R1_RESUME_STEP')})
        if 'CHESS_R1_SEGMENT_TARGET' in os.environ:
            env['CHESS_R1_SEGMENT_TARGET'] = str(segment_target)
        env['PYTHONPATH'] = source
        for key in ('MAX_JOBS','FLASHINFER_NVCC_THREADS','TORCHINDUCTOR_COMPILE_THREADS',
                    'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS','FLASHINFER_WORKSPACE_BASE',
                    'TORCHINDUCTOR_CACHE_DIR','TORCH_EXTENSIONS_DIR','VLLM_CACHE_ROOT'):
            if key in os.environ:
                env[key] = os.environ[key]
        runtime = Path(os.environ.get('LLMGO_RUNTIME_ROOT', '/user/lyh4170/LLMGo!/var/r'))
        runtime.mkdir(parents=True, exist_ok=True)
        ray.init(address='local', num_gpus=gpus, namespace=namespace,
                 _temp_dir=str(runtime/namespace), runtime_env={'env_vars': env})
        with initialize_config_dir(config_dir=str(Path(verl.__file__).parent/'trainer/config'), version_base=None):
            runtime_recipe = {**r, 'resume_from_path': resumed['checkpoint']}
            cfg = compose(config_name='ppo_trainer', overrides=overrides(runtime_recipe, a.output))
        atomic(a.output/'resolved-config.json', OmegaConf.to_container(cfg, resolve=True))
        progress('initializing_native_verl')
        run_ppo(cfg, task_runner_class=ray.remote(num_cpus=1)(ChessR1TaskRunner))
        workers = [json.loads(p.read_text()) for p in (a.output/'worker-compatibility').glob('*.json')]
        if len(workers) != gpus or not all(w['mrope_guard_installed'] and
                                          w['streamed_full_vocabulary_statistics'] and
                                          w['inference_jagged_positions_repaired'] and
                                          w['weight_remap_cache_flush'] for w in workers):
            raise RuntimeError('Compatibility not verified in every actual GPU worker')
        if r.get('mode') == 'long_context_acceptance':
            result = json.loads((a.output/'long-context-acceptance.json').read_text())
            assert result['status'] == 'passed'
            result['gpu_worker_compatibility'] = workers
            atomic(a.output/'result.json', result)
            progress('completed', step=2)
            return
        receipts = list((a.output/'checkpoints').glob('receipt-*.json'))
        if not receipts or not list((a.output/'updates').glob('step-*.json')):
            raise RuntimeError('No complete optimizer/checkpoint evidence')
        if not (a.output/'checkpoints'/f'receipt-{segment_target:08d}.json').is_file():
            raise RuntimeError('Segment ended without its complete checkpoint')
        result_path = a.output/'result.json' if segment_target == r['steps'] else a.output/'segment-result.json'
        atomic(result_path, {'status': 'completed' if segment_target == r['steps'] else 'segment_complete', 'steps': segment_target, 'run_id': r['run_id'],
                                       'gpu_worker_compatibility': workers,
                                       'checkpoint_receipts': [json.loads(x.read_text()) for x in receipts]})
        progress('completed' if segment_target == r['steps'] else 'segment_complete', step=segment_target)
    except BaseException as error:
        atomic(a.output/'failure.json', {'type': type(error).__name__, 'error': str(error)[-6000:]})
        progress('failed', error_type=type(error).__name__)
        raise
    finally:
        ray.shutdown()
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        log.close()


if __name__ == '__main__':
    main()
