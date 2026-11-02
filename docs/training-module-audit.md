# `src/llmgo/training` file audit

This directory contains the GRPO implementation, its reward service, and the compatibility code needed by the pinned remote environment. The imported class is named `RayPPOTrainer` in verl, but the selected advantage estimator is GRPO; this is not the removed complete-game PPO pipeline.

| File | Decision | Reason |
| --- | --- | --- |
| `__init__.py` | Keep | Package marker. |
| `chess_r1.py` | Keep, freeze in Phase 1 | Shared FEN/SAN prompt and answer protocol used by GRPO reward and evaluation. It is not an optimizer; moving it during the fixed-prompt phase would complicate snapshot identity. |
| `chess_r1_agent.py` | Keep | verl single-turn sampling adapter. |
| `chess_r1_agent.yaml` | Keep | verl agent registration and sampling penalties. |
| `chess_r1_attention.py` | Keep | Qwen3.5 attention compatibility and GPU acceptance. |
| `chess_r1_compat.py` | Keep | Worker-local compatibility installer. |
| `chess_r1_config.py` | Keep | Native verl GRPO configuration. |
| `chess_r1_memory.py` | Keep as a diagnostic | The 16K synthetic backward/remap acceptance is called only in `long_context_acceptance` mode. Its name is historical; it is not the old RSI Memory system. |
| `chess_r1_oracle.py` | Keep | CPU service for the 270M reward model. |
| `chess_r1_resume.py` | Keep | Checkpoint identity and resume validation. |
| `chess_r1_reward.py` | Keep | GRPO reward manager. |
| `chess_r1_verl.py` | Keep | GRPO runner and native verl integration. |
| `chess_r1_vocab.py` | Keep | Memory-bounded log-probability and entropy calculations. |
| `chess_r1_worker.py` | Keep | GPU worker compatibility hooks. |
| `export_weights.py` | Keep | HF model export from native verl checkpoints. |
| `game.py` | Move to `src/llmgo/rsi/game.py` | Complete chess games, referee events and replay belong to evaluation, not optimizer code. |
| `verl_memory.py` | Keep | vLLM/verl cache release during weight remapping; unrelated to old RSI Memory. |
| `vllm_compat.py` | Keep | Pinned vLLM/Transformers compatibility used by training workers. |

The training modules can be reviewed for simplification after the first successful full GRPO run. In particular, keep the long-context acceptance and compatibility hooks until GPU evidence confirms they are unnecessary.
