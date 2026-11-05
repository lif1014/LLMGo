# Meta-Agent research scope

## Phase 1: RL with a fixed prompt

The active `chess-grpo-v1` campaign is `phase1_rl_only`. The Meta-Agent may change the RL algorithm, reward or value estimate, sampling and curriculum, optimizer settings, training code, and selection of training positions. Each experiment must keep its own recipe, logs, checkpoints and model under `var/runs/<experiment-id>/`. It may submit L2, L2.5 or L3 development validation; a complete comparable panel uses both L2 and L3. It must inspect every move in losing games through `games/<id>/events.jsonl` and `summary.json` before proposing the next change.

The Qwen prompt is fixed to `src/llmgo/training/chess_r1.py` with `enable_thinking=False`. The exact system message is:

```text
You are a helpful assistant who plays chess professionally.
Given the FEN position and the list of legal moves, choose the best move.
Output exactly one legal SAN move inside \boxed{...}, for example \boxed{Nf3}.
The box must contain only the SAN move; output nothing else.
```

The user message is `Current FEN string: {fen}\nLegal moves: {legal SAN moves separated by spaces}`. The model must return only `\boxed{SAN}`. Training and evaluation use the non-thinking setting. Phase 1 proposals cannot change the prompt or answer parser, add `harness.py`, or populate `skills.json`; it stays `[]`. They must not change dataset prompt fields or training settings to bypass this rule. The Controller checks the frozen prompt source, recipe protocol, harness absence and empty skills before accepting a candidate. The official evaluator uses the default messages for Phase 1.

The original student model is `assets/models/Qwen3.5-4B`, training data is `assets/datasets/chess-r1/boxed-nonthinking-v2`, and the reward model is `assets/models/searchless-270M/270M`. The default GRPO entry is `scripts/run_grpo.py`. SFT and supervised distillation remain forbidden. DeepSeek Flash non-thinking is the held-out final opponent; its result cannot be used to tune the candidate.

The initial Qwen3.5-4B checkpoint cannot be replaced or overwritten. Continued training can use its derived research checkpoints. The default remains the filtered dataset; the full Lichess training-only pool is optional. The Meta-Agent decides whether and how to use it, and may still change RL algorithms, hyperparameters and reward. Every round delivers a selection program/query, selected ID list, category statistics and a hypothesis grounded in losing-game evidence. No data-selection random seed or compute-budget comparison is required. See [Lichess data](lichess-data.md) for the pool, exclusion rules and tools.

## Phase 2: fixed weights, open harness and skills

Phase 2 uses a separate `phase2_prompt_only` campaign, starting from either the base Qwen3.5-4B model or an explicit Phase 1 HF checkpoint. The Meta-Agent may freely develop harnesses, appended prompts, arbitrary skills and helper resources. Training or weight updates are forbidden and invalidate the stage score (0); this research rule is conveyed through the Meta-Agent prompt only, without training detection or command admission restrictions. The selected checkpoint, non-thinking mode, boxed SAN output and official evaluation rules remain research instructions.

Each candidate is evaluated against L2/L3 (40 games each), followed by another Agent decision. Preserve the Phase 1 campaign and all its evidence; Phase 2 has independent Memory, workspaces, Meta sessions and results. Deliver the selected checkpoint reference together with its frozen harness/skills source. Keep DeepSeek Flash non-thinking as the independent final test. See [Phase 2 operations](phase2-prompt-research.md) and the [complete Meta-Agent prompt](phase2-meta-prompt.md).
