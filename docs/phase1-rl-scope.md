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

## Phase 2: planned RL and harness research

Phase 2 is not active. At the transition, preserve the Phase 1 campaign, candidate snapshots, validation traces and selected model. Start an explicit new campaign/scope and record its parent candidate and source hashes. Give the Meta-Agent a prompt construction interface and skills format, then permit coordinated RL and harness changes in candidate snapshots. Keep the referee, legal-move validation, engine settings, answer attribution and raw per-move evidence outside the candidate's control.

Use matched L2/L3 validation seeds and four ablations where practical: Phase 1 model with fixed prompt, Phase 1 model with new harness, Phase 2 model with fixed prompt, and Phase 2 model with new harness. This separates policy improvement from prompt effects. Permit L2.5 as a diagnostic. Choose the final candidate using development validation, then run DeepSeek Flash non-thinking once as the held-out test. Do not reuse that test for further iteration.
