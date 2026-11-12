---
name: rsi-meta-execution
description: Understand API or on-demand GPU Meta execution without modifying Student research rules.
---

The operator selects `meta_execution=api` (DeepSeek API) or `gpu_job` (Qwen3.5-4B thinking).
The framework owns Meta inference submissions; do not submit another inference card yourself.
One request may include many inference/tool calls; its one GPU allocation ends at handoff.
Meta cards use HIGH/model-autosft or PREEMPTABLE/model-new, model-claw-tool, model-codeagent2.
The framework prefers available H800s and shares the single HIGH slot with Student training.
When training occupies HIGH, Meta uses a PREEMPTABLE pool. A killed attempt is invalid and its edits/results
are not adopted; the framework restores the request baseline and reselects a pool.
Student training remains RL-only, fixed prompt and non-thinking in Phase1. Register Student
training/development evaluation in proposal.json; Controller/Executor submits it. Do not
run Student training on the Meta inference card or change a running frozen experiment.
Fresh studies use only their own Memory and evidence; do not inspect other study records.
See docs/meta-execution.md for operational details when present.
