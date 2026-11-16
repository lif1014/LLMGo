---
name: rsi-meta-execution
description: Understand API or on-demand GPU Meta execution without modifying Student research rules.
---

The operator selects `meta_execution=api` (DeepSeek API) or `gpu_job` (Qwen3.8-27B thinking).
The framework owns Meta inference submissions; do not submit another inference card yourself.
One request may include many inference/tool calls; its one GPU allocation ends at handoff.
Meta cards use NORMAL/model-autosft or PREEMPTABLE/model-new, model-claw-tool, model-codeagent2.
HIGH is reserved for Student training; a busy HIGH slot does not block NORMAL Meta inference.
The framework prefers available H800s; NORMAL Meta inference remains eligible while training occupies HIGH. A killed attempt is invalid and its edits/results
are not adopted; the framework restores the request baseline and reselects a pool.
Student training remains RL-only, fixed prompt and non-thinking in Phase1. Phase2 keeps Student
weights fixed and changes harness/prompts/skills; it permits no training. Register the appropriate
development evaluation in proposal.json; Controller/Executor submits it. Do not
run Student training on the Meta inference card or change a running frozen experiment.
Fresh studies use only their own Memory and evidence; do not inspect other study records.
See docs/meta-execution.md for operational details when present.
