# Qwen3.8-27B Meta-Agent

Meta uses Qwen3.8-27B thinking on one on-demand H800. Student remains the original
Qwen3.5-4B, fixed Chess-R1 prompt and non-thinking mode. This profile does not start
or resume a study by being installed. DeepSeek API execution remains supported.

## Inference profile

`configs/rsi/site.qwen38-meta.json` declares:

| Setting | Value |
| --- | --- |
| Model | Qwen/Qwen3.8-27B; pinned source revision in download receipt |
| Initial precision | BF16, TP=1, language-model-only, max_num_seqs=1 |
| Context window | 131,072 tokens, including input/history/output |
| Maximum output per request | 65,536 tokens, including thinking |
| Codex compaction trigger | 57,344 tokens (not an independent input allowance) |
| Thinking | enabled, reasoning_effort=xhigh, preserve_thinking=true |
| Sampling | temperature=1, top_p=.95, top_k=20, min_p=0, presence_penalty=0, repetition_penalty=1 |
| Meta Python | /user/lyh4170/venvs/llmgo-qwen38-meta-20261113/bin/python |

These sampling settings follow the [official model card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices).
The 128K/64K/56K limits are this project's engineering profile. Output capacity is
not a promise that every call generates 64K tokens. The independent Meta environment
uses vLLM0.19.1/Transformers5.8.1 with the image's Torch2.10 CUDA12.9; deployment must
record actual import and GPU acceptance results. Student inference/training environments
are not upgraded. BF16 feasibility on80GB H800 is measured, not inferred from model size.
If another precision is tested, store a separate profile and results; do not mix it into BF16.

## Current allocation policy

All new jobs, including CPU maintenance, downloads, Student training/evaluation and
Meta inference, use PREEMPTABLE in beijing/model-new, model-claw-tool or model-codeagent2.
No new model-autosft submission is allowed. The submission boundaries enforce this;
legacy HIGH admissions and historical task records remain readable but cannot authorize
new jobs. Meta selects the most available H800 headroom, then shorter queues. Unknown
telemetry never means zero or free capacity. Platform polls remain600 seconds apart.
Preempted Meta attempts and their edits are invalid; restore the request baseline and
reselect among the three authorized pools. No other user's task is stopped or modified.

One card is held for a complete research/review request and its tools, then released.
It is not a permanent devspace and not a new allocation for each token/API call. The
existing limitation remains: a Student allocation waits at checkpoint reviews until a
valid decision; this feature does not migrate Student state across allocations.

## Schema and factual checks

`check_research_proposal.py` and `finish_research.py` inspect the actual workspace context.
Normal research uses the executable proposal contract. A checkpoint review instead requires
`training_review.request_id`, `action`, nonempty `reason` and an evaluated `selected_step`.
An empty research proposal cannot masquerade as a review. Invalid handoffs return an error
inside the same Meta turn; they do not authorize an automatic operator decision.

The optional Lichess selection helper remains optional. The Agent may retain default data,
use category mixtures or write its own selector. Every claimed data change must be reflected
in the submitted recipe and delivery receipts, not only narrative text.

## Independent acceptance and comparison

`scripts/check_meta_capability.py` exercises20 synthetic factual/recipe/review cases,
three repetitions by default, with a fresh workspace and Codex home per case. It starts
no Student work. Both compared models use the same context/output/compaction limits,
sampling, tools, schema checker and fixtures. Existing study results are not inputs.

It records factual completion, first required schema-check success, correction attempts,
input-file integrity, per-case latency, token usage when reported and category counts.
First-check rate is not a measurement of unseen drafts. Repetitions of one case are not
independent samples; report per-case paired differences rather than claiming significance
from a single successful readiness check. Engineering failures and preemptions are separate
from completed capability scores. Long-context checks use real ~60K/~120K inputs, with
64K/8K declared reply budgets, and verify a middle-of-document key; they do not claim64K
output tokens were actually generated. Compaction thresholds are recorded in the Codex
config; a threshold is not evidence that compaction happened in a short case.

```bash
python scripts/check_meta_capability.py \
  --site configs/rsi/site.qwen38-meta.json --name qwen38-capability-v1 \
  --repetitions 3 --long-context
```

This command explicitly allocates GPU acceptance work. It never starts the formal study.
Results are in `var/operator-checks/<name>/meta_requests/.../attempts/.../benchmark-result.json`;
platform-Succeeded request-bound attempts are the only adoptable results. Operator deployment
and comparison receipts are under `var/runs/qwen38-meta-20261113/`.

## New study

Prepare a new clean source (no old Memory, sessions, hypotheses or trained Student):

```bash
cd '/user/lyh4170/LLMGo!'
/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python scripts/prepare_qwen_research.py
```

After reviewing acceptance results, the user explicitly starts:

```bash
bash '/user/lyh4170/LLMGo!/var/clean-starts/qwen38-27b-meta-v1/scripts/start_rsi.sh' \
  --site '/user/lyh4170/LLMGo!/var/clean-starts/qwen38-27b-meta-v1/configs/rsi/site.qwen38-meta.json' \
  --campaign chess-grpo-qwen38-27b-meta-v1
```

Fresh initialization measures Student L2/L3 baselines before the first research proposal.
Do not import the old Qwen4B teacher's checkpoint or records. Its paused campaign may be
marked archived to hide it from active Web selectors; original receipts and its Student
checkpoint remain retained separately. DeepSeek history is not archived by this action.
Shared-disk routing is not a security sandbox: the new instructions prohibit reading other
studies, but account-level read access still exists. A study is not initialized during
installation, and its name must not be reused to overwrite existing evidence.

The common Web panel shows model, thinking, precision, context/output limits, execution
stage, exact pool/task ID, attempt history and measured cost for GPU/API modes. An archived
campaign remains accessible by its known URL but is omitted from the active selector.
