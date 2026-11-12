# Meta execution: API and on-demand Qwen GPU

The existing DeepSeek API profile, `configs/rsi/site.beijing.json`, remains the default.
`meta_execution` defaults to `api`; existing campaigns and sessions continue through the
same Codex runner and Responses-to-Chat bridge. No DeepSeek game evaluator is changed.
The Qwen profile, `configs/rsi/site.qwen-meta.json`, selects `gpu_job`, the original
Qwen3.5-4B, thinking enabled, 65,536 context tokens, 45,000-token compaction trigger,
and a 16,384-token maximum model reply. Student non-thinking is independent.

A GPU allocation covers one complete research request or checkpoint review, including
multiple model calls and tool execution. It ends after handing over its answer and
stopping vLLM. This is a batch job, not a GPU devspace or a permanent model server.
Codex, the private bridge and vLLM run together; model traffic stays on loopback.
A paused campaign is never resumed by installing this feature.

## Resource policy

Meta inference always requests exactly one H800. It never requests HIGH.

| model-autosft H800 headroom / queue | model-new H800 headroom / queue | Selection |
|---|---|---|
| At least1 free, or fewer than5 queued people | Any | model-autosft / NORMAL |
| Fewer than1 free AND at least5 queued people | At least1 free, or fewer than5 queued people | model-new / PREEMPTABLE |
| Fewer than1 free AND at least5 queued people | Zero free AND at least5 queued people | model-autosft / NORMAL |

Headroom is the current pool H800 quota `capacity-used` in beijing, not a promise of
immediate placement. Queue people are distinct visible submitters of queued H800 jobs,
not GPU count or duplicate jobs by the same person. `meta_min_free_gpus=1` defines
"too few" for this one-card request. Missing/incomplete telemetry conservatively selects
NORMAL/autosft; it never authorizes HIGH or guesses zero queue. Other users' task details,
commands and credentials are neither modified nor fed to the research model.

Platform polls are persisted and at least600 seconds apart, including Controller restart.
Before each new allocation and after a killed attempt, both pools are re-evaluated. If a
still-queued own Meta task belongs in the other pool, its identity is checked, it is stopped,
and a replacement is submitted only after the original is confirmed terminal. Running
tasks are not killed merely because queue conditions changed.

Killed/stopped allocations are invalid even if `result.json` already exists. Their edits
are archived as `discarded-workspace`; the registered workspace is restored to the request
baseline before retry. Each attempt gets a new Codex home/session and prompt attempt nonce.
Only a platform-Succeeded allocation with the matching request result can be adopted.
Other execution failures stop for diagnosis rather than endlessly repeating deterministic
errors. Ambiguous create responses retain the intent and reconcile the exact own task;
there is no blind duplicate submission. The existing single-HIGH training admission is unchanged.

## New study and launch

On the remote filesystem:

```bash
cd '/user/lyh4170/LLMGo!'
/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python -B scripts/prepare_qwen_research.py
```

This builds `var/clean-starts/qwen-meta-v1`, with no Git history, prior campaign Memory,
Codex sessions, benchmark recipes or research conclusions. It retains common implementation
fixes, the default GRPO template, generic skills/tests and clean research instructions.
`assets` and `var` route to the project assets/artifact roots so existing workers and Web
APIs can read the new outputs. This is clean input, not a security sandbox: same-account
filesystem/network access can still reach other studies. The new instructions forbid that;
strict read isolation requires a separate filesystem/container policy.

Start the new study explicitly, from its prepared source:

```bash
bash '/user/lyh4170/LLMGo!/var/clean-starts/qwen-meta-v1/scripts/start_rsi.sh' \
  --site '/user/lyh4170/LLMGo!/var/clean-starts/qwen-meta-v1/configs/rsi/site.qwen-meta.json' \
  --campaign chess-grpo-qwen-meta-v1
```

Fresh initialization measures Student L2/L3 baselines and imports no old trained checkpoint
or result. Start commands are not run by the installer. For the old DeepSeek study, keep
using its original site and campaign; creating another API study is also supported.
Changing models inside an existing frozen campaign is not the switch mechanism.

An explicit engineering check, which allocates one GPU but does not train a Student:

```bash
/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python -B scripts/check_gpu_meta.py \
  --site configs/rsi/site.qwen-meta.json
```

It uses the same pool policy and worker, starts thinking/tool-capable Qwen, and requires
Codex to actually execute a tool and copy a nonce file before passing. The check's600-second
model/tool deadline is specific to this acceptance test; production research has no overall
deadline. Launching the normal study does not allocate a separate startup-probe GPU.

## Artifacts and Web

`var/campaigns/<campaign>/meta_requests/<request-hash>/` stores the request and immutable
baseline; `attempts/0001/` etc. store reports, source, submission intent/receipt, worker/vLLM
logs, private Codex home, session and result. All paths stay under `/user/lyh4170`.
The GPU worker uses short `var/runtime/mqwen` for vLLM sockets/cache and a per-attempt bridge
log directory. No source credentials or old DeepSeek config are copied into its new home.

The read-only `/api/rsi/campaigns/<campaign>/meta` endpoint projects both modes. A small
`meta-overview.json` compatibility artifact lets existing Web Python processes serve the
new panel without a backend restart. The page reads disk every15 seconds and never queries
cctl or invokes a model. Older campaigns without request records display legacy status.

The common panel shows model, thinking, API/GPU, queue/load/reasoning/tool/finish stages,
individual attempts, pool/priority/task ID, reported tokens and original artifact references.
Campaign switching resets selection and cancels stale rendering. Existing thinking/tool
feeds and Student replay remain shared. API GPU cost is "not applicable", not zero compute.
Meta GPU-hours describe worker lifetime including startup/tools; queue is excluded, invalid
attempts remain in cost. The panel covers the most recent30 requests and labels this limit.
Student training/evaluation costs remain in the existing GRPO and evaluation panels.

The current Student allocation still waits at its safe checkpoint while a Meta task queues
and runs. This change removes a permanent Meta GPU; it does not release/reallocate the8
Student cards at every review. Cross-job Student resume can be a separate future change.
