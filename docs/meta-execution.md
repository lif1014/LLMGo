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

Meta inference always requests exactly one H800. The November12 operator update allows:

| Pool | Meta priority |
|---|---|
| model-autosft | HIGH |
| model-new | PREEMPTABLE |
| model-claw-tool | PREEMPTABLE |
| model-codeagent2 | PREEMPTABLE |

Choose the pool with the most available H800 quota headroom; ties prefer fewer queued
people, then fewer queued tasks, then table order. If none has a free card, choose the
shortest queue. This replaces the earlier two-pool/five-person threshold policy.
Headroom is `capacity-used` in beijing, not a guarantee of immediate placement. Queued
people are distinct visible H800 submitters, excluding CPU jobs and other GPU types.

HIGH admission is shared with Student training: one active or uncertain HIGH submission
across both. If that slot is occupied, exclude autosft and select a PREEMPTABLE pool.
Selection checks availability and submission rechecks under the existing shared file lock;
an uncertain HIGH create retains its reservation until reconciled. A queued own HIGH Meta
task does not count as a competing HIGH task when deciding whether to move pools.
A failed pool query excludes its telemetry; if all queries fail, retain the warnings and
queue in model-new/PREEMPTABLE without assuming available capacity or claiming a free HIGH
slot. Existing NORMAL attempts retain their recorded priority and remain reconcilable.
Other users' tasks are never stopped or modified; queue summaries alone inform selection.

Platform polls are persisted and at least600 seconds apart, including Controller restart.
Before each new allocation and after a killed attempt, all four pools are re-evaluated. If a
still-queued own Meta task can move to a pool with strictly more free H800s (at least one),
its identity is checked, it is stopped, and a replacement is submitted only after the
original is confirmed terminal. Equally full pools retain queue position rather than
cycling due to small changes in queue length. Running
tasks are not killed merely because queue conditions changed.

Killed/stopped allocations are invalid even if `result.json` already exists. Their edits
are archived as `discarded-workspace`; the registered workspace is restored to the request
baseline before retry. Each attempt gets a new Codex home/session and prompt attempt nonce.
Only a platform-Succeeded allocation with the matching request result can be adopted.
Other execution failures stop for diagnosis rather than endlessly repeating deterministic
errors. Ambiguous create responses retain the intent and reconcile the exact own task;
there is no blind duplicate submission. Training and Meta use the same HIGH admission registry.

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
Codex to actually execute a tool and copy a nonce file before passing. The check's1,800-second
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

Qwen's chat template requires system instructions at the beginning. For Qwen only, the
bridge combines Codex's interleaved system/developer instruction messages into leading
Responses.instructions before the pinned LiteLLM conversion. User/assistant/tool ordering,
call IDs and content remain intact; the original request is retained as evidence. DeepSeek
API input conversion is unchanged. GPU acceptance identified this requirement before any
research was started.

Qwen streaming replies send SSE comment heartbeats every20 seconds while awaiting upstream
creation or the next event. Comments do not fabricate model output or extend token limits;
they prevent Codex from retrying a valid long first generation after its stream idle timeout.
The engineering tool-check deadline is30 minutes; production research has no overall deadline.
