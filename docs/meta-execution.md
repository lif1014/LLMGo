# Meta execution: API and on-demand GPU

The current GPU teacher is Qwen3.8-27B thinking. The common Codex/bridge/controller
interfaces still support DeepSeek API execution. See [Qwen3.8 deployment and acceptance](qwen38-meta.md)
for the current128K context,64K output,56K compaction profile and exact launch commands.
All new jobs use PREEMPTABLE/model-new, model-claw-tool or model-codeagent2; model-autosft
is disabled. Existing paused studies are not resumed by installation.

The sections below explain common behavior and historical implementation. Old Qwen3.5-4B
Meta settings and HIGH/model-autosft descriptions are historical, not current launch policy.


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

Only beijing/model-new, model-claw-tool, model-codeagent2 PREEMPTABLE submissions are
currently authorized for Meta and Student jobs, including CPU maintenance. Choose available
H800 headroom first, then shorter queues. GPU platform polls persist600 seconds across
restarts. A killed/stopped Meta attempt is invalid even with a result file; archive its
edits, restore its baseline and reselect. Never move a running task merely for a better
queue, and never stop another user's work. Legacy task reads retain their recorded identity.

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
