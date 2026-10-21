# 已成功的任务格式、Agent 提交方式与本次文件清单

更新：2026-10-21。下列记录包括先前成功的CPU检查/配置复制与后续目录核验，失败尝试也保留；均不是GPU训练结果。运行框架在本机实现，由用户上传；没有把本机测试冒充远端GPU实验。

## 1. 已成功的任务和可核对事实

| Task ID | 资源 | 用途与结果 |
|---|---|---|
| 1310237 | GPU0、CPU1、2GiB | 只读检查，Succeeded；结束过快，未取得原始归档日志，不用它推断详细路径事实 |
| 1310259 | GPU0、CPU1、2GiB | 只读检查，Succeeded；实际取得目录、来源配置/程序及模型路径的存在性日志 |
| **1310452** | GPU0、CPU1、2GiB | Codex 配置复制到用户自己的目录，校验内容一致、权限及来源未变，Succeeded |
| **1310479** | GPU0、CPU1、2GiB | 修正回执 JSON 末尾换行并独立复核持久副本 SHA/权限，Succeeded |
| 1318148 | GPU0、CPU1、2GiB | 后续部署核验的首次脚本；Failed，归档确认入口换行转义SyntaxError，没有执行检查；未发送stop |
| **1318184** | GPU0、CPU1、2GiB | 入口先本机编译再提交的修正版，Succeeded；自己目录缺少当前RSI入口、工具、venv和BT4，原学生路径存在；回执只写自己的rsi_readiness目录 |

上述任务均为 `project=rsi`、`cluster=beijing`、`resourcePoolId=525`（`model-autosft`）、`priority=NORMAL`，项目关联共享文件系统385。既有 **devspace1309690** 为0GPU；它与这些独立 job 是不同实例。

1310452 的真实 description 是 `llmgo-codex-config-copy-20261021-000212`，平台创建时间 `2026-10-20T16:02:13.198492Z`，校验回执时间 `2026-10-20T16:02:55.182872+00:00`。复制结果：

```text
来源（只读）：/user/liyaohu4170/.codex/config.toml
目标：/user/lyh4170/.codex-rsi/config.toml
SHA256：9d4360ad1ca94a19278d1689530de2626282148eb5f3bdc393b53b66738a0cb9
文件权限：0600；目标目录：0700
source_bytes_unchanged：true
authentication_tested：false
回执：/user/lyh4170/.codex-rsi/config-copy-receipt.json
```

没有把源配置正文、密钥或认证 token 放进项目。配置复制已经完成，不应每轮重新复制。程序安装、认证和真实强模型调用仍要在自己的 devspace 验证。

本机保留的操作证据在 `/tmp/llmgo-config-copy-submission.json`、`/tmp/llmgo-config-copy-1310452-logs.jsonl`、`/tmp/llmgo-config-copy-1310452-receipt.json`、`/tmp/llmgo-config-copy-receipt-verification-submission.json`、`/tmp/llmgo-config-copy-1310479-logs.jsonl`；这些是本机临时证据路径，不要求远端存在，也不复制认证内容入库。

后续核验回执：远端 `/user/lyh4170/rsi_readiness/llmgo-readiness-20261021T042452714112.json`，本机 `/tmp/llmgo-readiness-result-1318184.json`。检查容器包版本与devspace环境分开记录，详见plan.md第24节。真实API还证明job list只返回tasks/ID资源名；已修复Executor的句柄规范化/详情补查/别名去重/不完整枚举处理，并只读找回原任务1310479，无重复创建。该修复后的完整Python回归为169通过、3默认跳过。

## 2. 之前成功提交采用的格式

实际命令的资源部分如下；`ENTRY` 当时是经过 `shlex.join` 转义的 `python3 -B -u -c SCRIPT`，SCRIPT 只执行获授权的检查/复制工作。

```text
cctl job create
  --project rsi
  --cluster beijing
  --resource-pool model-autosft
  --billing-account-id N00069
  --priority NORMAL
  --image b-side/vllm/vllm-openai:v0.23.0-x86_64-cu129-ubuntu2404
  --gpu 0 --cpu 1 --memory 2
  --description UNIQUE_DESCRIPTION
  --entry ENTRY
  --no-input -o json
```

这是参数结构说明，逐行展示未加 shell 续行符。`--memory 2` 在返回对象中为2GiB。镜像被成功用于这些 CPU 任务，并不证明它已经安装全部训练依赖。

创建返回的是 JSON对象，不是任务 ID 的纯数字文本。1310452 提交响应的关键真实字段为：

```json
{
  "id": 1310452,
  "kind": "BATCH",
  "project": "rsi",
  "cluster": "beijing",
  "resourcePoolId": 525,
  "filesystemId": 385,
  "priority": "NORMAL",
  "status": "Queued",
  "resourcesPerNode": {"cpu": 1, "memory": "2Gi"},
  "description": "llmgo-codex-config-copy-20261021-000212"
}
```

`Queued` 只是提交瞬间的状态；必须后续 get 确认 `Succeeded` 并核对产物，不能仅凭获得 ID 宣称完成。

### Agent 手写独立 CPU 诊断任务时

这是可复用的参数数组示例，运行它会真的提交一项新任务；示例只检查自己的目录是否存在，不重复历史配置复制：

```python
import datetime
import json
import os
import shlex
import subprocess

script = "from pathlib import Path; import json; print(json.dumps({'own_root_exists': Path('/user/lyh4170').is_dir()}), flush=True)"
entry = shlex.join(["python3", "-B", "-u", "-c", script])
description = "llmgo-own-check-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
argv = [
    "cctl", "job", "create", "--project", "rsi", "--cluster", "beijing",
    "--resource-pool", "model-autosft", "--billing-account-id", "N00069",
    "--priority", "NORMAL",
    "--image", "b-side/vllm/vllm-openai:v0.23.0-x86_64-cu129-ubuntu2404",
    "--gpu", "0", "--cpu", "1", "--memory", "2",
    "--description", description, "--entry", entry, "--no-input", "-o", "json",
]
env = {**os.environ, "CCTL_NO_UPDATE_CHECK": "1"}
# 先将 description/argv/时间作为 intent 保存到自己目录，再调用；网络不确定时先对账。
result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=90, env=env)
task = json.loads(result.stdout)
print(json.dumps({"id": task["id"], "description": description, "status": task["status"]}))
```

使用 `subprocess.run(argv)`，不使用 `shell=True`。外层没有 shell 不代表 entry 不经远端 shell；因此 entry 内仍要 `shlex.join`，不能用 JSON 转义替代 shell 转义。先查 `cctl pool list --own`，不要自行换池、加优先级或改 billing account。

查询与日志的已验证格式：

```bash
CCTL_NO_UPDATE_CHECK=1 cctl pool list --own --no-input -o json
CCTL_NO_UPDATE_CHECK=1 cctl job get 1310452 --no-input -o json
CCTL_NO_UPDATE_CHECK=1 cctl job list --own --cluster beijing --project rsi \
  --pool model-autosft --page 1 --limit 100 --no-input -o json
CCTL_NO_UPDATE_CHECK=1 cctl job logs 1310452 --pod job-rsi-1310452-main-0 --page-size 200
```

注意 `job logs` 这里不带 `-o json` 或 `--no-input`；当前安装版本的日志 helper 会把不支持的参数向下传而报错。历史 CPU 检查脚本曾在输出后保留约120秒以抓实时日志，这只是日志获取措施，不是强 Agent 的研究时限，也不是新训练框架必须 sleep 的步骤。

本机当时无法直连 devspace 内网 SSH；CLI 未提供已验证的 exec。实际采用共享盘上的独立 CPU job完成检查/复制，没有把 `cctl job` 描述为已进入既有 devspace 执行命令。

## 3. 新框架中 Agent 应如何提交

一般 **不手写 `cctl job create`**。在自己的0GPU devspace启动 `scripts/rsi.py run CAMPAIGN` 后：

1. Controller 先完成现任20局基线，为强 Agent 建立 workspace。
2. 强 Agent 阅读 `context.json`、Memory 与必要的原始证据，编辑 `skills.json`、`curriculum.jsonl`、`proposal.json`。
3. 强 Agent 返回 `ready` 和 proposal绝对路径；Controller校验并冻结它，生成有哈希的 `jobs/jNNNNNN/manifest.json`。
4. Controller 保存 `submission.json` intent，再调用cctl提交；强 Agent 不对同一实验另建第二个任务。
5. Worker核对manifest、源码、skills、课程与模型身份；计算完成写 `result.json` 和对应产物哈希。Controller同时核对平台状态和产物，再验收。

proposal例子和RL字段见 [操作文件第4–5节](rsi-operations.md)。新增提交代码生成的GPU任务结构为：

```text
cctl job create --project rsi --cluster beijing --resource-pool model-autosft
  --billing-account-id N00069 --priority NORMAL
  --image SITE_IMAGE --gpu 1 --gpu-model h800 --cpu 16 --memory 128
  --description llmgo-rsi:CAMPAIGN:j000001:MANIFEST_SHA
  --entry "SITE_PYTHON FROZEN_SOURCE/scripts/rsi_worker.py --manifest MANIFEST_PATH --manifest-sha256 MANIFEST_SHA"
  --no-input -o json
```

上面是新实现的模板，**本次没有把它冒充已经提交成功的GPU任务**。真实 entry由参数数组转义生成，带`!`的路径也会正确处理。项目关联共享盘385；不是本地打包自动传输，必须先由用户上传项目。

运维想先查看具体提交参数，可在 Controller 未运行时：

```bash
cd '/user/lyh4170/LLMGo!'
python3 scripts/rsi.py init chess-rsi-001
python3 scripts/rsi.py step chess-rsi-001
python3 scripts/rsi.py preview chess-rsi-001
python3 scripts/rsi.py submit chess-rsi-001
python3 scripts/rsi.py run chess-rsi-001
```

这是另一套从新campaign开始的示例；若已经init，直接用现有campaign，不要再次init。`step`第一次只准备基线manifest，`preview`不提交，`submit`才产生任务。平常一条 `run`已经处理这些步骤。

提交响应逾时只做唯一description对账，不自动盲重提；找不到或找到多个会报告不确定。重新运行沿用intent。Controller进程锁、Codex进程身份和平台task ID共同用于恢复，不能仅看过期锁文件就重启训练。失败恢复见操作文件第6节。

## 4. 此次新增文件

以下为本轮实现新增；`plan.md`与`skills_docs.md`在先前规划阶段已经存在，不能仅因Git尚未追踪就说它们都是本轮新建。

| 文件 | 用途 |
|---|---|
| `scripts/rsi.py` | devspace Controller命令行入口 |
| `scripts/rsi_worker.py` | 冻结manifest驱动的GPU Worker入口 |
| `configs/rsi/site.beijing.json` | 自己目录、程序、模型及固定资源配置 |
| `requirements-rsi-control.txt` | Controller/网页依赖 |
| `requirements-rsi-worker.txt` | 训练额外依赖，尚非已验证lockfile |
| `src/llmgo/rsi/__init__.py` | RSI包 |
| `src/llmgo/rsi/contracts.py` | 路径边界、协议、内容哈希、原子JSON、模型身份 |
| `src/llmgo/rsi/artifacts.py` | 源码快照和manifest校验 |
| `src/llmgo/rsi/memory.py` | 原始事件、SQLite索引、锁、状态重放、handoff |
| `src/llmgo/rsi/ladder.py` | L1及以上难度、20局统计、候选比较、通关规则 |
| `src/llmgo/rsi/evaluator.py` | 20局正式对局、实际检查点重载、引擎身份与节点记录 |
| `src/llmgo/rsi/meta.py` | Codex app-server、可见交互、thread/turn恢复 |
| `src/llmgo/rsi/controller.py` | 基线、候选、训练、验收、回归、扶正、难度晋级状态机 |
| `src/llmgo/rsi/worker.py` | Worker执行、离线标注/训练/评测分派、成功与失败产物 |
| `src/llmgo/rsi/cli.py` | doctor、init/run/status、Memory、预览提交、恢复、课程提取 |
| `src/llmgo/remote/__init__.py` | 远端执行包 |
| `src/llmgo/remote/cctl.py` | 资源校验、intent、防重复提交、身份核对与原始传输审计 |
| `src/llmgo/remote/lifecycle.py` | 专用Executor生命周期、独立日志、短报告、退避、收集及常驻服务 |
| `src/llmgo/training/__init__.py` | 离线训练包 |
| `src/llmgo/training/rollout.py` | 保留历史的课程与正式Harness同源prompt |
| `src/llmgo/training/rewards.py` | BT4原生WDL、固定学生视角Q、TD(0)与协议边界 |
| `src/llmgo/training/ppo.py` | 固定Lc0 critic的多轮token clip PPO、LoRA、原子检查点与HF导出 |
| `src/llmgo/server/rsi.py` | 只读campaign、事件、棋局和文本产物API |
| `src/llmgo/store/activity.py` | 强Agent/Executor有界显示投影；保留原文定位，避免HTTP重扫原日志 |
| `src/llmgo/server/static/rsi.html` | RSI总览页面 |
| `src/llmgo/server/static/rsi.css` | RSI页面样式 |
| `src/llmgo/server/static/rsi.js` | 日志增量直播、棋盘、Memory、候选得分/成本视图 |
| `.agents/skills/cctl/SKILL.md` | 可上传的本项目cctl约束与查询/提交知识 |
| `.agents/skills/rsi-memory/SKILL.md` | 跨轮检索、按需读原文、追加证据笔记 |
| `.agents/skills/rsi-submit/SKILL.md` | proposal到Controller提交、防重与恢复 |
| `.agents/skills/rsi-rl-review/SKILL.md` | on-policy、无SFT、reward视角与正式评测隔离 |
| `.agents/skills/rsi-executor/SKILL.md` | 先读任务短报告、异常按需取证，防止轮询污染研究上下文 |
| `tests/test_rsi.py` | 状态、版本、验收、回归、恢复、提交防重和只读API测试 |
| `tests/test_rsi_meta.py` | 真实子进程JSONL传输、恢复和正常/失败退出尾部归档测试，不调用强模型 |
| `tests/test_rsi_gradient.py` | CPU策略梯度与KL方向测试 |
| `tests/test_rsi_ui.js` | RSI日志增量展示与棋盘/流式内容的DOM测试 |
| `tests/test_rsi_executor.py` | 371次模拟轮询隔离、真实子进程交接、重启/退避/超时/产物核验 |
| `tests/test_rsi_activity.py` | 活动分页、原文保真、响应预算、只读索引、工具用量与任务文件 |
| `tests/test_rsi_browser.py` | 可选真实Chromium集成测试，含移动布局；不调用模型/集群 |
| `docs/rsi-operations.md` | 上传、启动、proposal、RL、Memory、恢复和已知限制 |
| `docs/rsi-executor.md` | Executor的组件/进程分离、独立日志、启动与恢复 |
| `docs/rsi-observability.md` | 本轮网页字段、低负担设计、索引升级、只读API和验证方法 |
| `docs/rsi-submission-and-files.md` | 本文：已成功任务、Agent提交方式与文件清单 |

## 5. 修改的既有文件

| 文件 | 变化 |
|---|---|
| `README.md` | RSI启动、架构、验收、网页及详细文档入口 |
| `plan.md` | 更新实施状态，区分已写代码和仍需远端验收的研究计划 |
| `harness/skills.py`（在`src/llmgo/`下） | 候选卡片校验与显式skills输入，避免不同候选混用全局缓存 |
| `harness/agent.py` | 接收冻结卡片，保持原单题协议 |
| `harness/player.py` | 可选模型采样seed；旧默认不变 |
| `runner/harness_loop.py` | 冻结skills传递、引擎全历史与搜索节点诊断 |
| `runtime/uci_player.py` | nodes模式、禁用重复降强、完整历史和搜索信息；兼容python-chess管理variant |
| `eval/elo.py` | 未校准级别保留W/D/L，不对None锚点造Elo或报错 |
| `server/app.py` | `/rsi`、只读API、可选统一难度参数 |
| `server/static/index.html` | RSI入口及普通对局L1–L8选择 |
| `server/static/app.js` | 传递可选难度，未选择时沿用旧协议 |
| `scripts/eval_harness.py` | 可选统一难度/节点基数；旧命令保留 |
| `scripts/arena_ladder.py` | 复用难度注册表，支持节点升级与无Elo锚点报告 |

`skills_docs.md`保留；学生初始13张卡片的正文没有为本次框架开发被改写。PostTrainBench不属于此次变更范围。旧模型路径、旧启动脚本保持现状。

## 6. 验证范围

本机测试覆盖规则、HTTP模型接口到真实Fairy-Stockfish、状态恢复、Memory重建、版本不匹配、20局判定、晋级和旧级退步、提交不确定性、Codex JSONL模拟服务、网页API与DOM、CPU策略梯度。具体测试命令及结果见交付说明；GPU依赖不在本机，因此不能将这些结果当作真实Qwen RL、BT4推理或20局提升证据。

本轮记录：

| 检查 | 结果 |
|---|---|
| `PYTHONPATH=src /tmp/llmgo-harness-venv/bin/python -m pytest -q` | 140 passed、2 skipped；跳过项为该venv缺Torch的梯度模块、缺本机字体的旧中国象棋视频测试 |
| 使用本机已有Torch及该venv依赖运行 `tests/test_rsi_gradient.py` | 2 passed；奖励梯度方向与父模型KL方向正确 |
| 最后恢复逻辑调整后重跑 `tests/test_rsi_meta.py` | 1 passed；无需实际API凭证 |
| `node tests/test_harness_ui.js` 与 `node tests/test_rsi_ui.js` | 均通过，含流式内容与有界DOM检查 |
| 四份新增skill的 `quick_validate.py` | 均通过 |
| Python编译、JS语法、`git diff --check` | 通过 |
| 真引擎nodes探针 | `nodes=1000, depth=None, time=None` 实际返回1001nodes、depth6及合法步，未被旧depth5压住 |

这些临时venv路径只是本机验证记录，不要复制为远端环境配置。后续修改应按影响范围重跑相关检查；远端训练另做GPU验收。

首次远端部署还须：准备完整Codex程序及可用认证；上传唯一网络并准备Lc0；固定Worker训练环境；验证Qwen采样/反传/导出/重载；验证共享盘恢复；实跑基线和候选。框架保存拒绝和不明确结果，不保证每个实验都成功改善。

## 7. 后续补充：Executor分离

任务生命周期现经`JobExecutor`执行，旧提交格式/资源约束保留。默认一条`run`托管组件；独立进程模式与`executor run/status/step`命令见[Executor说明](rsi-executor.md)。高频平台原文移到独立执行日志，研究Memory只接收状态转换与小型结果报告；这不改变原始强Agent对话的保存。

本次补充修改Controller/Memory/CLI、cctl传输、站点配置及网页，新增lifecycle模块、Executor操作skill、专项测试和文档，已在上方文件清单列出。仅本机改动，没有新增远端提交、取消任务或改模型路径。

Executor补充后的完整Python回归：**151 passed、2 skipped**；专项含11项生命周期/隔离测试。两套网页DOM测试、新增Executor skill及修改的Memory/submit skills校验均通过。371次查询为模拟状态回覆；真实子进程测试使用隔离平台fixture，没有调用集群。

## 8. 后续补充：详细观测与低负担网页

本轮新增`src/llmgo/store/activity.py`、`tests/test_rsi_activity.py`、`tests/test_rsi_browser.py`和`docs/rsi-observability.md`。补充修改了`rsi/memory.py`、`meta.py`、`cli.py`、`worker.py`、`remote/cctl.py`、`lifecycle.py`、`server/rsi.py`、`static/rsi.html/css/js`、相关测试及README/操作文件/本计划。

强Agent协议事件、工具起讫/输出/用量/异常，Executor逐次轮询与命令耗时，Worker阶段均可在`/rsi`查看。新`index-activity`只补建旧日志显示索引。默认最新页、3秒增量、80卡片上限、串流合并、详情延迟展开、64KiB原文分页；网页不查集群、不调用模型、不向Meta上下文追加轮询日志。旧记录不能补造未采集事件，冻结Worker源码不自动替换。

最终本机验证：完整Python回归 **157 passed、3 skipped**（其中2项Torch梯度测试和1项可选浏览器测试）；使用本机Torch环境单独执行梯度测试 **2 passed**；启用真实Chromium测试 **1 passed**。两套Node DOM测试通过，新增测试覆盖长原文/索引只读/历史与实时切换/5,000片段合并/异步次序/手机宽度。这些都是隔离fixture，无新远端提交或取消任务；本轮没有修改RL算法。

补充修复：强Agent收到轮次终态后，在退出收尾阶段仍归档已读取的协议通知与stderr，保存最后用量；关闭输入后不再发送RPC。新增正常/失败两条真实子进程回归，验证尾部记录、用量更新、脱敏及不重复归档。此修复后完整Python回归为159通过、3默认跳过；训练算法与远端任务未改动。

## 9. 清理已废弃入口与一次性文件

2026-10-21清理本地LLMGo!。删除前已检查源码、测试、配置、skills、脚本以及PostTrainBench打包器的引用；现行pipeline没有调用以下入口：

| 删除文件 | 原因与现行替代 |
|---|---|
| `docs/2026-10-05-pipeline-plan.md` | 早期双模型中国象棋方案，参数及范围已过时；现行说明在README、plan.md及RSI操作文件 |
| `scripts/run_five.py` | 早期网页中国象棋五局脚本；国际象棋批测使用eval_harness.py |
| `scripts/run_parallel.py` | 早期中国象棋双模型并行补局脚本；现行批测及RSI使用各自正式入口 |
| `scripts/run_arena_match.sh` | 旧机器列表上的SSH/Docker调度；已有服务可直接交给arena_ladder.py，RSI计算由Executor提交job |
| `scripts/serve_arena_fleet.sh` | 仅供上述旧SSH入口调用的Docker批量启动器 |
| `scripts/serve_arena_one.sh` | 仅供上述旧Docker启动链调用；保留serve_vllm.sh和eval_harness.py现有服务启动能力 |
| `integrations/posttrainbench-single-choice.patch` | 一次性整合内容已在当前PostTrainBench中；baseline由package_seal.py更新 |
| `integrations/posttrainbench-compact-events.patch` | 精简日志、裸UCI和tokenizers依赖已合入当前PostTrainBench，无需重复apply |

README已移除旧patch命令，保留当前打包方法。现行源码/训练模块、模型与引擎路径、技能与棋谱、对局原文和分析证据均保留。已有campaign的冻结源码没有改动。

清理后验证：Python完整回归157通过、3默认跳过；Harness与RSI两套Node DOM测试通过；eval_harness.py、arena_ladder.py、rsi.py、rsi_worker.py的帮助入口均通过；对当前实际源码创建并验证了85文件的RSI快照；在隔离临时目录调用真实PostTrainBench packager并验证封装后的评测入口。未修改现有PostTrainBench seal或提交远端任务；这些检查不等于实际GPU训练/棋力验收。

## 6. 本次PPO迁移文件

新增`src/llmgo/training/ppo.py`、`tests/test_rsi_ppo.py`、`tests/test_rsi_qwen_ppo.py`；重写`training/rewards.py`、`tests/test_rsi_gradient.py`；修改`training/rollout.py`、`rsi/controller.py`、`rsi/worker.py`接线及规则边界，`server/rsi.py`增加最近PPO轨迹文件指针、`server/static/rsi.js`显示批次和更新进度，`tests/test_rsi.py`迁移公式与配置断言，`requirements-rsi-worker.txt`移除旧verl数据依赖。`README.md`、`plan.md`、操作/观测文件及`rsi-rl-review`、`rsi-memory`技能同步。

移除先前未提交的`training/grpo.py`、`verl_adapter.py`、`verl_reward.py`，旧配置明确失败，不偷偷改变冻结历史实验。PPO只训练actor，Lc0固定、不做SFT。新增本机检查不等于已完成真实Qwen GPU运行或Elo提升；本次没有提交GPU训练。

## 7. PPO之后的pipeline复核

本轮修改`rsi/controller.py`（连续重试保留检查点、训练回执核对）、`rsi/cli.py`（有界可选Codex登录诊断/平台解析）、`scripts/eval_harness.py`（loopback就绪请求不经代理），新增`tests/test_pipeline_audit.py`并扩充`tests/test_rsi.py`。README与操作文档更新远端事实。另清除5个已删程序的无源bytecode；未再删除有调用方的源码或历史证据。

远端仅提交0GPU检查1321064与只读回执读取1321123，均NORMAL/beijing/model-autosft；没有GPU训练或其他任务停止。读取确认复制的config默认custom/deepseek-v4-pro，而本站点Codex程序/控制venv/RSI/Lc0/BT4均尚未部署；详见操作文档第9节。完整回归190通过、4默认跳过，另有11项真实CPU PPO/微型Qwen整合与Chromium/DOM验证；未声称远端闭环或Elo提升。
