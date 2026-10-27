# 自主研究协议与恢复

## 1. 权限与目标

Meta-Agent 只能通过 RL 和改进 skills 提升 Qwen3.5-4B，可单独或组合使用；禁止 SFT，使用 SFT 最终得0分。可以改本轮 workspace 的研究代码和配置，包括 RL 算法、reward、数据、课程，以及支持 RL/skills 的 Harness、prompt、工具、skills 内容/格式/触发/选择。无需预先声明固定 mode。既有 training_config 的参数范围仅属于初始 PPO 实现，可以改它或换 RL trainer。

禁止 SFT 预热、辅助 SFT 损失、行为克隆及监督蒸馏。离线教师可用于 RL 奖励/价值估计、局面分析和提炼 skills，不可用教师走法/思考作为监督微调目标。RL 策略梯度及价值函数拟合不属于上述 SFT 禁令。方法限制和0分规则写在 Meta-Agent 的 developer/任务 prompt 与研究 skills 中；目前不自动检测任意训练代码是否含 SFT，也不自动据此将棋局分数归零。

不能改 campaign/source 中的受保护评测副本、正式裁判/引擎/协议与历史正式结果。workspace 是可写副本，同名 evaluate.py 在研究副本中改动不会改变正式裁判；正式任务始终使用 campaign/source 的脚本。各次提交记录源码和权重身份。

最终固定 L1–L8；最低档 -9/depth5/.05秒。初始基线只测L1–L4，每档10局、白黑各5局、共40局；后续正式评测每档20局，均分只有全八档才存在。没有自动晋级/通关阈值，没有每轮必须成功的要求；不能把只刷最低档当最终成果。正式玩家必须是 Qwen 玩家，不用另一引擎替它选步。

## 2. workspace 与 proposal

Controller 在每轮建立新的源码副本，context.json 给出 Memory、工作版本、最佳版本和受保护源码路径。可以新增任意研究模块/数据；快照排除环境、模型、runs、engines、.git 等大产物，实际模型通过 model_path 交付。

```json
{
  "hypothesis": "诊断失败并尝试新的训练目标或Harness",
  "evidence_refs": ["jobs/j000001/output/metrics.json"],
  "jobs": [
    {
      "label": "任意训练、采样或格式转换",
      "argv": ["{python}", "research/train.py", "--output", "{output}"],
      "resources": {"gpu_count": 8, "cpu": 64, "memory_gib": 512}
    }
  ],
  "evaluation": {"levels": [1,2,3,4,5,6,7,8], "tracks": ["player","model"]},
  "evaluation_batch": 0,
  "recommend": false,
  "recommendation": "如推荐此版本，给出证据与仍不确定的地方"
}
```

jobs 可以为空；无需训练即可提交 Harness/skills 修改。evaluation 可为 false（只研究、不正式测试），也可只测试部分难度；这两种都不会更新最佳完整成绩。自动生成的 proposal 默认只测 player L1；手写提案若省略 evaluation 则仍按 player 全八档处理。model_path 可指定自己的已有导出模型，省略则沿用工作版本。来源 base_model 只读。

argv 按数组直接执行，没有隐式 shell。需要 shell 时显式 `["bash","-lc","..."]`，自行正确转义。占位：

| 占位/环境 | 内容 |
|---|---|
| `{python}` | Worker overlay 的真实解释器 |
| `{candidate}` / LLMGO_CANDIDATE | 已冻结候选源码根目录 |
| `{output}` / LLMGO_OUTPUT | 本 job 的可写输出目录 |
| `{model}` / LLMGO_MODEL | 当前候选模型 |
| `{source}` | 受保护评测源码根目录 |
| LLMGO_SITE | 本 job manifest，site 在 payload.site |
| LLMGO_RESUME_FROM | 显式恢复的原 job output，未恢复为空 |

多个任务顺序执行。每个任务若产生新模型，在自己的 output/submission.json 写：

```json
{"model_path":"/user/lyh4170/checkpoints/unique-experiment/final_model","method":"自选算法"}
```

随后任务和评测使用该模型。没有该文件则沿用原模型。模型应为完整可部署 HF 导出，保留 Qwen3.5 架构、tokenizer 和必要 processor；不要用单独 LoRA adapter 冒充完整模型。

已冻结源码用于复现，计算日志、缓存和权重写 output 或自己目录。若任务生成代码改进，将其产物交回 Meta-Agent，下一轮纳入 workspace 再提交；不要在运行期间改已冻结源码。skills.json 是初始 Harness 的顶层覆盖，可删除或改格式，只需同步改候选 Harness 的加载逻辑。

## 3. 研究与评测状态

BASELINE：L1–L4初始 player 基线（40局） → PLANNING：Meta-Agent自由研究 → EXPERIMENT：提交任意命令 → EVALUATING：所选档位/轨道逐项测试 → PLANNING。

每档评测单独一个 job，基线10局、后续20局均可逐局查看。player 完整均分超过旧实测最高就记录 new_best；这不是显著性证书。保存 working、best、recommended 三个概念，失败/部分测试也可成为下一轮工作分支。

训练/候选评测失败：保存日志、失败实验和已完成评测，回到 Meta-Agent，避免永久停在 FAILED；未跑完的玩家面板不能更新 best。初始基线失败：NEEDS_INPUT，先排查环境和路径。复用旧基线成功档位，恢复失败档。

固定评测资源：thinking=True、200半步、32768生成tokens/手、540秒/请求、1800秒/手。完整玩家的内部 Harness 可变化，外部裁判执行真正的推理和合法步判断。model 轨道始终使用创建 campaign 时的参考 Harness，不能把候选 skills 混进去。

初始10局基线使用五对白黑种子，后续20局使用十对；候选按同一 batch-index 对齐，前五对种子与基线相同。基线与候选的比较标识区分记录，比较时应考虑基线样本更少。development 成绩会被反复查看，不能当独立盲测或校准 Elo。最终推荐版本使用新 evaluation_batch 复测，并报告各档退步。

## 4. 任意测试调用

可把受保护入口放进 research job：

```json
{"label":"L1快速诊断","argv":["{python}","{source}/scripts/evaluate.py","--model-path","{model}","--candidate-root","{candidate}","--levels","1","--track","player","--output","{output}/probe"]}
```

这类 probe 写自己的 panel.json，不直接进入 campaign 正式最佳榜；要进入正式成绩，使用 proposal.evaluation。不能通过自写 metrics.json 冒充注册评测。完整 both 为320局，可以按研究需求调整测试频率。


## 单难度评测接口

Meta-Agent 在研究 workspace 执行 `python scripts/request_evaluation.py --level L2`，即可保留既有RL任务并把 proposal.json 设为仅测L2。也接受数字 `--level 2`；`--track model` 测固定Harness模型对照，`--track both` 测两轨，`--full-panel` 登记L1–L8完整面板。后续每档每轨20局。

该命令只准备提案并输出标准 ready JSON；Meta-Agent 将其作为最终答复，Controller/Executor 自动提交和监控GPU任务。它不会在devspace直接推理，也不会让Meta-Agent轮询。可单独评测skills/Harness，也可保留proposal.jobs先训练再评测。`--hypothesis "..."` 更新假设，`--batch-index 1` 选择新复测种子批次。

完成L1–L4初始40局后立即进入研究；基线面板显示为4/8、无完整总分，best暂为空。单难度结果仍会保存并进入Memory，但只有完整L1–L8 player结果才更新完整实测最佳。

## 5. 默认 verl PPO + LC0 配方

详细配置、样例、资源和恢复见 [rsi-verl-ppo.md](rsi-verl-ppo.md)。在workspace准备 skills.json 和 ppo.json，不再要求单步课程：

```json
{"training":{"backend":"verl_ppo","mode":"full_game_ppo_lc0","games_per_batch":2,"concurrent_games":2,"steps":2,"ppo_epochs":1,"gamma":1.0,"gae_lambda":0.95},"teacher":{"mode":"lc0_value","nodes":10000,"backend":"cuda-auto"}}
```

在proposal.jobs登记 `{"argv":["{python}","scripts/train_ppo.py"],"resources":{"gpu_count":8}}`。默认7卡verl FSDP2 actor/rollout/reference＋1卡LC0。同一策略从标准开局采样完整局，中间环境r=0、终局胜/负/和为+1/−1/0；LC0给固定学生视角的V，按棋步计算gamma1/lambda0.95的GAE，不训练critic、不做SFT。学生走子加Fairy回复构成一个转移，终局bootstrap0。

完整局展开为逐手与逐请求，优势映射到实际模型tokens；重试合并为一手等权，padding零mask。verl负责clipped PPO、梯度累积、原生优化器、分布式同步、checkpoint及HF导出。默认whole_batch_update=true、epochs1，每完整批次一次optimizer更新。训练及内部完整局验证采样temperature1/top_p1/top_k-1，记录实际behavior logprobs，reference KL独立检查。正式受保护评测保留原采样。旧TD/guarded/balanced配方和旧HF入口已删除，旧recipe显式报错，不能混用旧优化器身份续跑。

每任务最多8GPU；评测默认8副本，可用 `python scripts/request_evaluation.py --level L2 --gpus 4` 调整并行数，比赛数不变。训练产物、教师标注、失败与恢复规则见专文。Meta-Agent可以继续修改RL和skills方案，但禁止SFT、正式评测禁用教师的边界不变。

正式评测的vLLM副本采用TP=1、BF16、max-num-seqs=8和默认编译/CUDA Graph（不传enforce-eager），沿用原上下文长度、thinking与生成预算。每卡仍一次进行一局。该推理配置已在H800任务1354496完成单请求及4请求测速，未验证长棋局等比例提速；首次请求可能有编译/内核暖机等待。训练rollout的配置仍由verl训练配方独立管理。从旧eager版本恢复现有campaign，先确认旧任务和Controller已停止，再运行一次 `bash scripts/start_rsi.sh --upgrade --web`；后续普通恢复无需反复upgrade。

## 6. Memory、Executor 与恢复

run 重启从持久日志恢复，真实进程锁防双重 Controller，提交 intent 防重复任务。长期研究没有总 turn timeout，RPC 确认仍有界；Executor退避轮询独立于LLM上下文。

Memory的JSONL是权威记录，SQLite为可重建索引。实时append已落盘后，并发replay可能先把同一事件加入索引；索引写入应接受完全相同的事件，不因重复ID退出。新实现使用幂等插入并核对完整事件；真实ID/内容冲突仍报错。遇到`UNIQUE constraint failed: events.id`先保存日志和索引，升级控制端后重建/恢复，不删除原始JSONL或伪造结果来清错。

```bash
python3 scripts/rsi.py status CAMPAIGN
python3 scripts/rsi.py memory CAMPAIGN --kind decision --limit 5
python3 scripts/rsi.py note CAMPAIGN --text '证据、假设、反证与下一步'
```

基线任务确认失败后可以 `resume-failed CAMPAIGN`；NEEDS_INPUT 修复后可 `resume CAMPAIGN` 重建该档任务，不会复用失败 intent。候选失败已交回研究，下一次提案在 argv 中显式指定旧checkpoint路径恢复，不盲目重跑旧训练。观察失败和 submission_uncertain 不是任务失败，不重复提交。

研究阶段的 `NEEDS_INPUT` 经操作员修复后，用 `bash scripts/start_rsi.sh --resume --web` 恢复默认campaign；其他名称加 `--campaign NAME`。CLI `resume` 会记录 `operator_resume` 并增加 `meta_request_revision`，使同一workspace开启新的Meta请求，保留thread、基线和所有历史。普通重启仍会恢复已完成请求的结果，避免Controller崩溃后重复研究或提交；不要删除 `meta-session.json` 来绕过这个机制。已有研究workspace时不需要也不能用 `--upgrade` 代替恢复。

Meta研究不向上游强制最终JSON schema。一次`turn/completed`只是模型回合结束，Controller继续判断是否明确交接；不能凭proposal草稿存在就登记任务。原始回答、续行、工具与Token用量均保留。

### 明确完成研究

每次研究有独立的`campaign/meta-requests/REQUEST_ID/request.json`。修改、测试及proposal全部完成后，Meta执行prompt给出的绝对路径命令：

```bash
# 使用当前prompt给出的控制Python与request路径，不用旧轮次request。
python /user/lyh4170/LLMGo!/scripts/finish_research.py \
  --request /user/lyh4170/rsi_campaigns/CAMPAIGN/meta-requests/REQUEST_ID/request.json \
  --summary '完成的实验、验证结果与证据路径'
```

默认`--status ready --proposal proposal.json`；真正缺少用户输入时用`--status needs_input --summary '具体问题'`。命令原子写同目录`result.json`，不创建任务、不递归进入Controller；模型随后普通结束发言即可。完成记录绑定request ID、workspace、proposal SHA与研究源码SHA；提交后又改源码须重新执行，context与缓存按既有快照规则排除。之后仍经过Controller原有提案、资源与路径校验。

兼容最终JSON：`{"status":"ready","summary":"...","proposal_path":"绝对路径"}`。只解析最后一条助手消息，接受唯一完整JSON围栏，拒绝歧义或不完整对象；完成文件与最终JSON的状态/目标冲突时要求修正。若Controller在接收最终回复前退出，已记录的Meta子进程已结束且完成记录哈希有效时，普通重启可离线接手，不重新调用模型。

### 操作员维护暂停

```bash
bash '/user/lyh4170/LLMGo!/scripts/start_rsi.sh' --pause-research --web
# 新版Controller已加载时继续监控/收集既有任务，不启动Meta和新提交。
bash '/user/lyh4170/LLMGo!/scripts/start_rsi.sh' --resume-research --web
```

两条命令只改变独立research-control.json，不改campaign phase、workspace或冻结身份；不需要取得Controller长期writer锁。恢复是显式动作，普通启动、--resume、自动重试都不会清除维护暂停。未确认task ID的prepared/uncertain任务保持原样，不在暂停期间发起新的create。

Meta在启动、线程续接/新turn和等待流事件时检查暂停；活动turn请求interrupt并结束自己创建的backend子进程，保存可见事件，不将操作员暂停记成模型故障或进入自动重试。心跳仍最多每30秒，暂停检查不会制造逐秒日志。任务已登记并完成时可以正常收集与结束候选，后续研究停在边界。

旧Controller不能热加载新逻辑。live codex_bridge的app-server启动闸门先按真实cwd确定所属campaign，再在读取凭证/启动桥之前检查暂停；它只阻止该站点研究workspace的新app-server，其他campaign和普通版本/连接检查不受该闸门影响。它为等待原devspace重启提供硬阻断，但旧Controller可能暂时将退出码78当连接失败。应在WAITING且无Meta进程时执行已有`--restart-control --web`，加载新版后保留原task ID；内嵌Executor的轮询有短暂交接，不停止GPU Worker。启动器在维护暂停下跳过Meta推理检查，只恢复控制监控。没有原devspace执行通道的CPU job不能跨容器发信号，需在原devspace执行该命令。

网页兼容读取小型research-control.json与controller-status.json，展示持久暂停原因；不扫描原始上下文或查询cctl。任务监控仍按原5–10分钟节奏。单看标志文件表示暂停已请求，检查Controller回执与会话后才能报告完整控制逻辑已载入。

### 自动恢复与停止边界

| 情况 | 控制端行为 |
|---|---|
| 无明确交接，但该回合有工具执行 | 原会话续接，无研究总回合或时间限制 |
| 连续2次无工具、无完成记录 | 换新会话读取原workspace/Memory与已有任务；单次backend调用最多换2次 |
| 断流、暂时服务错误、交接协议持续失败 | 持久保存`controller_recovery`，依次等60/120/240/480/600秒，最多自动重试5次 |
| 在等待重试时重启 | 沿用计数与绝对下次时间，不立即重问模型 |
| 401/403、无效key、协议配置问题、未知异常、产物身份不符 | `NEEDS_INPUT`显示具体阶段/类别；保留证据，不伪造成功 |
| GPU创建响应不确定 | Executor保留intent并对账，Controller不重复创建 |
| 快照已发布但状态未保存 | 只复用提案、源码、skills相同的快照；冲突保留原件并停止检查 |

`controller_retry/recovered/blocked`、`meta_continuation`、`meta_protocol_recovery`与`meta_completion_committed/rejected`进入Memory及网页有界索引。网页显示恢复原因、次数与下一次时间，并分开`active_job`和`latest_job`；旧任务Killed不表示当前有活动任务。HTTP刷新只读磁盘，不查cctl或扫描完整日志。旧Python网页服务通过小型`controller-status.json`兼容读取新恢复状态，不必为此停止已有网页；新版接口直接在overview提供这些字段。浏览器需刷新以加载新版JS。

上述间隔用于控制端故障恢复，GPU任务仍每5–10分钟轮询。操作员中断不会被自动吞掉；真正用户问题可暂停。处理具体原因后`--resume --web`清除恢复计数，保留workspace和已有任务ID。旧版因`format_retry`失败且仍为PLANNING的会话，新版会迁移后续接，无需init、upgrade或丢弃第四轮提案。

连接验收：`python3 scripts/check_meta_connection.py --sandbox-check --tools-check`。`model_tool_calls_tested=true` 要求模型产生真实成功命令事件和随机测试文件的正确副本，仅返回READY无法通过；启动器默认执行这项检查。`--offline` 不调用模型，不能验证模型工具选择。

### 推理摘要串流诊断

`ReasoningSummaryDelta without active item`需核对bridge原始事件，而不是停止Executor或直接隐藏stderr。当前固定转换器曾输出reasoning开始项却漏掉必填`summary`数组，也没有在摘要delta前输出`response.reasoning_summary_part.added`。另有空/role-only首区块过早初始化正文，导致后续推理ID从未开始；对固定LiteLLM 1.105.2响应实例延后这类初始化，仍保留区块的usage与累积处理，不改安装文件或全局类。`responses_stream.py`再逐请求补空数组与缺失part-start，保持序号递增；已有合法start不重复添加，工具和模型文字等有效载荷原样保留。无匹配item、索引冲突或关闭后的delta会明确报错，不猜测关联。

bridge的`events.jsonl`中`stream`仍是原转换事件；`stream_repair`是结构修正元数据，`stream_normalization`是每次请求的修正计数。`/health`的`stream_protocol`标识已载入版本。转接服务随Meta backend子进程启动；同步live repo后自然下次启动生效，不热改当前流、不重启Controller/Executor、不动已冻结评测源码和Worker。当前仍在使用旧bridge的研究可能继续产生旧警告，不能说同步后已立即消失。协议依据见[OpenAI Responses streaming events](https://developers.openai.com/api/reference/resources/responses/streaming-events)。

2026-10-25隔离0GPU任务1365204使用真实Codex CLI、固定LiteLLM与本地受控Chat串流做对照，涵盖推理首区块和空role首区块：旧桥6次警告、交付5个摘要delta；修正后0次警告、交付全部11个delta，两边真实工具命令均成功。回执为`/user/lyh4170/rsi_readiness/reasoning-lifecycle-20261025/validation-v2/result.json`。该测试不调用正式模型、不提交训练，也不停止生产进程；不把它当作正在运行的旧bridge已热更新。

### 走棋超时与部分诊断

`test_external_turn_timeout_preserves_partial_request_diagnostics`在第四轮及未改动的第三轮均失败，原始异常为`StreamCancelled: request_cancelled`，并非缺少部分输出。裁判已保存诊断，却重新抛出了继承`asyncio.CancelledError`的子类；Python 3.12的`asyncio.timeout`只转换类型完全相同的取消异常，导致原本应成为`TimeoutError`的超时以取消异常外泄。本机Python 3.13改用子类判断，掩盖了这个跨版本问题。官方实现可对照[Python 3.12](https://github.com/python/cpython/blob/v3.12.3/Lib/asyncio/timeouts.py#L103)与[Python 3.13](https://github.com/python/cpython/blob/v3.13.0/Lib/asyncio/timeouts.py#L106)。它属于基础设施错误处理问题；没有证据把第三轮权重退化归因于它。

`CandidateSeat._dialog`现在先写完权威取消诊断，再标准化取消异常类型，保留原异常因果链。真实超时正常转换为`TimeoutError`，操作员/调用方主动取消仍传递取消；不吞掉取消、不更改生产超时、预算或评测规则。原测试另有冷启动超过3秒便无法进入推理的脆弱假设：新版测试在stream开始时才安排外部超时，给冷启动单独保护，覆盖0秒/3.2秒启动延迟、子进程清理及部分诊断；另直接校验普通取消与带诊断取消的精确类型，避免Python 3.13再次掩盖回归。

2026-10-26香港时间的隔离0GPU任务1365275使用现有控制环境Python 3.12.13：旧代码/旧测试复现相同`StreamCancelled`失败，修复后17项候选适配器测试全部通过，包含真实Fairy接口。本机全套369通过、4跳过。文件在`/user/lyh4170/rsi_readiness/timeout-diagnostics-20261025/validation/`，`result.json`记录版本、源码/测试SHA、前后返回码和日志路径；原第四轮失败证据另在同目录上层`failure-evidence.json`。

这项修改位于评测适配器，必须区分live仓库、Meta的研究workspace和campaign冻结源。同步live仅提供已验证修复，正在运行或已排队的任务仍用登记时的源码；不热改快照、哈希或已完成成绩，也不为它停止Executor。新campaign会采用新源码；现有campaign的迁移需在无在途任务且研究状态允许时单独安排。当前`upgrade`拒绝已有研究workspace，不能删除workspace绕过它，也不能把普通重启说成评测已升级。

### CUDA Graph诊断的答案检查器

远端第四轮自建的`graph_ab.py`曾把`AskResult.output`（推理与答案的展示拼接）交给`parse_move`，并用字面的`</think>`计数边界。当vLLM分别返回reasoning/content时，合法最终答案也可能被判失败；无字面标签不代表思考未完成。j000013的0/4不能当作真实合法率，其报告未保存回答内容，无法事后算出正确比例，也不能直接改成4/4。

仓库现提供修正版`scripts/graph_ab.py`：检查规范化的`answer.text`及normalization_error，保留请求错误、未闭合思考、非法/含糊答案与length截断的失败判定。`has_boundary`表示已观察到可用的推理/答案边界，原字面标签改由`has_think_tag`单独记录，独立的纯UCI答案仍按Harness接受；`protocol_check=normalized-final-v1`与`schema_version=2`区分新旧结果。

每笔完成或返回错误的HTTP请求在`output/responses/graph-NN.json`或`eager-NN.json`保存完整AskResult，包括raw_response、text、reasoning、output、token_ids、usage、finish_reason、请求payload、错误与延迟，并保留prompt、合法步列表及判定结果。摘要的response_path指向这些文件，可离线重检；不从推理里搜索某个看似合法的棋步代替最终答案。

调用仍用既有research job：`argv=["{python}","scripts/graph_ab.py"]`，由Worker传入LLMGO_SITE/OUTPUT/MODEL/CANDIDATE；该诊断只申请1GPU、TP1，并行最多4个样本，服务器容量8。运行两个模式各自第一批，`timing_scope=first_batch_includes_lazy_initialization`；它还不是暖机后多轮吞吐，不能把首次延迟推论为稳态性能。真实prompt token数来自usage，不再用字符数/4冒充。

修正版只部署live仓库，不修改已冻结r000004源码、j000013结果或在途j000014。Meta后续若复用自己的旧研究脚本，应先将live的`scripts/graph_ab.py`与`tests/test_graph_ab.py`复制进新的可编辑workspace并验证，再经正常提案/快照提交；不在冻结目录热修，也不需要为修检查器停止当前训练或重新做初始基线。超时兼容修复的启用边界见上一节。

验收：本机381通过、4跳过；隔离0GPU任务1365383使用远端Python 3.12.13，检查器与超时适配器共29项通过。`/user/lyh4170/rsi_readiness/graph-checker-20261026/validation/result.json`保存版本、SHA与pytest.log路径。它验证真实HTTP适配器的受控响应、检查器和回执，不代表重新生成了j000013的八份回答或完成了新的GPU测速。

### 已完成Meta提案的恢复

当 `phase=PLANNING`、没有active_job、Meta进程已退出，但原始 `turn/completed` 为completed、最终回复含ready提案时，可用：

```bash
bash scripts/start_rsi.sh --recover-meta --web
```

底层命令是 `python3 scripts/rsi.py recover-meta CAMPAIGN`。它只读取已登记workspace与当前session匹配的thread/turn，要求日志中的完成事件和最终agentMessage，允许最终JSON外有Markdown代码围栏及说明，但拒绝歧义或不完整结果。恢复前在campaign的 `recoveries/` 备份campaign和session，恢复后照常校验proposal、冻结candidate源码/skills、记录Memory；不会修改既有基线或已完成评测，不触发新研究推理。若最近500个协议事件中没有完整终态证据则停止，需先定位原始trace，不能用一个不明来源的proposal代替完成证据。

恢复后由Controller依次进入EXPERIMENT、EVALUATING并登记新job。第二轮原提案是L2、L3各20局，8 GPU，一个难度完成并收集后再提交下一档。重复执行recover-meta会因阶段已改变而拒绝，防止重复冻结或提交；此时普通 `bash scripts/start_rsi.sh --web` 即可。

### 如何恢复评测

| 当前情况 | 操作 | 保留与恢复范围 |
|---|---|---|
| 仅SSH/浏览器断线，Controller仍持锁 | 查看 `--status`、网页或 `--logs`，无需再启动 | nohup后台进程继续运行 |
| Controller退出，阶段为WAITING/EVALUATING/EXPERIMENT | `bash scripts/start_rsi.sh --web` | 读取已有intent/task ID和回执，继续原任务及后续难度 |
| 已完成Meta回复因JSON包装解析失败，仍为PLANNING | 修复解析后 `--recover-meta --web` | 从原完成事件恢复提案，不重做研究 |
| 真正缺少输入，NEEDS_INPUT | 解决具体缺失项后 `--resume --web` | 授权新Meta请求，保留会话与历史 |
| 当前GPU任务确认终止失败，仍登记为active_job | 查看失败原因，修好后用 `python3 scripts/rsi.py resume-failed CAMPAIGN`，再普通启动 | 新任务登记resume_from，Worker仅复用身份匹配且有效的逐局回执 |

不要kill仍在运行的任务来“恢复”，不要删除submission intent、Memory或改变回执hash。若失败候选已经被Controller交回PLANNING且candidate为空，应由Meta登记新提案，不能直接resume-failed旧候选。源码/协议不兼容的旧棋局不会被强行复用。

### 重复生成的检测与恢复

控制端`generation_guard.py`只检查`item/agentMessage/delta`，按消息ID隔离，保留8192字符滑动窗口，每累计256字符检查一次。8–256字符的完全相同短周期重复至少12次且覆盖至少2048字符才触发；纯数字、空白或标点不单独作为循环证据。它不检测命令stdout，不用“很久没有工具调用”推定卡死，不设研究总时间上限。

触发时先持久化原事件及周期样本/摘要，再发送`turn/interrupt`。等原turn确认interrupted（或竞态中已经completed），才新建thread，以原prompt、同一workspace及Memory继续；恢复提示要求先核对已有任务和产物，禁止重提。中断确认有独立的协议等待上限，失败时不会盲目并发新会话。每个研究请求最多切换两次，第三次重复则返回明确needs_input，由Controller暂停。原线程、所有已收到的原始输出、工作文件与集群job均保留。

研究面板事件：`meta_generation_loop`（异常）、`meta_generation_recovery`（生命周期）、`meta_generation_exhausted`（异常），并记录thread/turn及恢复次数。新线程不携带整段污染的对话历史，依赖持久Memory与已有文件接续，避免反复把循环文本塞回上下文。

### 等待计算任务时加载控制端修复

```bash
bash scripts/start_rsi.sh --restart-control --web
```

只在原devspace使用。此入口要求Controller确实持锁、主机/PID/cwd/命令匹配本campaign、权威状态为WAITING、无活动Meta，以及active_job已有匹配description的task ID。它以非阻塞方式取得Executor lifecycle锁和主journal写锁，确保没有平台命令或主日志写入在途，备份状态/session/进程身份后，仅向已核对的Controller发送SIGTERM，并等待其锁释放；不停止Worker或集群job。随后普通启动器读取原intent恢复监控。

若Meta仍在正常工作、锁忙、任务提交不确定或主机不符，命令会拒绝且不发送信号。Controller早已退出时无需再停止，直接继续启动。`--restart-control`不能与`--resume`、`--recover-meta`、`--upgrade`或状态查看混用。此操作加载控制端代码，不更新冻结的正式评测源码。

旧自动晋级 state 不迁移，新建 campaign，旧证据保留。只写 /user/lyh4170、NORMAL/beijing/model-autosft、devspace0GPU、不停止别人任务的约束继续有效。
