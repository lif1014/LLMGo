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

详细配置、样例、资源和恢复见 [rsi-verl-ppo.md](rsi-verl-ppo.md)。在workspace准备 curriculum.jsonl、skills.json 和 ppo.json：

```json
{"training":{"backend":"verl_ppo","steps":16,"ppo_epochs":4},"reward":{"mode":"lc0_bootstrap","nodes":10000,"backend":"cuda-auto"}}
```

在proposal.jobs登记 `{"argv":["{python}","scripts/train_ppo.py"],"resources":{"gpu_count":8}}`。默认7卡verl FSDP2 actor/rollout/reference＋1卡LC0。批量自动适配actor卡数（8卡任务batch56/minibatch14）。训练每个样本是一手棋，LC0离线给V=P(win)−P(lose)，非终局A=V_after−V_before，终局真实结果减V_before，gamma1/lambda0，不训练critic、不做GRPO、不做SFT。

该A广播至本次走棋的response tokens，verl负责clipped PPO、多epoch、优化器、分布式同步、checkpoint及HF导出。采样使用temperature1/top_p1/top_k-1，保存vLLM实际behavior logprobs并作为old logprobs，固定reference KL。LC0值已形成TD误差，不再把它叠加到另一个价值差奖励。

每任务最多8GPU；评测默认8副本，可用 `python scripts/request_evaluation.py --level L2 --gpus 4` 调整并行数，比赛数不变。训练产物、教师标注、失败与恢复规则见专文。Meta-Agent可以继续修改RL和skills方案，但禁止SFT、正式评测禁用教师的边界不变。

正式评测的vLLM副本采用TP=1、BF16、max-num-seqs=8和默认编译/CUDA Graph（不传enforce-eager），沿用原上下文长度、thinking与生成预算。每卡仍一次进行一局。该推理配置已在H800任务1354496完成单请求及4请求测速，未验证长棋局等比例提速；首次请求可能有编译/内核暖机等待。训练rollout的配置仍由verl训练配方独立管理。从旧eager版本恢复现有campaign，先确认旧任务和Controller已停止，再运行一次 `bash scripts/start_rsi.sh --upgrade --web`；后续普通恢复无需反复upgrade。

## 6. Memory、Executor 与恢复

run 重启从持久日志恢复，真实进程锁防双重 Controller，提交 intent 防重复任务。长期研究没有总 turn timeout，RPC 确认仍有界；Executor退避轮询独立于LLM上下文。

```bash
python3 scripts/rsi.py status CAMPAIGN
python3 scripts/rsi.py memory CAMPAIGN --kind decision --limit 5
python3 scripts/rsi.py note CAMPAIGN --text '证据、假设、反证与下一步'
```

基线任务确认失败后可以 `resume-failed CAMPAIGN`；NEEDS_INPUT 修复后可 `resume CAMPAIGN` 重建该档任务，不会复用失败 intent。候选失败已交回研究，下一次提案在 argv 中显式指定旧checkpoint路径恢复，不盲目重跑旧训练。观察失败和 submission_uncertain 不是任务失败，不重复提交。

研究阶段的 `NEEDS_INPUT` 经操作员修复后，用 `bash scripts/start_rsi.sh --resume --web` 恢复默认campaign；其他名称加 `--campaign NAME`。CLI `resume` 会记录 `operator_resume` 并增加 `meta_request_revision`，使同一workspace开启新的Meta请求，保留thread、基线和所有历史。普通重启仍会恢复已完成请求的结果，避免Controller崩溃后重复研究或提交；不要删除 `meta-session.json` 来绕过这个机制。已有研究workspace时不需要也不能用 `--upgrade` 代替恢复。

Meta研究不向上游强制最终JSON schema，最终JSON仍在本地验证。没有工具动作的 `needs_input` 会收到一次 `meta_continuation` 提醒，要求实际执行已授权工作或说明明确的用户问题；再次确有问题时正常暂停，不无限重试。无效最终JSON最多补正一次。这些是结束协议补正次数，不是研究时间限制。原始回答、后续turn、工具及Token用量均保留在Memory和原会话中。

连接验收：`python3 scripts/check_meta_connection.py --sandbox-check --tools-check`。`model_tool_calls_tested=true` 要求模型产生真实成功命令事件和随机测试文件的正确副本，仅返回READY无法通过；启动器默认执行这项检查。`--offline` 不调用模型，不能验证模型工具选择。

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

旧自动晋级 state 不迁移，新建 campaign，旧证据保留。只写 /user/lyh4170、NORMAL/beijing/model-autosft、devspace0GPU、不停止别人任务的约束继续有效。
