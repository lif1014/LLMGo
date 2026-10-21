# 自主研究协议与恢复

## 1. 权限与目标

Meta-Agent 可以改本轮 workspace 的任何研究代码和配置，包括 RL/SFT、reward、PPO/其他算法、数据、课程、Harness、prompt、工具、skills 内容/格式/触发/选择。无需预先声明固定 mode。既有 training_config 的参数范围仅属于初始 PPO 实现，可以改它或换 trainer。

不能改 campaign/source 中的受保护评测副本、正式裁判/引擎/协议与历史正式结果。workspace 是可写副本，同名 evaluate.py 在研究副本中改动不会改变正式裁判；正式任务始终使用 campaign/source 的脚本。各次提交记录源码和权重身份。

最终固定 L1–L8；最低档 -9/depth5/.05秒。每档20局，均分只有全八档才存在。没有自动晋级/通关阈值，没有每轮必须成功的要求；不能把只刷最低档当最终成果。正式玩家必须是 Qwen 玩家，不用另一引擎替它选步。

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
      "resources": {"gpu_count": 1, "cpu": 16, "memory_gib": 128}
    }
  ],
  "evaluation": {"levels": [1,2,3,4,5,6,7,8], "tracks": ["player","model"]},
  "evaluation_batch": 0,
  "recommend": false,
  "recommendation": "如推荐此版本，给出证据与仍不确定的地方"
}
```

jobs 可以为空；无需训练即可提交 Harness/skills 修改。evaluation 可为 false（只研究、不正式测试），也可只测试部分难度；这两种都不会更新最佳完整成绩。默认 evaluation 是 player 全八档。model_path 可指定自己的已有导出模型，省略则沿用工作版本。来源 base_model 只读。

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

BASELINE：八档初始 player 基线 → PLANNING：Meta-Agent自由研究 → EXPERIMENT：提交任意命令 → EVALUATING：所选档位/轨道逐项测试 → PLANNING。

每档评测单独一个 job，20局可逐局查看。player 完整均分超过旧实测最高就记录 new_best；这不是显著性证书。保存 working、best、recommended 三个概念，失败/部分测试也可成为下一轮工作分支。

训练/候选评测失败：保存日志、失败实验和已完成评测，回到 Meta-Agent，避免永久停在 FAILED；未跑完的玩家面板不能更新 best。初始基线失败：NEEDS_INPUT，先排查环境和路径。复用旧基线成功档位，恢复失败档。

固定评测资源：thinking=True、200半步、32768生成tokens/手、540秒/请求、1800秒/手。完整玩家的内部 Harness 可变化，外部裁判执行真正的推理和合法步判断。model 轨道始终使用创建 campaign 时的参考 Harness，不能把候选 skills 混进去。

20局是十对白黑种子，所有候选按同一 batch-index 对齐。development 成绩会被反复查看，不能当独立盲测或校准 Elo。最终推荐版本使用新 evaluation_batch 复测，并报告各档退步。

## 4. 任意测试调用

可把受保护入口放进 research job：

```json
{"label":"L1快速诊断","argv":["{python}","{source}/scripts/evaluate.py","--model-path","{model}","--candidate-root","{candidate}","--levels","1","--track","player","--output","{output}/probe"]}
```

这类 probe 写自己的 panel.json，不直接进入 campaign 正式最佳榜；要进入正式成绩，使用 proposal.evaluation。不能通过自写 metrics.json 冒充注册评测。完整 both 为320局，可以按研究需求调整测试频率。

## 5. 初始 PPO 配方

可用 CLI 创建课程：`python3 scripts/rsi.py curriculum --count 16 --output curriculum.jsonl`，或从自己的对局 events 提取。课程校验仅针对这个配方，其他训练数据格式可以自由使用。

workspace 中新增 ppo.json，例如：

```json
{"training":{"steps":16,"batch_size":8,"ppo_epochs":4},"reward":{"mode":"lc0_bootstrap","nodes":10000,"backend":"cuda-auto"}}
```

jobs 配成 `{"argv":["{python}","scripts/train_ppo.py"]}`。该实现是课程单步 HF PPO：当前策略采样，Lc0 离线给 V=P(win)−P(lose)，非终局 A=V_after−V_before，终局采用真实结果减 V_before；gamma1/lambda0，无训练critic。标签不进入正式 prompt。

初始策略A广播到response tokens，逐tokenclip、每回答平均，behavior/reference/A在同批epochs固定。学生采样→CPU offload→Lc0评分释放→学生更新，最后完整导出。检查采样分布、clip/KL/真实更新及 checkpoint；配方细节见 rsi-training skill。它可修改/替换，不再禁止其他方法。

LC0/BT4仅这条可选配方需要；不选择它不影响控制器或正式评测。BT4权重 SHA256：e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618。

## 6. Memory、Executor 与恢复

run 重启从持久日志恢复，真实进程锁防双重 Controller，提交 intent 防重复任务。长期研究没有总 turn timeout，RPC 确认仍有界；Executor退避轮询独立于LLM上下文。

```bash
python3 scripts/rsi.py status CAMPAIGN
python3 scripts/rsi.py memory CAMPAIGN --kind decision --limit 5
python3 scripts/rsi.py note CAMPAIGN --text '证据、假设、反证与下一步'
```

基线任务确认失败后可以 `resume-failed CAMPAIGN`；NEEDS_INPUT 修复后可 `resume CAMPAIGN` 重建该档任务，不会复用失败 intent。候选失败已交回研究，下一次提案在 argv 中显式指定旧checkpoint路径恢复，不盲目重跑旧训练。观察失败和 submission_uncertain 不是任务失败，不重复提交。

旧自动晋级 state 不迁移，新建 campaign，旧证据保留。只写 /user/lyh4170、NORMAL/beijing/model-autosft、devspace0GPU、不停止别人任务的约束继续有效。
