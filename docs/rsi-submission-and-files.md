# 集群任务提交、环境与本次文件变化

## 已成功的提交格式

使用 cctl skill：[`.agents/skills/cctl/SKILL.md`](../.agents/skills/cctl/SKILL.md)。本地源路径是 `/home/lif/.local/bin/.cursor/skills/cctl/SKILL.md`。

所有创建操作先 `cctl pool list --own`，确认 model-autosft。固定授权：NORMAL、beijing、model-autosft、自己的 /user/lyh4170；devspace 必须0GPU，不停止他人任务。

已成功任务实例：1310237/1310259/1310452/1310479 为此前CPU核对/配置复制任务；1321064/1321123 为此前CPU检查/回执读取，均已成功。**1324638** 使用本次指定的 ptdbench-verl-vllm017 镜像，GPU0、CPU2、8Gi，已成功并将版本回执写入自己的 rsi_readiness；平台归档日志不可读时，应读取该文件，不能凭任务成功就推断训练环境已完整。

与该实例一致的格式如下；字符串 entry 使用实际 argv 的 `shlex.join` 构建：

```bash
CCTL_NO_UPDATE_CHECK=1 cctl pool list --own -o json
cctl job create \
  --project rsi --cluster beijing --resource-pool model-autosft \
  --billing-account-id N00069 --priority NORMAL \
  --image share-dev/ngc-docker/ptdbench-verl-vllm017:dev-957fee7-cb-bf632efb \
  --gpu 0 --cpu 2 --memory 8 \
  --description llmgo-own-unique-inspection \
  --entry 'python3 /user/lyh4170/your-inspection.py' \
  --no-input -o json
```

正式Worker相同资源授权，默认GPU1/H800、CPU16、128Gi；research可用resources字段选择GPU数量、CPU和内存。固定评测始终1GPU。GPU只能job/pytorchjob，不加到devspace。

```bash
cctl job get TASK_ID --no-input -o json
cctl job logs TASK_ID --pod job-rsi-TASK_ID-main-0 --page-size 200
```

logs子命令不用 `-o json` 或 `--no-input`。日志归档可能不可用，应用同时把完整日志与回执写共享盘。不要因日志暂不可读或查询失败就重提。

## Agent 如何提交

常规流程在workspace写proposal.json，使用 `.agents/skills/rsi-submit/SKILL.md`；Controller快照、Executor提交。`preview CAMPAIGN` 可查看真实argv但不创建任务。每次manifest的description含campaign、job ID与SHA，先持久保存submission intent，再调用create，超时按description对账，不盲目重复。

新容器入口为源快照的 scripts/worker_entry.py，参数 `--manifest ABS_PATH --manifest-sha256 HASH`。它先核对manifest，在指定基底镜像建立/核验共享Worker overlay，随后exec真正的worker解释器。不能只在devspace pip安装就假定GPU job也拥有环境。

允许Meta-Agent自行提交独立研究任务；应持久记任务ID、输出和结论，避免让原始轮询填满研究上下文。常规任务优先Executor。Controller持锁调用Meta时不要递归run/prepare/step以免抢同一写锁。

## 本次新增文件

| 文件 | 用途 |
|---|---|
| scripts/evaluate.py | 受保护L1–L8评测入口，player/model两轨与完整面板 |
| src/llmgo/rsi/candidate.py | 裁判侧候选进程代理，执行推理与外部预算 |
| src/llmgo/rsi/candidate_bridge.py | 候选侧JSONL适配，加载自由修改的Harness |
| src/llmgo/rsi/service.py | 从旧入口提取的vLLM生命周期，无隐式安装或另一路评测 |
| scripts/bootstrap_worker.py | 在用户指定镜像安装自己的共享overlay，保存约束和环境回执 |
| scripts/worker_entry.py | 每个新容器先验证/安装overlay，再启动Worker |
| scripts/train_ppo.py | 可自由修改的现有HF PPO/LC0起始配方入口 |
| scripts/package_source.py | 打包、逐文件SHA/大小/权限核验，排除凭证和运行数据 |
| .agents/skills/rsi-training/SKILL.md | RL/skills方法边界、SFT禁令与可选PPO配方说明 |
| tests/test_rsi_worker.py | 任意研究命令、模型回执与幂等恢复检查 |
| tests/fixtures/qwen_stream_boundaries.json.gz | 独立于runs目录的11条原始GPU流式响应回归样本 |
| tests/test_rsi_candidate.py | 真子进程、预算、伪造事件、历史、超时和真实Fairy接口检查 |
| docs/removed-files.json | 删除/替换文件清单 |
| docs/environment-status.json | 本次远端环境检查/安装的实际状态回执 |
| docs/validation.json | 交付时测试与包验证记录 |

## 主要修改

- controller/meta/worker：研究代码与argv保持可配置，不要求固定提案mode、没有自动晋级；prompt要求只用RL/skills，禁止SFT且使用SFT最终得0分。保留working/best/recommended、完整评测才更新最佳；目前没有自动检测SFT或归零机制。
- ladder/contracts/evaluator：固定L1–L8面板，每档20局、独立轨道和资源约束；对手与正式结果保持外部控制。
- artifacts/memory：研究源码完整快照、身份、交接与短上下文；历史记录和Executor隔离继续保留。
- cctl：基底镜像共享overlay入口、任意research资源，仍只用授权集群/池/优先级。
- server：只保留此RSI面板、成绩矩阵、对局与日志查看；增量与懒加载。
- README、plan、first-run、operations、executor、observability、skills_docs与Agent skills：全面替换旧约束与启动说明。
- 测试保留初始Harness/规则/串流/PPO覆盖，替换已删除游戏创建/自动晋级/十局入口的测试，新增自主研究验收。

完整删除清单见 removed-files.json；包括其他棋种、旧Docker服务/十局评测/arena入口、旧多棋种网页和其专用测试。旧runs不删，PostTrainBench原目录不修改。

## 验收范围

本机CPU、模拟模型与真实Fairy只能证明程序接口、边界与恢复；不代表4B GPU训练通过、真实LC0评分或棋力提升。真实镜像的版本/依赖安装状态在environment-status.json；仍需GPU job完成Qwen生成、训练导出和重载验收。
