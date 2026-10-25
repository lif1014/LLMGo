# 集群任务提交、环境与本次文件变化

## 已成功的提交格式

使用 cctl skill：[`.agents/skills/cctl/SKILL.md`](../.agents/skills/cctl/SKILL.md)。本地源路径是 `/home/lif/.local/bin/.cursor/skills/cctl/SKILL.md`；完整原文随项目保存在 [`.agents/skills/cctl/references/cctl-full.md`](../.agents/skills/cctl/references/cctl-full.md)，摘要入口链接到它，按需阅读。该参考文件与入口一同打包并复制到每轮研究工作目录，远端无需访问本机源路径。

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

正式Worker相同资源授权，默认GPU8/H800、CPU64、512Gi；research可用resources字段选择GPU数量、CPU和内存。评测默认8GPU、可配置1–8GPU。GPU只能job/pytorchjob，不加到devspace。

新的手写提交格式先加 `--dry-run` 检查实际request，再正式create。JSON `-f` 的环境变量字段是 `envVars: [{"name":"KEY","value":"VALUE"}]`；`envs`可能被忽略，不要混用文件主体与`--env`来假定覆盖。纯flags的`--env KEY=VALUE`仍可正常使用。平台限制每任务最多64个环境变量，每个value最多4096字符；大资产应正常上传共享盘，而不是作为环境变量提交。create后附带的expose失败不代表任务未创建，必须先按唯一description对账；不要重提或停止其他任务来腾端口。

```bash
cctl job get TASK_ID --no-input -o json
cctl job logs TASK_ID --pod job-rsi-TASK_ID-main-0 --page-size 200
```

logs子命令不用 `-o json` 或 `--no-input`。日志归档可能不可用，应用同时把完整日志与回执写共享盘。不要因日志暂不可读或查询失败就重提。

## Agent 如何提交

常规流程在workspace写proposal.json，使用 `.agents/skills/rsi-submit/SKILL.md`；Controller快照、Executor提交。`preview CAMPAIGN` 可查看真实argv但不创建任务。每次manifest的description含campaign、job ID与SHA，先持久保存submission intent，再调用create，超时按description对账，不盲目重复。

完成后优先用prompt给出的`finish_research.py --request ABS_REQUEST_JSON --summary '...'`保存本请求的完成记录，绑定proposal与源码哈希，不提交任务。仍兼容三字段最终JSON。完成后又改源码应重新确认，草稿存在不等于完成。恢复规则见[rsi-operations.md](rsi-operations.md)。

新容器入口为源快照的 scripts/worker_entry.py，参数 `--manifest ABS_PATH --manifest-sha256 HASH`。它先核对manifest，在指定基底镜像建立/核验共享Worker overlay，随后exec真正的worker解释器。不能只在devspace pip安装就假定GPU job也拥有环境。

允许Meta-Agent自行提交独立研究任务；应持久记任务ID、输出和结论，避免让原始轮询填满研究上下文。常规任务优先Executor。Controller持锁调用Meta时不要递归run/prepare/step以免抢同一写锁。

## 本次新增文件

| 文件 | 用途 |
|---|---|
| src/llmgo/rsi/codex_sandbox.py | 统一正式会话、恢复、每轮与执行检查的隔离策略；当前依赖服务器提供的外层隔离 |
| src/llmgo/rsi/codex_auth.py | 从私有TOML/环境解析认证，仅注入子进程并脱敏 |
| src/llmgo/rsi/connection_check.py、scripts/check_meta_connection.py | 真实 app-server READY 连接测试与私有协议日志 |
| src/llmgo/rsi/codex_bridge.py、scripts/serve_meta_bridge.py | 保留 Codex 工具和会话，Responses 转 Chat Completions |
| src/llmgo/rsi/meta_protocol.py、scripts/finish_research.py | 请求绑定的研究完成记录、源码哈希检查与故障分类 |
| tests/test_research_completion.py、tests/test_controller_recovery.py | 交接崩溃、过期记录、持久重试与候选原子发布回归 |
| configs/rsi/meta-bridge-requirements.txt | 独立转接环境固定依赖版本，不影响训练环境 |
| tests/test_meta_connection.py、tests/test_codex_bridge.py | 配置认证、脱敏、URL规范化、流式及工具事件回归 |
| scripts/request_evaluation.py、src/llmgo/rsi/evaluation_request.py | 单难度或全八档评测提案接口，保留RL任务并交给Executor |
| src/llmgo/remote/cctl_env.py、tests/test_cctl_env.py | 解析cctl地址、修正旧裸别名，并显示脱敏诊断；保留原profile/token文件 |
| scripts/start_rsi.sh、tests/test_start_rsi.py | 一条短命令检查、初始化、后台启动/恢复loop与可选网页，验证重复启动和端口占用处理 |
| scripts/evaluate.py | 受保护L1–L8评测入口，player/model两轨与完整面板 |
| src/llmgo/rsi/candidate.py | 裁判侧候选进程代理，执行推理与外部预算 |
| src/llmgo/rsi/candidate_bridge.py | 候选侧长度前缀JSON通讯，加载自由修改的Harness；完整长输出和事件通过独立管道接收 |
| src/llmgo/rsi/service.py | 从旧入口提取的vLLM生命周期，无隐式安装或另一路评测 |
| scripts/bootstrap_worker.py | 在用户指定镜像安装自己的共享overlay，保存约束和环境回执 |
| scripts/worker_entry.py | 每个新容器先验证/安装overlay，再启动Worker |
| scripts/train_ppo.py | 可自由修改的verl PPO/LC0起始配方入口 |
| scripts/package_source.py | 打包、逐文件SHA/大小/权限核验，排除凭证和运行数据 |
| scripts/install_teacher.py | 核验完整包内LC0/BT4身份，安装到现有site路径，不覆盖不同文件 |
| .agents/skills/rsi-training/SKILL.md | RL/skills方法边界、SFT禁令与可选PPO配方说明 |
| tests/test_rsi_worker.py | 任意研究命令、模型回执与幂等恢复检查 |
| tests/fixtures/qwen_stream_boundaries.json.gz | 独立于runs目录的11条原始GPU流式响应回归样本 |
| tests/test_rsi_candidate.py | 真子进程、预算、伪造事件、历史、超时和真实Fairy接口检查 |
| docs/removed-files.json | 删除/替换文件清单 |
| docs/environment-status.json | 本次远端环境检查/安装的实际状态回执 |
| docs/validation.json | 交付时测试与包验证记录 |

## 主要修改

- controller/meta/worker：研究代码与argv保持可配置，不要求固定提案mode、没有自动晋级；prompt要求只用RL/skills，禁止SFT且使用SFT最终得0分。保留working/best/recommended、完整评测才更新最佳；目前没有自动检测SFT或归零机制。
- ladder/contracts/evaluator：固定L1–L8面板，初始基线每档10局、后续评测20局，独立轨道和资源约束；对手与正式结果保持外部控制。
- artifacts/memory：研究源码完整快照、身份、交接与短上下文；历史记录和Executor隔离继续保留。
- cctl：基底镜像共享overlay入口、任意research资源，仍只用授权集群/池/优先级。
- server：只保留此RSI面板、成绩矩阵、对局与日志查看；增量与懒加载。
- README、plan、first-run、operations、executor、observability、skills_docs与Agent skills：全面替换旧约束与启动说明。
- 测试保留初始Harness/规则/串流/PPO覆盖，替换已删除游戏创建/自动晋级/十局入口的测试，新增自主研究验收。

完整删除清单见 removed-files.json；包括其他棋种、旧Docker服务/十局评测/arena入口、旧多棋种网页和其专用测试。旧runs不删，PostTrainBench原目录不修改。

## 验收范围

本机CPU、模拟模型与真实Fairy只能证明程序接口、边界与恢复；不代表4B GPU训练通过、真实LC0评分或棋力提升。真实镜像的版本/依赖安装状态在environment-status.json；仍需GPU job完成Qwen生成、训练导出和重载验收。


默认多卡（2026-10-23）：每项任务最多8 GPU，site默认8；评测为每卡独立Qwen副本并行棋局，`request_evaluation.py --level L2 --gpus N`可选1–8卡。默认训练已改为verl PPO+LC0，8卡任务使用7卡actor/rollout/reference＋1卡LC0，详见docs/rsi-verl-ppo.md。逐局回执核验后恢复，完整批次才合并成正式成绩。轮询仍由Executor处理，网页只读进度摘要和按需日志。已有campaign需显式upgrade，不重写历史快照。

本次新增的verl入口与实现：`requirements-rsi-verl.txt`、`scripts/bootstrap_verl.py`、`src/llmgo/training/verl_ppo.py`（配置、数据、分布式训练、恢复、导出）、`verl_registration.py`（注册TD0与记录PPO批次/指标）、`verl_reward.py`（专用LC0 GPU与逐手评分）、`vllm_compat.py`（固定版本配置兼容）；`scripts/train_ppo.py`已切到这套实现。`docs/rsi-verl-ppo.md`给出完整提交样例与验收证据。新增`tests/test_verl_pipeline.py`，覆盖资源划分、任意合法批量的分块、断点完整性、并行游戏、重试与结果核验。

现有入口更新：`scripts/start_rsi.sh --upgrade --web`可在旧loop退出后，使用专用控制Python升级并恢复现有campaign。`scripts/request_evaluation.py --level L2 --gpus 8`登记单档20局；`scripts/evaluate.py --gpus N`供GPU job中的独立测试使用。Evaluator/Controller/CctlExecutor/Executor、网页进度和研究skills同时支持1–8卡评测与逐局恢复。GPU轮询300–600秒，CPU任务保留较短间隔。指定镜像、原始模型路径、Fairy难度、正式比赛协议均保留。
