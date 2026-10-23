# LLMGo!：自主国际象棋 RSI

强 Meta-Agent 在自己的远端 **0 GPU devspace** 研究，以 Qwen3.5-4B 为起点，**只能通过 RL 和改进 skills 提升能力；禁止 SFT，使用 SFT 最终得0分**。可自主修改 RL 算法、reward、数据、课程，以及支持这两种方法的 Harness、prompt、工具、skills 及触发逻辑；计算通过集群 job 执行。默认训练配方为 **verl PPO＋LC0**，支持多卡。每项任务最多8 GPU，默认评测8个单卡Qwen副本；默认训练7卡actor/rollout/reference＋1卡LC0。

上述方法与0分规则已写入 Meta-Agent 的 developer prompt、每轮任务 prompt 和研究 skills；当前没有自动识别任意训练代码是否包含 SFT 的检测器，也没有程序自动执行的 SFT 归零逻辑。禁止范围包括 SFT 预热、辅助监督损失、行为克隆及监督蒸馏。离线教师仍可用于 RL 奖励/价值估计、局面分析和提炼 skills，RL 策略梯度及价值函数拟合可用。

最终用独立评测比较不同难度的 Fairy-Stockfish。**没有胜率通关、80%自动升档或必须每轮提升才能继续研究的规则。**只刷最低档不能取得完整成绩。

## 运行方式

程序在本机开发，你上传后在远端运行。唯一默认可写远端根目录 `/user/lyh4170`；原学生路径保持 `/user/liyaohu4170/models/Qwen3.5-4B`，只读。所有任务使用 **beijing / model-autosft / NORMAL**，GPU 只用于 job/pytorchjob，不停止他人任务。

按 [首次部署](docs/rsi-first-run.md) 安装控制环境与基于指定镜像的 Worker overlay，然后：

```bash
bash '/user/lyh4170/LLMGo!/scripts/start_rsi.sh' --web
```

脚本使用现有专用控制Python，检查集群认证和Meta连接，首次初始化，随后在后台启动/恢复同名loop；`--web`同时启动网页。默认campaign为 `chess-autonomous-v2`，可用 `--campaign NAME` 选择自己的名称。启动后关闭终端不会结束后台loop。重复执行会检查真实进程锁，不重建campaign、不重复启动Controller；external Executor模式也会自动启动/复用独立Executor。已有端口上的服务保持原样。 若cctl的server配置缺少协议前缀，启动器会在本进程补齐 `https://`；旧的裸地址 `cybertron-hcloud.modelbest.co` 因其HTTPS证书过期而改用已验证的正式入口 `https://cybertron.modelbest.co`。已有完整http(s)地址保持原样，profile与token文件不变。集群查询失败会显示脱敏后的具体错误。

查看状态：`bash scripts/start_rsi.sh --status`；跟随日志：`bash scripts/start_rsi.sh --logs`，Ctrl-C只退出日志查看。Controller日志在campaign目录的 `controller.log`；网页日志在站点runtime_root下 `launcher/web-7860.log`。需要选择其他控制解释器时设置 `RSI_PYTHON`，不改变系统或其他环境。

`init`只执行一次，恢复时沿用原campaign。已有campaign保持冻结设置，旧自动晋级campaign请另起名称，不原地混用旧状态。手动CLI操作见[首次部署](docs/rsi-first-run.md)。

启动前可单独验证 Meta-Agent：在远端配置 `OPENAI_API_KEY`，或在当前 provider 的 `experimental_bearer_token` 填入 key 后，运行 `python3 scripts/check_meta_connection.py`。配置内 token 优先，测试与正式 Meta-Agent 使用相同解析逻辑；即使旧 env_key 行仍在，也无需再次 export。它使用站点配置中的 Codex、模型和独立 CODEX_HOME，通过 app-server 请求结构化 READY；只有真实模型请求成功才通过。`--offline` 仅检查本地程序与协议，不证明认证有效。每次测试的脱敏原始事件和结果保存在自己目录 `rsi_readiness/meta-connection/`。目前自己的 provider 使用 `https://yeysai.com/v1`、模型 `deepseek-v4.1-flash`，已真实通过 READY 请求。独立 Chat 转接层保留 Codex 工具和会话，将 Codex Responses 请求转为上游 `/v1/chat/completions`；无需让该站点支持 Responses。完整 endpoint 也会被规范化，避免重复拼接。安装与配置见[首次部署](docs/rsi-first-run.md)。 服务器已限制可写目录，因此 RSI 使用 `meta_sandbox_mode: "external"`，由外层容器/服务器提供隔离，避免容器中再次建立 bubblewrap 沙箱。正式会话、会话恢复和检查脚本使用相同策略；可运行 `python3 scripts/check_meta_connection.py --sandbox-check` 验证 shell 和模型连接。 已在指定镜像的0GPU job验证真实模型工具写入/读取文件，以及重启 Codex 后恢复同一会话继续执行工具；原始事件保留在自己目录 readiness/Memory 中。

默认先完成 L1–L4 的初始玩家基线，每档10局、白黑各5局，共40局，然后调用 Meta-Agent。之后由它决定代码改动、任意计算任务、测试范围和下一步。失败实验会保留产物并交回研究；基线基础设施失败则等待修复。没有 Meta-Agent 总研究时间上限。

Executor默认embedded，与Controller同进程，但轮询/原始日志已从Meta上下文分离；需要独立0GPU进程时，在init前设置 `executor_mode: "external"`，分别运行 `scripts/rsi.py executor run CAMPAIGN` 和 `scripts/rsi.py run CAMPAIGN`。两种模式的研究、基线和恢复规则相同。GPU任务每300秒查询集群，状态稳定时退避至450、600秒；纯CPU任务保留较短间隔（默认120秒）。


## 单难度评测接口

Meta-Agent 在研究 workspace 执行 `python scripts/request_evaluation.py --level L2`，（默认8卡，可加 `--gpus 1` 至 `--gpus 8`）即可保留既有RL任务并把 proposal.json 设为仅测L2。也接受数字 `--level 2`；`--track model` 测固定Harness模型对照，`--track both` 测两轨，`--full-panel` 登记L1–L8完整面板。后续每档每轨20局。

该命令只准备提案并输出标准 ready JSON；Meta-Agent 将其作为最终答复，Controller/Executor 自动提交和监控GPU任务。它不会在devspace直接推理，也不会让Meta-Agent轮询。可单独评测skills/Harness，也可保留proposal.jobs先训练再评测。`--hypothesis "..."` 更新假设，`--batch-index 1` 选择新复测种子批次。

完成L1–L4初始40局后立即进入研究；基线面板显示为4/8、无完整总分，best暂为空。单难度结果仍会保存并进入Memory，但只有完整L1–L8 player结果才更新完整实测最佳。

## LC0 / BT4 完整上传包

`LLMGo-rsi-full-20261022.tar.gz` 在源码外包含 `teacher_assets/`：Linux x86_64 LC0 v0.32.1、对应源代码/构建输入、BT4-it332权重和SHA回执。LC0是在指定Worker镜像中为H800（sm_90）构建的离线教师程序；BT4是它加载的神经网络权重，用于初始PPO的局面价值估计。正式对手仍是Fairy-Stockfish，正式玩家不使用LC0替Qwen选步。

上传完整包并解压到 `/user/lyh4170` 后，在仓库运行：

```bash
python3 scripts/install_teacher.py --verify-only
python3 scripts/install_teacher.py
```

第二条将资产安装到原site的 `lc0_bin` 与 `reward_weights` 路径，遇到不同的已有文件会拒绝覆盖；不改系统或其他venv。编译/version/动态库检查不能替代GPU推理验证，详细范围见包内构建回执。源码小包不含这些资产，学生Qwen权重不包含在任一上传包。`teacher_assets/`不进入Git，也不复制到每轮研究快照，避免365 MiB教师反复占用磁盘。

## 固定评测

正式入口是 **`scripts/evaluate.py`**。受保护范围包括它使用的裁判、棋规、引擎、协议和固定参考 Harness；候选在单独进程中载入可修改代码，不能通过 Python import 改动裁判。所有走法由裁判重新检查。

| 轨道 | 提交 | 用途 |
|---|---|---|
| `player` | 候选模型＋候选 Harness＋skills | 主要实际玩家能力 |
| `model` | 候选模型＋创建 campaign 时固定的参考 Harness/skills | 权重能力对照 |

后续正式评测每档20局、白黑各10局；初始基线每档10局、白黑各5局。两者均为200半步、thinking=True。每手总生成上限32768 tokens，单请求540秒、整手1800秒；推理请求由评测端执行并记录。候选可自主改变内部阶段、工具及分配，但不能突破外部资源上限。正式下棋不用另一个引擎替代 Qwen 选步；离线教师与训练方案须遵守上面的 RL/skills 方法边界。

| 难度 | Skill Level | depth | 时间上限/步 |
|---|---:|---:|---:|
| L1 | -9 | 5 | 0.05秒 |
| L2 | -5 | 5 | 0.10秒 |
| L3 | -1 | 5 | 0.15秒 |
| L4 | 3 | 5 | 0.15秒 |
| L5 | 7 | 5 | 0.15秒 |
| L6 | 11 | 5 | 0.15秒 |
| L7 | 15 | 5 | 0.15秒 |
| L8 | 20 | 5 | 0.15秒 |

总分是八档得分率 `(胜+0.5×和)/20` 的等权平均；**缺任何一档，总分为 null，不能更新实测最佳**。初始基线只测L1–L4，每档分母为10、共40局；这不是完整八档成绩；网页按实际局数显示。基线样本更少，与后续20局结果比较时须注意波动。两条轨道独立计分。20局有波动，实测最高不代表显著进步；最终推荐模型应另用新的 batch-index 完整复测。难度不是已校准 Elo，不报告虚构的 Elo。

在 GPU job 中可调用评测查看实力，例如：

```bash
python3 /PATH/TO/PROTECTED/source/scripts/evaluate.py   --model-path /user/liyaohu4170/models/Qwen3.5-4B   --candidate-root /user/lyh4170/research-checkout   --track both --levels 1,2,3,4,5,6,7,8   --output /user/lyh4170/probes/unique-name
```

`--levels 1` 是快速单档探针；`--track both` 完整共320局。每次输出使用新目录。Meta-Agent 可通过任意 research job 调用受保护入口；正式 campaign 成绩只采纳 Controller 注册、Executor 核验的评测。

同一 Unix 用户下的独立进程和哈希验证是防误修改边界，**不是对恶意代码的 OS 沙箱**。若要防止主动篡改，用独立评测账户/服务和只读挂载部署评测副本。不要只把 evaluate.py 标成唯读，却让其裁判或引擎可替换。

## 研究提交与观察

[操作与 proposal 格式](docs/rsi-operations.md)；[任务提交样例与本次文件变化](docs/rsi-submission-and-files.md)；[Executor](docs/rsi-executor.md)；[网页与日志](docs/rsi-observability.md)；[初始棋类 skills](skills_docs.md)。

开启网页：

```bash
cd '/user/lyh4170/LLMGo!'
export LLMGO_RSI_ROOT=/user/lyh4170/rsi_campaigns
PYTHON=/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python bash scripts/run_web.sh
```

通过平台转发7860端口，访问 `/rsi` 或 `/`。页面显示研究假设、工作/最佳/推荐版本、跨难度矩阵、Meta-Agent可见输出与命令、Executor轮询、任务日志、棋盘和模型流式输出。网页只读磁盘；3秒增量、有界窗口、正文按需加载，隐藏页面暂停刷新，不发 cctl 查询。

## Worker 镜像与安装

默认且本次指定基底：

```
share-dev/ngc-docker/ptdbench-verl-vllm017:dev-957fee7-cb-bf632efb
```

`worker_entry.py` 在这个镜像内使用专属 Worker overlay `/user/lyh4170/venvs/llmgo-rsi-20261022`；正式推理使用独立 inference overlay（Transformers 4.57.6）。默认 verl PPO 另用 `/user/lyh4170/venvs/llmgo-verl-ppo-tf54-20261024`，固定 verl 0.8.0、Transformers 5.4.0、mistral-common 1.10.0，保留原镜像 Torch/CUDA/vLLM。所有安装、缓存和临时文件在本研究自己的目录；安装器有归属标记、文件锁、基础栈约束和导入回执，不修改系统或其他任务环境。依赖从中科大镜像下载。

## 本地检查与重新打包

```bash
PYTHONPATH=src python3 -m pytest -q
node tests/test_rsi_ui.js
python3 scripts/package_source.py --output /home/lif/LLMGo-rsi-autonomous-20261022.tar.gz
# 已有完整教师资产目录时，额外生成完整包
python3 scripts/package_source.py --teacher-assets /PATH/TO/teacher_assets --output /PATH/TO/LLMGo-rsi-full-20261022.tar.gz
```

源码小包包含当前源码、tests、隐藏 `.agents`、文档、Fairy 和 NNUE；排除 `.git`、runs、凭证、Python环境及学生/LC0权重。带 `--teacher-assets` 的完整包额外包含经核验的LC0/BT4教师资产，学生权重仍不打包。旧对局不会删除或打进包。本次删除了其他棋种、旧十局/自动晋级入口、旧游戏创建网页及其专用测试；逐手查看集成在 RSI 网页，删除清单见 [removed-files.json](docs/removed-files.json)。

本机无 Qwen4B GPU 实战验收。CPU/模拟服务/真实 Fairy 接口测试只验证程序链路，不证明模型棋力提高，也不证明远端镜像训练已成功。

## verl PPO、多卡与已有 campaign 升级

新训练入口 `scripts/train_ppo.py` 自动切换到自己的独立 verl 环境；安装脚本为 `scripts/bootstrap_verl.py`，依赖见 `requirements-rsi-verl.txt`。基于指定 ptdbench 镜像，固定 verl 0.8.0 / Transformers 5.4.0 / 原镜像 Torch 2.10、vLLM 0.17；不改变评测推理环境。详细配方、资源划分、恢复和证据见 [PPO 与并行操作](docs/rsi-verl-ppo.md)。

已有 campaign 的源码和站点配置被冻结，不会因覆盖仓库而自动升级。旧任务确认停止、Controller/Executor退出后，执行一次：

```bash
cd '/user/lyh4170/LLMGo!'
bash scripts/start_rsi.sh --upgrade --web
```

脚本自动使用已安装的控制环境Python。upgrade保留旧任务、原始基线和参考Harness，新增版本快照，升级默认8卡并为失败基线登记新任务。失败旧版本没有可验证的逐局回执时，不复用其未核验结果。正式比赛数和协议不变：初始L1–L4每档10局，之后单档20局。


8GPU训练验收已通过（任务1350755）：7卡PPO＋1卡LC0，2个训练步、每步2个PPO epoch，第二步重新采样，非零有限梯度、完整分片checkpoint与HF导出均完成。正式评测的多难度完整比赛尚未重跑；这次短验收不代表棋力提升。回执在 `/user/lyh4170/rsi_readiness/verl8-gpu-smoke-v5/`，详细证据见 `docs/environment-status.json`。
