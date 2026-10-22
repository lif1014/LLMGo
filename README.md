# LLMGo!：自主国际象棋 RSI

强 Meta-Agent 在自己的远端 **0 GPU devspace** 研究，以 Qwen3.5-4B 为起点，**只能通过 RL 和改进 skills 提升能力；禁止 SFT，使用 SFT 最终得0分**。可自主修改 RL 算法、reward、数据、课程，以及支持这两种方法的 Harness、prompt、工具、skills 及触发逻辑；计算通过集群 job 执行。当前 HF PPO＋LC0 保留为可选起始配方。

上述方法与0分规则已写入 Meta-Agent 的 developer prompt、每轮任务 prompt 和研究 skills；当前没有自动识别任意训练代码是否包含 SFT 的检测器，也没有程序自动执行的 SFT 归零逻辑。禁止范围包括 SFT 预热、辅助监督损失、行为克隆及监督蒸馏。离线教师仍可用于 RL 奖励/价值估计、局面分析和提炼 skills，RL 策略梯度及价值函数拟合可用。

最终用独立评测比较不同难度的 Fairy-Stockfish。**没有胜率通关、80%自动升档或必须每轮提升才能继续研究的规则。**只刷最低档不能取得完整成绩。

## 运行方式

程序在本机开发，你上传后在远端运行。唯一默认可写远端根目录 `/user/lyh4170`；原学生路径保持 `/user/liyaohu4170/models/Qwen3.5-4B`，只读。所有任务使用 **beijing / model-autosft / NORMAL**，GPU 只用于 job/pytorchjob，不停止他人任务。

按 [首次部署](docs/rsi-first-run.md) 安装控制环境与基于指定镜像的 Worker overlay，然后：

```bash
cd '/user/lyh4170/LLMGo!'
export PYTHONPATH="$PWD/src"
export CODEX_HOME=/user/lyh4170/.codex-rsi
export PATH=/user/lyh4170/tools/bin:$PATH
python3 scripts/rsi.py doctor --platform --codex-auth
python3 scripts/rsi.py init chess-autonomous-v2
python3 scripts/rsi.py run chess-autonomous-v2
```

请用安装了控制依赖的 Python；也可将上述 `python3` 换成 `/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python`。`init` 只执行一次，恢复时重新 `run`。旧的自动晋级 campaign 保留证据，**新建 v2 campaign，不原地混用旧状态**。

默认先完成 L1–L8 的初始玩家基线，每档20局，共160局，然后调用 Meta-Agent。之后由它决定代码改动、任意计算任务、测试范围和下一步。失败实验会保留产物并交回研究；基线基础设施失败则等待修复。没有 Meta-Agent 总研究时间上限。

## 固定评测

正式入口是 **`scripts/evaluate.py`**。受保护范围包括它使用的裁判、棋规、引擎、协议和固定参考 Harness；候选在单独进程中载入可修改代码，不能通过 Python import 改动裁判。所有走法由裁判重新检查。

| 轨道 | 提交 | 用途 |
|---|---|---|
| `player` | 候选模型＋候选 Harness＋skills | 主要实际玩家能力 |
| `model` | 候选模型＋创建 campaign 时固定的参考 Harness/skills | 权重能力对照 |

每档20局、白黑各10局、200半步、thinking=True。每手总生成上限32768 tokens，单请求540秒、整手1800秒；推理请求由评测端执行并记录。候选可自主改变内部阶段、工具及分配，但不能突破外部资源上限。正式下棋不用另一个引擎替代 Qwen 选步；离线教师与训练方案须遵守上面的 RL/skills 方法边界。

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

总分是八档得分率 `(胜+0.5×和)/20` 的等权平均；**缺任何一档，总分为 null，不能更新实测最佳**。两条轨道独立计分。20局有波动，实测最高不代表显著进步；最终推荐模型应另用新的 batch-index 完整复测。难度不是已校准 Elo，不报告虚构的 Elo。

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

`worker_entry.py` 在这个镜像内用其 Python 建立共享 overlay `/user/lyh4170/venvs/llmgo-rsi-20261022`，保留镜像 Torch/CUDA/vLLM，建立两个隔离 overlay：HF训练用 Transformers 5.2.0，vLLM推理保留镜像的 Transformers 4.57.6；另加装控制依赖、PEFT/accelerate，使用中科大镜像。环境、缓存和临时文件均在此RSI专用目录，安装器拒绝修改未标记归属的已有环境。推理解释器路径由 site.inference_python 指定。文件锁避免并发安装；保存基础版本约束、完整 freeze 与导入验收。依赖冲突或 Qwen3.5 不支持会明确失败，不会自动换镜像或升级 GPU 栈。详见首次部署文档。

## 本地检查与重新打包

```bash
PYTHONPATH=src python3 -m pytest -q
node tests/test_rsi_ui.js
python3 scripts/package_source.py --output /home/lif/LLMGo-rsi-autonomous-20261022.tar.gz
```

包包含当前源码、tests、隐藏 `.agents`、文档、Fairy 和 NNUE；排除 `.git`、runs、凭证、Python环境及学生/LC0权重。旧对局不会删除或打进包。本次删除了其他棋种、旧十局/自动晋级入口、旧游戏创建网页及其专用测试；逐手查看集成在 RSI 网页，删除清单见 [removed-files.json](docs/removed-files.json)。

本机无 Qwen4B GPU 实战验收。CPU/模拟服务/真实 Fairy 接口测试只验证程序链路，不证明模型棋力提高，也不证明远端镜像训练已成功。
