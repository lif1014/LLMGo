# 第一次在服务器启动 RSI

适用目录：`/user/lyh4170`。Controller、强Agent与网页在0GPU devspace；对局、Lc0与PPO通过NORMAL/beijing/model-autosft的GPU job。本文是操作步骤，不表示远端依赖已经安装或真实4B GPU验收已经通过。

## 1. 上载程序：Git 或 tar 选一种

服务器尚没有项目目录时，可直接拉取GitHub：

```bash
cd /user/lyh4170
git clone https://github.com/lif1014/LLMGo.git 'LLMGo!'
```

也可上传本机 `/tmp/llmgo-rsi-upload-20261021/LLMGo-rsi-source-20261021.tar.gz` 至自己远端目录，然后：

```bash
cd /user/lyh4170
tar -xzf /user/lyh4170/LLMGo-rsi-source-20261021.tar.gz -C /user/lyh4170
```

压缩包内部已带`LLMGo!`目录，不要解压到`LLMGo!`里再套一层。首次安装目录不存在时使用这些命令；已有版本/研究记录则先核对，不能盲目覆盖。Git安装后可用`git pull --ff-only`更新工作仓库，冻结中的campaign不会自动升级为新源码。

tar有源码、隐藏`.agents`、文档、Fairy及NNUE；没有`.git`、认证、Python环境、学生权重或BT4。Git与tar都是部署程序的方式，不需要两者都做。

## 2. 先补齐启动条件

| 项目 | 站点需要的位置/条件 |
|---|---|
| 学生Qwen3.5-4B | 沿用只读`/user/liyaohu4170/models/Qwen3.5-4B`，不复制或改路径 |
| Fairy | 项目`engines/fairy-stockfish`，同目录保留NNUE；Git/tar已带 |
| Codex客户端 | `/user/lyh4170/tools/bin/codex`可执行，支持app-server且有可用provider认证 |
| cctl | devspace的PATH可找到，`cctl pool list --own`能看到model-autosft |
| Lc0程序 | `/user/lyh4170/tools/bin/lc0`，必须支持BT4及Worker所用GPU backend |
| BT4网络 | `/user/lyh4170/reward_models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz` |
| Worker环境 | `site.image`与`site.python`在每个新GPU容器中可运行Qwen/vLLM/PEFT及全部Worker依赖 |

BT4另行上传，本机文件为：

```text
/home/lif/.cache/llmgo/reward_models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz
SHA256 e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618
```

只上传tar还不能启动完整训练。2026-10-21核对时，自己的远端目录尚缺Codex程序、控制Python、Lc0和BT4。现有Codex配置已复制到`.codex-rsi/config.toml`，不要重复覆盖；它目前指向custom/deepseek-v4-pro，`meta_model=null`会沿用该模型。

### 控制环境与Codex

在0GPU devspace执行：

```bash
cd '/user/lyh4170/LLMGo!'
mkdir -p /user/lyh4170/tmp /user/lyh4170/cache /user/lyh4170/tools/bin
export TMPDIR=/user/lyh4170/tmp
export XDG_CACHE_HOME=/user/lyh4170/cache
export CODEX_HOME=/user/lyh4170/.codex-rsi
export PATH=/user/lyh4170/tools/bin:$PATH
python3 -m venv /user/lyh4170/venvs/rsi-control
/user/lyh4170/venvs/rsi-control/bin/pip install -r requirements-rsi-control.txt
```

若已有Node/npm，可在自己的前缀下安装Codex。以下版本是本机使用并确认npm存在的CLI版本，不代表其远端provider推理已验证：

```bash
export npm_config_cache=/user/lyh4170/cache/npm
npm install --global --prefix /user/lyh4170/tools @openai/codex@0.163.0
/user/lyh4170/tools/bin/codex --version
/user/lyh4170/tools/bin/codex app-server --help
/user/lyh4170/tools/bin/codex login status
```

CLI支持npm安装；登录方式按自己已有provider/账号配置，不要粘贴凭证进仓库或日志。[官方安装说明](https://learn.chatgpt.com/docs/codex/cli)，[认证说明](https://learn.chatgpt.com/docs/auth)。自定义provider可能还依赖配置指定的环境变量或桥接服务；不要凭`login status`通过就断言推理可用，也不要为了登录覆盖原provider配置。

在准备好认证后，可手动验证一次真实强模型调用（会使用相应模型额度）：

```bash
CODEX_HOME=/user/lyh4170/.codex-rsi /user/lyh4170/tools/bin/codex exec \
  --ephemeral --sandbox read-only --skip-git-repo-check \
  --cd '/user/lyh4170/LLMGo!' '只回复READY，不调用工具，不修改文件。'
```

这条实际请求成功后，再把强Agent视为具备可用的推理入口；它仍不能代替RSI app-server闭环验证。若要使用其他强模型，先明确provider与模型支持范围，再在init前配置`meta_model`；本说明不偷偷替换现有默认模型。

### Worker环境是独立条件

控制venv只负责调度/网页，不会自动安装GPU容器依赖。`requirements-rsi-worker.txt`需落实到每个新job可用的环境；默认站点镜像尚未验证为完整训练镜像。

可采用已准备并验证的镜像，或自己的共享Worker环境。在init前核对`site.image`、`site.python`和Worker PATH；vLLM启动器与Python必须匹配。仅在一次性job里pip安装，而后续仍启动原始镜像，不会使安装自动保留。不要直接假定devspace的`export PATH`会传给新job。更具体的边界见[操作文档第2节](rsi-operations.md#2-上传与环境准备)。

正式run前先在自己的GPU job验证`doctor --worker`，然后验证真实Qwen生成、BT4原生WDL和PPO单卡导出/重载；该验证不能在devspace申请GPU，也不能停止他人任务。所有任务继续用NORMAL/beijing/model-autosft。

## 3. 启动常驻循环

准备齐上述条件后，在0GPU devspace中执行；可用自己的tmux会话保持进程在线：

```bash
cd '/user/lyh4170/LLMGo!'
export TMPDIR=/user/lyh4170/tmp XDG_CACHE_HOME=/user/lyh4170/cache
export CODEX_HOME=/user/lyh4170/.codex-rsi
export PATH=/user/lyh4170/tools/bin:$PATH
source /user/lyh4170/venvs/rsi-control/bin/activate
python scripts/rsi.py doctor --platform --codex-auth
```

doctor失败退出码为1。先处理失败项目，再执行以下两条；`init`只在第一次运行：

```bash
python scripts/rsi.py init chess-rsi-001
python scripts/rsi.py run chess-rsi-001
```

`run`会真实提交GPU任务；当前默认embedded Executor，不需要另启动executor进程，也不需要手工启动`serve_vllm.sh`。评测Worker会启动和停止自己创建的学生服务。

如果重新运行同一个campaign，只执行`run chess-rsi-001`，不要再次init。它读持久Memory与已有job报告继续，任务提交不确定时先对账，不重复创建。更改站点配置/框架后应使用新campaign，不修改冻结实验。

## 4. 启动后会发生什么

1. 提交第一个GPU job：原Qwen3.5-4B对L1 Fairy，20局（10白10黑），200半步上限、thinking开启。
2. Executor自动提交/轮询/收集，Meta-Agent不会一直问job是否结束；等待GPU排队时可能较久。
3. 未达到升级条件时，强Agent读完整证据和Memory，选择改skills、PPO或两者组合；不是每轮都必定训练。
4. 有PPO时，GPU job做当前策略采样→学生移CPU→Lc0对实际选择前后评分→Lc0退出→学生回GPU做多轮token clip→导出完整学生模型。
5. 候选重新打20局，同级比较不明确时补匹配批次；同级有实测改善、已通关级回归无退步才扶正。失败/无改善候选保留证据，现任不被替换。
6. 现任连续两批满足胜至少16/20、得分率85%、黑白各60%才升级Fairy，再测新档基线。循环没有预设轮数终点，也不保证每轮都提高Elo。

当前PPO是课程单步反馈，A=V_Lc0(after)−V_Lc0(before)，不是完整多步GAE；正式下棋没有Lc0参与。

## 5. 网页和状态

在同一个devspace另开一个终端，使用控制环境：

```bash
cd '/user/lyh4170/LLMGo!'
LLMGO_RSI_ROOT=/user/lyh4170/rsi_campaigns PYTHONPATH=src \
  /user/lyh4170/venvs/rsi-control/bin/python -m uvicorn llmgo.server.app:app \
  --host 0.0.0.0 --port 7860
```

通过自己的devspace访问/端口转发方式打开`/rsi`。网页端口不会由以上命令自动在平台公开。可看到强Agent事件/工具/用量、Executor任务与轮询、Worker阶段、已完成对局、当前棋盘/思考、候选得分和Memory；大日志按需分页读取。PPO日志可看批次、优势、loss/KL/clip及原文文件，不是训练token数组全量自动推送。

只读状态：

```bash
python scripts/rsi.py status chess-rsi-001
python scripts/rsi.py executor status chess-rsi-001
```

进程中断后用相同run恢复；平台Failed/OOM则先查failure.json并解决原因，再显式`resume-failed chess-rsi-001`。不要同时开第二个Controller、删锁文件或盲目重提任务。完整恢复说明见[操作文档](rsi-operations.md)。
