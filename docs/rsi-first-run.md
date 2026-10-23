# 远端首次部署

本机负责实现与打包；运行环境基于用户指定镜像：

`share-dev/ngc-docker/ptdbench-verl-vllm017:dev-957fee7-cb-bf632efb`

不要把旧源代码 tar 和新自主研究 pipeline 混合解压。已有仓库先备份或移到自己目录的另一个位置，再把新包解压到站点配置的 repo 路径。原始模型路径不变。

## 1. 上传与控制环境

上传包有两种：`LLMGo-rsi-autonomous-20261022.tar.gz` 是源码小包；`LLMGo-rsi-full-20261022.tar.gz` 额外包含LC0/BT4教师资产，初始PPO建议使用完整包。两者都不含学生模型/认证信息/历史runs。以下使用完整包，上传到 /user/lyh4170：

```bash
tar -xzf /user/lyh4170/LLMGo-rsi-full-20261022.tar.gz -C /user/lyh4170
cd '/user/lyh4170/LLMGo!'
python3 -m venv --without-pip --system-site-packages /user/lyh4170/venvs/llmgo-rsi-control-20261022
/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python -m pip install -r requirements-rsi-control.txt
export PATH=/user/lyh4170/tools/bin:$PATH
export CODEX_HOME=/user/lyh4170/.codex-rsi
export TMPDIR=/user/lyh4170/llmgo-rsi-runtime-20261022/tmp
export XDG_CACHE_HOME=/user/lyh4170/llmgo-rsi-runtime-20261022/cache
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
```

这是新建的专用控制环境；若同名目录已存在且用途不明，不覆写，先选择自己的另一新目录。此控制环境不需要 torch/GPU。控制器、Meta-Agent、Memory 和网页都在自己0GPU devspace，使用安装好控制依赖的解释器。

Codex 配置在 `/user/lyh4170/.codex-rsi/config.toml`，来源目录保持只读。当前沿用用户指定的 `custom/deepseek-v4.1-flash`；`site.meta_model=null` 时使用此模型。2026-10-23 已验证该 key 的 Chat Completions 文本请求、工具调用和工具结果回传，随后经 Codex app-server 成功取得结构化 READY。

当前入口 `/user/lyh4170/tools/bin/codex` 指向专用 launcher，使用从来源复制的独立 CLI 0.162.1：`/user/lyh4170/tools/codex-reused-20261023/codex-native`。原失效链接和旧 launcher 在自己目录 readiness 中保留。源码包不包含 CLI 或任何认证信息。

这个 CLI 使用 Responses，而当前上游提供 Chat Completions；把完整 Chat URL 直接填给原生 CLI 不能改变协议。因此 launcher 自动启动只监听 `127.0.0.1` 的短生命周期转接服务：Codex → 本地 Responses → LiteLLM → `https://yeysai.com/v1/chat/completions`。无需单独启动服务；每次 CLI 退出时清理自己的子进程。模型选择沿用当前配置，不自动替换。

服务器已由用户确认限制只能修改 `/user/lyh4170`，因此 RSI 当前使用外层容器/服务器隔离，站点设置 `meta_sandbox_mode: "external"`。`thread/start` 和 `thread/resume` 使用 `sandbox="danger-full-access"`，每轮及执行检查明确使用 `sandboxPolicy={"type":"externalSandbox","networkAccess":"enabled"}`；不再在容器中建立 bubblewrap 沙箱。审批策略仍为 never，服务器权限边界、只写自己目录、来源只读、资源授权和SFT禁令继续有效。这些目录权限由外部服务器执行，Codex 不额外验证它们。 2026-10-23 的0GPU任务1344064已通过 shell 执行检查、真实模型文件写入/读取，以及重新启动 Codex 后的同线程恢复和第二次工具操作，两个命令的退出码均为0。完整事件在 `/user/lyh4170/rsi_readiness/codex-reuse-20261023/agent-probe-4c60d45bd8/memory`，回执为同 readiness 根目录下 `external-agent-probe.json`；这不代表4B GPU训练或正式评测已验收。

之前测试容器报 `bwrap: Failed to make / slave: Permission denied`，这是内层沙箱所需挂载权限被拒绝；本次改为使用已存在的外层隔离，不调整宿主机或容器权限、不安装Docker。专用前缀此前安装的 bwrap 不再是 external 模式的依赖。如在另一台支持内层沙箱的环境运行，可明确设置 `meta_sandbox_mode: "workspace-write"`；未知值直接报错，不静默切换模式。既有 campaign 的站点快照若没有这个新字段，也采用 external，以适配当前服务器部署；重启 Controller 后载入新代码，Memory 和已完成结果不会清空。

依赖在专用独立环境 `/user/lyh4170/venvs/llmgo-meta-bridge-20261023`，已用指定 Worker 镜像的0GPU任务安装 LiteLLM 1.105.2、FastAPI 0.161.1、Uvicorn 0.54.0；版本清单在 `configs/rsi/meta-bridge-requirements.txt`。未改系统包、Worker训练/推理环境或他人任务。换镜像要重新确认 Python ABI，不能假定共享venv在任何镜像都能运行。 实际Ubuntu24.04 devspace与Ubuntu22.04 Worker的Python标准库不同：复制来的3.12.13虽能print，匯入ctypes/LiteLLM时仍可SIGSEGV。为devspace另建 `llmgo-meta-bridge-devspace-20261023`，用该镜像的Python3.12.3和标准库，并通过.pth唯读复用固定cp312依赖。launcher只有在操作系统、Python版本和base_prefix与安装回执匹配时才选择它；原Worker转接环境保留。启动失败现在会记录退出码、信号、faulthandler输出及脱敏stderr尾部。

配置示意如下，key 仍由你保存在当前 provider 中，不要复制示意占位符覆盖已有 key：

```toml
model = "deepseek-v4.1-flash"
model_provider = "custom"

[model_providers.custom]
name = "custom"
base_url = "https://yeysai.com/v1"
wire_api = "responses"
# experimental_bearer_token 使用你已有的私有值；不写进仓库。
```

`wire_api="responses"` 是 Codex 面向本地转接服务的协议，上游实际请求仍为 Chat Completions。`base_url` 保存 API 根路径；也兼容输入 `https://yeysai.com/v1/chat/completions` 并规范化，不重复添加 `/v1`。只写 `base_url = "..."` 一项；`base_trl = base_url = "..."` 不是合法 TOML。

每次真实转接请求、流式返回和错误都保存在专用前缀 `_runtime/bridge/<唯一目录>/events.jsonl`，服务日志在同目录 `bridge.log`，均先脱敏。上游 key 只传给转接服务的子进程环境；Codex 连接本地服务使用随机临时 token。集群白名单代理仅对本入口子进程生效。原始事件可能含研究内容，目录与文件使用私有权限。

在启动 Controller 的同一远端终端设置所配置 provider 的有效 key，再做最小请求；不要把 key 写进源码、命令历史或发到对话中：

```bash
read -r -s -p 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY
printf '\n'
cd '/user/lyh4170/LLMGo!'
python3 scripts/check_meta_connection.py
```

也支持直接在 `/user/lyh4170/.codex-rsi/config.toml` 当前 provider 的 `experimental_bearer_token` 字段保存 key，此时只需执行 `python3 scripts/check_meta_connection.py`，无需再次 export。配置内 token 优先；测试与正式 Meta-Agent 使用同一解析逻辑，兼容仍保留 env_key/requires_openai_auth 的原配置。必要的环境映射仅发生在 Codex 子进程内，不修改全局环境或另存 key。配置文件用 `chmod 600` 限制权限，不打包或提交到 Git。

只有输出 `"ok": true`、`"inference_tested": true` 和 `"reply": "READY"` 才确认模型连通。脚本不接收命令行明文 key；配置和环境中的已知 key 值都会从原始协议事件中移除，再写入 `rsi_readiness/meta-connection/<唯一测试目录>/trace.jsonl`，结果在同目录 result.json。默认180秒，仅限制此连接测试，不限制研究时间；可用 `--timeout 540` 调整。`--offline` 不需要 key，仅检查 CLI 与 app-server 协议，不能代替真实连接测试。401/403、模型或路径不可用、额度与网络错误会给出对应提示，不会自动更换模型或站点。

在实际运行 Controller 的 devspace 中检查当前隔离模式下的 shell 执行及模型连接：

```bash
python3 scripts/check_meta_connection.py --sandbox-check
# 仅验证 CLI/shell 执行，不发模型请求，也不需要 key：
python3 scripts/check_meta_connection.py --offline --sandbox-check
```

`--sandbox-check` 通过 app-server 的 `command/exec` 执行无副作用的 `printf READY`，使用与正式 Meta-Agent 相同的隔离策略；external 模式不会调用 `codex sandbox linux` 或 bubblewrap。失败时停止，不继续花模型 token；完整命令结果进入私有 trace。通过时 `shell_execution_tested=true`，这只证明执行环境可用，`model_tool_calls_tested=false`，不等于模型已正确选用工具或已验证线程续接。未加此参数的 READY 测试只验证模型与协议。

终端环境变量不会因复制文件自动迁移，换终端或重启后须重新提供，或使用自己目录中受权限保护的启动配置。在同一终端完成 READY 测试后，按第4节运行 doctor/init/run，Controller 的 Meta-Agent 子进程会继承该环境变量。

若 own tools/bin/codex 尚不存在，并且镜像有 Node/npm，可安装此前验证的 CLI 版本：

```bash
export npm_config_cache=/user/lyh4170/llmgo-rsi-runtime-20261022/cache/npm
npm install --global --prefix /user/lyh4170/tools/llmgo-rsi-20261022 @openai/codex@0.163.0
/user/lyh4170/tools/llmgo-rsi-20261022/bin/codex --version
/user/lyh4170/tools/llmgo-rsi-20261022/bin/codex app-server --help
CODEX_HOME=/user/lyh4170/.codex-rsi /user/lyh4170/tools/llmgo-rsi-20261022/bin/codex login status
```

上例为避免干扰其他任务使用新的独立CLI前缀；采用它时，将自己的site配置中codex_bin设为 `/user/lyh4170/tools/llmgo-rsi-20261022/bin/codex`，且在init前完成。已有默认 `/user/lyh4170/tools/bin/codex` 可用时直接只读复用，无需重新安装或覆盖。

`doctor --codex-auth` 检查已配置 provider 凭证；只有没有配置 key 时才检查 CLI login status。自定义 token 不需要 CLI OAuth 登录。doctor 不发模型请求，不代表 key 有效；真实认证与推理以 `check_meta_connection.py` 为准。上面的 npm 安装只提供原生 CLI；Chat-only 站点仍需要上述专用转接入口。

## 2. 在指定基底镜像安装 Worker overlay

每个新GPU容器都从基底镜像启动，不能依赖之前容器里临时 pip 的结果。现在的 worker_entry 会建立并复用共享目录中的隔离 overlay，不替换镜像：

```bash
# 这条安装命令应在指定镜像内执行；可用0GPU job完成安装/import检查。
python3 scripts/bootstrap_worker.py --prefix /user/lyh4170/venvs/llmgo-rsi-20261022
```

本次专用路径为 `/user/lyh4170/venvs/llmgo-rsi-20261022` 与其 `-inference` 同级目录，pip/cache/tmp 全部在训练前缀的 `_runtime` 内。安装器只接受带 `llmgo-owner.json` 标识的已有环境，遇到其他环境会拒绝修改。不会激活、升级、卸载系统Python包或其他venv。

下载固定走中科大 `https://mirrors.ustc.edu.cn/pypi/simple`；先测试直连，不通再测试集群白名单代理。两条路均不通会明确报错，不继续重复pip。

安装器：

1. 使用镜像自己的 Python 创建 `--system-site-packages` venv。
2. 保留镜像 Torch 2.10.0+cu129、vLLM 0.17.0、CUDA相关包与tokenizers，不替换GPU栈。
3. 通用Worker overlay保留Transformers5.2；默认verl PPO使用另一个独立环境（见rsi-verl-ppo.md），固定Transformers5.4与mistral-common1.10。原镜像4.57.6没有HF Qwen3.5模型类。
4. 推理overlay `llmgo-rsi-20261022-inference` 保留 Transformers 4.57.6：vLLM 0.17声明依赖<5，且自带Qwen3.5 config/native模型实现。正式服务从site.inference_python启动，不能混用训练解释器。
5. 安装 requirements-rsi-worker/control 中缺少的控制组件、PEFT、accelerate。
6. 分别验证HF Qwen3.5/PEFT与vLLM native Qwen3.5 config。
7. 在 venv 写入使用该解释器的 vllm launcher、环境回执、base-stack.constraints.txt 和 installed.freeze.txt。
8. 通过安装锁避免多个任务同时写同一个环境。

不支持 Qwen3.5 或依赖不兼容时会失败并保留原因，不自动升级 torch/CUDA/vLLM。此时应在同一基底上制作明确版本的派生环境，再验证；不能把“镜像名字有vllm”当作兼容证据。镜像检查/安装是否实际完成，以 docs/environment-status.json 的回执为准，代码实现本身不代表远端安装成功。

镜像ensurepip不可用，因此venv用without-pip并继承镜像已有pip；安装路径和pip缓存均在自己目录。每次 worker 启动会先做安装/核验，再 exec overlay 解释器。基底镜像必须相同；不要把共享 venv 接到不同 Python ABI 的其他镜像。site.worker_venv 可指向自己目录的新前缀重新安装。

在真正GPU job内：

```bash
/user/lyh4170/venvs/llmgo-rsi-20261022/bin/python scripts/rsi.py doctor --worker
# 若选择初始LC0 PPO，再加 --ppo 检查可选教师资产
/user/lyh4170/venvs/llmgo-rsi-20261022/bin/python scripts/rsi.py doctor --worker --ppo
```

0GPU import 检查不能替代GPU生成、训练、导出和重载检查。devspace 不申请GPU。

## 3. 模型与引擎

- 学生只读来源：`/user/liyaohu4170/models/Qwen3.5-4B`。
- Fairy 与 NNUE：包内 engines，站点保持 `/user/lyh4170/LLMGo!/engines/fairy-stockfish`。
- 初始 PPO 可选 Lc0：`/user/lyh4170/tools/bin/lc0`，需支持BT4和所选backend。
- 可选 BT4：`/user/lyh4170/reward_models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`。

完整包已把BT4、Linux LC0程序、对应源代码和构建回执放在 `LLMGo!/teacher_assets`。解压后执行：

```bash
cd '/user/lyh4170/LLMGo!'
python3 scripts/install_teacher.py --verify-only
python3 scripts/install_teacher.py
```

安装器核对全部资产SHA，把LC0和BT4复制到上面的原site路径；相同资产直接复用，不同已有文件拒绝覆盖，不修改系统/其他任务环境。LC0程序针对指定Worker镜像和H800/sm_90构建，GPU推理是否实际验证以build-result.json为准；不能只看编译成功。`teacher_assets/lc0-source.tar.gz`包含匹配的LC0/common/Abseil源代码、构建说明/脚本和Meson/Ninja wheel。

源码小包不包含教师资产。单独上传时，LC0放到site.lc0_bin并设为可执行，BT4放到site.reward_weights；BT4保持gzip原文件，不解压，SHA256必须为 `e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618`。本机BT4位于 `/home/lif/.cache/llmgo/reward_models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`。不使用该配方时无需LC0/BT4。

含义与来源：[Lc0引擎](https://github.com/LeelaChessZero/lc0)、[BT4-it332网络](https://lczero.org/play/networks/bestnets/)。BT4是LC0加载的棋类神经网络权重；当前训练从其原生WDL得到V=P(win)−P(lose)，不是Qwen权重，也不是正式对手。

## 4. 启动与恢复

最短启动方式（包含环境/Meta检查、首次初始化、后台loop及网页）：

```bash
bash '/user/lyh4170/LLMGo!/scripts/start_rsi.sh' --web
```

相同命令可恢复；`--campaign NAME`选择其他campaign，`--status`查看状态，`--logs`跟随日志。默认使用已有专用控制解释器，可通过 `RSI_PYTHON`指定；脚本不安装或升级依赖。Controller与网页使用独立锁，关闭终端后仍在后台运行，已被其他任务占用的网页端口不会被终止。`executor_mode=external`时脚本同时启动/复用Executor；默认embedded无需额外进程。

若cctl报 `unsupported protocol scheme ""`，说明server地址缺少协议前缀。启动器会按CCTL_SERVER、CCTL_PROFILE/当前profile、旧modelbest配置的顺序读取既有地址，仅在本进程补齐https://；Controller/Executor也使用相同解析。旧裸地址 `cybertron-hcloud.modelbest.co` 的HTTPS证书在2026-10-21过期，因此该裸别名会改用已实测有效的正式入口 `https://cybertron.modelbest.co`。显式指定的完整http(s)地址保持原样，凭证不变。不会修改全局profile、另存token或把Qwen模型API key当作cctl凭证。若修正地址后出现401/403，再检查cctl认证或卡池权限；doctor会显示脱敏的具体错误。

若 `CCTL_SERVER=https://cybertron.modelbest.co cctl pool list --own -o json` 返回 `[]`，说明当前CLI没有列出该入口下自己的卡池，不能把它当作解析失败或直接跳过权限校验。先用 `cctl profile current` / `cctl profile list` 核对profile，再在正确账号和入口重新登录或由管理员核对model-autosft成员权限。Qwen/Meta模型API key与集群cctl凭证独立，复制Codex配置不会迁移cctl登录。

如需手动逐步操作：

```bash
cd '/user/lyh4170/LLMGo!'
export PYTHONPATH="$PWD/src"
export CODEX_HOME=/user/lyh4170/.codex-rsi
export PATH=/user/lyh4170/tools/bin:$PATH
/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python scripts/rsi.py doctor --platform --codex-auth
/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python scripts/rsi.py init chess-autonomous-v2
/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python scripts/rsi.py run chess-autonomous-v2
```

站点默认路径见 configs/rsi/site.beijing.json。需要自定义时复制为自己的 site，CLI 的 --site 放在子命令前。campaign创建后站点冻结，保持原配置恢复。

init创建新campaign时冻结site与初始参考，只执行一次；重复init同名会报错，不会清零。首次是L1–L4每档10局、白黑各5局，共40局player基线，之后Meta-Agent开始研究。再次run同名campaign接续持久状态、当前模型/源码和已有任务；控制器在基线阶段中断后，会接续监控原任务，已收集完成的档位不会因为重启重跑；若某个评测job本身失败，修复后可能重跑该档10局基线或20局候选评测，当前没有局内逐步断点恢复。换名称init才建立另一条初始研究，旧campaign不迁移。新的L1–L4、40局基线只用于新建campaign；既有未记录 baseline_games_per_level 的旧campaign保留原20局/档及冻结参考源码，不改旧manifest或结果。

Executor默认embedded，由单个run进程托管；轮询不经过LLM，原始日志独立保存。external是在init前将site.executor_mode改为external，再额外启动 `scripts/rsi.py executor run chess-autonomous-v2`（用同一控制venv），Controller照常run；二者都0GPU。没有启动external Executor时，Controller会等待报告。两种模式都不会重置基线或模型，GPU任务默认300秒查询、稳定时退避至600秒；纯CPU任务可更频繁查询（默认120秒）。已有campaign沿用冻结配置。

每个研究任务由Executor提交/监控/收集；评测按档位分job。不开独立 vLLM、不用旧 serve_vllm/eval_harness/arena 启动脚本。重启不等于自动恢复任意trainer：训练失败后由Meta提案明确旧checkpoint恢复路径。

网页：设置 LLMGO_RSI_ROOT=/user/lyh4170/rsi_campaigns，用控制venv运行 scripts/run_web.sh，平台转发7860；访问/rsi。

版本设计依据：[vLLM 0.17依赖声明](https://github.com/vllm-project/vllm/blob/v0.17.0/requirements/common.txt)、[其Qwen3.5配置实现](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/transformers_utils/configs/qwen3_5.py)、[Transformers 5.2 Qwen3.5](https://github.com/huggingface/transformers/tree/v5.2.0/src/transformers/models/qwen3_5)。这些源码说明接口设计，实际远端验证以回执为准。


默认多卡（2026-10-23）：每项任务最多8 GPU，site默认8；评测为每卡独立Qwen副本并行棋局，`request_evaluation.py --level L2 --gpus N`可选1–8卡。默认训练已改为verl PPO+LC0，8卡任务使用7卡actor/rollout/reference＋1卡LC0，详见docs/rsi-verl-ppo.md。逐局回执核验后恢复，完整批次才合并成正式成绩。轮询仍由Executor处理，网页只读进度摘要和按需日志。已有campaign需显式upgrade，不重写历史快照。
