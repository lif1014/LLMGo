# 远端首次部署

本机负责实现与打包；运行环境基于用户指定镜像：

`share-dev/ngc-docker/ptdbench-verl-vllm017:dev-957fee7-cb-bf632efb`

不要把旧源代码 tar 和新自主研究 pipeline 混合解压。已有仓库先备份或移到自己目录的另一个位置，再把新包解压到站点配置的 repo 路径。原始模型路径不变。

## 1. 上传与控制环境

将新的 LLMGo-rsi-autonomous-20261022.tar.gz 上传到 /user/lyh4170：

```bash
tar -xzf /user/lyh4170/LLMGo-rsi-autonomous-20261022.tar.gz -C /user/lyh4170
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

Codex 配置此前已经从来源目录复制到 /user/lyh4170/.codex-rsi/config.toml，保留它；不要覆盖来源目录或记录凭证。曾核对的配置默认 custom/deepseek-v4-pro，site.meta_model=null 会沿用配置，不会自动切为另一强模型。

若 own tools/bin/codex 尚不存在，并且镜像有 Node/npm，可安装此前验证的 CLI 版本：

```bash
export npm_config_cache=/user/lyh4170/llmgo-rsi-runtime-20261022/cache/npm
npm install --global --prefix /user/lyh4170/tools/llmgo-rsi-20261022 @openai/codex@0.163.0
/user/lyh4170/tools/llmgo-rsi-20261022/bin/codex --version
/user/lyh4170/tools/llmgo-rsi-20261022/bin/codex app-server --help
CODEX_HOME=/user/lyh4170/.codex-rsi /user/lyh4170/tools/llmgo-rsi-20261022/bin/codex login status
```

上例为避免干扰其他任务使用新的独立CLI前缀；采用它时，将自己的site配置中codex_bin设为 `/user/lyh4170/tools/llmgo-rsi-20261022/bin/codex`，且在init前完成。已有默认 `/user/lyh4170/tools/bin/codex` 可用时直接只读复用，无需重新安装或覆盖。

CLI login status 不验证模型推理，自定义 provider 可能需要自己的环境变量或服务。可自行用一次只回复READY的 ephemeral 请求检查，避免记录认证信息。没有实际请求通过前不要宣称远端强模型已可用。

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
3. HF训练overlay单独安装 Transformers 5.2.0；原镜像4.57.6没有HF Qwen3.5模型类。
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

本机已下载BT4位于 `/home/lif/.cache/llmgo/reward_models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`，不包含在源码tar。上传后核对SHA256：`e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618`。不使用该配方时无需LC0/BT4。

## 4. 启动与恢复

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

首次是160局player基线，之后Meta-Agent开始研究。每个研究任务由Executor提交/监控/收集；评测按档位分job。不开独立 vLLM、不用旧 serve_vllm/eval_harness/arena 启动脚本。恢复只重跑run，不再init。旧campaign不迁移。

网页：设置 LLMGO_RSI_ROOT=/user/lyh4170/rsi_campaigns，用控制venv运行 scripts/run_web.sh，平台转发7860；访问/rsi。

版本设计依据：[vLLM 0.17依赖声明](https://github.com/vllm-project/vllm/blob/v0.17.0/requirements/common.txt)、[其Qwen3.5配置实现](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/transformers_utils/configs/qwen3_5.py)、[Transformers 5.2 Qwen3.5](https://github.com/huggingface/transformers/tree/v5.2.0/src/transformers/models/qwen3_5)。这些源码说明接口设计，实际远端验证以回执为准。
