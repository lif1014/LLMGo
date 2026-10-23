# verl PPO + LC0 与多 GPU pipeline

默认路线是课程局面 → 当前Qwen策略采样一手 → LC0逐步打分 → verl PPO更新 → 完整HF导出 → Fairy-Stockfish实战。禁止SFT，使用SFT最终得0分。LC0不是正式玩家或正式对手，不向学生prompt提供最佳着法。

## 资源与环境

每项任务最多8 GPU，默认8。评测每卡启动独立Qwen/vLLM副本，同时处理棋局；10局初始档仍是白黑各5局，20局候选档仍是白黑各10局。并行不增加比赛数，不改种子或难度。

训练用verl 0.8.0的RayPPOTrainer、FSDP2和vLLM：8卡任务中7卡用于actor/rollout/reference，1卡固定分配给LC0。支持2–8卡，actor数=总卡数−1；单卡可以运行，但必须显式选择LC0 CPU后端，速度会慢。预留教师GPU避免LC0与rollout争用显存。

指定镜像中原来没有安装verl，独立环境为`/user/lyh4170/venvs/llmgo-verl-ppo-tf54-20261024`。`scripts/train_ppo.py`自动调用`bootstrap_verl.py`建立/检查此环境，再切换解释器。依赖在`requirements-rsi-verl.txt`，固定verl0.8.0、Transformers5.4.0、mistral-common1.10.0，保留镜像Torch2.10/CUDA/vLLM0.17/tokenizers。由于vLLM0.17的包元数据声明Transformers<5，而Qwen3.5训练需要5.x，这个组合已在任务1350755完成8GPU、2步/每步2epoch的实际PPO与完整HF导出；这不代表完整棋力评测已完成。评测继续使用原独立inference环境。安装只写本研究专属venv/cache，不改系统、原控制环境或其他任务环境。

安装使用中科大PyPI镜像，直连不通时使用集群已有代理。安装失败不会创建成功marker，可修复后重试。不要安装`verl[vllm]` extras去降级镜像GPU栈。

## 训练提案

在研究workspace准备`curriculum.jsonl`、`skills.json`、`ppo.json`。课程每行保存initial_fen、完整moves、来源及split=train，不含教师答案；可以从自己的events提取局面。默认训练输入仍是现有合法着法选择题prompt，thinking开启。

```json
{
  "training": {"backend":"verl_ppo","steps":16,"ppo_epochs":4,"learning_rate":0.00001},
  "reward": {"mode":"lc0_bootstrap","nodes":10000,"backend":"cuda-auto"}
}
```

默认8卡时batch_size=56、minibatch_size=14、每卡microbatch=1；根据actor卡数自动调整默认batch。若显式设置，batch必须整除minibatch，minibatch必须是actor卡数的倍数。满参数FSDP2训练，默认不使用LoRA。

```json
{
  "hypothesis":"根据实战错误提炼skills并用LC0逐步反馈改善选步",
  "jobs":[{
    "label":"verl PPO + LC0",
    "argv":["{python}","scripts/train_ppo.py"],
    "resources":{"gpu_count":8,"cpu":64,"memory_gib":512}
  }],
  "evaluation":{"levels":[2],"tracks":["player"]},
  "evaluation_gpus":8
}
```

Executor提交、轮询和收集，Meta-Agent无需周期性查询。默认site也是8GPU/64CPU/512Gi，可按需降低CPU、内存和GPU请求。所有任务仍为rsi / beijing / model-autosft / NORMAL；devspace保持0GPU。

## 逐步反馈与PPO

V=P(win)−P(lose)，保持同一学生行棋方视角。学生走一步后棋盘轮到对方，但LC0值必须转回原行棋方。

- 非终局：A=V(s_after)−V(s_before)，环境奖励0，gamma1/lambda0。
- 真实终局：A=胜负值−V(s_before)，胜1、和0、负−1，bootstrap为0，不重复计算价值差。
- 第200半步按协议截断为和棋；非法、缺失完整thinking结束标记、生成长度截断视为协议失败，目标−1，不生成随机替代步。

一手棋为一个训练segment，不是逐token棋步奖励，也没有对手应手或整盘GAE。自定义`lc0_td0`直接把该A广播到本次response的有效tokens，不做GRPO分组/归一化。不训练critic网络；LC0充当外部固定价值估计器，返回的token returns不会用于拟合价值头。

verl执行实际clipped PPO、多epoch更新、reference KL和分布式同步。训练采样temperature1/top_p1/top_k−1，不裁掉策略支持集；使用vLLM保存的实际behavior logprobs作为old logprobs（bypass PPO-clip模式）。KL独立施加，不再从教师TD误差中重复扣除或重算bootstrap。

LC0是强教师的价值近似，不是当前学生策略的校准Vπ。不能仅凭reward/loss变好宣称棋力提升，必须看多个Fairy难度的实战。

## 证据与恢复

`output/training/`保留：

- `identity.json`：输入模型、课程、skills、reward、训练配置、GPU数、verl版本。
- `resolved-config.json`、`teacher.json`、`model-preflight.json`：实际verl配置、LC0身份、真实Qwen processor预检；identity同时固定代码和运行时版本。
- `lc0/sample-*.json`：逐手原始输出、response token IDs、前后FEN、原生WDL、V、target、A、失败/终局原因。
- `batches/step-*.pt`：更新前实际response、old/rollout/ref logprobs、response mask及advantage张量；`batch-latest.json`索引。
- `rollouts/*.jsonl`：verl生成日志；`metrics.jsonl`、`latest.json`：每步PPO指标。
- `checkpoints/global_step_N/`：全rank模型、optimizer、extra/RNG和dataloader状态。
- `model/`、`training-result.json`：经过合并、完整HF身份核验的导出，成功后写`output/submission.json`。

恢复只接受身份相同、并行数相同且所有rank文件齐全的verl checkpoint。旧HF LoRA checkpoint不能直接恢复为verl。新的独立research提案可以在argv前用`env LLMGO_RESUME_FROM=/user/lyh4170/.../output {python} scripts/train_ppo.py`指定原输出；常规失败重试由Worker传递该变量。变更reward/课程/卡数应作为新实验开始并记录区别。

评测逐局写带身份和日誌SHA的`game_results`。每GPU顺序处理其分配棋局，各GPU并行；单局/服务故障最多尝试3次，旧未完成日志移到`incomplete_games`，异常写`attempts`。同任务或显式失败恢复只复用身份一致且原始日志未变的完成局。全部局数、种子、白黑配对齐全才发布metrics；不把部分结果当成功。没有该回执格式的旧版本结果不能当缓存。

网页显示每GPU状态、当前局、重试次数、完成数和PPO步骤；日志只在点击时加载，目录列表有限，不周期性读全部训练/思考原文。平台轮询仍留在Executor：GPU任务300秒起退避至600秒，纯CPU任务保留较短间隔。

## 已有 campaign 升级

旧源码/协议和初始参考Harness不可原地改写。自己的旧GPU任务确认停止、Controller和独立Executor退出后，执行以下脚本（自动使用已安装控制环境）：

```bash
bash scripts/start_rsi.sh --upgrade --web
```

升级备份旧state，增加新版本快照，更新默认资源和研究副本来源，保留原始基线、参考Harness和所有任务证据。失败基线登记新的job ID与intent。若仍有活动任务或锁，升级拒绝执行。新campaign无需upgrade，直接start。


## 已定位的兼容故障（2026-10-24）

首个8GPU验证任务1348078在vLLM配置初始化时失败：vLLM 0.17的Qwen3.5回移植配置传入list，而Transformers 5.2进行set并集。`llmgo.training.vllm_compat`只在当前训练进程内、固定版本组合、vLLM Qwen3.5配置类上将忽略键转成set，通过verl的external_lib覆盖各模型/采样服务进程；不写site-packages、不改学生config.json、不改变RoPE数值。启动还会在创建Ray worker前预检真实模型配置。

旧任务输出保留。后续GPU验证以各自readiness/result.json为准，不能把import和配置检查当成完整PPO通过。


第二个8GPU验证任务1349962已通过模型和vLLM初始化，随后在7条样本/4个AgentLoop worker的分块处失败。配置现在根据batch与可用worker上限的最大公约数选择worker数，保证均分；已加入1–8GPU、三种批量倍率的回归检查。任务1350249随后在分布式actor前向遇到CUDA错误；后续定位与最终通过记录见下一节。


## 8GPU验收通过（2026-10-24）

任务1350755成功：verl0.8.0 / Transformers5.4.0 / Torch2.10.0+cu129 / vLLM0.17.0 / tokenizers0.22.2 / Ray2.54.0，8卡中7卡actor、1卡LC0。实际执行2个训练步、每步2个PPO epoch，正常异步CUDA，第二步使用更新策略重新采样。两步actor grad_norm分别为21.8264、21.1634；已保存old/ref logprobs、教师标注、优化器与完整分片checkpoint，并成功合并完整HF模型。验收用64-token生成上限和LC0 nodes1，只验证运行链路；默认研究配方的2048 tokens/nodes10000不变，不把验收模型作为棋力提升结论或正式champion。

此前1350249通过采样与LC0评分、在分布式actor前向出现CUDA非法访存；独立单卡回放1350451可完成同一笔资料的策略前向/反向，故不能把报错行直接认定为LC0奖励问题。依据[上游同堆叠问题](https://github.com/verl-project/verl/issues/6549)另建Transformers5.4环境；旧5.2环境保留。1350597暴露旧mistral-common1.9.1与5.4 processor的导入冲突，已在专属环境固定1.10.0。1350733实际验证native/vLLM processor和thinking模板；rollout明确language_model_only，模型结构和原权重路径保持不变。

最终checkpoint：`/user/lyh4170/rsi_readiness/verl8-gpu-smoke-v5/training/checkpoints/global_step_2`；HF导出在同目录的`training/model`，主回执`result.json`、指标`training/metrics.jsonl`。正式多难度评测仍由独立裁判执行，未拿此短验收冒充实战改善。
