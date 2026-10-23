---
name: rsi-training
description: 设计verl PPO与LC0多卡强化学习实验，检查逐步价值视角、采样概率与恢复证据。
---

只能通过RL和改进skills提升Qwen3.5-4B。禁止SFT、SFT预热、辅助SFT损失、行为克隆、监督蒸馏；使用SFT最终得0分。离线教师可用于RL奖励/价值估计和提炼skills，不能把教师答案当学生监督目标。可修改RL算法、reward、课程、Harness与skills触发；正式评测不加载教师。

先读 docs/rsi-verl-ppo.md。默认入口 scripts/train_ppo.py，配置 ppo.json、curriculum.jsonl、skills.json。真正调用verl 0.8.0 RayPPOTrainer、FSDP2、vLLM，不再默认使用自写HF PPO。GPU任务默认8卡：7卡actor/rollout/reference，1卡LC0；每任务最多8卡，可指定2–8卡（留1卡LC0），单卡需显式CPU教师后端。batch/minibatch必须整除actor并行数，默认自动匹配。

非终局A=V_after−V_before，V=P(win)−P(lose)，前后保持同一行棋方视角。终局用胜1/和0/负−1减V_before；非法/生成截断是协议惩罚。gamma1/lambda0，一手一segment，没有对手应手和整盘GAE，也不训练critic。教师V不是学生Vπ；只凭reward/loss不能宣称棋力提升。

自定义lc0_td0估计器使用教师算好的TD误差，不做组归一化；verl使用实际vLLM behavior logprobs作old policy、多epoch clipped PPO，reference KL独立。采样默认temperature1/top_p1/top_k-1；不要改了采样却继续使用错误old logprobs。保留thinking分隔符，截断不做随机后备，教师不输出最佳走法。

样本原文/响应token/LC0前后值/优势在training/lc0，rollouts保存verl行为日志；metrics.jsonl/latest.json和resolved-config.json记录训练。完整分片模型、optimizer、RNG及dataloader checkpoint用于恢复，身份/并行数不同则拒绝。最终合并完整HF模型才写submission.json。旧HF LoRA checkpoint不能作为verl checkpoint。

训练通过proposal.jobs交给Executor，不反复shell轮询。用request_evaluation.py --level L2 --gpus 8做20局探针；最终需要不同难度的实战证据，只刷最低档无效。所有计算beijing/model-autosft/NORMAL，devspace0GPU，只写/user/lyh4170，不停止他人任务，不改其他环境。
