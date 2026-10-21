---
name: rsi-training
description: 使用现有HF PPO与LC0起始配方，或设计其他训练方法时核对概率、价值视角和训练证据。
---

你可以自主选择 PPO、其他 RL、SFT、课程或混合方案，也可以只改 Harness/skills。无需沿用此配方；以下说明仅描述仓库保留的初始实现。

scripts/train_ppo.py 从 ppo.json、curriculum.jsonl、skills.json 读取配方，通过 research job 执行。它调用 llmgo.training.ppo：每批当前策略真实采样、thinking 开启、Lc0 BT4 离线评分、LoRA 多 epoch 逐 token clipped PPO，然后完整 HF 导出。

初始非终局 A=V_after−V_before，V=P(win)−P(lose)，始终从原行棋方看；gamma=1/lambda=0。after 是学生一步后，没有对手应手和整盘 GAE。终局用胜1/和0/负−1减 V_before；非法/截断是协议惩罚。该 teacher value 不是经过校准的学生 Vπ，不能凭 reward 或 loss 宣称棋力提升。

现有配方保存实际采样 behavior logprobs、固定 reference、advantages、clip/KL 和真实 optimizer updates；恢复完整 checkpoint，未完成批重新采样。修改 reward/advantage 时检查是否重复计入价值差；修改采样时检查 old policy 概率是否匹配实际分布。不要将随机后备或教师答案说成学生生成。

初始配方的数据校验、超参范围是该实现的接口，不是你研究的权限边界；如需不同算法、带标签数据或其他设置，可修改研究副本的实现或新增 trainer。使用 docs/rsi-operations.md 的任意 research 命令提交，不必把新方法伪装成 hf_ppo。

训练产物和依赖须可重现。正式评测由独立裁判启动学生，不加载训练用教师；最终比较固定多难度的实战结果，保留每档退步和失败证据。
