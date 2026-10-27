---
name: rsi-training
description: 使用verl执行完整棋局PPO，核对终局奖励、LC0学生视角、move-level GAE、真实行为概率及多卡恢复证据。
---

先读 docs/rsi-verl-ppo.md 和当前任务 identity/resolved-config。允许修改自己的训练Python源码、RL配方、Harness、prompt与skills；只能通过RL和skills提升Qwen，禁止SFT/预热/辅助SFT/行为克隆/监督蒸馏，使用SFT最终0分。教师不提供监督答案或正式最佳着法。受保护裁判及历史快照不可改。

默认入口 scripts/train_ppo.py、模式 full_game_ppo_lc0，候选需 ppo.json（training/teacher）和skills.json。真正使用verl0.8、FSDP2、vLLM与原生PPO优化器。原始模型+r000002 Harness/skills可作干净起点，但保留当前训练修复；不要恢复其旧training目录。选择起点按[rsi-startpoint](../rsi-startpoint/SKILL.md)动态比较证据。

每批冻结同一策略，从标准开局下完白黑均衡完整局；一条转移包含学生走子及Fairy回复，直到再次轮到学生。中间r=0、终局胜/负/和为+1/-1/0，终局bootstrap=0；总计200半回合上限为协议和棋。LC0给V=P(win)-P(lose)，固定该局学生颜色。按每局倒序算delta=r+gamma*(1-done)*V_next-V，再A=delta+gamma*lambda*(1-done)*A_next，默认gamma1/lambda0.95。没有可学习critic、没有LC0价值差环境奖励、没有旧guarded/balanced中心化。LC0不等于学生Vπ；lambda1完整局退化为真实回报减状态baseline，可作为明确的新实验。

完整局结束后再展开各手与重试请求。prompt、工具、Fairy及padding不承担策略损失，真实请求保留thinking和EOS；每手等权，重试按该手生成tokens分配权重。默认whole_batch_update=true、epochs1，即每批完整局做一次梯度累积更新；关闭时记录实际minibatch/optimizer步数，不把2批下棋说成只有2次更新。训练/内部验证采样均temperature1/top_p1/top_k-1，与真实behavior logprobs和actor/ref重算匹配；正式评测采样保持冻结协议。

预算内重试成功可继续，全部请求保留；耗尽重试/预算仍无合法答案时，从全部合法步按种子随机选一步，作为学生该手继续；原始请求正常参与终局reward/LC0/GAE/actor loss，不屏蔽整手、不伪造随机UCI的token/logprob。保留model_uci、action_source、fallback_reason及random_rate；失败请求不改记成功。服务/引擎/网络错误为aborted，不能造奖励。不能筛掉负局再训练。完整局采样期间不切权重，不能用旧策略前半局拼新策略后半局。按请求、回合与整局区分合法率。

训练前/退化时读[rsi-ppo-audit](../rsi-ppo-audit/SKILL.md)。默认internal_validation=false，不打更新前/后内部验证局，训练批次仍检查实际reference KL、协议、actor/rollout一致性及有限值。默认KL警戒0.05/停止0.1、同步平均绝对差停止0.1；这是可调初始设置，不是棋力保证。关闭时导出有同实验有限optimizer更新记录的最新完整checkpoint，回执必须标记disabled、passed=null；不得称内部验证通过，棋力另做正式评测。显式internal_validation=true可恢复固定完整局验证及已验证checkpoint选择。恢复保存开关、模型/优化器/RNG/episode批次身份，启用时另存原始验证基线；代码/目标改变另开实验，旧HF LoRA或单步优化器不能接入。

默认最多8卡：7 actor/rollout/reference+1 LC0；concurrent_games默认actor卡数（7），games_per_batch默认两倍（14，白黑均衡）。旧ppo.json显式2局配置需改为14/7并设internal_validation=false，不因live默认更新就自动采用。2–8卡留一张教师卡，单卡明确CPU LC0。保持CUDA Graph、wake缓存释放和HF导出兼容修复。只用beijing/model-autosft/NORMAL、只写/user/lyh4170、devspace0GPU，不改他人环境或停止他人任务。

提交通过proposal.jobs交Executor，GPU每300–600秒轮询。正式单档接口request_evaluation.py --level L2 --gpus 8仍为20局；用户授权的10局快速探针可用scripts/probe_strength.py，明确不是正式候选扶正。最终需多个难度实战，不以LC0均值或训练loss代替棋力。

证据在training/episodes/*/{events.jsonl,trajectory.json,summary.json}、requests、lc0、batches、metrics/health、validation、checkpoint-selection和training-result。网页读小摘要，原文按需；只加载可信自有.pt。暂停不会因为维护验证完成而自动解除。
