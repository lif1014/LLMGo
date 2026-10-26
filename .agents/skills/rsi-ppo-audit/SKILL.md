---
name: rsi-ppo-audit
description: 根据真实Worker身份、采样、LC0反馈、更新前张量与实战对照诊断国际象棋PPO，发现非法输出被正向奖励、更新未生效或策略退化；适用于训练前检查、训练异常和棋力下降调查。
---

先确认campaign、研究暂停状态和实际job。原训练实现不是可靠的不可改基线：用户明确允许Meta修改训练Python源码、RL算法、reward/advantage、采样、优化器及训练配置；继续限于RL与skills，禁止SFT，正式裁判和已冻结产物保持原样。修改在自己的可编辑workspace进行，代码/目标改变后登记新实验。

## 按证据链观察

先读Executor短报告、manifest、training/identity.json、resolved-config.json和latest.json，核对实际模型、训练代码、recipe、卡数与环境。proposal或总结只表示意图，不证明实际运行。正常监控交给Executor，GPU每5–10分钟查询；只在定位异常后按step/sample_path读取相关完整样本、日志与batch，避免把全部原始输出放进Meta上下文。

1. **采样与协议。** 读quality-latest.json/quality.jsonl的合法率、失败类型、长度上限、tokens及实测generation时间。区分invalid_final_answer、incomplete_thinking、length与服务/基础设施错误；未知故障不能直接视为坏棋给奖励。看完整response、thinking结束标记和候选列表，不能仅凭日志截断认定模型格式错误。
2. **教师反馈。** 对同一sample核对前后FEN、原行棋方视角、V_before/V_after、终局target与原始TD。V=P(win)−P(lose)，换行棋方后要转回原视角；终局只计算一次target−V_before。教师V不是学生策略的校准Vπ，不能把教师分数改善当成胜率改善。
3. **真正进入优化的优势。** 读protocol-advantages和batches/step-*.pt；核对response_mask、token_level_rewards、最终advantages/returns及failure_reason。默认guarded模式下，合法TD只减合法均值，协议失败最终优势固定为负，padding为0；检查全部失败样本，而非只看平均reward。旧全批中心化可能把非法输出的原始TD=0变成正优势。只读回放可用scripts/check_ppo_protocol.py，输入仅限可信自有pickle。
4. **实际更新。** 分开看actor/lr、grad_norm、ppo_kl/clip比例、reference的actor/kl_loss、checkpoint和导出。非零梯度不证明优化器使用了非零lr，也不证明BF16参数真的变化；需要时比较可复现的权重样本或完整张量。不要以lr×steps代替实际参数位移，不把PPO相邻策略KL与reference KL混称。
5. **actor与rollout一致性。** 看sync-check.jsonl及错误。bypass模式下old_log_probs来自rollout，两者相等不证明当前actor确实同步；需要actor对相同response重算logprob。比较同一mask范围，区分服务错误、诊断器形状错误和真实分布偏移。
6. **产物与实战。** 平台Succeeded、Worker成功、HF完整导出、推理栈可加载、棋力提升分别验证。核对export-preflight、training-result、模型SHA与评测manifest；不同权重、Harness、skills或协议同时变化不能归因于单个因素。用同档、同白黑/种子/预算的实战判断，区分胜率和含和棋的得分率，20局不能给出稳定Elo。

## 发现问题后

保留原版本与反例，先用CPU回放/回归验证优化目标和诊断，再做必要的短GPU验收及实战。新代码默认值不会覆盖旧recipe里的显式模式；live发布也不会更新旧workspace或冻结Worker。检查新manifest与训练identity，证明采用了修复。研究暂停时写诊断和交接，不自行解除暂停或重复提交维护任务。

实际发现路径、21个非法正优势的张量证据、已有修复及仍未证明的结论，见[2026-10-26故障案例](references/20261026-protocol-advantage.md)。选择历史较佳起点时读[rsi-startpoint](../rsi-startpoint/SKILL.md)，不要把旧训练实现随旧Harness一起恢复。
