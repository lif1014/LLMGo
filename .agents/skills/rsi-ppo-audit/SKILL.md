---
name: rsi-ppo-audit
description: 根据真实Worker身份、采样、LC0反馈、更新前张量与实战对照诊断国际象棋PPO，发现非法输出被正向奖励、更新未生效或策略退化；适用于训练前检查、训练异常和棋力下降调查。
---

先确认campaign、研究暂停状态和实际job。原训练实现不是可靠的不可改基线：用户明确允许Meta修改训练Python源码、RL算法、reward/advantage、采样、优化器及训练配置；继续限于RL与skills，禁止SFT，正式裁判和已冻结产物保持原样。修改在自己的可编辑workspace进行，代码/目标改变后登记新实验。

## 按证据链观察

先读Executor短报告、manifest、training/identity.json、resolved-config.json和latest.json，核对实际模型、训练代码、recipe、卡数与环境。proposal或总结只表示意图，不证明实际运行。正常监控交给Executor，GPU每5–10分钟查询；只在定位异常后按step/sample_path读取相关完整样本、日志与batch，避免把全部原始输出放进Meta上下文。

1. **采样与协议。** 读quality-latest.json/quality.jsonl的合法率、失败类型、长度上限、tokens及实测generation时间。区分invalid_final_answer、incomplete_thinking、length与服务/基础设施错误；未知故障不能直接视为坏棋给奖励。看完整response、thinking结束标记和候选列表，不能仅凭日志截断认定模型格式错误。
2. **教师反馈。** 当前模式在lc0/sample和episodes/EPISODE/trajectory中核对原始棋谱、student_color、value、next_value、reward、delta与bootstrap_mask；V=P(win)−P(lose)固定学生颜色，对手回复后才到下一学生状态，终局bootstrap0。不能将原生E=P(win)+0.5P(draw)当作V。教师V不等于学生Vπ，不以LC0均值改善代替真实棋力。历史单步任务另查V_before/V_after与原始TD，不混用当前字段。
3. **真正进入优化的优势。** 当前 full_game_ppo_lc0 先读 episodes/EPISODE/trajectory.json，再核对 batches/step-*.pt。普通中间步r=0，终局仅一次±1/0；V固定学生颜色，delta含对手应手后的下一学生状态，终局bootstrap0；每局倒序GAE默认gamma1/lambda0.95。核对原始A、重试权重、分布式padding零mask及最终token优势，不能跨局、重复终局或在hook覆盖成单步TD。不把负均值、非法A=0或正A单独当bug；新目标不做中心化。旧guarded/balanced及其非法正优势缺陷仅解释历史任务，当前已移除，不再运行旧check_ppo_protocol.py。
4. **实际更新。** 分开看actor/lr、grad_norm、ppo_kl/clip比例、reference的actor/kl_loss、checkpoint和导出。非零梯度不证明优化器使用了非零lr，也不证明BF16参数真的变化；需要时比较可复现的权重样本或完整张量。不要以lr×steps代替实际参数位移，不把PPO相邻策略KL与reference KL混称。
5. **actor与rollout一致性。** 看sync-check.jsonl及错误。bypass模式下old_log_probs来自rollout，两者相等不证明当前actor确实同步；需要actor对相同response重算logprob。比较同一mask范围，区分服务错误、诊断器形状错误和真实分布偏移。
6. **产物与实战。** 平台Succeeded、Worker成功、HF完整导出、推理栈可加载、棋力提升分别验证。核对export-preflight、training-result、模型SHA与评测manifest；不同权重、Harness、skills或协议同时变化不能归因于单个因素。用同档、同白黑/种子/预算的实战判断，区分胜率和含和棋的得分率，20局不能给出稳定Elo。

## 发现问题后

保留原版本与反例，先用CPU回放/回归验证优化目标和诊断，再做必要的短GPU验收及实战。新代码默认值不会覆盖旧recipe里的显式模式；live发布也不会更新旧workspace或冻结Worker。检查新manifest与训练identity，证明采用了修复。研究暂停时写诊断和交接，不自行解除暂停或重复提交维护任务。

实际发现路径、21个非法正优势的张量证据、已有修复及仍未证明的结论，见[2026-10-26故障案例](references/20261026-protocol-advantage.md)。选择历史较佳起点时读[rsi-startpoint](../rsi-startpoint/SKILL.md)，不要把旧训练实现随旧Harness一起恢复。


2026-10-26后续：r11/j000028采用原始权重及guard修复后仍在第17批退化到27/56合法，并被闸门拦截。旧非法正优势未复发；重点检查reward尺度、loss归约、累计reference KL和停止时点。整段负优势及负均值本身不是bug的充分证明，系数总量不是参数梯度占比。新诊断修复signed同步ratio漏报，新增health/固定验证/已验证checkpoint选择；方法和取证入口见docs/ppo-stability-20261026.md。CPU回放通过与GPU训练、模型导出、Fairy棋力提升分别验收。主研究仍按research-control.json暂停，维护任务完成不自动恢复Meta。

2026-10-27切换：完整局采样→终局奖励→LC0 value→move-level GAE→verl PPO，细节见live/docs/rsi-verl-ppo.md。旧ppo.py/rewards.py/advantages.py已退出当前源码，不能从历史Harness恢复旧训练。whole_batch_update=true时每完整批次每epoch一次optimizer更新；检查实际设置。完整局验证也保存reference KL与同步；协议forfeit和基础设施aborted分开，后者不赋胜负。不能将仅CPU通过或GPU任务提交当作棋力改善。

完整局基线实际反例：同一局面c6a5为合法步，但模型在</think>后继续解释并列举多步，r2严格解析器拒绝，两次正常stop均未达上限，最终协议负局。这发生在optimizer更新前。分析invalid_final_answer先区分“棋步非法”和“最终回答附解释/多候选”，不要误诊为PPO训练后退化；当前对照不可中途放宽解析器，研究解析鲁棒性时记录新的Harness身份。详见live/docs/full-game-ppo-validation-20261027.md。


初始门槛误拦案例：1375521在任何PPO更新之前，完整白黑验证为35/39合法（89.7436%），KL0.0004804、同步差0.013159正常，但绝对90%门槛提前退出。它不是训练退化，也没有新模型。当前默认绝对下限50%+更新后较原始基线最多降10个百分点，正式裁判不改。先看stage、step、update_completed和checkpoint，不能只看平台FAILED。复用更新前验证仅限无更新、模型/Harness/采样/环境一致、完整逐请求证据校验，记录baseline-import.json及原始来源，绝不将验证轨迹用于梯度或重抽好看的基线；见live/docs/full-game-ppo-validation-20261027.md。所有阈值是内部防崩溃配置，不是棋力提升证据。


2026-10-27用户更改训练环境：无有效答案在重试/预算耗尽后随机选全部合法步之一，归属学生该手继续，参与终局reward/LC0/GAE/actor loss。保留原始token/behavior概率与失败类型，不伪造随机UCI输出，不屏蔽整手。记录action_source/random_rate；随机后备后获胜也可能使原失败回答获得正优势，是该终局目标的可解释结果，不等同旧中心化优势bug。历史1376133仍按旧协议负局采样，live更新不热改在途Worker。检查identity.environment，不能复用旧forfeit验证或混接optimizer状态。
