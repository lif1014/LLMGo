# 完整对局 PPO + LC0 GAE：实施与验收

更新时间：2026-10-27。当前默认模式已实现为 `full_game_ppo_lc0`；完整配方和接口见 [docs/rsi-verl-ppo.md](docs/rsi-verl-ppo.md)。本文件记录实施状态，不能把“代码完成/CPU通过”误当作“GPU训练通过/棋力提升”。

## 已实现的目标

Qwen3.5-4B 从标准国际象棋开局与 Fairy-Stockfish 下到终局；白黑均衡，同一批完整对局使用同一份冻结策略，结束后统一做 verl PPO 更新。

- RL 转移：学生当前决策→学生走子→Fairy 回复及必要强制步→下一学生决策。
- 中间环境 reward=0；终局从学生视角胜+1、负−1、和0。对手将杀也正确结束最后一个学生转移。
- 双方合计200半回合的协议上限为和棋。基础设施中断为aborted，不冒充和棋。
- LC0 native WDL 转为 V=P(win)−P(lose)，固定该局学生颜色；终局 bootstrap=0。
- 按棋步倒序计算 delta=r+gamma*(1-done)*V_next−V，以及 A=delta+gamma*lambda*(1-done)*A_next。当前默认 gamma=1、lambda=0.95，符合用户后续确认；lambda=1 可作为完整回报对照。
- LC0 只给离线状态值，不提供最佳步、不向正式选步提供 oracle、不训练可学习 critic。LC0 不是当前学生的校准 Vπ，使用bootstrap仍有价值不匹配的偏差。
- 不叠加LC0价值差环境奖励、不使用旧批次中心化/协议奖金。

## verl 接入

保留 verl0.8、Ray、FSDP2、vLLM、原生actor/reference、clipped PPO、优化器、checkpoint和HF导出。ResearchTaskRunner作用域内替换fit编排，环境和move-level GAE由本项目处理，未再实现一个平行PPO优化器。

完整局是采样单位，学生回合是归因单位，请求是张量记录单位。完整局结束后展开可变数量的逐手记录，不再依赖旧历史课程的一行输入对应一行输出。默认whole_batch_update=true、ppo_epochs1，通过原生microbatch1梯度累积，每批完整局只做一次optimizer更新；可按实验显式关闭。

- thinking开启；逐请求保留真实prompt/response token IDs、behavior logprobs、finish_reason和延迟。
- 预算内重试与首请求一起保存，每手等权，手内按生成tokens分配请求权重。
- prompt、工具、Fairy及padding的策略loss mask为0；补齐不复制真实样本贡献。
- 训练/内部完整局验证采用temperature1/top_p1/top_k-1，保证行为概率、actor/ref和同步检查一致；正式评测仍保留冻结参数。
- 使用既有7卡actor/rollout/reference+1卡LC0布局、CUDA Graph、wake缓存释放、Qwen兼容和导出归一化。

## 文件变更

| 范围 | 处理 |
|---|---|
| training/game.py | 新增完整棋局环境、对手应手/强制步、终止与逐手事件 |
| training/trajectories.py | 新增move-level GAE、轨迹视角/策略/边界校验和重试权重 |
| training/lc0.py | LC0值读取与WDL/POV，替代旧混合奖励实现 |
| training/verl_games.py | 新增完整局并行采样、变长batch、原生更新、验证/恢复编排 |
| training/verl_agent.py、verl_reward.py | 可审计请求、零占位奖励与完整局LC0值标注 |
| training/verl_ppo.py、verl_registration.py、scripts/train_ppo.py | 切换新目标、配置、训练身份与存档导出 |
| scripts/probe_strength.py | 用户授权的L2/L3各10局诊断；复用原受保护裁判，不修改正式20局入口 |
| server/rsi.py、静态页面 | 有界完整局进度、胜和负/得分率、KL和原始文件入口 |
| README、操作文档、研究skills、Meta/controller提示 | 更新新配方及旧字段迁移，允许研究修改训练代码，继续禁止SFT |

删除旧自写HF PPO `training/ppo.py`、旧单步TD `rewards.py`、guarded/balanced `advantages.py`、旧单局面validation_sampling.py及专属检查/采样脚本和测试。混合测试中的LC0视角、棋规、网页、恢复和导出覆盖保留并迁移；清单见docs/removed-files.json。历史只读快照、训练结果和模型不删除。

## 稳定性、恢复和可观察性

1. 更新前检查合法率、参考KL、actor/rollout实际重算差；更新后固定完整局再验证协议、KL和同步。
2. 原始r、V、delta、GAE、请求权重和最终token优势分别保存。不能把负均值或非法A=0一概当bug。
3. 耗尽重试/模型预算仍无合法答案时，均匀随机选择全部合法步中的一步，归属学生该手并继续完整局；不追加中间处罚，原始请求参与GAE/loss并记录后备来源；正式裁判规则不变。网络/服务/引擎异常为aborted，不进入梯度。
4. checkpoint需完整全rank分片及games.json；仅有通过更新后完整局验证回执才能导出。普通运行错误不转换为成功。
5. 保存原始验证基线、已更新episode清单和下一批种子；恢复身份不符拒绝，不重新设较差基线，不混合不同策略半局。
6. 网页只定期读小摘要；原始思考/棋谱/张量按需读取。Executor独立负责GPU每300–600秒的查询，不挤占Meta上下文。

## 本次验证安排与当前证据

- 旧A/B/C已取消：任务1374863确认STOPPED，旧维护Executor已退出；主Meta研究保持操作员暂停。
- 本机全套检查：415通过、6跳过；跳过项包含本机缺少Torch/verl的检查，不能当GPU证据。
- 远端0GPU接口检查1375314：78通过，使用实际verl/Torch环境；后续修正的远端检查为83通过，覆盖首轮GPU发现的RewardManager构造参数兼容问题；旧训练1375421未执行更新，新训练1375521，训练前评测1375420保持运行。
- 本次GPU短训使用原始Qwen权重+r000002 Harness/skills+当前训练修复；少量完整局/少量PPO批次，不缩短棋规终局。
- 训练前后使用相同r2框架、白黑、种子和预算，对L2/L3各10局快速对照。记录胜/和/负、得分率、后备/协议失败、自然结束率与模型/源码身份。10局只作探针，不作为20局正式扶正或完整面板分数。
- 是否真正训练/导出通过、KL是否稳定、棋力是否提高，以后续GPU回执及实战结果填写，当前不预先声称提升。

## 保留约束

仅国际象棋，方法限RL/skills，禁止SFT及监督蒸馏。Meta可改研究训练代码、Harness、prompt、工具及skills/触发环境，正式裁判/协议和历史证据受保护。

只写/user/lyh4170，沿用已有远端模型/LC0/venv路径；beijing、model-autosft、NORMAL、每任务最多8GPU、项目名rsi，devspace0GPU，不停止他人任务。使用指定ptdbench镜像和专属venv，不改系统或其他任务环境。无80%自动晋级，固定多档评测继续保留。维护验证完成不自动恢复主Meta Loop。
