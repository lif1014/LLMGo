# 完整对局 PPO、终局奖励与 LC0 value

当前训练模式为 `full_game_ppo_lc0`：从标准初始局面下完 Qwen-vs-Fairy 对局，再做 verl PPO 更新。禁止 SFT、行为克隆和教师答案蒸馏。LC0 只离线提供 value，不给学生最佳着法，也不参与正式选步。

## 一批训练

1. 冻结当前 Qwen、Harness/skills 和对手配置。
2. 白黑均衡地并行下完整局。学生决策→学生走子→对手回复→下一次学生决策，构成一条转移；强制唯一合法步由环境执行，不创建虚假的模型样本。
3. 中间环境奖励为 0，最后一个学生转移按学生视角给胜 +1、负 −1、和 0。双方总计 200 半回合上限按和棋；对手刚刚将杀也写入最后一个学生转移。
4. LC0 用 native WDL 标注每个学生决策状态，`V=P(win)−P(lose)`，执黑仍固定学生视角。
5. 按每局从后向前计算 GAE，再展开为逐手/逐请求的 token 训练数据。
6. 使用 verl 原生 actor/reference、clipped PPO、FSDP2、优化器、权重同步及 checkpoint。完整局采样期间不更新权重，下一批才使用新策略。

```text
delta_t = r_t + gamma * (1-done_t) * V_LC0(s_next) - V_LC0(s_t)
A_t     = delta_t + gamma * lambda * (1-done_t) * A_next
```

默认 `gamma=1、gae_lambda=0.95`。终局 bootstrap 为 0，不叠加价值差奖励，不中心化、不覆盖成旧协议优势，也不新增可学习 critic。LC0 并不是当前学生的校准 Vπ，bootstrap 存在教师/学生策略不匹配的偏差；完整局 `lambda=1` 可作后续对照，此时 A=实际终局得分−当前 V。和棋的环境奖励为 0，也可能产生非零优势。

## 入口和配方

入口仍是 `scripts/train_ppo.py`，在注册 Worker 中读取 `LLMGO_MODEL / LLMGO_CANDIDATE / LLMGO_SITE / LLMGO_OUTPUT`，自动切换既有独立 verl 环境。候选目录准备 `skills.json` 与 `ppo.json`，不再要求 curriculum.jsonl。

小规模配方例子（训练步是采完一批完整游戏后的一次更新，不是棋步）：

```json
{
  "training": {
    "backend": "verl_ppo",
    "mode": "full_game_ppo_lc0",
    "games_per_batch": 2,
    "concurrent_games": 2,
    "opponent_level": 1,
    "steps": 2,
    "whole_batch_update": true,
    "ppo_epochs": 1,
    "learning_rate": 0.000001,
    "gamma": 1.0,
    "gae_lambda": 0.95,
    "validation_games": 2,
    "validation_every": 2,
    "checkpoint_every": 1,
    "cudagraph_capture_sizes": [1, 2, 4, 8]
  },
  "teacher": {"mode": "lc0_value", "nodes": 10000, "backend": "cuda-auto"}
}
```

默认 8 GPU：7 张 actor/rollout/reference、1 张 LC0；每任务最多 8 张，可选择 2–8 张；单卡必须显式选 LC0 CPU 后端。保留用户指定 ptdbench 镜像、已有路径和独立环境，不重装系统或其他任务的 GPU 栈。CUDA Graph 默认开启，保留 wake 前缓存释放与导出兼容修复。

默认 `whole_batch_update=true`：本批全部真实请求做一个全局 minibatch，native verl 以 microbatch=1 梯度累积；`ppo_epochs=1` 时每批一次 optimizer step，避免两个完整局展开后悄悄变成几十次更新。可显式关闭以研究多个 minibatch；`minibatch_size` 默认按 actor 数匹配，亦作为补齐粒度。

thinking 开启。训练每请求最多 16384 tokens，每手共享 32768 tokens、最多 2 次请求，请求/整手超时为 540/1800 秒。每次请求保留真实上下文、token IDs、behavior logprobs 和原文；失败后成功的重试也计入同一手的训练样本。一手各请求按生成 token 数分配权重，再按手等权，不因重试更多而多算棋步。padding 行 response mask/reward/advantage 均为 0，不重复最后一手凑样本。

训练和内部完整局验证均使用 temperature=1、top_p=1、top_k=-1，确保 actor/ref 重算与 behavior logprobs 一致。内部验证固定独立种子、白黑均衡，不用来更新模型；它不请求LC0标注，轨迹中的value=0明确标记为占位，不当作教师预测；正式受保护评测仍使用其冻结的 0.9/0.95/20 和原预算。二者的得分不可直接混为同一统计。

## 终止、诊断与恢复

- 用尽重试/固定模型预算仍无合法答案时，从当前全部合法走子中按episode种子均匀随机选一步，作为学生这一手继续整局；不再因格式失败直接判负，不追加中间奖励或处罚。正式裁判原有随机后备规则不变。棋盘没有合法步时仍按将杀/逼和终局规则处理，不尝试随机走子。
- 引擎/网络/服务故障或集群停止属于 aborted，不编造输棋或和棋奖励；未完成局不进入梯度。负局和协议失败样本不按结果筛除。
- `quality_gate_min_valid` 与内部 `validation_min_valid` 默认 0.5，拒绝全部失败的批次；更新后的固定验证还要求合法率不低于原始验证基线减 `validation_max_drop=0.1`（10个百分点）。不要求原模型先达到90%才能学习。训练批次与固定验证的局面不同，不直接比较两者合法率。reference KL 警戒/停止默认 0.05/0.1，actor/rollout 平均绝对 logprob 差停止默认 0.1；异常值和缺失证据不会默认通过。KL 是独立正则，不是环境奖励。受保护正式评测规则保持不变。
- 不把所有负优势、非法输出 A=0 或 GAE 下的个别正优势一概当成程序错误；检查实际 r/V/delta/A/λ/边界。旧非法输出被全批中心化推正的故障已随旧目标移除。
- 导出只接受有通过更新后完整局/协议/KL/同步检查回执的完整全 rank checkpoint。安全停止只能回退至此前已验证且完整的 checkpoint；普通 OOM/运行错误不得转换为成功导出。
- 恢复要求模型、代码、Harness/skills、配方、教师及卡数身份相同。`games.json` 保存已更新 episode 清单、下一批种子和原始验证基线；恢复不重置基线，不将旧策略前半局与新策略后半局混用。中断的未完成局保留 aborted 证据，重采完整局。
- 从 r000002 框架起跑指原始 Qwen 权重+r2 Harness/skills+当前修复后的训练代码，不能将 r2 的旧训练实现一并恢复。

## 保存的证据

| 路径（相对 output/training） | 内容 |
|---|---|
| identity.json / resolved-config.json / teacher.json | 实际模型、训练源码、环境、配方与教师身份 |
| episodes/EPISODE/events.jsonl | 逐手工具观测、prompt、请求、原始思考/答案、token IDs/logprobs、对手着法和终止原因 |
| episodes/EPISODE/trajectory.json | 完整棋谱状态、r/V/delta/GAE、重试请求及学生视角 |
| agent_requests/*.json | 请求完成即保存的原始token/思考/logprobs；同批其他请求或后续奖励接口失败不会抹掉已完成输出 |
| requests/*.pt / batches/step-*.pt | 实际采样张量与最终进入优化器的张量；仅加载可信自有 pickle |
| lc0/*.json | 每个学生决策的 native WDL、value、搜索节点及延迟 |
| game-progress.json / health.json / latest.json | 有界进度与质量/KL/同步摘要 |
| validation/step-*.json / validation-baseline.json | 固定完整局验证与原始基线 |
| sync-check.jsonl / update-diagnostics/*.json | actor/rollout 对照、reference KL、思考/答案的系数质量 |
| checkpoint-selection.json / training-result.json | 实际选择的已验证存档与完整 HF 模型 |

网页周期性读小摘要，逐局原文按需分页。GPU 平台轮询仍由独立 Executor 每 300–600 秒处理，不让 Meta 或页面直接高频查询 cctl。

## 已移除的旧入口

旧 `training/ppo.py` 自写 HF/LoRA PPO、`rewards.py` 单手 TD、`advantages.py` guarded/balanced 中心化，以及旧单步检查/验证采样脚本已退出当前路径；有效 LC0/棋规/梯度映射/恢复覆盖迁移至完整局测试。旧配方字段显式报错，不能静默切换目标续写同一实验。历史只读快照、旧模型和实测成绩不删除或覆写。

## 验收边界

CPU 数学/棋规测试、远端实际 verl 接口检查、真实 GPU 更新导出与实战棋力是四项独立证据。此次维护先少量更新，再用同一 r2 Harness、种子、白黑和正式预算对比训练前后 L2/L3；用户授权的每档 10 局属于快速探针，不作为 20 局候选扶正或完整 L1–L8 排名。


## 随机后备与训练归属（2026-10-27）

按用户要求，随机后备归属学生玩家的这一手，参与同一条完整轨迹的终局reward、LC0 value、GAE与actor loss，整手不屏蔽。`uci`是实际执行步，`model_uci`保留解析出的模型答案（后备时为null），`action_source=random`、`fallback_reason`和fallback事件记录来源；summary与metrics分别统计random_moves/random_rate。失败请求的failure_reason不改成成功，不伪造随机UCI的token或behavior logprob；优化器仍使用这一手实际生成的原始回答及其真实概率。

这相当于环境把无效回答映射成随机合法落子。中间reward仍为0，只有真实终局给±1/0；因此随机后备后的胜局也可能给失败回答正优势。这是当前终局奖励与用户要求的归属方式的结果，不能当作实现自动过滤了非法正优势。应同时看合法率、后备率和实战表现，不能只看胜率。

环境语义写进training identity；原协议负局版本的optimizer状态/初始验证不能直接冒充新版续训基线。旧冻结任务、旧checkpoint和正在运行的评测不热改。当前研究起点与已导出模型见[rsi-research-baseline.md](rsi-research-baseline.md)。
