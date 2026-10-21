---
name: rsi-rl-review
description: 准备或复盘国际象棋PPO候选时检查固定Lc0价值、on-policy概率、无SFT与正式评测隔离。
---

学生能力改进仅允许RL与条件skills。禁止教师答案/思考的交叉熵、SFT、行为克隆或下载已微调学生。

当前唯一训练后端为`hf_ppo`，固定Lc0 BT4-it332作为critic，不训练额外value head。先读工作目录`docs/rsi-operations.md`第5节的实际配置与边界。旧GRPO/verl配置会被拒绝；不要把旧检查点接到PPO。

- 课程只有可重放局面、完整可用历史和来源，无最佳步/教师解释/value；split必须train。优先从失败对局提取，留出集隔离。
- 每批当前策略真实生成，prompt复用正式全部合法UCI与条件卡片，thinking开启。保留真实生成token、behavior/reference logprobs、mask、策略版本、skills/critic身份。教师value不进prompt。
- `V=P(win)−P(lose)=2E−1`，前后始终从原学生执棋方看。γ=1、λ=0；非终局`A=V_after−V_before`。不再把ΔV加一次reward。外部critic不是已校准的学生V^π，改善须看实战。
- 当前训练是课程单步，after是学生刚走完后的局面，没有对手应手/整盘GAE。规则终局r=胜1/和0/负−1，continuation=0；200半步协议和棋同样为0。非法/截断是−1协议惩罚，不是真实输棋；引擎故障不补0。
- 决策A广播给真实response token，逐tokenclip，再对每回答取均值。old分母/reference/A全epochs固定，不是联合序列ratio、token critic或组内归一化。采样temperature1/top_p1/top_k0，评分须匹配实际分布。
- GPU分时：学生采样→移CPU→Lc0评分并退出→学生更新。全部A=0跳过；整轮0更新不导出。检查loss、clip fraction、KL、非法/截断率与长度，不能用loss下降宣布棋力上升。
- 轨迹原文在rollouts，训练汇总在training.jsonl/latest.json；按需读取，不将大量token数组灌入上下文。恢复只认完整批次checkpoint及文件哈希；中断批重采，不重放旧策略buffer。
- 正式评测独立重载学生，对Fairy下20局，禁用Lc0/标签。候选扶正与对手晋级分别判定，不改分/删失败局。CPU测试不等于Qwen GPU通过，真实Elo提升需独立比赛。
