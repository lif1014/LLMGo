# 完整对局 PPO 实施与验证记录

状态快照：2026-10-27 香港时间。此记录区分代码检查、GPU运行和棋力结论；最终异步结果以文末产物为准。

## 已实现

- 原始Qwen3.5-4B，从标准开局对战Fairy；本批完整对局采样期间冻结策略，全部结束后再做verl PPO。
- 环境中间reward=0，终局胜/负/和为+1/−1/0；一条转移包含学生落子和对手回复，双方总计200半回合上限按和棋。
- LC0提供固定学生视角V=P(win)−P(lose)，默认gamma1/lambda0.95，按每局倒序GAE；无可学习critic、无价值差环境奖励、无SFT。
- 原生verl actor/reference/FSDP2/vLLM/优化器/分布式checkpoint，默认整批梯度累积，每批每epoch一次optimizer更新；重试按棋步等权，padding零mask。
- 更新前及更新后完整局检查协议质量、reference KL和实际权重同步；仅导出通过验证的完整存档。恢复略过缺少games.json的半成品，保留原始验证基线。
- 旧自写ppo.py、单步rewards.py、guarded/balanced优势及专属脚本测试已移除；保留LC0/WDL、棋规、导出、显存和同步的有效修复。
- README、操作文档、Meta提示及研究skills已更新。主Meta仍暂停，维护任务由独立Executor每300秒查询。

具体配方和文件接口见[rsi-verl-ppo.md](rsi-verl-ppo.md)，删除清单见[removed-files.json](removed-files.json)。

## 已验证的范围

| 项目 | 证据 | 含义 |
|---|---|---|
| 本机全套 | 428通过、7跳过（本轮完整套件；新增旧schema兼容另有定向回归） | CPU/控制/网页检查；本机缺少Torch/verl的项目不冒充GPU通过 |
| 前端DOM及JS语法 | 通过 | 有界刷新、完整局进度、历史兼容与原文分页 |
| 远端实际verl/Torch | 1375607：83通过；1375736日志兜底回归84通过 | 包括真实DataProto、重试/补齐/奖励、原生actor接口与整批元数据、RewardManager构造和恢复闸门 |
| 远端部署 | deployment.json、deployment-final.json | live代码/文档/skills已同步，冻结Worker未热改；不证明旧Controller已重载 |
| Meta通知 | .codex-rsi/AGENTS.md与Memory | 已发布新目标/兼容修复，暂停期间未启动Meta，不声称新会话已读到 |

首次GPU训练1375421在首轮验证因奖励管理器未接受verl传入的reward_router_address而退出，未执行optimizer更新、未导出候选；已补构造兼容并加入真实verl回归。失败记录保留。

## 正在执行的棋力验证

| 对象 | 平台ID | 设置 |
|---|---:|---|
| 训练前探针 | 1375420 | 原始权重+r000002 Harness/skills，L2/L3各10局，白黑各5局 |
| 修正后短训练 | 1376133 | 同一起点，8GPU（7 actor/rollout/ref + 1 LC0），每批2完整局、2批更新、epochs1；另有更新前/后各2局内部验证 |
| 训练后探针 | 待训练通过后提交 | 保持同一评测Harness/skills、种子、白黑和正式预算，只替换导出权重 |

实际训练冻结源码：

`/user/lyh4170/rsi_campaigns/chess-autonomous-v2/maintenance/full-game-20261027-v2/source-v4r1`

SHA：`0dd314029ce5925b09dba3d7bcd8ba2d6cfbca2c4b87899f555ca151308f6e94`。

后续live的存档状态闸门及验证占位标记不热改该Worker；它们不改变reward/GAE数值。训练与内部验证采样1/1/−1；正式L2/L3保持冻结0.9/0.95/20，不能混用两种得分。

2026-10-27 03:31快照：1375521更新前门槛问题已修正；1376133已提交8GPU短训，原模型与r000002 Harness/skills保持一致。102项远端真实verl回归通过，且原35/39验证已完成身份及完整性校验，记录在baseline-reuse-v4r1.json。训练前L2已完成8胜1和1负（得分率85%），L3尚在进行。尚无新训练更新后的胜率结论。

训练通过后，Executor还会以分块读取的safetensors张量数值、基模型存储精度比较语言模型权重，单纯dtype转换不计作有效更新，排除只有配置/文件布局变化却没有有效参数变化的“训练成功”。通过后才登记训练后评测。10局结果仅为快速探针，不自动更新champion，也不能据此认定稳定Elo提升。

## 产物与查看

远端根目录：

`/user/lyh4170/rsi_campaigns/chess-autonomous-v2/maintenance/full-game-20261027-v2`

- `jobs/train-v4r1/output/training/`：逐局events/trajectory、LC0值、最终更新张量、quality/KL/sync、checkpoint与HF导出。
- `jobs/eval-before/output/evaluation/`：训练前L2/L3进度、完整比赛与统计。
- `jobs/eval-after-v4r1/output/evaluation/`：训练后同设置评测，待有效训练产物后创建。
- `weight-change.json`：语言模型张量数值/存储精度变化审计。
- `comparison-v4r1.json`：前后探针最终对照，完成后自动发布，并写Memory研究摘要。
- `deployment-final.json`、`cpu-final.log`：部署和实际verl接口检查回执。

网页的维护验证区域读取小型进度摘要，逐局原始prompt、思考、答案、棋谱和教师值按需查看，不周期性拉取大日志或直接轮询cctl。正式玩家的既有streaming保持原样；verl内部采样原文按请求完成落盘。

## 更新前的协议反例

2026-10-27 02:33:40快照中，黑方初始验证在第7次决策耗尽两次尝试后记协议负局。原文两次均给出合法的c6a5，但在thinking结束后继续列举/解释候选，未满足当前r2解析器要求的纯UCI或单字段JSON。两次输出分别7914和3694 tokens，均正常stop，未达单请求16K上限。此例应归类为回答协议失败，不能误称棋步本身非法，也不能据此判断PPO退化，因为当时尚未更新。

为隔离权重变化，本次前后评测保持同一r2解析器，不在中途放宽格式。后续Meta可研究prompt/skills或解析鲁棒性，但须另记框架身份与新对照，不能把放宽解析带来的合法率提升全归因于RL。

live新增agent_requests逐请求兜底：生成完成即保存tokens、原文和行为概率；同批另一请求或奖励接口失败时仍可追溯。当前冻结Worker不热改，后续任务采用这项日志加固。


## 初始门槛修正与证据复用

训练1375521在更新前完成白黑两局，共39次请求、35次合法回答（89.7436%），两局均协议负；reference KL约0.0004804，同步平均绝对差0.013159。固定90%门槛提前拒绝了原模型，未发生任何PPO更新，不能将其误报成训练退化。

内部门槛改为绝对下限50%与更新后相对原始基线最多下降10个百分点，保留KL、同步、非有限值及全失败停止规则；50%是可配置的防崩溃下限，不是棋力验收标准。正式评测未改。修正后这批基线对应后续最低合法率79.7436%，不是重新抽样来挑更高的基线。

`training/baseline_import.py`只允许在没有任何更新/存档时，核对模型、运行库、skills、Harness、采样预算、Fairy身份、完整棋局和逐请求数量后，重用初始验证。采集函数必须相同；agent仅允许显式审核的日志改动SHA对。新目录保存全部来源SHA和原始路径，旧失败记录不覆盖；这些轨迹不进入RL梯度，训练仍重新on-policy采样。恢复或预置基线均检查实验身份。该功能属于操作员维护，不是自动放宽失败标准或任意复用旧策略样本。

本轮远端CPU任务1376122：102通过，已验证baseline导入。GPU训练1376133与原训练前评测1375420由独立维护Executor每300秒监控；无自动恢复主Meta、无自动扶正。新baseline-import.json保存原始失败任务1375521的证据与模型身份，源记录保持不变。
