# 当前维护研究的起点与模型身份

核对日期：2026-10-27。此文描述操作员的完整局 PPO 对照实验，不表示主 Meta campaign 已选择新 working/best；主研究仍暂停。r000002 是 Harness/skills 的来源，不是这次短训新建的轮次编号。

## 干净对照与训练输入

| 部分 | 实际来源 |
|---|---|
| 模型权重 | 原始 HF 目录 `/user/liyaohu4170/models/Qwen3.5-4B`，未接入历史有问题的 RL 权重 |
| Harness | `r000002`：`/user/lyh4170/rsi_campaigns/chess-autonomous-v2/candidates/r000002/source/src/llmgo/harness` |
| skills | `/user/lyh4170/rsi_campaigns/chess-autonomous-v2/candidates/r000002/skills.json` |
| 训练代码 | 本次修复后的完整局 verl PPO＋LC0；不复制 r000002 的旧 training 实现 |
| 训练前对照 | 平台1375420，L2/L3各10局，白黑各5局 |

模型 SHA：`0c3129c0a669beca441f07bfa7bb9e11c7d9ee626419e40195a95de7461feee7`。

r000002 整体 source SHA：`ee655c54a9d9064d2ec1ae872f7127863e6da688909bdd5765ae5b94077e4bb2`。已逐文件核对，本次训练 source-v4r1 的14个 Harness 文件与 r000002 完全一致；整体源码 SHA 不同，是因为训练代码及文档不同。

skills 文件 SHA：`60df0ee9f3010766e64f8a96f305e834f7f20d3e14b18cd2bbabf7e19f867ea3`；训练 identity 的卡片语义摘要：`d393e74da1972aa9b291d5c69c49a99c7b0afd8c01e4e22cc3d2871d2997006f`。

## 已完成的2次更新与当前训练后评测

训练平台任务1376133，独立维护编号train-v4r1；从上面的原始模型开始新的优化器及reference，未从r000007/r000010恢复RL optimizer。

训练冻结源：`/user/lyh4170/rsi_campaigns/chess-autonomous-v2/maintenance/full-game-20261027-v2/source-v4r1`；SHA：`0dd314029ce5925b09dba3d7bcd8ba2d6cfbca2c4b87899f555ca151308f6e94`。

相对维护根目录 `/user/lyh4170/rsi_campaigns/chess-autonomous-v2/maintenance/full-game-20261027-v2`：

- 完整 PPO checkpoint：`jobs/train-v4r1/output/training/checkpoints/global_step_2`。
- HF 导出模型：`jobs/train-v4r1/output/training/model`。
- 导出模型 SHA：`e41483ae23adf194f7b0950b299efd6517bc1566241d720d27599a26822ad58b`。
- 训练后评测：1377822，L2/L3各10局，使用同一r000002 Harness/skills和受保护裁判，只替换模型。

1376133已完成2次真实optimizer更新并通过内部完整局验证、KL、同步及导出加载检查。数值张量审计与训练后实战单独验收，不仅凭不同文件SHA宣布有效提升。

## 新随机后备代码与旧实验的边界

1376133的冻结训练仍使用旧“无有效答案则协议负局”环境。2026-10-27新live源码改成随机合法后备、继续整局，后备归属学生该手，原始输出正常参与终局reward/LC0/GAE/loss。此更新不会修改过去checkpoint、在途评测或其成绩，也不会自动从global_step_2发起新训练。

若后续从global_step_2的HF模型研究新环境，需要显式指定导出目录作为新实验输入，重新记录模型/reference/环境身份；不能把旧环境的optimizer或初始验证直接当作新版恢复状态。主Meta仍可按可比证据选择其他起点，本文件不是永久最佳模型排行榜。
