# 当前实现：自主国际象棋研究与固定评测

此文件替代旧RL-only/自动晋级计划，实际操作以README和docs/rsi-operations.md为准。

- Meta-Agent在0GPU devspace，自主修改训练、reward、数据、Harness、prompt、工具和skills/触发逻辑；SFT与其他算法不再被框架禁止。
- 现有HF PPO/LC0保留可选配方，课程单步TD0优势；不是唯一方案。
- 任意研究任务通过proposal.jobs提交，Executor独立监控。研究失败也是证据，可继续下一轮。
- 固定裁判控制Fairy、棋规、资源和结果。候选程序独立加载，不在裁判中import可写Harness。
- player与model两轨独立；L1–L8每档20局、全八档才有均分。取消胜率晋级，不将难度当校准Elo。
- 保存working/best/recommended、源码/模型/环境身份及原始日志；最终推荐追加新batch复测。
- 网页展示研究和矩阵，有界增量读取；Executor原始轮询不进入Meta上下文。
- Worker以用户指定ptdbench-verl-vllm017镜像安装共享overlay，保留GPU栈版本，实际兼容情况由环境回执记录。
- 清理其他棋种和旧平行入口；旧runs保留证据但不打包。旧campaign新建v2，不混用旧状态。

交付验证涵盖恢复/身份/预算/真实候选子进程/真实Fairy接口、网页与可选CPU PPO；不把这些测试说成4B模型GPU训练或棋力提升的证据。
