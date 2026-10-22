# 当前实现：自主国际象棋研究与固定评测

此文件描述当前RL/skills自主研究与固定评测，实际操作以README和docs/rsi-operations.md为准。

- Meta-Agent在0GPU devspace，只能通过RL和改进skills提升Qwen3.5-4B；禁止SFT，使用SFT最终得0分。可自主修改RL、reward、数据及支持RL/skills的Harness、prompt、工具和skills/触发逻辑。
- 方法与0分规则通过developer prompt、每轮任务prompt及研究skills传达；尚无任意训练代码的自动SFT检测或归零机制。禁止SFT预热、辅助SFT、行为克隆和监督蒸馏；离线教师可用于RL奖励/价值、分析与提炼skills，RL策略梯度和价值函数拟合可用。
- 现有HF PPO/LC0保留可选配方，课程单步TD0优势；不是唯一方案。
- 任意研究任务通过proposal.jobs提交，Executor独立监控。研究失败也是证据，可继续下一轮。
- 固定裁判控制Fairy、棋规、资源和结果。候选程序独立加载，不在裁判中import可写Harness。
- player与model两轨独立；L1–L8每档20局、全八档才有均分。取消胜率晋级，不将难度当校准Elo。
- 保存working/best/recommended、源码/模型/环境身份及原始日志；最终推荐追加新batch复测。
- 网页展示研究和矩阵，有界增量读取；Executor原始轮询不进入Meta上下文。
- Worker以用户指定ptdbench-verl-vllm017镜像安装共享overlay，保留GPU栈版本，实际兼容情况由环境回执记录。
- 可选完整上传包包含LC0 Linux程序、对应源代码与BT4权重；install_teacher核验后安装到既有site路径，不覆盖不同文件，教师资产不进入每轮源码快照。
- 专案名保持rsi；新campaign每120秒查询集群，状态稳定时退避至600秒，已有campaign保留冻结配置。
- 清理其他棋种和旧平行入口；旧runs保留证据但不打包。旧campaign新建v2，不混用旧状态。

交付验证涵盖恢复/身份/预算/真实候选子进程/真实Fairy接口、网页与可选CPU PPO；不把这些测试说成4B模型GPU训练或棋力提升的证据。
