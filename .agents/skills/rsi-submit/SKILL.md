---
name: rsi-submit
description: 为自主棋力研究登记任意计算命令、模型或Harness提交和多难度测试，交给Executor管理任务。
---

workspace 是自由研究副本：训练方法、reward、数据、Harness、prompt、工具、skills 与触发条件均可改。当前 PPO/LC0 是可选起点，不是方法限制。不能修改受保护正式评测、对手、裁判及历史正式结果。读取 docs/rsi-operations.md 的 proposal 格式。

proposal.json 可登记任意 argv 任务；每项通过独立 job 执行，支持 {python}/{candidate}/{output}/{model}/{source} 占位。训练、标注、转换都可以登记；向 LLMGO_OUTPUT 写日志/结果，模型导出后写 submission.json 的 model_path。源码提交后快照不再改变；要改代码，在下一个研究 workspace 改。需要任务中生成 Harness 时把代码产物交回 Meta-Agent，再纳入下一次提交。

正式最终测试固定 L1–L8，每档20局。evaluation 可选择 levels 和 tracks 做探针；只有覆盖八档的 player 结果能更新实测最佳。model 对照使用固定参考 Harness，单独计分。资源边界和成功 cctl 格式见 docs/rsi-submission-and-files.md。

Controller 调用你时持有 campaign 写锁，避免在该 turn 递归运行 prepare/step/run；完成研究后返回 ready 和 proposal 绝对路径，由 Controller 快照，Executor 提交/轮询/收集。也允许自行提交独立研究任务，但应登记任务 ID、输出与结论，避免重提；常规任务建议走 Executor 以免轮询占用上下文。

提交响应不确定时保留 intent、按 description 查询自己的任务并对账；不要盲目重提。只用 beijing/model-autosft/NORMAL，devspace 0 GPU，只写 /user/lyh4170，不停止他人任务。需要原生操作时读同级 cctl/SKILL.md。

环境互不干扰：使用本RSI专用环境，若实验需要不同依赖，另建自己的实验venv/cache；不升级系统Python或其他task的环境，也不原地改正在被任务使用的共享环境。中科大镜像为 https://mirrors.ustc.edu.cn/pypi/simple。
