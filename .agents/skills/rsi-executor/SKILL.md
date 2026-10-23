---
name: rsi-executor
description: 在国际象棋 RSI 的任务交接或异常恢复时读取独立 Executor 报告，按需查原始执行证据，避免反复轮询进入研究上下文。
---

常规提交、监控、产物核验由Python Executor完成；Meta-Agent不承担周期性shell轮询，也不用另一个LLM每隔几十秒查询任务。

先看`context.json`的`executor.latest/report_path`，或运行`python3 scripts/rsi.py executor status CAMPAIGN`。这里是短报告，不是原始日志。
状态未完成时等待Controller/Executor继续处理，不反复询问任务是否结束。

异常时按报告的`evidence`读取对应`failure.json`、训练日志或`executor/events.jsonl`中相关事件；只把证据路径、短结论和下一项可检验假设写研究Memory。保留原文，不把完整执行历史粘回教练会话。

- `retrying`是观察失败，原task可能仍在运行；不重提、不自动取消。
- `submission_uncertain`保留intent并对账，不能盲目创建第二份。
- `collected`仅代表平台与产物核验通过；是否更新实测最佳由完整多难度比赛结果决定，研究可继续。
- `artifact_invalid`不能当成功评测；基线基础设施失败可确认终止后`resume-failed`；候选失败会交回Meta-Agent，下一提案明确恢复路径。

独立`external`模式下，运维进程运行`executor run CAMPAIGN`；Controller只读其报告。`embedded`模式由单个`run CAMPAIGN`托管同一组件。不要在Meta-Agent turn内启动第二个Controller或循环执行step。

详细文件布局、启动与恢复见工作目录`docs/rsi-executor.md`；cctl原始操作约束见同级`cctl/SKILL.md`。资源固定beijing/model-autosft/NORMAL、devspace0GPU，只写自己的`/user/lyh4170`，不得停止他人任务。


默认多卡（2026-10-23）：每项任务最多8 GPU，site默认8；评测为每卡独立Qwen副本并行棋局，`request_evaluation.py --level L2 --gpus N`可选1–8卡。默认训练已改为verl PPO+LC0，8卡任务使用7卡actor/rollout/reference＋1卡LC0，详见docs/rsi-verl-ppo.md。逐局回执核验后恢复，完整批次才合并成正式成绩。轮询仍由Executor处理，网页只读进度摘要和按需日志。已有campaign需显式upgrade，不重写历史快照。

GPU任务提交后只每5–10分钟查询一次（默认300秒，稳定后450/600秒）；纯CPU任务可较频繁查询。交给Executor处理，不用Meta-Agent重复shell轮询。
