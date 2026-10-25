---
name: rsi-submit
description: 为自主棋力研究登记训练任务和单难度或完整面板评测提案，交给Executor管理任务。
---

workspace 是研究副本：只能通过RL和改进skills提升Qwen3.5-4B，禁止SFT，使用SFT最终得0分。在此范围内RL算法、reward、数据、Harness、prompt、工具、skills与触发条件均可改。当前PPO/LC0是可选起点，不限于此RL配方。不能修改受保护正式评测、对手、裁判及历史正式结果。读取 docs/rsi-operations.md 的 proposal 格式与SFT禁令说明。

proposal.json 可登记任意 argv 任务；每项通过独立 job 执行，支持 {python}/{candidate}/{output}/{model}/{source} 占位。训练、标注、转换都可以登记；向 LLMGO_OUTPUT 写日志/结果，模型导出后写 submission.json 的 model_path。源码提交后快照不再改变；要改代码，在下一个研究 workspace 改。需要任务中生成 Harness 时把代码产物交回 Meta-Agent，再纳入下一次提交。

初始基线固定 L1–L4，每档10局、白黑各5局，共40局；后续正式最终测试每档20局。基线样本更少，比较时注意波动。evaluation 可选择 levels 和 tracks 做探针；只有覆盖八档的 player 结果能更新实测最佳。model 对照使用固定参考 Harness，单独计分。资源边界和成功 cctl 格式见 docs/rsi-submission-and-files.md。

每轮可以只测一个难度，不必重跑全八档。在当前 workspace 中使用：

```bash
python scripts/request_evaluation.py --level L2
python scripts/request_evaluation.py --level L4 --track model
python scripts/request_evaluation.py --full-panel
```

第一条把 proposal.json 的 evaluation 更新为 player/L2，保留既有 jobs、模型路径、假设和其他设置；后续该档20局、白黑各10局。`--hypothesis "..."` 可同时写假设，`--batch-index 1` 可选择新复测批次。`--track both` 会安排两条轨道，各20局。命令仅登记提案，不在0GPU devspace启动推理，也不直接提交集群；把命令输出的 ready JSON（含 proposal_path）作为最终答复，Controller 会快照并交给Executor执行，下轮从Memory读取结果。不要递归启动Controller。

四档初始基线完成即开始研究，不会等待L5–L8；四档基线只记分档结果，完整总分与初始best为空；后续单档测试不覆盖已有的完整best。只有完整八档player评测才更新best；最终应调用 `--full-panel`，单测简单档不能作为完整实力证据。

Controller调用你时持有campaign写锁，避免递归运行prepare/step/run。完成修改、验证和proposal后，优先执行当前prompt给出的`finish_research.py --request ABS_REQUEST_JSON --summary '完成内容与证据'`。它保存绑定本请求与源码/提案哈希的完成记录，不提交任务；模型随后普通结束发言即可。完成后再修改文件须重新执行，不使用旧轮次request。仍兼容最后三字段JSON：status=ready、summary、proposal_path绝对路径。草稿存在不等于完成；Controller校验后快照，Executor提交/轮询/收集。

也允许自行提交独立研究任务，但应登记任务ID、输出与结论。续接或换会话时先读现有workspace笔记、已有任务与产物，不重做已完成步骤、不重复提交。

提交响应不确定时保留 intent、按 description 查询自己的任务并对账；不要盲目重提。只用 beijing/model-autosft/NORMAL，devspace 0 GPU，只写 /user/lyh4170，不停止他人任务。需要原生操作时读同级 cctl/SKILL.md。

环境互不干扰：使用本RSI专用环境，若实验需要不同依赖，另建自己的实验venv/cache；不升级系统Python或其他task的环境，也不原地改正在被任务使用的共享环境。中科大镜像为 https://mirrors.ustc.edu.cn/pypi/simple。


默认多卡（2026-10-23）：每项任务最多8 GPU，site默认8；评测为每卡独立Qwen副本并行棋局，`request_evaluation.py --level L2 --gpus N`可选1–8卡。默认训练已改为verl PPO+LC0，8卡任务使用7卡actor/rollout/reference＋1卡LC0，详见docs/rsi-verl-ppo.md。逐局回执核验后恢复，完整批次才合并成正式成绩。轮询仍由Executor处理，网页只读进度摘要和按需日志。已有campaign需显式upgrade，不重写历史快照。

GPU任务提交后只每5–10分钟查询一次（默认300秒，稳定后450/600秒）；纯CPU任务可较频繁查询。交给Executor处理，不用Meta-Agent重复shell轮询。
