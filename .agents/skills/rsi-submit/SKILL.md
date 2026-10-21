---
name: rsi-submit
description: 在 RSI 工作目录准备可冻结的 skills 或 RL 实验，交由 Controller 提交并在重启时核对已有任务。
---

默认路径是本轮 workspace 中的 `skills.json`、`curriculum.jsonl` 和 `proposal.json`。
读取 `docs/rsi-operations.md` 的 proposal 字段，再准备 `skills`、`rl` 或 `rl_and_skills` 候选。
不直接编辑既有权重、evaluator、基础 prompt、协议、已冻结源码或其他候选。

Controller 正在调用你时，它持有 campaign 写锁。不要从该 turn 调用 `prepare`、`step`、`submit` 或 `run`，也不要重复 `cctl job create`。
完成后按指定JSON返回`ready`和proposal绝对路径；Controller负责冻结与研究验收，专用Executor负责提交、轮询和产物交接。等待任务不需要Meta-Agent反复发shell指令；原始查询与平台日志不写进研究Memory。`needs_input`只用于具体缺失项。

独立运维场景、Controller 未运行时，可以：

```bash
python3 /user/lyh4170/LLMGo!/scripts/rsi.py preview CAMPAIGN
python3 /user/lyh4170/LLMGo!/scripts/rsi.py submit CAMPAIGN
python3 /user/lyh4170/LLMGo!/scripts/rsi.py run CAMPAIGN
```

提交响应不确定时，保留 `submission.json`，按 description 查自己的任务并对账；查不到也不能盲目重提。平台失败须先确认原任务终止，再 `resume-failed`，保留原始失败产物。不要删锁文件来绕过仍存活的进程。

需要任务诊断时先读`rsi-executor/SKILL.md`和短report；`external`模式的独立Executor可在Controller持锁时工作，不应再开启第二个提交循环。

资源边界、已成功任务样例和日志参数见 `docs/rsi-submission-and-files.md`；需原生 cctl 时再读同级 `cctl/SKILL.md`。
