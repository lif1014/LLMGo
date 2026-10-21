---
name: rsi-memory
description: 在国际象棋 RSI 每轮开始、恢复或重复实验前检索持久记忆，按需读取完整棋局证据并追加研究笔记。
---

读取本轮 `context.json`，确认 campaign、incumbent、当前难度、最近失败和正在等待的任务。
`context.json` 只是入口摘要；旧经验在 campaign 的 `events.jsonl` 与各 job 的产物中。

`executor.latest`只提供完成/异常短报告与证据路径；高频轮询、cctl原始回覆另存`executor/events.jsonl`，不属于默认研究检索。正常等待不要反复查询平台；异常再按需读取相关执行事件，避免把全部原文带回教练上下文。

- 先查最近 `decision`、`experiment_failed`、`proposal_invalid` 和相关假设；相同 fingerprint 再试必须在 proposal 的 `repetition_reason` 解释新增证据。
- 从 `short_events.jsonl` 定位问题，再按 request_id/ply 查看原始 `events.jsonl` 的 prompt、输出、usage、finish_reason、工具和后备原因。不要用摘要的截断文字断言模型没有考虑某步。
- 全部原始交互留在共享盘，不把全日志粘进下一轮 prompt。笔记写明局面/请求/文件位置、观察、假设与反证，区分模型错误、预计算局限、基础设施失败。
- 查看固定Lc0 TD(0)优势、clip fraction/KL、behavior重算差异、thinking截断和真实optimizer updates；不要将随机后备步或教师标签说成学生生成。

CLI 可在 Controller 运行期间只读查询或追加笔记；脚本路径与 site 从 campaign 的 `site` 取得：

```bash
python3 /user/lyh4170/LLMGo!/scripts/rsi.py memory CAMPAIGN --search queen --limit 10
python3 /user/lyh4170/LLMGo!/scripts/rsi.py memory CAMPAIGN --kind decision --limit 5
python3 /user/lyh4170/LLMGo!/scripts/rsi.py note CAMPAIGN --text '观察：...；证据：...；下轮假设：...'
```

若用了自定义 site，把 `--site /user/lyh4170/site.json` 放在子命令前。不得覆盖 journal、campaign、已冻结候选或历史结论。具体格式见工作目录 `docs/rsi-operations.md`。
