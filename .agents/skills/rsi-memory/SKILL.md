---
name: rsi-memory
description: 在自主国际象棋研究的轮次开始、恢复或实验设计前检索持久记忆和跨难度证据。
---

先读本轮 context.json，确认 working、best、recommended、当前任务及失败。best 是完整 L1–L8 玩家评测的实测最高分，不代表统计显著提高；没有自动通关或升档。

同时读最新handoff中的operator_notes及research_note，区分新指令、历史总结与已验证事实。working只是最近研究候选；每轮结合最新可比评测重新选择起点，不固定继承最新版本或历史r7。r7的16胜记录只是已核对示例，选择方法见[rsi-startpoint](../rsi-startpoint/SKILL.md)。旧训练实现有已验证问题并允许修改，不能因过去摘要说“训练没有问题”而跳过[rsi-ppo-audit](../rsi-ppo-audit/SKILL.md)的实际张量检查。

context 只是入口。按需检索 campaign 的 decision、hypothesis、evaluation、experiment_failed 和 research_note；相同思路可以重试，但先说明新证据。短日志定位局面，再按 request_id/ply/raw_ref 读取完整输出、prompt、usage、finish_reason、工具和后备。不要以截断日志断言模型没有考虑某步。

原始轮询和平台输出位于 executor/events.jsonl；正常等待只读 executor 短报告，不反复发 shell 查询。把路径、观察、反证和下一步持久化，不把全日志粘进下一轮。

可在 Controller 持锁期间查询或追加笔记：

```bash
python3 /user/lyh4170/LLMGo!/scripts/rsi.py memory CAMPAIGN --search queen --limit 10
python3 /user/lyh4170/LLMGo!/scripts/rsi.py note CAMPAIGN --text '观察、证据路径、假设、反证和下一步'
```

自定义 site 放在子命令前。完整玩家与固定参考 Harness 的模型对照应分开分析；部分难度测试没有完整总分。源代码、产物、环境及历史结果是复现证据，保留旧版本再在新研究副本继续修改。细节见 docs/rsi-operations.md。
