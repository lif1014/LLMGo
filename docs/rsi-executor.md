# Executor：独立管理任务生命周期

Executor 是无LLM的Python组件，负责manifest校验、持久提交intent、cctl提交、对账、退避轮询、终止日志收集和结果哈希验证。Meta-Agent仅接收任务变化、短报告和产物路径；高频查询不进入默认研究上下文。

默认embedded：单个 `rsi.py run CAMPAIGN` 托管Executor。external模式在init前设置site.executor_mode=external，然后分别运行：

```bash
python3 scripts/rsi.py executor run CAMPAIGN
python3 scripts/rsi.py run CAMPAIGN
```

两个控制进程都0GPU；正式计算通过job。独立Executor可在Controller长时间持锁研究时工作。状态查询只读本地report：`python3 scripts/rsi.py executor status CAMPAIGN`。

任务范围包括任意research命令和固定单档evaluate（初始基线10局、后续评测20局）。research可指定GPU/CPU/memory，评测默认8GPU、可配置1–8GPU；集群、池和优先级始终beijing/model-autosft/NORMAL。不会自动停止任务或抢占别人资源。

文件：executor/events.jsonl保存完整命令、查询和原始输出；jobs/ID/executor/report.json保存短状态；jobs/ID/output保存worker环境、进度、日志与结果；submission.json记录外部task ID和唯一description。GPU任务查询间隔300秒起，稳定时退避至450、600秒；纯CPU任务沿用poll_seconds（默认120秒）；Controller/服务可每60秒醒来读取本地状态，尚未到期不会查询集群。网页刷新不会查cctl。已有campaign沿用创建时冻结的站点设置，不原地改其manifest或任务身份。

- submission_uncertain：创建响应不确定，保留intent，枚举自己的任务对账，不能盲目重提。
- retrying：一次观察失败，原任务可能仍运行；保留ID继续观察。
- waiting_for_artifact：平台成功但共享盘结果未出现，继续等待。
- artifact_invalid：结果身份/哈希不符，不能作为正式成绩。
- collected：任务/结果收集完成；不代表棋力改善。

候选任务失败保留证据并回Meta-Agent；它可以修改方案或在新提案明确指定旧checkpoint恢复。初始基线失败进入NEEDS_INPUT，解决后可resume-failed（确认旧任务已终止）或resume创建该档新任务。锁由实际flock进程持有，不靠删除过期锁文件恢复。


默认多卡（2026-10-23）：每项任务最多8 GPU，site默认8；评测为每卡独立Qwen副本并行棋局，`request_evaluation.py --level L2 --gpus N`可选1–8卡。默认训练已改为verl PPO+LC0，8卡任务使用7卡actor/rollout/reference＋1卡LC0，详见docs/rsi-verl-ppo.md。逐局回执核验后恢复，完整批次才合并成正式成绩。轮询仍由Executor处理，网页只读进度摘要和按需日志。已有campaign需显式upgrade，不重写历史快照。
