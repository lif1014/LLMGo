# RSI网页：强Agent事件、Executor轮询与Worker进度

页面地址为`/rsi`。本次只增加观测与展示，不改变学生走子、奖励、RL目标或正式验收条件。PPO替换仍待单独实施。实际研究在自己的0GPU devspace；网页也不需要GPU。

## 1. 页面能看到什么

| 区域 | 信息与操作 |
|---|---|
| 强Agent状态 | 最近记录的状态、thread/turn、工作目录、状态时间、已记录子进程身份；服务实际返回的累计输入/缓存命中/输出/推理Token及最近请求Token |
| 强Agent时间线 | app-server请求与通知、文字及服务可见思考摘要、命令/工具开始与完成、可见输出增量、退出/失败、轮次状态、心跳；可按类别筛选 |
| Executor状态 | 平台task ID、平台状态与交接状态、累计检查次数、当前退避间隔、下次检查时间、报告摘要及独立服务心跳 |
| Executor时间线 | 每轮检查开始与完成、每条cctl命令开始/返回、耗时、退出码、stdout/stderr摘要、平台状态、查询错误、提交不确定、产物核验及交接 |
| Worker进度 | 已记录的阶段：正式评测、离线标注、训练、导出核验、完成或失败；课程局面数、已落盘正式对局数与最近阶段时间 |
| 任务文件 | manifest、提交凭据、Executor报告/平台日志、环境、progress、failure、训练日志、导出引用、标签索引、评测结果等已有文件；点击才读正文 |
| 学生 | 保留棋盘、模型可见思考/最终回答、工具库与请求/工具/后备诊断；可选任务、选对局或跟随最新对局 |
| Memory与表现 | 假设、笔记、验收与晋级事件；最近200条评测曲线及最多80张结果卡片。更早证据仍在完整日志 |

Token字段缺失时显示“未提供”，不从文字长度估算。页面只展示服务提供的可见内容，不能补出未公开的内部思考。心跳或PID文件只是最近记录，不是实时进程存活证明。Worker阶段是里程碑记录，不等于逐个训练step百分比；训练细节可打开对应日志。

后端在本轮正常完成或失败后，仍归档子进程退出时已读到的尾部用量和stderr；最终用量会写回会话快照，相关事件也进入时间线。已脱敏的退出诊断与普通事件一样可按需查原文，不另启动模型调用。

强Agent与Executor分开显示。等待GPU任务期间不因网页刷新调用强模型；轮询原文仍保存在Executor自己的记录，Meta-Agent默认仍只取得短报告和证据路径。

## 2. 控制长期运行负担

- 自动刷新每3秒一次；页面隐藏或点击“暂停刷新”时停止自动读取。不叠加尚未结束的一轮刷新，每个请求最多等待15秒。
- 强Agent/研究与Executor各有独立游标。初次进入直接取最新一页，后续只读新增索引行；每页最多80条、服务端显示预算约192KiB。元数据快照有缓存，不在每次HTTP读取时重建索引或重扫完整journal。
- 每个列表最多保留80张卡片；同一thread/turn/item的串流合并，当前文字窗口最多20,000字符。闭合的详情不生成大段JSON，展开才格式化。
- “较早事件”替换当前窗口并暂停该来源的自动追加；“返回实时”重新取最新窗口。后台响应不会覆盖之后切换的历史页或文件。
- 原始事件完整保留在追加日志。索引只存有界摘要及精确字节位置；点击“读取原始事件”时每段读64KiB，支持上一段/下一段。超长原始记录不会整条解析后传给浏览器。
- 大日志文件默认从末尾64KiB打开。任务列表只给文件名称、长度与更新时间，点击后才读取正文。查询对局限定在当前任务，避免每3秒遍历所有历史任务；评测曲线最多200条，每20秒更新一次，或收到新评测事件后更新。
- HTTP接口不执行cctl、不提交任务、不调用模型。网页的3秒刷新与Executor默认30至120秒退避是两个独立节奏；同时打开几个浏览器不会增加平台轮询次数。

这些限制约束的是显示与读取窗口，不删除原始证据。数据量很大时，新事件按页追赶；时间线会提示正在追赶。筛选作用于已载入的当前窗口，需要查看更早类别时可翻页。

## 3. 升级与旧记录

上传代码后重启网页服务，并让浏览器重新加载`/rsi`。后台仍沿用原来的启动方法：

```bash
cd '/user/lyh4170/LLMGo!'
LLMGO_RSI_ROOT=/user/lyh4170/rsi_campaigns PYTHONPATH=src \
  /user/lyh4170/venvs/rsi-control/bin/python -m uvicorn llmgo.server.app:app --host 0.0.0.0 --port 7860
```

若旧campaign提示“活动索引尚未建立”，在同一控制环境执行一次：

```bash
python3 scripts/rsi.py index-activity CAMPAIGN
# 原campaign用了自定义站点时，须沿用它：
python3 scripts/rsi.py --site /user/lyh4170/site.json index-activity CAMPAIGN
```

此命令重放研究和Executor现存日志，补建SQLite活动索引；不改原始journal，不调用强模型或集群，不启动训练。首次索引的耗时取决于旧日志大小，不把它放在HTTP请求中。新框架写入时同步维护索引；SQLite丢失后可从原始日志重建。

旧日志里未记录的“命令开始/轮询开始/轮询完成”无法补造。新增采集只对升级后运行相应代码的进程生效；Worker的`progress.json`同理。已有campaign冻结了计算源码，新代码不会自动替换运行中的GPU job。新建campaign能完整使用新Worker记录；为查看旧记录不需要重启或取消计算任务，也不要修改冻结源码混合实验版本。

## 4. 数据与只读接口

| 路径 | 内容 |
|---|---|
| `CAMPAIGN/events.jsonl` | 研究与强Agent可见协议的原始事实日志，原有脱敏继续生效 |
| `CAMPAIGN/executor/events.jsonl` | 独立提交/轮询/查询原文与起讫事件 |
| `CAMPAIGN/memory.sqlite`及`executor/memory.sqlite` | 可重建索引；`activity`保存显示投影、原始offset/length，`events`保留原有Memory检索 |
| `CAMPAIGN/meta-session.json` | 最近强Agent状态与会话恢复信息 |
| `jobs/JOB/executor/report.json` | Executor短报告与下一轮检查计划 |
| `jobs/JOB/output/progress.json` | Worker最新阶段与时间 |

以下路径均在`/api/rsi/campaigns/CAMPAIGN`下，只读：

| 接口 | 用法 |
|---|---|
| `/overview` | 小型状态、最近30个任务、成本与强Agent会话摘要 |
| `/activity?source=research`或`source=executor` | 默认最新页；`cursor=N`向后追新，`before=N`查看更早页，二者不能同时使用。N为索引序号，不是字节偏移 |
| `/executor` | 最近任务报告与服务心跳 |
| `/job-files?job_id=JOB` | 固定清单中的已有文件元信息与小型进度快照 |
| `/evaluations` | 最近200条已索引评测 |
| `/games?job_id=JOB`、`/game-events?folder=…` | 当前任务的棋局与学生事件；game-events仍用字节游标 |
| `/artifact?path=…&offset=…&length=…` | 经过campaign路径校验的原文分段读取 |

原`/events`、`/executor-events`接口保留用于按字节查原始证据，普通自动刷新不请求它们。

## 5. 本地验证

新增测试验证了：最新/历史/增量分页、Unicode字节定位、超大原文保真及重建、HTTP不扫描原始研究日志、响应预算、工具与用量事件、任务文件按需读取；DOM测试覆盖5,000段合并、80行上限、暂停/隐藏、筛选、历史切换与异步返回次序。

可选真浏览器测试使用临时campaign与本机HTTP服务，检查真实串流刷新、棋盘、历史页、原文分段和390px手机布局；不访问模型或集群：

```bash
PYTHONPATH=src python -m pytest -q tests/test_rsi_activity.py tests/test_rsi_meta.py
node tests/test_rsi_ui.js
# 仅需浏览器集成测试的开发环境安装Playwright和Chromium：
# pip install playwright
# python -m playwright install chromium --only-shell
LLMGO_BROWSER_TEST=1 PYTHONPATH=src python -m pytest -q tests/test_rsi_browser.py
```

这些是工程验证，不是实际远端Codex/集群耗时测量或学生棋力提升证据。Playwright不属于网页服务运行依赖。

### PPO文件与负担

Worker的`progress.json`显示sampling/labeling/updating/updated及批次进度；`job-files`增加`training/latest.json`和它指向的最近采样、targets及完整batch文件。最多读取一个64KiB有界指针及三个文件stat，不枚举历史rollouts。原始thinking/token数组只在选择文件后分页读取；训练loss、KL和clip fraction见training.jsonl，不进入Meta默认摘要。
