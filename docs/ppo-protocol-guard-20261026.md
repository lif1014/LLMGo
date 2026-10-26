# PPO协议优势修复与维护验收

本次只做用户授权的三项：持久暂停研究、修复非法输出正优势、回归与短训练对照。暂不做胜势转换或Harness/skills消融；不修改旧比赛、被冻结源码和正式裁判。

## 已确认的问题

旧4-step（j000016）与24-step（j000024）保存的更新前张量中，协议失败也参与全批TD中心化。当失败样本的原始TD为0、批次均值为负时，最终优势会变成正值。例如24-step第3步sample163原始TD=0、均值约−0.223，实际10344个response tokens都获得约+0.223优势。这是优化目标的错误；不能单凭此事断言它解释了全部棋力差异。

## 修复

- `advantages.py`在完整batch、actor更新前计算最终优势：合法样本TD减合法均值，协议失败固定−1，可配置正数`protocol_failure_penalty`。原始教师反馈保留；padding不参与优化。
- `verl_registration.py`先施加修复，再保存batch、做质量检查、调用原生verl actor更新。诊断记录最终实际使用的数值。全批失败、缺判定、未知故障类型不进入优化。
- 延续r000009已验证的导出权重命名/tokenizer修复、vLLM唤醒前缓存释放、同步诊断与recipe参数转发；不以旧本地训练代码覆盖这些修复。
- 新默认模式为`lc0_td0_protocol_guarded`，旧recipe显式模式不会自动迁移。修改模式或代码后从新实验开始，不拼接旧优化器状态。

## 验证证据

本地Python测试403通过、6跳过，网页DOM检查通过。本地没有Torch，对Torch的结论以指定Worker镜像中的专属控制环境测试为准：0GPU任务1371549完成73项测试。

同任务只读回放28批、1568个真实历史样本，其中55个协议失败；旧版有21个失败样本获得正优势，修复后为0，原始TD张量未改变。远端回执：

```
/user/lyh4170/rsi_readiness/ppo-protocol-guard-20261026/
  tests.log
  tests-result.json
  cpu-acceptance.json
  saved-batch-regression.json
```

另修正质量摘要里的`advantage_negative_fraction`：现在按有有效tokens的回答统计最终优势为负的比例；旧代码实际统计原始TD的负比例，现另名`raw_td_negative_fraction`保留。此项只影响诊断，不改更新目标。进行中的维护快照仍使用原冻结版本，该处的旧字段不能当作最终优势比例；最终优势以`protocol-advantages`和保存batch为准。

## 短训练与比较

独立维护目录冻结r000009的Harness、skills和课程，仅补入训练修复。顺序执行：2-step验收 → 4-step训练 → L2二十局（白黑各十局、固定旧种子及裁判设置）。两次训练都从原始Qwen3.5-4B开始，非从2-step继续4-step；学习率2e-6、每step两epoch、batch56/minibatch14、KL0.05、response16384、CUDA Graph开启、vLLM池0.45、7actor卡加1LC0卡。比较对象是旧4-step的j000022（16胜3和1负，得分率87.5%），不是r10的胜势转换Harness。

每阶段必须先确认Worker身份、实际最终优势、同步诊断、完整HF导出，再启动下一阶段。GPU平台每300秒轮询，提交意图、task_id和失败状态留在独立维护Executor中，不进入Meta上下文；没有通过就停止链条等待检查，不盲目重提。主campaign保持研究暂停，不自动扶正、不自动恢复Meta。短训练与实战结果须读各自实际回执，不能把上述CPU回归当作训练成功或棋力提高。

## 暂停与显式恢复

`research-control.json`独立于研究phase持久化。新版Controller在暂停期间仍收集已提交Worker，不准备新任务、不启动Meta；live Codex launcher也检查暂停。原devspace载入新版Controller使用`bash scripts/start_rsi.sh --restart-control --web`，不会重建campaign。只有用户明确恢复时才执行`bash scripts/start_rsi.sh --resume-research`；Controller已退出则随后普通`--web`启动。维护不改旧任务快照，live文件发布本身不保证后续候选自动采用修复。
