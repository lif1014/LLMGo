# 非法输出被PPO正向奖励：实际取证案例

这是历史故障证据，不代表每次棋力下降都由同一原因造成。训练代码允许修改，不能把它当作固定正确的裁判。

## 如何发现

1. 对比L2旧4-step评测j000022（16胜3和1负，得分率87.5%）与24-step评测j000026（12胜6和2负，75%），先核对模型、候选源码、Harness/skills、白黑、种子、协议。两次训练均从原始Qwen开始；24-step不只是多训，还把lr从2e-6改成5e-6，因此不是严格单变量步数实验。
2. 检查原始对局与回退原因：24-step并非“没有格式错误”。再读逐手LC0样本，发现有协议失败的原始TD为0，而训练整批均值为负。这只是疑点，尚不能证明actor真的用了正优势。
3. 在0GPU任务中读保存的更新前.pt，把sample_path、failure_reason、mask与advantages逐一对应，重建全批中心化。发现实际用于PPO的有效response tokens确实全为正优势，确认代码问题。

| 训练/step/sample | 失败原因 | V_before | 原始TD | 旧批次均值 | 实际最终优势 | 有效response tokens |
|---|---|---:|---:|---:|---:|---:|
| j000016 / 1 / 35 | invalid_final_answer | −1 | 0 | −0.1685357 | +0.1685357 | 6928 |
| j000024 / 3 / 163 | invalid_final_answer | −1 | 0 | −0.2229822 | +0.2229822 | 10344 |
| j000024 / 24 / 1340 | invalid_final_answer | −0.981 | −0.019 | −0.1587857 | +0.1397857 | 10043 |

“协议target=−1”不保证惩罚为负：旧代码先算target−V_before，再减全批均值。劣势V_before接近−1时，失败原始TD接近0，随后被负均值推成正优势。不能只看reward配置或原始样本里的advantage字段；更新前张量才是优化实际输入。

## 已验证的修复

`lc0_td0_protocol_guarded`在完整batch进入actor更新前处理：合法样本使用TD减合法均值；invalid_final_answer、incomplete_thinking、length直接设为负的protocol_failure_penalty（默认1），不再减教师基线或均值。全批失败拒绝优化，未知基础设施错误不当作棋步reward。原始TD保留。

实现：src/llmgo/training/advantages.py、verl_registration.py与verl_ppo.py。修复同时保留r9的权重命名/tokenizer导出修复、vLLM唤醒前缓存释放和同步检查；不要用旧本地模块覆盖这些改动。核心commit bba2d37；后续8fb4ba6把诊断的最终负优势比例与原始TD负比例分开。

0GPU任务1371549的73项测试通过，并回放28批1568个真实样本：55个协议失败中，旧版21个获得正优势，修复后0个，原始TD不变。live后续诊断测试64项通过（测试集合不同，不相加）。证据目录：

```
/user/lyh4170/rsi_readiness/ppo-4-vs24-20261026/
  saved-signal-check.json
  result.json
/user/lyh4170/rsi_readiness/ppo-protocol-guard-20261026/
  saved-batch-regression.json
  cpu-acceptance.json
  live-deployment.json
  diagnostic-deployment.json
```

维护任务1371578第一步实际更新：53/56合法，3个失败优势均−1；grad_norm约0.5104、实际lr=2e-6、actor/rollout平均绝对logprob差约0.01294且无同步错误。完整2-step、导出、4-step和评测状态应读最新回执，不从这条历史记录推断已经完成或棋力提高。维护验收由独立Executor处理，避免Meta重复提交；其计划位于上述protocol-guard目录的validation-plan.json。

## 因果结论的边界

旧训练代码确实有优化目标错误，但不能据此声称它解释了全部24-step棋力下降。还观察到reference KL增大、较长训练/较高lr，以及部分优势局未转化；胜势转换验证已被用户暂缓。20局配对比较的波动较大，不能断言24-step在所有难度全面退化。禁止重复旧总结中的错误说法：“4-step没有更新”“lr全为0”“24-step没有格式失败”“六盘和棋全部是优势局”。
