> 历史诊断：本文记录已停止的单步 PPO A/B/C，不是当前默认入口。2026-10-27 完整对局方案和实际配置见 [rsi-verl-ppo.md](rsi-verl-ppo.md)。

# PPO稳定性、固定验证与checkpoint选择

2026-10-26 的 r11/j000028 从原始权重和r2 Harness/skills起跑，采用修复后的协议负优势，仍在第17批跌至27/56合法而停止。非法正优势未复发；调查发现整段协议惩罚的系数占比很大、reference KL累积上升、停止阈值过晚。它们是优先验证的机制，不是已经完成的因果证明。新实现保留原模式供对照，不宣称CPU测试通过就代表棋力提高。

## 目标

原始棋步信号保持同一行棋方视角：非终局`TD=V_after−V_before`，V=P(win)−P(lose)；终局只计一次目标减V_before。一手棋对应一个segment，没有SFT、监督答案、学习中的critic或整盘GAE。

`lc0_td0_protocol_balanced`分开计算：

```
chess_i = valid_i * (TD_i − mean(TD_valid))
protocol_i = protocol_weight * (valid_i − mean(valid_active))
A_i = chess_i + protocol_i
```

有效样本按每手平均的A为零；混合批次中非法回答的A严格为负，合法回答保留TD排序并得到协议奖励。全部非法时没有可用的中心化信号，进入optimizer前强制停止。padding为0，未知基础设施失败不能当棋步奖励。

原来的`lc0_td0_protocol_guarded`仍支持固定`−protocol_failure_penalty`，便于受控对照。`protocol_weight`只影响balanced模式，不影响guarded对照。

默认`loss_agg_mode="seq-mean-token-mean"`先对每个回答的有效token求平均，再对走棋样本求平均。旧`token-mean`可显式选择；microbatch=1并不自动使样本等权。有效推理与答案共同参与RL，分段诊断只用于分析。

## 推荐配方

在现有ppo.json的training中设置；reward仍使用现有LC0配置：

```json
{
  "backend": "verl_ppo",
  "steps": 8,
  "batch_size": 56,
  "minibatch_size": 14,
  "ppo_epochs": 1,
  "learning_rate": 0.000001,
  "lr_warmup_steps_ratio": 0.25,
  "kl_beta": 0.05,
  "advantage_estimator": "lc0_td0_protocol_balanced",
  "protocol_weight": 0.1,
  "loss_agg_mode": "seq-mean-token-mean",
  "max_prompt_tokens": 6144,
  "max_response_tokens": 16384,
  "checkpoint_every": 2,
  "quality_gate_min_valid": 0.9,
  "reference_kl_warning": 0.05,
  "reference_kl_stop": 0.1,
  "preupdate_consistency_samples": 16,
  "sync_check_steps": 0,
  "sync_max_mean_abs_logprob_delta": 0.1,
  "validation_size": 56,
  "validation_every": 2,
  "validation_min_valid": 0.9,
  "validation_max_drop": 0.05,
  "export_validated_on_stop": true,
  "rollout_enforce_eager": false,
  "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
  "rollout_memory_utilization": 0.45,
  "rollout_wake_cache_flush": true,
  "seed": 4170
}
```

这些阈值是初始实验设置，可以研究调整；不属于已证明的最优参数。8卡仍是7卡actor/rollout/reference＋1卡LC0。已有显式recipe不会被新默认强行改写；修改优化目标/配方应登记新实验，不混用旧optimizer身份。

## 实际停止规则

- 更新前若没有合法回答，或训练批次合法率低于quality_gate_min_valid，则停止。
- 使用保存的rollout/ref logprobs计算非负K3参考散度，按每手平均；超过reference_kl_stop或证据缺失/非有限则停止。reference_kl_warning只记录警戒。它们不是未启用的algorithm.kl_ctrl.target_kl字段。
- 开启同步阈值后，每个更新都要有成功的actor重算logprob证据；失败或超过阈值停止。概率比使用signed logprob差，修正原来先abs后exp的上界漏报。
- 验证合法率不得低于`max(validation_min_valid, baseline_valid_rate−validation_max_drop)`。例如原始模型100%时，本例阈值95%。

## 固定验证与导出

validation_size默认0，代表未启用固定题集。显式启用时，按seed和局面内容确定独立分组，以棋盘FEN前四项把转置及不同计数/历史变体放在同组；验证组的全部记录从训练中移除。需要至少validation_size+1个不同局面组；不足时明确报错，不拿同一题同时训练和验证。

训练前用原始模型建立固定题集基线，之后按validation_every验证当前已更新权重。验证显式使用temperature0.9/top_p0.95/top_k20，与正式玩家采样参数一致，种子按validation_seed和position_id固定；保留原始输出、协议判定、LC0反馈和实际sample路径；全部实验使用同一题集与设置。验证指标用于开发选择，不是Fairy胜率。

validation_every须为checkpoint_every的倍数。通过更新后验证的step才登记在validated-checkpoints；baseline不算训练checkpoint。导出仅选择完整、身份匹配且有通过回执的checkpoint，按合法率优先、合法TD均值其次、回答长度再次排序；不会因为最新就选最新。

遇到明确的TrainingSafetyStop时，可按export_validated_on_stop从先前已通过验证的完整checkpoint导出；没有合格存档则失败。普通OOM、网络、模型运行异常仍报错，不能伪装成成功。training-result.json与checkpoint-selection.json显式记录提前停止、实际选择的step和证据。正式L2/L3评测后才能判断棋力是否提高，不因导出成功自动扶正。

## 三组对照

都从原始Qwen权重＋相同r2 Harness/skills开始，8步、lr1e-6、epochs1、相同seed/课程/固定验证，检查更新目标：

| 组 | 优势 | loss归约 | 对照目的 |
|---|---|---|---|
| A | guarded，失败−1 | token-mean | 保守更新是否足以稳定 |
| B | 同A | seq-mean-token-mean | 单独比较回答长度权重 |
| C | balanced，协议系数0.1 | 同B | 比较分离协议奖励 |

从稳定实验选候选，正式L2、L3各20局，与相同Harness/skills和固定种子下的原始权重比较。历史r2成绩只作参考；如需严格配对结论，用当前同条件基线。20局差距小应补测，不宣称稳定Elo提升。主Meta研究暂停期间由独立Executor提交维护任务，不自动解除暂停或改写历史结果。

## 观察文件与网页

training目录新增：health.json（小型当前状态）、update-diagnostics/step-*.json（分段与合法/非法系数统计）、validation-split.json、validation-baseline.json、validation.jsonl、validation-rollouts/、validated-checkpoints/、stability-stop.json、checkpoint-selection.json。原始LC0/响应和更新前.pt仍保留。

quality记录标记before_update，metrics标记update_completed，validation为baseline/after_update，避免错把采样批次当成该批更新之后的表现。系数总量不是参数梯度贡献率，不能用77%的系数占比声称77%的梯度来源。

网页的“PPO稳定性验证”读取campaign/maintenance/active.json中的独立任务清单，每20秒只读小型progress/health；它不扫描原始日志、不查询cctl。点击文件才分页读取原文；主研究任务仍在原任务列表。静态HTML/JS同步后刷新可生效，Python的job-files扩展需下次Web加载后生效。

可信自有保存batch可用scripts/check_ppo_protocol.py的`--mode lc0_td0_protocol_balanced --protocol-weight 0.1`只读回放。Python pickle只读取自己的可信训练产物。GPU平台轮询继续每300–600秒，由Executor独立处理。


## 首轮验证采样事故与修正

A1374321、B1374322、C1374323在step0原始权重验证分别仅18/56、6/56、10/56合法，失败主要为16k长度截断，均未进行optimizer更新。首轮新增验证继承了verl默认temperature0的贪心生成，与正式玩家和训练采样不一致。不能把它们作为三种PPO目标的训练效果；原始失败证据保留。

后续版本显式设置validation_temperature/validation_top_p/validation_top_k，默认0.9/0.95/20；validation_seed默认4170，按position_id生成每请求种子，不依赖副本编号、请求顺序和研究step。单turn代理继续调用verl原实现，仅在验证请求上加种子并记录实际参数，训练仍temperature1/top_p1/top_k−1，behavior logprobs路径不变。种子固定不保证跨硬件/批处理的逐位一致。

scripts/probe_validation_sampling.py用于单卡无更新探针：同一56题先比较贪心/正式采样/训练采样并保存完整响应，再决定是否重启三组8卡训练。原始模型在固定验证题集也不通过时，不通过降低闸门掩盖问题。step0安全停止且无合格checkpoint时，保留原始停止原因，不导出模型。


单卡探针首个任务1374566在输入编码阶段失败，未加载生成模型、未执行PPO。原因是直接对字符串content调用Transformers5.4多模态Processor的tokenize=True路径；修正为与verl Agent一致的“先渲染chat template，再processor编码”流程。0GPU任务1374615验证了真实Processor与Agent逐token一致，且全部56条输入编码成功（772–3217 tokens），随后才提交1374635重新做单卡探针。旧失败日志和manifest保留。
