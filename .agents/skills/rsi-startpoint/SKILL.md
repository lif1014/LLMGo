---
name: rsi-startpoint
description: 为国际象棋RSI从历史实测候选选择模型、Harness与skills研究起点，核对组合身份并在当前workspace分支；区分HF权重继续训练与完整PPO优化器恢复。
---

用户允许从较佳历史候选开始，不必继承最新working候选；也允许修改训练源码。当前明确可选的较佳L2起点是r000007/j000022，16胜3和1负；最新r000010/j000027为14胜5和1负。二者模型SHA相同，区别是Harness与skills；只换model_path不能恢复16胜的组合。这是20局L2的实测选择，不是完整L1–L8全局best或显著性结论。

先读[起点身份卡](references/research-startpoints.json)，再核对所列manifest、metrics、source与skills SHA。当前目录和存档是否仍存在必须检查；卡片是可验证的历史记录，不是实时任务状态。

## 在注册workspace中采用历史组合

Controller默认按最新working创建workspace，这只是初始副本，不是研究限制。Meta可在当前注册workspace恢复r000007的src/llmgo/harness目录和skills.json（含包内skills卡），去掉所选版本不存在的后续Harness文件；保留context.json、提案/完成交接机制、Memory与本轮目录。不要改Controller状态中的working/best或历史候选，不在冻结快照上直接编辑。

在proposal的model_path显式指定身份卡中的HF导出路径；hypothesis及evidence_refs说明实际恢复来源、原评测和后续改变。Controller的parent_id可能仍记录本轮注册工作流的父节点，另用研究证据说明逻辑研究起点，不篡改历史谱系。Harness、skills、模型分开登记；每次比较说明哪一项改变。

**训练模块独立处理。** 恢复r7 Harness/skills不意味着恢复当年的训练实现。旧PPO存在已证实的非法正优势问题，先读[rsi-ppo-audit](../rsi-ppo-audit/SKILL.md)；从live采用已修训练文件，或自主修改并验证新的训练代码。注意旧recipe显式lc0_td0_centered会覆盖新默认。保留已修的导出、缓存释放和同步诊断。

## 权重起跑和断点恢复

从r7导出的HF模型开始新RL实验是可行的起跑方式：设置model_path为所选HF目录，重新建立优化器、reference和训练身份；默认实现的reference以本次输入模型初始化，应记录这种基线改变。不要把HF目录当作完整PPO checkpoint。

继续旧PPO优化器/RNG/dataloader状态要求代码、reward、课程、配置和卡数身份匹配；改优势目标后不应直接拼接旧优化器断点。用户未要求本技能固定训练步数、学习率或必须使用某一算法，可按证据决定RL方案，仍禁止SFT。

研究当前处于操作员暂停时只准备选择和交接，不自行恢复Meta或提交GPU；恢复须按用户明确指令。已有维护验收不会因起点建议而被重启或替换。
