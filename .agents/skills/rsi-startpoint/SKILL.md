---
name: rsi-startpoint
description: 为国际象棋RSI从历史实测候选选择模型、Harness与skills研究起点，核对组合身份并在当前workspace分支；区分HF权重继续训练与完整PPO优化器恢复。
---

用户允许自主选择研究起点，不必继承最新working，也不固定从r7开始；训练源码同样可修改。每轮按最新可验证证据重新选择，发现更好的可比候选时可自行采用，无须仅为更换起点再次征求许可。

## 动态选择，避免固定锚点

先读最新context、handoff与Memory的evaluation/research_note，再查已完成评测的manifest和metrics。按同档位、track、协议、白黑、种子和样本量比较模型/Harness/skills组合；不同难度的胜局数不能直接排序，完整面板与单档证据分别看。更新本轮选择及证据路径，不把上一轮推荐当作永久最佳。

有新的、更好可比证据时优先考虑新候选；证据不足、波动较大或有特定研究目的时，也可选择其他起点并记录理由，不要求机械追逐最高一次分数。working、最新编号和某张历史身份卡都不具有永久优先权。

[起点身份卡](references/research-startpoints.json)只保存2026-10-26已核对的历史示例，不是完整候选列表或实时排行榜：r000007/j000022曾得L2 16胜3和1负，r000010/j000027曾得14胜5和1负，两者模型SHA相同，差别在Harness/skills。r7可作为参考或回退候选；有更佳证据时不因该卡继续固定选r7。采用任一组合前检查路径、manifest、metrics、source及skills SHA。

## 在注册workspace中采用历史组合

Controller默认按最新working创建workspace，这只是初始副本，不是研究限制。Meta可在当前注册workspace恢复本轮选定候选的src/llmgo/harness目录和skills.json（含包内skills卡），去掉所选版本不存在的后续Harness文件；保留context.json、提案/完成交接机制、Memory与本轮目录。不要改Controller状态中的working/best或历史候选，不在冻结快照上直接编辑。

在proposal的model_path显式指定身份卡中的HF导出路径；hypothesis及evidence_refs说明实际恢复来源、原评测和后续改变。Controller的parent_id可能仍记录本轮注册工作流的父节点，另用研究证据说明逻辑研究起点，不篡改历史谱系。Harness、skills、模型分开登记；每次比较说明哪一项改变。

**训练模块独立处理。** 恢复历史Harness/skills不意味着恢复当年的训练实现。旧PPO存在已证实的非法正优势问题，先读[rsi-ppo-audit](../rsi-ppo-audit/SKILL.md)；从live采用已修训练文件，或自主修改并验证新的训练代码。注意旧recipe显式lc0_td0_centered会覆盖新默认。保留已修的导出、缓存释放和同步诊断。

## 权重起跑和断点恢复

从选定候选导出的HF模型开始新RL实验：设置model_path为所选HF目录，重新建立优化器、reference和训练身份；默认实现的reference以本次输入模型初始化，应记录这种基线改变。不要把HF目录当作完整PPO checkpoint。

继续旧PPO优化器/RNG/dataloader状态要求代码、reward、课程、配置和卡数身份匹配；改优势目标后不应直接拼接旧优化器断点。用户未要求本技能固定训练步数、学习率或必须使用某一算法，可按证据决定RL方案，仍禁止SFT。

研究当前处于操作员暂停时只准备选择和交接，不自行恢复Meta或提交GPU；恢复须按用户明确指令。已有维护验收不会因起点建议而被重启或替换。
