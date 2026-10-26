# 给Meta的PPO观察技能与历史研究起点

本次按用户要求提供三项持久指引：PPO状态与故障取证技能、允许选择历史较佳组合、明确旧训练实现存在问题且可修改。研究仍保持操作员暂停，不自动回滚working、不恢复Meta、不替换正在运行的维护验收。

## 新增技能

- `.agents/skills/rsi-ppo-audit/SKILL.md`：从manifest/identity到原始response、LC0值、更新前优势、实际优化、权重同步、导出与正式对局建立证据链。具体案例放在references；不会把整个训练日志塞进prompt。
- `.agents/skills/rsi-startpoint/SKILL.md`：核对模型、Harness和skills的组合身份，在注册workspace从历史候选分支，区分HF模型起跑与完整PPO状态恢复。
- 原有`rsi-training`和`rsi-memory`增加训练源码可修改的明确授权、故障纠正及上述技能入口。

两项新技能也镜像到`/home/lif/skills_rsi`。远端发布到live仓库与实际CODEX_HOME的skills目录，并在AGENTS.md写带唯一标记的短指引；持久Memory的research_note和最新handoff/operator_notes同样记录入口。这样后续workspace即使继承旧候选，也可按绝对路径读取新技能，无须改变历史快照。

## 可选较佳起点

| L2二十局玩家评测 | 16胜组合 | 最新14胜组合 |
|---|---|---|
| candidate / job | r000007 / j000022 | r000010 / j000027 |
| 胜/和/负 | 16/3/1 | 14/5/1 |
| 胜率 | 80% | 70% |
| 得分率 | 87.5% | 82.5% |

两者实际模型SHA相同：`8df3af3df90383bfc061155375ea4053ec05807dbf1b4bfb624e16ad3df63292`。因此恢复16胜组合时要同时选择其Harness和skills，仅更换model_path不会改变此差异。

```
模型：/user/lyh4170/rsi_checkpoints/rsi-r7-stability4-keyfix/model
Harness来源：/user/lyh4170/rsi_campaigns/chess-autonomous-v2/candidates/r000007/source/src/llmgo/harness
skills：/user/lyh4170/rsi_campaigns/chess-autonomous-v2/candidates/r000007/skills.json
原评测：/user/lyh4170/rsi_campaigns/chess-autonomous-v2/jobs/j000022/output/metrics.json
```

source、skills、model和原评测文件SHA记录在`rsi-startpoint/references/research-startpoints.json`。这是已测L2候选间可用的较佳研究起点，不是完整L1–L8全局best或统计显著性结论。用户允许选择，并非要求每轮强制回滚；Meta可根据证据选择其他起点。

Controller默认从最近working拷贝workspace。Meta可在该可编辑副本中恢复历史Harness/skills，在proposal显式填写model_path并记录恢复来源；不修改冻结candidate、原成绩或Controller状态。若从HF导出继续新RL，优化器和reference按新实验重新建立；代码/目标已经变化时，不能把旧优化器状态冒充同一实验续训。

## 旧训练代码的问题与修改权限

已证实的旧漏洞是：协议失败先算target−V_before，再统一减全批均值，可能导致非法输出获得正优势。保存的真实更新前张量证明28批中有21例；修补回放降至0。不能因此断言全部棋力下降均由此造成。

用户明确允许修改训练Python代码、RL算法、reward/advantage、采样、优化器和配置；训练实现不是受保护裁判。修改仍限于RL与skills，禁止SFT，保留正式裁判/协议和冻结证据。恢复r7组合时，不盲目恢复旧训练模块；采用live的修补或自行实现经过验证的新方案。旧recipe显式指定centered时也要处理，不能仅依赖新默认。

发布通知、Meta会话已读取、后续任务实际采用是三个证据等级。本次保持暂停，只验证通知和技能已发布，不启动Meta来证明读取。恢复后仍应从新workspace及Worker identity/实际优势日志核对采用情况。
