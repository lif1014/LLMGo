# 本研究工作區的操作入口

適用於本工作區內的 Meta-Agent 研究。先讀以下檔案，再執行研究工具：

1. `context.json`，以及存在時的 `OPERATOR_GUIDANCE.md`。
2. `docs/meta-research-operations.md` 與 `docs/clean-grpo-pipeline.md`。
3. `.agents/skills/rsi-memory/SKILL.md`、`rsi-research-operations/SKILL.md`。

按 context 中的實際產物路徑讀證據；不要猜檔名、壓縮格式或工具名稱。
先核對正式勝和負與逐局收據，完整讀雙方走子，保留分析程式與可驗證假設。
本階段先提交並執行L2導向RL訓練，不先停在L2.5診斷。直接讀context.research_evidence提供的核驗報告，
不自行猜路徑、解析格式或安裝依賴。正式metrics是分數唯一來源，評測協議固定，不能改成4步。
自行登記已授權的RL／開發評測任務並用本次 finish_command 交接；
Meta 推理卡不執行學生訓練或正式 GPU 評測。只有缺少使用者獨有資訊／權限才 needs_input。
禁止 SFT；學生初始模型、prompt、non-thinking 固定。RL 策略由你自行研究。
只讀本 campaign 的研究記憶及產物，不讀其他研究的結果、checkpoint 或配方。
