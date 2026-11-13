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


可選資料選樣工具：`scripts/select_lichess_data.py` 已支援分類配比與難度範圍，
例如 `--mix fork=1 pin=2 --num-puzzles 30000 --rating-min 800 --rating-max 1600`。
先用 `--list-tags` 查真實分類、`--preview` 查配額是否可滿足；複合分類／各組難度用 `--spec` JSON。
用 `/user/lyh4170/venvs/llmgo-lichess-data-20261103/bin/python` 執行 CPU 選樣。
比例以 puzzle 計算，同題只加入一次，再保留其在訓練池中的全部局面；每輪保留交付檔。
詳見 `docs/lichess-data.md`。**不一定需要用這個腳本**：可以保留預設資料，也可以自行寫選樣程式。
是否使用、分類配比及難度由你根據本研究敗局自主決定，不是強制課程。
只修改實驗配方的 train_file，不改預設資料、固定學生 prompt、non-thinking 或正式評測協議。


2026-11-13 現行啟動規則：Meta 使用 Qwen3.8-27B thinking，128K context／64K最大輸出／56K compaction，
學生仍是固定 Qwen3.5-4B non-thinking。所有新任務（含CPU維護、Meta、學生訓練／評測）
只可在 model-new、model-claw-tool、model-codeagent2 使用 PREEMPTABLE；禁止 model-autosft，
此規則取代本文任何舊 HIGH／model-autosft 指示。新研究由使用者明確啟動，不恢復舊研究。
詳見 [Qwen3.8 操作說明](qwen38-meta.md)。
訓練審查交接必須包含當前context的 training_review.request_id、action、reason、selected_step；
執行 check_research_proposal.py 檢查後才交接，格式錯誤留在本回合修正。
