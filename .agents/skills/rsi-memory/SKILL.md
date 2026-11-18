---
name: rsi-memory
description: 從持久研究記憶、成長曲線與完整敗局形成對應 Phase 的下一輪研究假設。
---

先讀 context.json；Phase 2 讀 docs/phase2-prompt-research.md，Phase 1 讀 docs/meta-research-operations.md、docs/clean-grpo-pipeline.md；若存在 OPERATOR_GUIDANCE.md 也先讀。再讀 campaign 的 handoff、最近 decision 與 training_review_decision。working 是下一輪起點；best 是開發驗證的觀測最優；final_checkpoint 是已選定交付，三者可能不同。用證據路徑與 SHA 引用結果，保留失敗與下降點，不改舊 manifest 或分數。

Phase 2（phase2_prompt_only）讀同樣的逐手敗局證據，但提出 harness／skills 假設，不進行 RL，也不要求資料選樣交付或每5step審查。按 rsi-phase2 技能交付完整候選與開發驗證；final-candidate.json 才是模型與提示的完整交付。以下資料選樣和訓練曲線要求適用於 Phase 1。

每輪必須閱讀學生敗局每一步的 events.jsonl：FEN、合法 SAN、雙方走法、模型原文、解析、fallback、終局。Phase 1 RL 研究用失敗模式提出可驗證的資料／reward／超參假設。交付選樣程式或 selection.json、selected_ids.txt、selected_positions.txt、stats.json；新 RL 研究只能從完整 Lichess 訓練池自行選樣，沒有預選子集。後續沿用本研究已選資料也可用 report_training_data.py 記錄；不得讀取其他研究的選樣。選樣助手可選，也可自行寫查詢／程式。無需 random seed 或算力預算比較。Phase 2 不訓練、不要求資料選樣，依該階段的 harness／skills 研究合約交付。

每 5 step 的曲線是自適應開發驗證，不是獨立測試。保留 raw score、observed-best、checkpoint 身份、lineage step 與記錄階段 GPU 時數。跨改方案仍連到父 checkpoint，不把新實驗 step 0 當成重新訓練的進步。DeepSeek non-thinking 留給使用者最後測試。
