---
name: rsi-memory
description: 從持久研究記憶、學生成長曲線與完整敗局形成下一輪 RL 假設。
---

先讀 context.json、campaign 的 handoff、最近 decision 與 training_review_decision。working 是下一輪父模型；best 是開發驗證的觀測最優；final_checkpoint 是已選定交付，三者可能不同。用證據路徑與 SHA 引用結果，保留失敗與下降點，不改舊 manifest 或分數。

每輪必須閱讀學生敗局每一步的 events.jsonl：FEN、合法 SAN、雙方走法、模型原文、解析、fallback、終局。用失敗模式提出可驗證的資料／reward／超參假設。交付選樣程式或 selection.json、selected_ids.txt、selected_positions.txt、stats.json；沿用預設資料也可用 report_training_data.py 記錄。完整 Lichess 池可選，不必使用。無需 random seed 或算力預算比較。

每 5 step 的曲線是自適應開發驗證，不是獨立測試。保留 raw score、observed-best、checkpoint 身份、lineage step 與記錄階段 GPU 時數。跨改方案仍連到父 checkpoint，不把新實驗 step 0 當成重新訓練的進步。DeepSeek non-thinking 留給使用者最後測試。
