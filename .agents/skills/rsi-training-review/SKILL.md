---
name: rsi-training-review
description: 在每個已驗證的原生 checkpoint 上決定繼續訓練、改方案或交付模型。
---

先讀 context.json 的 training_review、curve 與 docs/training-reviews.md。請求包含 request_id、step、max_steps、可選 steps、checkpoint 收據和 L2/L3 metrics 引用。核對身份，讀每盤輸棋的 raw_events 所指 events.jsonl，逐手查看 FEN、雙方走法、模型原文、fallback 與敗因。訓練 reward 不代表棋力，最高觀測分包絡也不代表實際能力單調增加。

在本副本 proposal.json 寫 training_review 物件：request_id、action、reason、selected_step；用本輪指定的 finish_research.py 正常交接，status=ready。Controller 驗證後才把決策提交給 Worker，不直接寫受保護 request/decision 檔。

- continue：selected_step 是當前 step，保留凍結 recipe、程式、reward 與資料。Worker 恢復同一 actor、optimizer、RNG、dataloader，繼續下一個 5 step；達 max_steps 時不能 continue。
- replan：可選任一已完整驗證 checkpoint。寫明新假設並在此研究副本修改 RL 方案；Controller 保留副本作下一輪起點，從所選父模型開始新實驗、新 optimizer。下一輪仍需提交提案與選樣交付。
- finish：可選任一已完整驗證 checkpoint，包括初始 step 0。交付模型與完整曲線，由使用者另行做 DeepSeek non-thinking 最終測試。

審查期間當前 HIGH allocation 仍保留，但訓練／推理子程序已退出；不要提交第二個 GPU 任務。不改凍結源碼或裁判。Phase 1 固定初始 Qwen3.5-4B、prompt、non-thinking，skills.json=[]，不新增 harness.py；允許修改 RL 超參、reward、資料與課程。使用者已取消固定審查期限：Worker 保留 allocation 與完整 checkpoint，持續等待有效的 continue／replan／finish 決策，不因經過 900 秒或 15 分鐘自動過期。排程輪詢仍由 Executor 每 600 秒處理，與 Agent 審查等待分開。歷史已過期請求不能改寫，operator 可在 Worker 結束且交接身份核驗後恢復已完成的 replan。
