# 分段訓練與持續驗證

預設每 5 個 **optimizer update** 保存原生 checkpoint，退出訓練子程序並匯出 HF 模型，再完成 Fairy L2 / L3 各 20 局（各 10 個黑白配對）。初始 step 0 也用同樣協議。Lichess dev 不再是此流程每 5 step 的決策訊號；資料隔離照舊。初始 Qwen3.5-4B、Chess-R1 prompt 和 non-thinking 不變，預設仍用篩選資料，可選完整 Lichess 訓練池。

一個 HIGH job 保留 allocation，按「訓練 → 導出 → 驗證 → Meta 審查」循環，階段間釋放子程序。Controller 與外部 Executor 分開，Meta 思考時 Executor 仍能運作。排程器 GPU 查詢間隔 600 秒；審查請求經共享磁碟讀取，不受慢輪詢影響。同時最多一個活動或提交不確定的 HIGH job；遇到其他自己的 HIGH job 就等待，不停止任何其他任務。

Worker 在 `scaling/reviews/step-N/request.json` 發佈綁定 checkpoint 與評測 SHA 的請求。Controller 建立審查研究副本，Meta 在 `proposal.json.training_review` 寫：

```json
{"request_id":"請求中的ID","action":"continue","reason":"基於曲線與敗局的理由","selected_step":5}
```

Meta 使用該輪 `finish_research.py` 交接，Controller 校驗才提交不可覆寫的 decision；Worker 寫 ack。相同決策可重放，不會再跑已完成訓練或評測。

| 決策 | 結果 |
|---|---|
| continue | 同一配方、源碼、actor、optimizer、RNG、dataloader 接續下一段；不可修改程式／配方，達配方總步數後不可繼續 |
| replan | 選一個已評測父模型，保留審查副本的修改，下一輪新提案、新實驗與 optimizer；可改 reward、超參、資料、RL 實作 |
| finish | 選一個已評測 checkpoint（可以早於最後一步），campaign COMPLETE，交付模型與證據；不自動執行 DeepSeek 最終測試 |

每次模型身份均核對權重與配置 SHA。原生 actor/optimizer/extra_state shards、data.pt 與 checkpoint 收據不完整就拒絕接續。審查超時預設 900 秒，GPU job 釋放、保留 checkpoint 與 HF 模型，Controller NEEDS_INPUT；恢復連線後需顯式 resume 和新提案，此時不聲稱自動恢復原 optimizer。異常退出也需先核對提交身份與 checkpoint，不盲目再提交。

## 曲線與交付

每項實驗在 `/user/lyh4170/LLMGo!/var/runs/<campaign>-<job>/`：

| 路徑 | 內容 |
|---|---|
| `grpo/recipe.json`、`identity.json` | 凍結配方、訓練源碼身份 |
| `grpo/checkpoints/receipt-N.json`、`native-checkpoints/global_step_N/` | 原生完整續訓狀態 |
| `scaling/models/step-N/` 及旁邊的 `.submission.json` | 每個 checkpoint 的 HF 匯出與模型身份 |
| `scaling/evaluations/step-N/L2/`、`L3/` | metrics.json、每盤 games/*/events.jsonl、模型／對手／協議身份 |
| `scaling/reviews/step-N/` | request、decision、ack |
| `scaling/curve.json`、`best.json` | 全部原始分數、L2/L3 分數、描述性區間、觀測最優包絡、lineage step 與階段 GPU 時數 |
| `scaling/final.json`、根目錄 `submission.json` | 選定模型、所選 step、父系與證據路徑 |

網頁 GRPO 區顯示曲線與 checkpoint 表，可讀完整棋局事件。曲線允許下降；observed-best 只是開發驗證最高觀測值，不是能力單調提升的證明。重複自適應評測有選擇偏差，區間只作描述。改方案後繼承父模型的 lineage step 與 GPU 時數；時數只包含記錄的訓練、匯出、評測及審查階段，不含排隊、啟動與失敗階段耗時。

每輪另交付選樣程式／查詢、選中 ID、分類統計與敗局假設，見 [資料說明](lichess-data.md)。選定最終模型後由使用者另行執行 DeepSeek non-thinking 測試。測試 provider、請求模型名與回應模型名都應記錄，不能把供應商別名未經核實地當作同一模型。
