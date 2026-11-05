# 分段訓練與持續驗證

預設每 5 個 **GRPO step（verl `global_step`）** 保存原生 checkpoint，退出訓練子程序並匯出 HF 模型，再完成 Fairy L2 / L3 各 40 局（各 20 個黑白配對）。每個 GRPO step 包含一批 rollout，內部可能有多次 optimizer minibatch 更新，不能把兩者混算。初始 step 0 也用同樣協議。Lichess dev 不再是此流程每 5 step 的決策訊號；資料隔離照舊。初始 Qwen3.5-4B、Chess-R1 prompt 和 non-thinking 不變，預設仍用篩選資料，可選完整 Lichess 訓練池。

L2/L3 共用同一個 80 局佇列和 vLLM 副本池，每卡一個 TP=1 副本，max_num_seqs=20，每卡最多 20 局並行。8 卡時每卡約 10 局，1 卡最多並行 20 局；全部副本就緒後派發。兩檔各 40 局、白黑各 20 局，指標與逐手證據分開保存。完整 80 局及身份校驗完成後才叫醒 Agent。舊 10／20 局報告仍按原樣本數解析，不改寫舊成績；比較曲線須標示局數改變，不能把新舊面板當成等樣本測量。

首次基線也合併為一個 evaluate_panel 任務；單獨提交某一難度為 40 局。僅改評測排程與副本容量，沒有修改學生 prompt、reward 或對手強度。既有凍結任務沿用原版本；新版只對升級後建立的 manifest 生效。

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

每次模型身份均核對權重與配置 SHA。原生 actor/optimizer/extra_state shards、data.pt 與 checkpoint 收據不完整就拒絕接續。審查不設固定期限：Worker 在安全 checkpoint 保留 allocation，等待 Agent 提交有效決策，不因 15 分鐘或任何舊 review_timeout_seconds 設定自動退出。排程器輪詢間隔仍為 600 秒。異常退出也需先核對提交身份與 checkpoint，不盲目再提交。

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

評測任務表中 L2／L3 各佔一行，分別顯示有效完成局、勝／和／負、勝率與得分率；每個 checkpoint 的表格也列出兩檔各自的勝／和／負、勝率和得分率。共享 80 局排程不合併兩檔勝率。未完成整批校驗時，逐局統計標為暫時統計；完整 L2／L3 收集後才標為正式成績。

獨立的「GRPO · 訓練指標」區塊以 `grpo/metrics.jsonl`、`latest.json` 和已驗證更新收據顯示每項 RL 實驗。曲線每15秒增量讀取，支援 Reward/Q、Reference KL、entropy、合法／格式比例、回答 tokens、梯度范數、policy loss、零優勢組比例及任意額外標量；逐步表格每頁25步。完成步數依 `update/mean_abs_logprob_change` 或完整更新收據判斷，`progress.step` 僅為當前階段標記。Entropy coefficient=0 仍可量測策略 entropy。續訓時同一步採最新記錄，回退到 checkpoint 後不將舊分支的未保留更新冒充新分支。網頁透過既有 artifact API 讀取，舊 Web 行程無須重啟，訓練任務與凍結源碼不變。

每輪另交付選樣程式／查詢、選中 ID、分類統計與敗局假設，見 [資料說明](lichess-data.md)。選定最終模型後由使用者另行執行 DeepSeek non-thinking 測試。測試 provider、請求模型名與回應模型名都應記錄，不能把供應商別名未經核實地當作同一模型。

## 歷史審查過期的恢復

只有已結束且收集完成的舊 Worker、NEEDS_INPUT 審查邊界、匹配原 request_id 的 Agent 完成交接、未改動的提案／源碼哈希、有效 checkpoint 和評測身份全部成立，operator 才能用 `scripts/rsi.py recover-training-review <campaign> --job <job-id>` 接回已完成的 replan。此操作保留 expired.json、舊任務和成績，另存恢復回執與研究源碼，讓下一輪正常提交新實驗；不偽造舊 Worker 的 decision／ack，也不重做已完成研究。操作需先取得 Controller lock，原 Controller 已退出時可在另一個自己的 0 GPU 容器接管，保留現有 Executor 與網頁。

棋局規則更新：後續 L2/L3 驗證與 DeepSeek 獨立測試啟用五次重複局面自動和棋，上限為 300 個半步（150 回合）。逼和、常見子力不足、五十步自動申請判和的原設定保留；三次重複局面仍不啟用。達步數上限前先判定將死等正常終局，舊凍結任務及歷史結果保留原規則。

L2/L3 正式學生推論請求預設不設客戶端時間期限（`timeout=None`），等待完整回覆或實際連線錯誤；移除舊 600 秒限制。GPU 排程輪詢仍每 600 秒一次，兩者獨立。既有凍結任務保持原設定，後續需以含此修復的新受保護快照建立任務。
