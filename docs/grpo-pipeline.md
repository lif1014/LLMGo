# Chess-R1 GRPO pipeline

## 來源與資料

學生原始權重在 `/user/liyaohu4170/models/Qwen3.5-4B`（來源只讀），本專案核驗副本在 `assets/models/Qwen3.5-4B`。270M 權重原址是 `/user/lyh4170/models/chess-r1/searchless-270M-relayed/extracted/270M`，副本在 `assets/models/searchless-270M/270M`。上游程式在 `assets/vendor/Chess-R1-830c0597`。資料原址是 Chess-R1 實驗的 `data/boxed-nonthinking-v2`，副本在 `assets/datasets/chess-r1/boxed-nonthinking-v2`。

預設 prompt 要求唯一合法 SAN 步，以 `\\boxed{Nf3}` 作答，關閉 thinking。response 上限 16,384 tokens；temperature 1、top-p 0.95、top-k 20、presence penalty 1.5、repetition penalty 1。每題 8 個回答，GRPO 同題分組；過長的組整組拒收並補新批次。reward = 270M 的 Q(s,a) + 0.1 格式 + 0.1 英文；Lichess 標準答案僅作驗證指標。

## 執行與產物

`configs/grpo/default.json` 是新任務模板。提案用 `scripts/run_grpo.py` 登記標準訓練；受保護 Worker 的 `rsi/training_loop.py` 凍結配方，每 5 step 呼叫 GPU 入口、核對 checkpoint、合併 FSDP shards 並修正 Transformers 5.4 權重命名，再驗證與等待 Meta 決策。原生 checkpoint 含 actor、optimizer、RNG 和 dataloader；續訓只認同一實驗、同一配方及完整收據。改 reward、模型或配方時建立新實驗。

訓練路徑：`var/runs/<實驗ID>/grpo/{recipe.json,identity.json,progress.json,result.json,logs/train.log,oracle.log,updates/,tables/,sampling/,checkpoints/,native-checkpoints/}`。分段匯出在 `scaling/models/step-N/`，驗證與決策在 `scaling/evaluations/`、`scaling/reviews/`；Worker 收據在實驗根的 result.json、submission.json。兩種 result.json 位於不同目錄。導出前核對對應 step 收據與各 shard 大小。舊的非分段配方仍可使用 grpo/model/。

當前 W&B 預設 offline；更新 JSON、分組表和原生日誌是權威證據。要用遠端 W&B 時需另立新實驗和 run ID。

完整 Lichess 訓練池是額外選項，預設配方仍使用現有篩選資料。Meta-Agent 可自行選樣或沿用預設，並修改超參、reward 等 RL 設計；初始學生 checkpoint、prompt 與 non-thinking 模式固定。每輪資料選擇需要程式／查詢、ID 清單、分類統計與基於敗局的假設，不要求 random seed 或算力預算比較。詳見 [資料使用與轉檔](lichess-data.md)。

## 評測與 Meta-Agent

開發驗證固定 Fairy L2（skill -5、depth 5、0.10 秒）、L2.5（skill -3、depth 5、0.125 秒）、L3（skill -1、depth 5、0.15 秒）。L2/L3 基線每檔 20 局，候選每檔 20 局；L2.5 可作中間難度診斷，完整 L2/L3 才計算可比較的錨點分數。`ChessEpisode` 每手寫 `events.jsonl`，含初始局面、模型消息和原文、學生步、對手步、forced move、fallback 和終局。Phase 1 只允許 RL 研究，固定 Chess-R1 非思考 prompt；`skills.json` 為空，不能新增 `harness.py`。每輪研究應讀輸棋的全部步驟。具體規則與 Phase 2 設計見 [Meta-Agent 研究範圍](phase1-rl-scope.md)。

最終測試保留給選定的成果，使用 `scripts/run_final_test.py` 讓 Qwen 與 DeepSeek Flash non-thinking 對戰 10 局，白黑各五局。DeepSeek 走法沿用 `run_deepseek_loose.py` 的無歧義 boxed SAN 解析和代理設定；Qwen 仍使用嚴格 Chess-R1 格式。失效回覆使用有記錄的隨機合法步，雙方原始回覆、每步局面和終局寫入 `var/runs/<實驗ID>/final-test`。不要以此測試結果調整 Meta-Agent 的研究提案。

程序和網頁從 `var/runs` 讀日誌；活動索引在 `var/campaigns`。舊 PPO、LC0、舊 Harness 和舊 Memory 網頁面板已退出新 pipeline。

預設每 5 個 optimizer step 完成 L2/L3 各 20 局驗證，由 Meta-Agent 選擇 continue / replan / finish，保留中間 checkpoint、原始分數與 observed-best 曲線。Controller 使用外部 Executor，GPU 排程查詢間隔 600 秒。流程與輸出路徑見 [分段訓練說明](training-reviews.md)。
