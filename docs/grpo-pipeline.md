# Chess-R1 GRPO pipeline

## 來源與資料

本次對照的遠端 release 是 `/user/lyh4170/rsi_experiments/chess-r1-grpo-20261029/releases/a267c6e88a27e94c`。`scripts/train_chess_r1.py` 在同一 GPU job 啟動 CPU JAX 270M oracle，使用 `llmgo.training.chess_r1_verl.ChessR1TaskRunner` 呼叫 verl 0.8 的 `run_ppo`，其 advantage estimator 是 `grpo`。`run_ppo` 是 verl 的訓練器入口名稱。

學生原始權重在 `/user/liyaohu4170/models/Qwen3.5-4B`（來源只讀），本專案核驗副本在 `assets/models/Qwen3.5-4B`。270M 權重原址是 `/user/lyh4170/models/chess-r1/searchless-270M-relayed/extracted/270M`，副本在 `assets/models/searchless-270M/270M`。上游程式在 `assets/vendor/Chess-R1-830c0597`。資料原址是 Chess-R1 實驗的 `data/boxed-nonthinking-v2`，副本在 `assets/datasets/chess-r1/boxed-nonthinking-v2`。

預設 prompt 要求唯一合法 SAN 步，以 `\\boxed{Nf3}` 作答，關閉 thinking。response 上限 16,384 tokens；temperature 1、top-p 0.95、top-k 20、presence penalty 1.5、repetition penalty 1。每題 8 個回答，GRPO 同題分組；過長的組整組拒收並補新批次。reward = 270M 的 Q(s,a) + 0.1 格式 + 0.1 英文；Lichess 標準答案僅作驗證指標。

## 執行與產物

`configs/grpo/default.json` 是新任務模板。`scripts/run_grpo.py` 凍結配方，呼叫 `scripts/chess_r1_gpu_entry.py` 和訓練入口，最後核對最終 checkpoint 收據、合併 FSDP shards、修正 Transformers 5.4 權重命名，輸出完整 HF 模型和 `submission.json`。原生 checkpoint 含 actor、optimizer、RNG 和 dataloader；續訓只認同一實驗、同一配方及完整收據。改 reward、模型或配方時建立新實驗。

標準路徑：`var/runs/<實驗ID>/grpo/{recipe.json,identity.json,progress.json,result.json,logs/train.log,oracle.log,updates/,tables/,sampling/,checkpoints/,native-checkpoints/,model/}`；Worker 收據在實驗根 `var/runs/<實驗ID>/{result.json,submission.json,research.log}`。兩種 `result.json` 位於不同目錄。導出前核對 `grpo/checkpoints/receipt-<最終step>.json` 和各 shard 大小。

當前 W&B 預設 offline；更新 JSON、分組表和原生日誌是權威證據。要用遠端 W&B 時需另立新實驗和 run ID。

## 評測與 Meta-Agent

正式評測固定 Fairy L1–L8，默認 Chess-R1 非思考 prompt。基線每檔 10 局，後續每檔 20 局。`ChessEpisode` 每手寫 `events.jsonl`，含初始局面、模型消息和原文、學生步、對手步、forced move、fallback 和終局。Meta-Agent 可在候選根目錄建立 `harness.py`，提供 `build_messages(fen, legal_san, skills, default_messages)`，並寫 `skills.json`。候選 prompt builder 在隔離子行程執行；正式棋步仍由 Qwen 輸出，裁判和固定對手不可修改。每輪研究應讀輸棋的全部步驟。

程序和網頁從 `var/runs` 讀日誌；活動索引在 `var/campaigns`。舊 PPO、LC0、舊 Harness 和舊 Memory 網頁面板已退出新 pipeline。
