# LLMGo! · Chess-R1 GRPO

遠端專案根目錄是 `/user/lyh4170/LLMGo!`。以 Qwen3.5-4B 為初始學生，使用 Chess-R1 非思考 prompt、Lichess 棋題、searchless 270M reward model 與 verl GRPO 訓練。Meta-Agent 可研究 RL 和 skills，禁止 SFT。正式棋力評測是 Qwen 對 Fairy-Stockfish 的完整棋局，L1–L8 固定難度。

## 目錄

| 路徑 | 用途 |
| --- | --- |
| `assets/models/Qwen3.5-4B/` | 學生 HF 權重與 tokenizer |
| `assets/models/searchless-270M/270M/` | Chess-R1 動作價值 reward model |
| `assets/datasets/chess-r1/boxed-nonthinking-v2/` | GRPO 訓練與驗證 parquet |
| `assets/datasets/chess-r1/boxed-v2/` | 思考版歷史資料，供對照 |
| `assets/vendor/Chess-R1-830c0597/` | searchless Chess 上游原始碼 |
| `assets/engines/` | Fairy-Stockfish 與 NNUE |
| `configs/grpo/default.json` | GRPO 配方模板 |
| `scripts/run_grpo.py` | Worker 研究任務入口與模型導出 |
| `scripts/train_chess_r1.py` | verl GPU 訓練入口 |
| `src/llmgo/training/chess_r1_verl.py` | verl GRPO 實作 |
| `src/llmgo/training/chess_r1_reward.py` | 270M reward 呼叫與計分 |
| `scripts/run_deepseek_loose.py` | DeepSeek Flash 棋局評測 |
| `var/runs/<實驗ID>/` | 配方、訓練、評測、棋局事件與日誌 |
| `var/campaigns/<campaign>/` | Controller、Meta-Agent、Executor 狀態 |
| `var/workspaces/` | Meta-Agent 每輪可編輯副本 |
| `var/runtime/` | 暫存與服務日誌 |

大型權重、parquet、引擎、實驗輸出和憑證留在遠端，不進 Git；原始碼、配方、測試與文檔進 GitHub。逐檔 SHA-256 資產搬遷回執在遠端 `var/migrations/20261102/assets.json`。資料有 99,571 筆訓練、2,291 筆驗證、23,243 筆測試。詳細內容見 [GRPO pipeline](docs/grpo-pipeline.md)。

## 執行

在遠端自己的 GPU job 中，Worker 研究任務可使用：

```json
{"argv":["{python}","scripts/run_grpo.py","--output","{output}","--model","{model}"],"resources":{"gpu_count":8,"cpu":64,"memory_gib":512}}
```

配方凍結於 `var/runs/<實驗ID>/grpo/recipe.json`，原生 verl checkpoint、指標與 oracle 日誌在 `grpo/`，完整 HF 模型在 `grpo/model/`，供 Controller 收集的 `submission.json` 在實驗根。所有作業限 `beijing / model-autosft / NORMAL`，每項最多 8 GPU，devspace 0 GPU。

遠端已有專用 verl、teacher 與推理環境。首次新 campaign 用 `bash scripts/start_rsi.sh --campaign chess-grpo-v1 --web`；不沿用舊 campaign 的凍結快照。詳見 [RSI 操作](docs/rsi-operations.md)。網頁讀取 `var/runs/` 的指標和日誌，正式棋局的 `events.jsonl` 可逐手復盤，包含輸棋每一步、模型回覆和隨機後備。

DeepSeek 對照：在自有 CPU job 設 `CHESS_LEVEL=2` 或 `3` 執行 `scripts/run_deepseek_loose.py`。密鑰在 `/user/lyh4170/.config/llmgo/deepseek-api-key`；程式僅對 DeepSeek API 使用內部白名單代理；JSON 與日誌寫在 `var/runs/<實驗ID>/evaluations/`。

## 同步

遠端 `/user/lyh4170/LLMGo!` 是本次遷移的代碼準據。本機 `/home/lif/LLMGo!` 原有未提交改動需保留；在隔離工作樹完成、測試、提交和推送 GitHub 後，再以同一提交更新兩端。已啟動的 job 使用提交時的凍結快照；live 倉庫更新不會熱改已有實驗。
