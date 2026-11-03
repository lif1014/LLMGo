# LLMGo! · Chess-R1 GRPO

遠端專案根目錄是 `/user/lyh4170/LLMGo!`。以 Qwen3.5-4B 為初始學生，使用 Chess-R1 非思考 prompt、Lichess 棋題、searchless 270M reward model 與 verl GRPO 訓練。Phase 1 Meta-Agent 只研究 RL，固定 prompt、空 skills、無 harness，禁止 SFT；Phase 2 才開放 RL 和 harness 共同研究。開發驗證是 Qwen 對 Fairy-Stockfish L2、L2.5、L3；保留的最終測試是 Qwen 對 DeepSeek Flash non-thinking。詳見 [研究階段規則](docs/phase1-rl-scope.md) 和 [訓練目錄逐檔審核](docs/training-module-audit.md)。

## 目錄

| 路徑 | 用途 |
| --- | --- |
| `assets/models/Qwen3.5-4B/` | 學生 HF 權重與 tokenizer |
| `assets/models/searchless-270M/270M/` | Chess-R1 動作價值 reward model |
| `assets/datasets/chess-r1/boxed-nonthinking-v2/` | GRPO 訓練與驗證 parquet |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/` | 可選完整 Lichess 訓練池及分類索引，已排除現有 dev/test；使用前核對 verification.json |
| `assets/vendor/Chess-R1-830c0597/` | searchless Chess 上游原始碼 |
| `assets/engines/` | Fairy-Stockfish 與 NNUE |
| `configs/grpo/default.json` | GRPO 配方模板 |
| `scripts/run_grpo.py` | Worker 研究任務入口與模型導出 |
| `scripts/train_chess_r1.py` | verl GPU 訓練入口 |
| `src/llmgo/training/chess_r1_verl.py` | verl GRPO 實作 |
| `src/llmgo/training/chess_r1_reward.py` | 270M reward 呼叫與計分 |
| `scripts/run_deepseek_loose.py` | DeepSeek Flash 棋局評測 |
| `scripts/run_final_test.py` | 選定成果與 DeepSeek Flash 的保留最終測試 |
| `var/runs/<實驗ID>/` | 配方、訓練、評測、棋局事件與日誌 |
| `var/campaigns/<campaign>/` | Controller、Meta-Agent、Executor 狀態 |
| `var/workspaces/` | Meta-Agent 每輪可編輯副本 |
| `var/runtime/` | 暫存與服務日誌 |

大型權重、parquet、引擎、實驗輸出和憑證留在遠端，不進 Git；原始碼、配方、測試與文檔進 GitHub。逐檔 SHA-256 資產搬遷回執在遠端 `var/migrations/20261102/assets.json`。資料有 99,571 筆訓練、2,291 筆驗證、23,243 筆測試。詳細內容見 [GRPO pipeline](docs/grpo-pipeline.md)。

預設仍用現有篩選資料。Meta-Agent 可自行決定是否使用新的完整 Lichess 訓練池，以及如何篩選、混合或安排課程；也可修改超參、reward 等 RL 設計。初始 Qwen3.5-4B checkpoint、prompt 與 non-thinking 模式固定。每輪記錄選樣程式／查詢、ID 清單、分類統計及基於敗局的假設，不要求資料選樣 random seed 或算力預算比較。工具與範例見 [Lichess 資料使用](docs/lichess-data.md)。

## 執行

在遠端 **0 GPU devspace** 啟動研究（沿用現有網頁）：

```bash
cd '/user/lyh4170/LLMGo!'
export RSI_PYTHON=/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python
export CCTL_SERVER=https://cybertron.modelbest.co
export CCTL_NO_UPDATE_CHECK=1
bash scripts/start_rsi.sh --campaign chess-grpo-v1
```

此命令會檢查認證並背景啟動 Controller 和獨立 Executor；已有同名服務則沿用。若要在當前 devspace 額外啟動網頁，可加 `--web`，預設埠 7860。日誌分別在 `var/campaigns/chess-grpo-v1/controller.log`、`executor-service.log`，啟動檢查記錄在 `var/readiness`。`--status` 看狀態，`--logs` 跟隨日誌。

在遠端自己的 GPU job 中，Worker 研究任務可使用：

```json
{"argv":["{python}","scripts/run_grpo.py","--output","{output}","--model","{model}"],"resources":{"gpu_count":8,"cpu":64,"memory_gib":512}}
```

配方凍結於 `var/runs/<實驗ID>/grpo/recipe.json`，原生 verl checkpoint、指標與 oracle 日誌在 `grpo/`，每次驗證的 HF 模型在 `scaling/models/step-N/`，供 Controller 收集的 `submission.json` 在實驗根。作業限 `beijing / model-autosft`，預設 HIGH，同時最多一個活動或提交不確定的 HIGH 任務；每項最多 8 GPU，devspace 0 GPU。

遠端已有專用 verl、teacher 與推理環境。首次新 campaign 用 `bash scripts/start_rsi.sh --campaign chess-grpo-v1 --web`；不沿用舊 campaign 的凍結快照。詳見 [RSI 操作](docs/rsi-operations.md)。網頁讀取 `var/runs/` 的指標和日誌，驗證棋局的 `events.jsonl` 可逐手復盤，包含輸棋每一步、模型回覆和隨機後備。

Meta-Agent 可提交單檔 L2、L2.5、L3 驗證，或以 `--full-panel` 同時提交 L2/L3。只有 L2 和 L3 兩檔齊備時才更新可比較的驗證分數。選定成果後，在獨立的 1 GPU job 執行 `scripts/run_final_test.py`，固定 10 局、白黑各 5 局，對手為 DeepSeek Flash non-thinking；所有原始走棋與回覆存於 `var/runs/<實驗ID>/final-test/`。`run_deepseek_loose.py` 是 DeepSeek 對 Fairy 的參考流程，不是 Qwen 對 DeepSeek 的最終測試。密鑰在 `/user/lyh4170/.config/llmgo/deepseek-api-key`，只對 DeepSeek API 使用內部白名單代理。

## 研究起點

新 campaign 從 Qwen3.5-4B、固定 Chess-R1 非思考 prompt 開始，先做 L2、L3 各 20 局基線，再提出 Phase 1 RL 研究。現有 campaign 恢復持久狀態；已啟動的 job 使用提交時的凍結快照，live 倉庫更新不會熱改已有實驗。

預設每 5 個 optimizer step 完成 L2/L3 各 20 局驗證，由 Meta-Agent 選擇 continue / replan / finish，保留中間 checkpoint、原始分數與 observed-best 曲線。Controller 使用外部 Executor，GPU 排程查詢間隔 600 秒。流程與輸出路徑見 [分段訓練說明](docs/training-reviews.md)。

L2/L3 基線和驗證共用 40 局佇列，每卡一個 TP=1 vLLM 副本，所有副本就緒後開始派發。預設 8 卡時共 40 局同時進行，平均每卡 5 局；每副本 max_num_seqs=40，整個評測最多並行 40 局。兩個難度仍各自保存 20 局指標和完整棋譜。這是併發上限，不代表 GPU 計算利用率必定 100%。
