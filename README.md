# LLMGo! · Chess-R1 GRPO

遠端專案根目錄是 `/user/lyh4170/LLMGo!`。以 Qwen3.5-4B 為初始學生，使用 Chess-R1 非思考 prompt、Lichess 棋題、searchless 270M reward model 與 verl GRPO 訓練。Phase 1 Meta-Agent 只研究 RL，固定 prompt、空 skills、無 harness，禁止 SFT；Phase 2 是獨立的固定權重研究，可選 base 或 Phase 1 checkpoint，自由修改 harness、追加提示和 skills；禁止訓練的規則只透過 Meta-Agent prompt 約定，違反判 0 分。開發驗證是 Qwen 對 Fairy-Stockfish L2、L2.5、L3；保留的最終測試是 Qwen 對 DeepSeek Flash non-thinking。詳見 [研究階段規則](docs/phase1-rl-scope.md) 和 [訓練目錄逐檔審核](docs/training-module-audit.md)。

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

GRPO 預設 KL loss 係數為 `0.05`，entropy coefficient 為 `0`；實際 verl 設定由 `src/llmgo/training/chess_r1_config.py` 產生。Entropy 診斷仍保留。

訓練回答上限預設 16,384 tokens，可在新配方設定為 512–16,384 的整數；生成長度、上下文與整組截斷過濾共用相同設定。共用程式已整合研究中發現的首次啟動、權重索引與補採樣修復，採用方式及驗證見 [維護紀錄](docs/grpo-maintenance-20261104.md)。既有 campaign／任務使用原先凍結版本，發布 live 檔案不會自動替換其快照。

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

Meta-Agent 可提交單檔 L2、L2.5、L3 驗證，或以 `--full-panel` 同時提交 L2/L3。只有 L2 和 L3 兩檔齊備時才更新可比較的驗證分數。選定成果後，在獨立的 1 GPU job 執行 `scripts/run_final_test.py`，預設 10 局、白黑各 5 局，對手為 DeepSeek Flash non-thinking；所有原始走棋與回覆存於 `var/runs/<實驗ID>/final-test/`。`run_deepseek_loose.py` 是 DeepSeek 對 Fairy 的參考流程，不是 Qwen 對 DeepSeek 的最終測試。密鑰在 `/user/lyh4170/.config/llmgo/deepseek-api-key`，只對 DeepSeek API 使用內部白名單代理。

最終測試預設十局；`--games 20 --parallel-games 20` 可在單卡並行二十局、白黑各十局。使用同步排程和 PIECEWISE CUDA Graph，服務就緒後直接開始正式對局。任何基礎設施失敗保留原始記錄，不發布完整成績或算作棋局敗局。API provider 與模型名稱須先核實，詳見 [RSI 操作](docs/rsi-operations.md)。

## 研究起點

新 campaign 從 Qwen3.5-4B、固定 Chess-R1 非思考 prompt 開始，先做 L2、L3 各 40 局基線，再提出 Phase 1 RL 研究。現有 campaign 恢復持久狀態；已啟動的 job 使用提交時的凍結快照，live 倉庫更新不會熱改已有實驗。

預設每 5 個 GRPO step（verl `global_step`，每步含多次 optimizer minibatch 更新）完成 L2/L3 各 40 局驗證，由 Meta-Agent 選擇 continue / replan / finish，保留中間 checkpoint、原始分數與 observed-best 曲線。Controller 使用外部 Executor，GPU 排程查詢間隔 600 秒。流程與輸出路徑見 [分段訓練說明](docs/training-reviews.md)。

L2/L3 基線和驗證共用 80 局佇列，每檔 40 局、白黑各 20 局。每卡一個 TP=1 vLLM，max_num_seqs=20，每副本最多 20 局並行；8 卡可同時進行 80 局（每卡約 10 局），1 卡時最多並行 20 局，其餘排隊。兩檔各自保存指標和完整棋譜。新受保護驗證版本固定此政策，覆蓋相容 site 中的舊併發值；GRPO 訓練 rollout 384 seqs 和獨立 DeepSeek 測試不受此更改影響。

網頁評測表按 L2、L3 分行顯示勝／和／負、勝率與得分率；GRPO 的每個 checkpoint 表也分別列出兩檔成績。勝率為勝局／有效完成局，得分率為（勝局＋半數和局）／有效完成局。

「GRPO · 訓練指標」為獨立區塊，可選每次 RL 實驗，查看 Reward、270M Q、Reference KL、entropy、合法／格式比例、回答長度、梯度、policy loss 與零優勢組比例。每 15 秒從遠端指標檔增量讀取，支援任意標量曲線與逐步表格；沒有完成更新時顯示等待，不把採樣或準備階段的目標 step 當成已完成訓練。此顯示只讀檔案，不呼叫模型或查詢排程器。

審查沒有固定的 15 分鐘期限；每次 L2/L3 完整驗證後，Worker 在安全 checkpoint 等待 Meta-Agent 的有效決策。歷史過期交接可經 operator 驗證後恢復，詳見 [分段訓練說明](docs/training-reviews.md)。

DeepSeek 最終測試在 vLLM 服務就緒後直接開始正式對局，不額外執行 128-token／120 秒啟動生成檢查；正式學生與對手請求均不設客戶端逾時。既有排隊任務沿用原凍結程式，須新提交才採用更改。

棋局規則更新：後續 L2/L3 驗證與 DeepSeek 獨立測試啟用五次重複局面自動和棋，上限為 300 個半步（150 回合）。逼和、常見子力不足、五十步自動申請判和的原設定保留；三次重複局面仍不啟用。達步數上限前先判定將死等正常終局，舊凍結任務及歷史結果保留原規則。

## Phase 2：固定權重，研究 harness 與 skills

使用獨立的 `phase2_prompt_only` campaign。起點預設為 base Qwen3.5-4B，也可用 `init --checkpoint <Phase1-HF-checkpoint>` 指定 Phase 1 成果。每輪評測完成才喚醒 Agent，Agent 可繼續研究或交付目前／此前最佳候選。Phase 1 的 RL 流程仍為預設。

Meta-Agent 的完整 Phase 2 prompt 在 [phase2-meta-prompt.md](docs/phase2-meta-prompt.md)，專用技能在 [.agents/skills/rsi-phase2/SKILL.md](.agents/skills/rsi-phase2/SKILL.md)。禁止訓練、保持指定 checkpoint／non-thinking 等研究限制只寫在 prompt，不增加訓練行為偵測或命令攔截。harness 可自行載入任意 skills、輔助程式與資源，並取得完整走棋歷史。

網頁新增 Phase 2 候選區塊，分別呈現各版本 L2、L3 的勝／和／負與得分率，以及實際成績、觀測最佳曲線。最終 `final-candidate.json` 指向原 checkpoint 和完整凍結 harness／skills；DeepSeek 最終測試可用 `--final-candidate` 載入此組合。初始化、介面、停止與測試命令見 [Phase 2 操作](docs/phase2-prompt-research.md)。本分支只完成本機開發與測試，未部署遠端、未提交 GPU 任務。
