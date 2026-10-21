# LLMGo! 國際象棋 Harness

目前國際象棋的驗收玩家是選擇題框架：本地 vLLM 上的 Qwen3.5-4B 始終開啟
`enable_thinking=True`，每手只回答一道完整走法選擇題。Harness 自動檢查局面，
列出全部合法 UCI 走法，模型直接回答 `e2e4`。規則由
python-chess 執行；對手為 Fairy-Stockfish，預設第 1 檔（skill -9、depth 5、
每步 0.05 秒）。最多 **200 半步**，即 100 回合。中國象棋及五子棋保留相容入口。

## RSI：遠端強 Agent、RL 與條件技能

第一次部署請按[伺服器首次啟動步驟](docs/rsi-first-run.md)操作，涵蓋 Git／tar、必要依賴、常駐循環與網頁。

新增持久化 RSI Controller：強 Agent 在自己的 **0 GPU devspace** 讀 Memory、
準備 skills／RL 候選，再由 Controller 提交 GPU job。學生能力改進只允許 RL 和
條件 skills；正式下棋仍由 Qwen 自己作答，教師與 Lc0 不參與正式走子。

每批正式評測固定 **20 局（10 白、10 黑）**。同級證明改善才扶正候選；現任學生
連續兩批達到勝至少16局、得分率85%、黑白各60%，才升級對手。最低檔維持 L1
（-9／depth5／0.05秒），L4–L8 接到 skill20，L9 起增加 nodes 並移除 depth5
上限。沒有 random_rate 驗收門檻，診斷仍照常保存。

本機實作由使用者上傳到 `/user/lyh4170/LLMGo!`，實際研究不依賴開發機在線。
預設站點配置為 [`configs/rsi/site.beijing.json`](configs/rsi/site.beijing.json)，
只允許 beijing／model-autosft／NORMAL；所有遠端寫入位於 `/user/lyh4170`。
既有來源模型路徑保持唯讀。

在 devspace 準備好 Codex 程式／認證、cctl 及 Python 依賴，並核對站點配置後：

```bash
cd '/user/lyh4170/LLMGo!'
mkdir -p /user/lyh4170/tmp /user/lyh4170/cache
export TMPDIR=/user/lyh4170/tmp XDG_CACHE_HOME=/user/lyh4170/cache
python3 -m venv /user/lyh4170/venvs/rsi-control
/user/lyh4170/venvs/rsi-control/bin/pip install -r requirements-rsi-control.txt
/user/lyh4170/venvs/rsi-control/bin/python scripts/rsi.py doctor --platform --codex-auth
/user/lyh4170/venvs/rsi-control/bin/python scripts/rsi.py init chess-rsi-001
/user/lyh4170/venvs/rsi-control/bin/python scripts/rsi.py run chess-rsi-001
```

`run` 會提交實際 GPU 任務。Worker 需有相容的 Qwen3.5／CUDA／vLLM／PEFT 環境；
[`requirements-rsi-worker.txt`](requirements-rsi-worker.txt) 為額外依賴清單，尚非驗證過的
GPU lockfile。**訓練已切換為 `hf_ppo`：固定 Lc0 critic、單步 TD(0)、多輪 token
clip 更新；GRPO 及舊 verl GRPO 入口已移除。** 不訓練額外 value head，也沒有 SFT。
非終局使用 `A = V_Lc0(after) − V_Lc0(before)`，其中 `V=P(win)−P(lose)`；
前後均從學生執棋方看。終局使用真實勝負／和棋並將 continuation 歸零。
目前為課程局面的單步採樣，不是整盤 rollout 或 GAE；完整邊界和配置見
[PPO 操作說明](docs/rsi-operations.md#5-固定lc0-critic的ppo)。

唯一離線 critic 網路為已下載的 Lc0 BT4-it332，需連同 Lc0 程式上傳至站點指定的
自己的目錄。每批先 Qwen 採樣並移至 CPU，再啟動 Lc0 評分、退出，最後載回
Qwen 更新。PPO 的真實 Qwen GPU 訓練／vLLM 重載及棋力改善仍待遠端驗收。
2026-10-21 的遠端核對確認：只有 Codex 設定檔已複製；站點指定的 Codex 程式、
控制環境、RSI 源碼和 Lc0/BT4 尚未部署。該設定預設 `custom/deepseek-v4-pro`，
`meta_model=null` 會沿用它；Codex 是執行客戶端，不代表實際推理模型必然是 OpenAI 模型。
未驗證認證或真實模型請求。部署缺口與本次修復見[流程複核](docs/rsi-operations.md#9-整条pipeline复核与远端就绪状态)。

devspace 的只讀 RSI 網頁不需要 GPU：

```bash
LLMGO_RSI_ROOT=/user/lyh4170/rsi_campaigns PYTHONPATH=src \
  /user/lyh4170/venvs/rsi-control/bin/python -m uvicorn llmgo.server.app:app --host 0.0.0.0 --port 7860
```

開啟 **`/rsi`**，可查看強 Agent 的文字／工具起訖／用量／錯誤、Executor 的逐次輪詢、
cctl 命令與耗時，以及 Worker 階段、學生棋盤、Memory 和最近200條候選評測。
每3秒增量刷新；預設最新事件，每個列表最多80張卡片，可篩選、暫停、翻歷史。
完整交互保存在共享盤，大段原文點開才按64KiB分段讀取，不會灌入強 Agent 上下文。
舊 campaign 若缺顯示索引，可執行 `python3 scripts/rsi.py index-activity CAMPAIGN`；
詳細欄位、負擔控制與升級方式見 [網頁觀測說明](docs/rsi-observability.md)。
重啟使用同一 `run CAMPAIGN`；提交回應不確定時先對帳，避免重複任務。

任務生命週期已抽成專用 **Executor**：提交、輪詢、產物核驗和原始cctl日誌使用
獨立記錄，Meta-Agent只取得短摘要與證據路徑，等待期間不呼叫模型。預設單命令
仍可啟動；新campaign若在站點設定`executor_mode=external`，可另開
`python3 scripts/rsi.py executor run CAMPAIGN`常駐進程，Controller僅讀其本地報告。
網頁有獨立Executor區，原始記錄按需載入。參見 [Executor說明](docs/rsi-executor.md)。

詳細操作、proposal 範例、Memory、訓練和恢復見
[RSI 操作文件](docs/rsi-operations.md)。**之前成功提交的任務格式、Agent 提交方式，
以及此次新增／修改檔案清單**見 [任務提交與檔案說明](docs/rsi-submission-and-files.md)。
研究設計與尚待遠端驗收項目保留在 [plan.md](plan.md)。

目前已實作並完成本機流程測試；尚未在使用者上傳後的 GPU 環境完成 Qwen 訓練及
20 局前後比較，因此不能宣稱已提升 Elo。

## 伺服器啟動

沿用現有模型、引擎路徑和啟動腳本；不需將模型改成本地開發機的路徑。

```bash
bash scripts/serve_vllm.sh
bash scripts/run_web.sh
```

網頁預設國際象棋 Harness 對 Fairy-Stockfish，可切換模型執白或執黑。
國際象棋即使選 vLLM 後端，也會使用 Harness；腳本後端保留作為規則 smoke test。
網頁的其他棋類仍沿用原有流程。

`LLMGO_CONFIG` 可選既有 YAML；`LLMGO_CHESS_PROTOCOL` 可選既有國際象棋協議。
`LLMGO_RUNS` 可讓網頁讀取指定評測輸出目錄，預設仍是此專案的 `runs/`。

## 對局與診斷

```bash
python3 scripts/eval_harness.py --model-path /user/liyaohu4170/models/Qwen3.5-4B \
  --output runs/harness-metrics.json
```

腳本預設從 `/user/liyaohu4170/PostTrainBench/protocol.json` 讀取協議，也能用
`--protocol` 指定現有協議；`--base-url http://127.0.0.1:8001/v1` 可重用本地
vLLM。未給 URL 時，每張選定 GPU 啟動一個模型服務。`--gpus 1` 限制為一張卡。
`scripts/arena_ladder.py` 的模型座位也已改用相同 Harness。

每局 `config.json` 保留完整協議、生成預算、fallback seed 和 harness 檔案雜湊。
`events.jsonl` 逐次保存：

- `turn_started`：本手設定及工具庫。
- `model_request started/finished`：實際 prompt、請求參數、完整 reasoning/content、
  原始 HTTP／SSE 回應、usage、延遲、首段文字延遲、finish_reason、解析結果與重試序號。
- `token`：模型生成中的文字片段，包含 request_id、reasoning/content 通道、片段序號、
  時間及必要的文字替換標記；片段數不等於 tokenizer 的 token 數。
- `tool_call started/finished`：工具名稱、參數、完整回傳與計算時間。
- `choice`：全部完整合法步與最終選擇，唯一合法步記為 `forced`。
- `fallback`：隨機後備的原因、範圍及剩餘預算；舊日誌的 `route_recovery` 仍可回放。
- `attempt`、`position`、`game_finished`：實際走子、局面與結果。

每筆有時間戳及 sequence。每局還會同步寫入 **`short_events.jsonl`**：每次模型請求
僅保存一份輸出的前 1000 與後 1000 tokens，中間以省略數量標示；不足 2000 時
完整保存且不重複。usage、延遲、finish_reason、解析結果、錯誤、後備原因和棋盤
事件仍保留。截取使用現有 `--model-path` 下的 `tokenizer.json`，也可用
`LLMGO_TOKENIZER` 指定既有 tokenizer 檔案或模型目錄；網頁啟動使用既有 YAML 中
的 `model_path`。執行 Python 環境需要 `tokenizers`（評測腳本會自動補裝）。
新請求會要求 vLLM 回傳原始 token IDs，優先依生成 IDs 截取，再以模型 tokenizer
解碼為文字，並保留頭尾 IDs。舊紀錄或服務不提供 IDs 時，對可見輸出重新分詞，
`scope` 會標示來源；生成 token 用量仍以原始 usage 為準。
若 tokenizer 無法載入，會明確記為 `unit=characters, exact=false`，不能視作
精確的文字 token 窗口；若服務提供生成 IDs，頭尾各 1000 個 IDs 仍會保存。
可在模型所在的伺服器重新產生可讀的精簡紀錄。

完整 `events.jsonl` 保存在本機並已加入 `.gitignore`，GitHub 追蹤精簡紀錄、設定
與結果。已有的原文不會從磁碟刪除。舊對局可在**停止寫入後**一次轉換：

```bash
python3 scripts/compact_events.py runs \
  --tokenizer /user/liyaohu4170/models/Qwen3.5-4B
```

**伺服器首次拉取這個版本前，請先備份原始日誌。** `git rm --cached` 在開發機
保留原文，但 Git pull 會刪除先前已追蹤、此次除名的檔案。停止舊評測後，在專案
目錄執行以下步驟即可保留原文；備份留在專案旁邊：

```bash
events_backup_dir=$(mktemp -d ../llmgo-events.XXXXXX)
find runs -type f \( -name events.jsonl -o -name events.json \) -print0 \
  | tar --null -T - -cf "$events_backup_dir/events.tar"
git pull --ff-only
tar -xf "$events_backup_dir/events.tar"
```

網頁預設讀取精簡紀錄，以位元組游標分頁；診斷欄開啟時才格式化，完整事件按需
從產生對局的機器載入。只有精簡紀錄的 GitHub checkout 也能回放。
直播文字以有界視窗顯示，完整串流仍寫入原始日誌；側欄只保留最近六手的折疊
區，較早走法可用逐手回放查看。網頁可展示工具庫、
條件式 skills、生成預算及選擇完成率。模型請求使用 SSE 串流，網頁透過 WebSocket
即時顯示思考與最終回答。命令列進行中的棋局也透過 WebSocket 增量讀取 JSONL，
約每 0.1 秒檢查新記錄。重新連線先載入精簡歷史，再接上目前生成的內容；
停止、逾時或斷線後，已生成的片段仍保留供回放。精確 token 數以串流末尾的 usage
為準，服務未提供 usage 時明確標示估算。重啟網頁服務並啟動新對局後即可使用，
既有 vLLM 服務可繼續重用；舊評測進程需重新啟動，才會開始寫入串流片段。
舊對局仍可回放，但當時沒有保存的診斷不會被補造。

每批結果除 `--output` 外，也保存於 `harness_batches/<batch>.json`，避免被下一批
覆蓋。棋盤合法率包含程式後備；`model_choice_rate` 的分母只計需要模型作答的手數，
排除 `forced_moves`，`model_decisions` 也排除隨機後備；請同時查看 fallback_reasons。

## 工具、預算與 skills

每手由 Harness 自動執行兩項檢查：`position` 記錄將軍、易位權利／當前合法易位
及子力；`candidates` 列舉**全部**合法步，對每一步試走，檢查合法回吃、立即子力
變化、對手一步將殺及終局。工具不調用外部引擎、開局 oracle 或殘局庫，不排序
推薦著，也不刪除看起來差的合法步。完整結果保存於工具事件；prompt 只附簡短
候選觀察。局部回吃估算不代表完整交換搜尋或安全保證。

合法步以棋子種類、起點及 UCI 固定分組，例如：

```text
knight b1: b1a3 b1c3
knight g1: g1f3 g1h3
pawn e2: e2e3 e2e4
```

這是同一道題的分組；模型直接選完整 UCI，不需選工具或先選棋子。棋盤以具名棋子
座標按子力種類合併表示，FEN 保留於日誌。預設要求單獨的合法 UCI；程式仍相容
舊 JSON 回答，只解析最終
答案。兼容 vLLM 的分離 reasoning 欄位，以及混在 content 中、以 `</think>` 結束的
思考；截斷／未完成的思考不會被當作答案。原始 HTTP 回應完整保留。
請求同時設定 `include_reasoning=True`、`skip_special_tokens=False`，避免服務在沒有
分離 reasoning 欄位時又剝除思考標記。若仍然沒有明確答案邊界，會記錄
`incomplete_thinking`，不從思考中的候選 JSON 猜答案。這些請求欄位依據既有
[vLLM 0.20.1 協議](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/entrypoints/openai/chat_completion/protocol.py)。

若服務仍以空文字 delta 回傳思考結束 token，Harness 現在依**既有模型 tokenizer**
與原始生成 IDs 恢復分隔；打包的回應需先驗證重新解碼文字與收到的文字吻合。
`boundary_source`、`boundary_error`、`tokenizer_info` 記錄分隔來源與 tokenizer 指紋。
僅用 `--base-url` 時，也請附既有 `--model-path` 或設定 `LLMGO_TOKENIZER`；Arena
同樣可用 `--model-path`。服務沒有傳 IDs／明確分隔，或 tokenizer 不吻合時仍會拒收。
這是客戶端修正，重啟評測與網頁程序後即可使用，既有 vLLM 服務可繼續重用。

網頁「目前選中對局」顯示 game ID；未完成的棋局從該局事件產生暫時統計，分別
列出已結束／未結束請求，token 合計只計已結束請求。全部對局匯總位於獨立折疊區，
切換對局不沿用其他局的統計。[最新兩局的完整分析與重播結果](docs/2026-10-18-latest-runs.md)。

`src/llmgo/harness/settings.json` 的 `move_tokens` 預設為 32768；整手生成與格式
重試共用這個額度，thinking 始終開啟。最多兩次請求，第二次附格式修正提示。
單次請求逾時 540 秒，整手逾時 1800 秒。唯一合法步不呼叫模型，單獨統計
`forced_moves`；模型成功選擇與 `random` 後備也分開統計。
舊版的 `tool_tokens`、`piece_tokens`、`preview_tokens`、`max_lookups` 設定已移除；
若協議中有自行覆蓋這些欄位，請改用 `move_tokens`。

`skills/cards.json` 的 13 張卡保留觸發條件、規則、反例、來源與棋譜證據；每題最多
按條件載入兩張。六張開局卡涵蓋中央起手、黑方應對、雙王兵、西西里、卡羅－康
及后兵開局。防一步殺卡只在預計算確實發現候選允許一步殺時載入。
特定開局卡觸發時不再重複載入通用 development 卡，戰術卡仍保留條件觸發。
`skills/master_games.pgn` 是研究及測試素材，執行時不查表套棋譜。

## RSI 與測試

PostTrainBench 使用同一個選擇題 runner；Meta-Agent 可改該輪的 `harness/` 及
skills，亦可選擇後訓練。提交與 seal 打包方式見 `../PostTrainBench/README.md`。

PostTrainBench 已包含選擇題、精簡紀錄、裸 UCI 與 tokenizer 依賴的整合，舊的一次性
patch 已移除。伺服器更新兩個倉庫後，在 RSI 輪次之間重新打包 baseline：

```bash
cd ../PostTrainBench
python3 package_seal.py --llmgo '../LLMGo!'
```

現行入口為 `eval_harness.py`（批量評測）、`arena_ladder.py`（重用已部署服務），以及
`rsi.py`／`rsi_worker.py`（devspace 研究與 GPU 任務）。早期中國象棋五局測試和
直接 SSH／Docker 的 Arena 啟動腳本已移除；模型、引擎及站點路徑沿用現有設定。

```bash
PYTHONPATH=src python3 -m pytest -q tests
node tests/test_harness_ui.js
node --check src/llmgo/server/static/app.js
```

真實 Qwen 對局與 Elo 改善需在伺服器驗證；CPU／mock 測試只驗證流程與診斷。
