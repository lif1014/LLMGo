# LLMGo! 國際象棋 Harness

目前國際象棋的驗收玩家是選擇題框架：本地 vLLM 上的 Qwen3.5-4B 始終開啟
`enable_thinking=True`，每手只回答一道完整走法選擇題。Harness 自動檢查局面，
列出全部合法 UCI 走法，模型直接回答 `e2e4`。規則由
python-chess 執行；對手為 Fairy-Stockfish，預設第 1 檔（skill -9、depth 5、
每步 0.05 秒）。最多 **200 半步**，即 100 回合。中國象棋及五子棋保留相容入口。

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
這是對可見模型輸出的重新分詞，生成 token 用量仍以原始 usage 為準。
若 tokenizer 無法載入，會明確記為 `unit=characters, exact=false`，不能視作
精確的 1000-token 窗口；可在模型所在的伺服器重新產生精簡紀錄。

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

`src/llmgo/harness/settings.json` 的 `move_tokens` 預設為 32768；整手生成與格式
重試共用這個額度，thinking 始終開啟。最多兩次請求，第二次附格式修正提示。
單次請求逾時 540 秒，整手逾時 1800 秒。唯一合法步不呼叫模型，單獨統計
`forced_moves`；模型成功選擇與 `random` 後備也分開統計。
舊版的 `tool_tokens`、`piece_tokens`、`preview_tokens`、`max_lookups` 設定已移除；
若協議中有自行覆蓋這些欄位，請改用 `move_tokens`。

`skills/cards.json` 的 13 張卡保留觸發條件、規則、反例、來源與棋譜證據；每題最多
按條件載入兩張。六張開局卡涵蓋中央起手、黑方應對、雙王兵、西西里、卡羅－康
及后兵開局。防一步殺卡只在預計算確實發現候選允許一步殺時載入。
`skills/master_games.pgn` 是研究及測試素材，執行時不查表套棋譜。

## RSI 與測試

PostTrainBench 使用同一個選擇題 runner；Meta-Agent 可改該輪的 `harness/` 及
skills，亦可選擇後訓練。提交與 seal 打包方式見 `../PostTrainBench/README.md`。

伺服器更新 LLMGo! 後，也要同步 PostTrainBench 的 Meta-Agent 任務與 baseline。
本倉庫附有針對上一版 chess harness 整合的文字更新 patch：

```bash
cd ../PostTrainBench
git apply --unidiff-zero --check '../LLMGo!/integrations/posttrainbench-single-choice.patch'
git apply --unidiff-zero '../LLMGo!/integrations/posttrainbench-single-choice.patch'
python3 package_seal.py --llmgo '../LLMGo!'
```

若已套用，無需重複 apply。此次精簡紀錄、裸 UCI 與 tokenizer 依賴的增量同步：

```bash
cd ../PostTrainBench
git apply --check '../LLMGo!/integrations/posttrainbench-compact-events.patch'
git apply '../LLMGo!/integrations/posttrainbench-compact-events.patch'
python3 package_seal.py --llmgo '../LLMGo!'
```

每個 patch 只套用一次；每次更新 baseline 後重新執行 packager 即可。請在 RSI
輪次之間打包，原有模型／引擎路徑保持不變。

```bash
PYTHONPATH=src python3 -m pytest -q tests
node tests/test_harness_ui.js
node --check src/llmgo/server/static/app.js
```

真實 Qwen 對局與 Elo 改善需在伺服器驗證；CPU／mock 測試只驗證流程與診斷。
