# LLMGo! 國際象棋 Harness

目前國際象棋的驗收玩家是選擇題框架：本地 vLLM 上的 Qwen3.5-4B 始終開啟
`enable_thinking=True`，先選工具，再選棋子，最後選該子的合法著。規則由
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
  原始 HTTP 回應、usage、延遲、finish_reason、解析結果與重試序號。
- `tool_call started/finished`：工具名稱、參數、完整回傳與計算時間。
- `choice`：工具／試走候選／棋子／走法的選項和選擇。
- `route_recovery`、`fallback`：路由恢復或隨機走子的原因、階段、範圍及剩餘預算。
- `attempt`、`position`、`game_finished`：實際走子、局面與結果。

每筆有時間戳及 sequence。網頁可逐手回放完整請求與工具結果，並展示工具庫、
條件式 skills、生成預算及選擇完成率。命令列進行中的棋局會以增量輪詢顯示。
舊對局仍可回放，但當時沒有保存的診斷不會被補造。

每批結果除 `--output` 外，也保存於 `harness_batches/<batch>.json`，避免被下一批
覆蓋。合法率包含程式後備；請同時查看選擇完成率、random_rate 和 fallback_reasons。

## 工具、預算與 skills

工具庫包含 legal、hanging、king、material、preview。`preview` 由模型選候選步，
以棋規試走後列出合法回吃、子力變化、對手一步將殺及終局；不調用外部引擎、
開局 oracle 或殘局庫，也不替模型推薦最佳著。

king 分開顯示 in_check、易位權利及當前合法易位；hanging 使用合法回吃而非
幾何保護者，並明示局部交換估算的限制。

`src/llmgo/harness/settings.json` 控制本地 vLLM 生成：工具階段 4096、選子 8192、
試走候選 4096、選步 16384；整手仍共用 32768，並為後續階段保留額度。這些是
各階段額度，不是每階段都可另用整手預算。唯一選項自動前進。思考始終開啟。
單次模型請求逾時為 540 秒（9 分鐘），整手逾時為 1800 秒（30 分鐘）。

`skills/cards.json` 保留觸發條件、規則、反例、來源和棋譜證據；每次最多按條件
載入三張。`skills/master_games.pgn` 是研究及測試素材，執行時不查表套棋譜。
另有六張前幾步的開局知識卡，涵蓋中央起手、黑方應對、雙王兵、西西里、卡羅－康
及后兵開局；在選子／選步時提供典型短線與目的。它們只在開局結構符合時載入，
被將軍時停用，優先級低於戰術提醒；模型仍需依實際合法選項決定走法。

## RSI 與測試

PostTrainBench 使用同一個選擇題 runner；Meta-Agent 可改該輪的 `harness/` 及
skills，亦可選擇後訓練。提交與 seal 打包方式見 `../PostTrainBench/README.md`。

```bash
PYTHONPATH=src python3 -m pytest tests/test_harness.py tests/test_harness_v2.py tests/test_arena.py tests/test_games.py
node --check src/llmgo/server/static/app.js
```

真實 Qwen 對局與 Elo 改善需在伺服器驗證；CPU／mock 測試只驗證流程與診斷。
