# RSI 操作

Executor 使用專案自有的 `/user/lyh4170/tools/bin/cctl`。無人值守的 token 位於 `/user/lyh4170/.config/llmgo/cybertron-token`（0600），只注入 cctl 子程序環境，不進 Git、提案、模型 prompt 或任務 entry。明確傳入的 CYBERTRON_TOKEN 優先；不改 HOME 或其他任務的登入配置。doctor --platform 會實測自己的資源池權限。

`--own` 還需要登入核實過的 user_id，保存在同目錄的 cctl-identity.json。每個執行容器使用命名的 `llmgo-rsi` CLI profile 快取身份，token 只連結回專案的私有檔案。CLI 透過 current-profile 辨識 own：空容器首次初始化為本專案，已有的預設 profile 保持不變。更新認證時須同步核實身份；不能因空資源池清單而省略權限檢查。

遠端 0 GPU devspace 用 `bash scripts/start_rsi.sh --campaign chess-grpo-v1 --web` 建新 campaign。`configs/rsi/site.beijing.json` 固定 `beijing / model-autosft，預設 HIGH（同時最多一個活動或提交不確定的 HIGH 任務）`，每項最多 8 GPU。Controller 把 Meta-Agent 每輪副本放在 `var/workspaces`；Worker 與 Executor 把任務結果放在 `var/runs/<campaign>-<job>`。每項 job 有源碼快照與配方身份。舊 campaign 的凍結源碼不會被 live 倉庫更新。

現有網頁可直接讀共享結果，沿用它時省略 `--web` 即可；README 有完整啟動命令。預設 L2/L3 合併為一個 40 局共享佇列，每卡一個 vLLM，8 卡時每卡約 5 局；單副本容量和整體並發上限均為 40。所有副本先就緒，L2/L3 各 20 局完成並驗證後才呼叫 Agent。Controller 仍是普通 Python，未新增 Agent Runner 服務。

Phase 1 Meta-Agent 只透過 RL 提升模型，固定 Chess-R1 非思考 prompt、空 `skills.json`，禁止候選 `harness.py` 和 SFT。默認訓練提案：

初始 Qwen3.5-4B checkpoint 固定，可持續訓練它派生的研究 checkpoint；允許改超參與 reward 等。預設篩選資料不變，完整 Lichess 訓練池由 Meta-Agent 自行決定是否使用及如何選樣。每輪資料選擇交付項目與工具見 [Lichess 資料](lichess-data.md)。

```json
{"hypothesis":"驗證 Chess-R1 GRPO 非思考基線","evidence_refs":[],"jobs":[{"label":"Chess-R1 GRPO","argv":["{python}","scripts/run_grpo.py","--output","{output}","--model","{model}"],"resources":{"gpu_count":8,"cpu":64,"memory_gib":512}}],"evaluation":{"levels":[2],"tracks":["player"]},"recommend":false}
```

`{python}`、`{candidate}`、`{output}`、`{model}`、`{source}` 由 Worker 代換；命令以 argv 執行。研究只寫 `/user/lyh4170`；來源 `/user/liyaohu4170` 只讀。每個實驗有獨立 ID、配方、結果與日誌，不覆蓋前一實驗。Meta-Agent 可執行 `python scripts/request_evaluation.py --level L2`、`L2.5` 或 `L3`；`--full-panel` 是 L2/L3 雙檔驗證。完成後以 prompt 中的 `finish_research.py` 保存交接記錄，Controller 再交 Executor 提交、輪詢和收集。

選定成果後，另開獨立 1 GPU HIGH job 執行 `python scripts/run_final_test.py --model <導出HF模型> --candidate <凍結候選副本> --output /user/lyh4170/LLMGo!/var/runs/<實驗ID>/final-test --inference-python /user/lyh4170/venvs/llmgo-rsi-20261022-inference/bin/python`。測試固定 Qwen 對 DeepSeek Flash non-thinking 十局；Meta-Agent 的驗證提案不能提交 DeepSeek 測試。原始事件、棋局、結果和 vLLM 日誌都在同一實驗的 `final-test/`。

最終測試 CLI 預設請求 `deepseek-v4.1-flash` 並禁止 thinking；可用 `--opponent-base-url` 指定支援此名稱的 provider、`DEEPSEEK_KEY_FILE` 指定該 provider 的密鑰。不能假設官方 DeepSeek endpoint 接受自訂 provider 的同名別名，執行前先確認模型可用性。protocol 記錄請求模型／base URL，逐手事件另記 response_model；不把別名當作已核實的模型版本。舊參考 pipeline 的預設 `deepseek-flash` 保留，可透過 DEEPSEEK_MODEL、DEEPSEEK_BASE_URL 明確指定。

網頁 `/rsi` 的「Chess-R1 · verl GRPO」讀取 `var/runs` 的當前實驗和原始日誌，棋局頁按每一步回放模型、引擎、forced move 和 fallback。Meta-Agent 應讀輸棋的 `games/<id>/events.jsonl` 全部步驟與 `summary.json`，再提出下一輪假設。Phase 1 的提示文字和 Phase 2 開放方案見 [Meta-Agent 研究範圍](phase1-rl-scope.md)。`training` 逐檔保留與搬移理由見 [訓練目錄審核](training-module-audit.md)。

預設每 5 個 optimizer step 完成 L2/L3 各 20 局驗證，由 Meta-Agent 選擇 continue / replan / finish，保留中間 checkpoint、原始分數與 observed-best 曲線。Controller 使用外部 Executor，GPU 排程查詢間隔 600 秒。流程與輸出路徑見 [分段訓練說明](training-reviews.md)。
