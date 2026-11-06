# RSI 操作

Executor 使用專案自有的 `/user/lyh4170/tools/bin/cctl`。無人值守的 token 位於 `/user/lyh4170/.config/llmgo/cybertron-token`（0600），只注入 cctl 子程序環境，不進 Git、提案、模型 prompt 或任務 entry。明確傳入的 CYBERTRON_TOKEN 優先；不改 HOME 或其他任務的登入配置。doctor --platform 會實測自己的資源池權限。

`--own` 還需要登入核實過的 user_id，保存在同目錄的 cctl-identity.json。每個執行容器使用命名的 `llmgo-rsi` CLI profile 快取身份，token 只連結回專案的私有檔案。CLI 透過 current-profile 辨識 own：空容器首次初始化為本專案，已有的預設 profile 保持不變。更新認證時須同步核實身份；不能因空資源池清單而省略權限檢查。

遠端 0 GPU devspace 用 `bash scripts/start_rsi.sh --campaign chess-grpo-v1 --web` 建新 campaign。`configs/rsi/site.beijing.json` 固定 `beijing / model-autosft，預設 HIGH（同時最多一個活動或提交不確定的 HIGH 任務）`，每項最多 8 GPU。Controller 把 Meta-Agent 每輪副本放在 `var/workspaces`；Worker 與 Executor 把任務結果放在 `var/runs/<campaign>-<job>`。每項 job 有源碼快照與配方身份。舊 campaign 的凍結源碼不會被 live 倉庫更新。

現有網頁可直接讀共享結果，沿用它時省略 `--web` 即可；README 有完整啟動命令。預設 L2/L3 各40局，合併為80局共享佇列，每卡一個 vLLM、最多20個請求；8卡時每卡約10局，1卡最多同時20局。所有副本先就緒，L2/L3 各 40 局完成並驗證後才呼叫 Agent。Controller 仍是普通 Python，未新增 Agent Runner 服務。

Phase 1 Meta-Agent 只透過 RL 提升模型，固定 Chess-R1 非思考 prompt、空 `skills.json`，禁止候選 `harness.py` 和 SFT。默認訓練提案：

初始 Qwen3.5-4B checkpoint 固定，可持續訓練它派生的研究 checkpoint；允許改超參與 reward 等。預設篩選資料不變，完整 Lichess 訓練池由 Meta-Agent 自行決定是否使用及如何選樣。每輪資料選擇交付項目與工具見 [Lichess 資料](lichess-data.md)。

```json
{"hypothesis":"驗證 Chess-R1 GRPO 非思考基線","evidence_refs":[],"jobs":[{"label":"Chess-R1 GRPO","argv":["{python}","scripts/run_grpo.py","--output","{output}","--model","{model}"],"resources":{"gpu_count":8,"cpu":64,"memory_gib":512}}],"evaluation":{"levels":[2],"tracks":["player"]},"recommend":false}
```

`{python}`、`{candidate}`、`{output}`、`{model}`、`{source}` 由 Worker 代換；命令以 argv 執行。研究只寫 `/user/lyh4170`；來源 `/user/liyaohu4170` 只讀。每個實驗有獨立 ID、配方、結果與日誌，不覆蓋前一實驗。Meta-Agent 可執行 `python scripts/request_evaluation.py --level L2`、`L2.5` 或 `L3`；`--full-panel` 是 L2/L3 雙檔驗證。完成後以 prompt 中的 `finish_research.py` 保存交接記錄，Controller 再交 Executor 提交、輪詢和收集。

選定成果後，另開獨立 1 GPU job 執行 `python scripts/run_final_test.py --model <導出HF模型> --candidate <凍結候選副本> --output /user/lyh4170/LLMGo!/var/runs/<實驗ID>/final-test --inference-python /user/lyh4170/venvs/llmgo-rsi-20261022-inference/bin/python`。預設 Qwen 對 DeepSeek Flash non-thinking 十局；`--games 20` 為白黑各十局，`--parallel-games 20` 可在同一個 TP=1 vLLM 副本並行二十局，並發最多四十。所有局數必須為偶數，配對 seed 沿用既有協議，分數以實際計畫局數為分母；任何局失敗不發布完整成績。Meta-Agent 的驗證提案不能提交 DeepSeek 測試。原始事件、每局收據、進度、結果和 vLLM 日誌都在同一實驗的 `final-test/`。

2026-11-04 使用者另行授權對第六輪 step 5／10 各做二十局獨立最終測試；兩個 1 GPU 任務用 NORMAL，保留主研究的 HIGH 名額。結果只作使用者最終測試，不寫入 Meta-Agent 的開發驗證分數或調參回饋。這是本次額外測試的排程選擇，不修改研究任務的 HIGH 預設。

首批任務 1460436／1460438 在本地學生首步請求 600 秒逾時，各 0/20 局完成；DeepSeek 首步回覆正常。保留失敗輸出，不計為棋局敗局。最終測試使用同步排程及 PIECEWISE CUDA Graph。依使用者要求，已移除額外的 128-token／120 秒啟動生成檢查；服務就緒後直接開始正式對局。正式每手仍為最多 16384 tokens，正式學生與 DeepSeek 對手請求均不設客戶端時間期限（timeout=None）；L2/L3 正式學生推論請求現亦預設 `timeout=None`；學生權重、prompt、non-thinking 與對手設定不變。GET /models 就緒不保證生成成功，若正式對局發生推論故障，保留錯誤與未完成紀錄，不計為敗局。同步排程及 PIECEWISE 尚未證實解決首步卡住，根因仍待診斷。

最終測試 CLI 預設請求 `deepseek-v4.1-flash` 並禁止 thinking；可用 `--opponent-base-url` 指定支援此名稱的 provider、`DEEPSEEK_KEY_FILE` 指定該 provider 的密鑰。不能假設官方 DeepSeek endpoint 接受自訂 provider 的同名別名，執行前先確認模型可用性。protocol 記錄請求模型／base URL，逐手事件另記 response_model；不把別名當作已核實的模型版本。舊參考 pipeline 的預設 `deepseek-flash` 保留，可透過 DEEPSEEK_MODEL、DEEPSEEK_BASE_URL 明確指定。

網頁 `/rsi` 的「Chess-R1 · verl GRPO」讀取 `var/runs` 的當前實驗和原始日誌，棋局頁按每一步回放模型、引擎、forced move 和 fallback。Meta-Agent 應讀輸棋的 `games/<id>/events.jsonl` 全部步驟與 `summary.json`，再提出下一輪假設。Phase 1 的提示文字和 Phase 2 開放方案見 [Meta-Agent 研究範圍](phase1-rl-scope.md)。`training` 逐檔保留與搬移理由見 [訓練目錄審核](training-module-audit.md)。

預設每 5 個 optimizer step 完成 L2/L3 各 40 局驗證，由 Meta-Agent 選擇 continue / replan / finish，保留中間 checkpoint、原始分數與 observed-best 曲線。Controller 使用外部 Executor，GPU 排程查詢間隔 600 秒。流程與輸出路徑見 [分段訓練說明](training-reviews.md)。

棋局規則更新：後續 L2/L3 驗證與 DeepSeek 獨立測試啟用五次重複局面自動和棋，上限為 300 個半步（150 回合）。逼和、常見子力不足、五十步自動申請判和的原設定保留；三次重複局面仍不啟用。達步數上限前先判定將死等正常終局，舊凍結任務及歷史結果保留原規則。

Phase 2 使用獨立 `phase2_prompt_only` campaign，先以 `rsi.py init --scope phase2_prompt_only` 初始化，再用上述 launcher 啟動；`--checkpoint` 可指定 Phase 1 HF 模型。Agent 只研究 harness／skills，不訓練；規則只透過 prompt 告知，違反判 0 分。操作與交付見 [Phase 2](phase2-prompt-research.md)。本機開發不會自動部署到 devspace。

DeepSeek 棋局可直接在網頁「DeepSeek · 最終測試」查看。選擇實驗後，專欄顯示完成局數、勝和負、勝率、得分率、對手與規則、最後落盤時間；每次重試各自統計。未發布完整結果時只顯示已完成局的暫時統計，程式異常不計為棋局敗局。選某局「復盤」即可在「學生 · 對局與復盤」查看棋盤、逐手提示、回答與 fallback，支援進行中的棋局與已完成歷史。範圍選單的「DeepSeek 最終測試」對應上方選中的實驗。瀏覽器透過既有只讀 `/api/grpo/runs` 與 artifact API 分段讀原始 JSONL，保留 UTF-8 位元組游標，不需要重啟 Python 網頁程序，不會啟動 Meta／GPU 或查詢 cctl。

前端回歸測試：`node tests/test_final_replay_ui.js`、`node tests/test_evaluation_counts_ui.js`。前者驗證暫時／正式統計、異常局排除、中文長回答分段、共用棋盤與原始事件引用；後者驗證20／40局歷史混合顯示。

復盤右側的「DeepSeek · 對手輸出」直接讀 `opponent_move` 的 diagnostics，保留模型原始回答及解析／fallback 診斷。選擇對手落子時顯示該手回答；選擇隨後的學生落子時仍標示最近對手步的半步編號，避免誤認為新的回答。「讀取本手對手原始事件」開啟相同 JSONL 行的完整位元組紀錄。這些資料原本已由評測保存，此顯示更新不需要重新評測。

2026-11-05 更新：Qwen 與 DeepSeek 共用「最後一個 boxed SAN」解析器 `last-box-san-v1`，接受盒子前後文字和不同的早期盒子。Prompt 保持原文；最後一盒仍需是合法標準 SAN，截斷、未閉合或非法最後答案仍觸發隨機後備。新 protocol 記錄解析器版本；已提交的凍結任務不會熱換解析器，不能把舊棋局當作新規則測試。

「DeepSeek · 最終測試」每局新增學生／DeepSeek 隨機後備次數，復盤「本局診斷」也顯示雙方累計及原因。這是實際走出的隨機後備步數，不是用新解析器重判舊回答，也不包含 forced move。歷史索引可用 `scripts/report_random_moves.py <runs-root> [--run RUN_ID]` 產生；只有衍生 `final-test/random-moves.json` 會寫入，原始 summary、game_results、protocol、events 與勝負不變。缺索引時瀏覽器按字節游標增量讀取原始事件並快取計數，已完成棋譜不會在每次刷新重掃。


2026-11-05：1472149／j000010 完成 5 step 及模型匯出後，L2/L3 驗證因學生 HTTP 請求的舊 600 秒期限失敗，保留 75/80 局。已移除 `evaluator.request_completion` 的預設 socket 期限；後續包含新版 evaluator 的受保護快照才生效。GPU 排程輪詢仍為 600 秒。舊任務、凍結源碼與已完成棋局不重寫；live 更新不等於正在執行的 Controller 或其舊受保護快照已採用。

2026-11-06：主研究暫停後，驗證的學生／DeepSeek HTTP、vLLM 就緒總等待、harness 子程序與 UCI 通訊均改為無時間期限。使用 `Controller.upgrade_validation()` 建立新受保護快照並刷新既有研究副本的 context；保留原快照、歷史任務、模型與棋局，不解除 research hold，也不啟動 Meta。若 meta-session 殘留 process，先以原宿主任務的終態證明程序已消失，備份會話後才清除失效程序欄位；不能只憑舊 updated 時間判定死亡。
