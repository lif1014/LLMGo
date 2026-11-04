---
name: rsi-submit
description: 登記 Chess-R1 GRPO 訓練和固定多難度評測的研究提案，交 Executor 執行。
---

讀 `docs/rsi-operations.md`、`docs/training-reviews.md` 和 `configs/grpo/default.json`。Phase 1 只用 RL 提升 Qwen3.5-4B，禁止 SFT。可改 RL 算法、reward、超參與資料；初始 checkpoint、Chess-R1 prompt、boxed SAN 協議與 non-thinking 固定。不可新增 harness.py，skills.json 必須為 []。Phase 2 才開放 harness。不能修改正式裁判、固定對手或歷史結果。

`proposal.json` 的 `jobs` 登記 argv 陣列，支持 `{python}`、`{candidate}`、`{output}`、`{model}`、`{source}`。標準訓練命令是 `['{python}','scripts/run_grpo.py','--output','{output}','--model','{model}']`，由受保護 Worker 監管每 5 step 的訓練與 L2/L3 各 40 局驗證；不要繞過 Worker 直接啟動 training_review 配方。基線每檔 40 局。L2.5 可作額外診斷。輸出在 `LLMGo!/var/runs/<實驗ID>`。DeepSeek non-thinking 最終測試由使用者另行執行，不能作調參依據。每輪檢查輸棋的全部 events.jsonl。

完成修改、驗證與提案後執行本輪 prompt 指定的 `finish_research.py`，再讓 Controller 快照交 Executor。只用 beijing/model-autosft，預設 HIGH，同時最多一個活動或提交結果不確定的 HIGH 任務，每項最多 8 GPU，devspace 0 GPU，只寫 /user/lyh4170。提交不確定時先核對同一 description 和任務 ID，不盲目重提，不停止任何他人的任務。
