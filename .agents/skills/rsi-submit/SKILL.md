---
name: rsi-submit
description: 登記 Chess-R1 GRPO 訓練和固定多難度評測的研究提案，交 Executor 執行。
---

讀 `docs/rsi-operations.md` 和 `configs/grpo/default.json`。workspace 是每輪可編輯副本；只用 RL 與 skills 提升 Qwen3.5-4B，禁止 SFT。可改 RL 算法、reward、資料、prompt、skills，也可在候選根建立 `harness.py` 的 `build_messages(fen, legal_san, skills, default_messages)`；默認使用 Chess-R1 prompt。不能修改正式裁判、固定對手或歷史結果。

`proposal.json` 的 `jobs` 登記 argv 陣列，支持 `{python}`、`{candidate}`、`{output}`、`{model}`、`{source}`。標準訓練命令是 `['{python}','scripts/run_grpo.py','--output','{output}','--model','{model}']`；輸出與收據在 `LLMGo!/var/runs/<實驗ID>`。`python scripts/request_evaluation.py --level L2`、`L2.5` 或 `L3` 可調整單檔，`--full-panel` 登記 L2/L3 完整驗證；基線每檔 10 局，後續每檔 20 局。DeepSeek Flash non-thinking 只供選定成果後的獨立最終測試。每輪檢查輸棋的所有 `events.jsonl` 步驟。

完成修改、驗證與提案後執行本輪 prompt 指定的 `finish_research.py`，再讓 Controller 快照交 Executor。只用 beijing/model-autosft/NORMAL，每項最多 8 GPU，devspace 0 GPU，只寫 /user/lyh4170。提交不確定時先核對同一 description 和任務 ID，不盲目重提。
