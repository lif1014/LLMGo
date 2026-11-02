# RSI 操作

遠端 0 GPU devspace 用 `bash scripts/start_rsi.sh --campaign chess-grpo-v1 --web` 建新 campaign。`configs/rsi/site.beijing.json` 固定 `beijing / model-autosft / NORMAL`，每項最多 8 GPU。Controller 把 Meta-Agent 每輪副本放在 `var/workspaces`；Worker 與 Executor 把任務結果放在 `var/runs/<campaign>-<job>`。每項 job 有源碼快照與配方身份。舊 campaign 的凍結源碼不會被 live 倉庫更新。

Meta-Agent 只透過 RL 和 skills 提升模型，禁止 SFT。默認訓練提案：

```json
{"hypothesis":"驗證 Chess-R1 GRPO 非思考基線","evidence_refs":[],"jobs":[{"label":"Chess-R1 GRPO","argv":["{python}","scripts/run_grpo.py","--output","{output}","--model","{model}"],"resources":{"gpu_count":8,"cpu":64,"memory_gib":512}}],"evaluation":{"levels":[2],"tracks":["player"]},"recommend":false}
```

`{python}`、`{candidate}`、`{output}`、`{model}`、`{source}` 由 Worker 代換；命令以 argv 執行。研究只寫 `/user/lyh4170`；來源 `/user/liyaohu4170` 只讀。每個實驗有獨立 ID、配方、結果與日誌，不覆蓋前一實驗。提交前可執行 `python scripts/request_evaluation.py --level L2` 調整單難度評測，最終多難度用 `--full-panel`。完成後以 prompt 中的 `finish_research.py` 保存交接記錄，Controller 再交 Executor 提交、輪詢和收集。

網頁 `/rsi` 的「Chess-R1 · verl GRPO」讀取 `var/runs` 的當前實驗和原始日誌，棋局頁按每一步回放模型、引擎、forced move 和 fallback。Meta-Agent 應讀輸棋的 `games/<id>/events.jsonl` 全部步驟與 `summary.json`，再提出下一輪假設。需要 skills 或 prompt harness 時，可在 workspace 新增 `skills.json` 和 `harness.py`；默認只用 Chess-R1 prompt。
