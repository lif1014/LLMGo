---
name: rsi-training
description: 在 LLMGo! 的 Chess-R1 GRPO 研究中凍結配方、執行與核對 reward、更新、checkpoint 和模型導出。
---

先讀 `docs/grpo-pipeline.md`、當前實驗的 `grpo/recipe.json`、`grpo/identity.json` 和源碼快照。默認入口 `scripts/run_grpo.py`，資料與模型在專案 `assets/`。禁止 SFT、行為克隆和監督蒸餾。270M 只作 RL reward 與分析，正式棋步由 Qwen 決定。

核對每題 8 個回答、過長組拒收、實際 optimizer 更新、GRPO advantage、reward 中的 Q/格式/英文部分，以及 `updates/`、`tables/`、`sampling/` 的收據。只有原生 checkpoint 的 actor、optimizer、RNG、dataloader shard 齊全，且待驗證 step 收據完整，才可導出 HF 模型。模型權重鍵須與初始 Qwen 匹配。新配方或 reward 須開新實驗，不沿用舊 optimizer 狀態。

提交給 Executor；只用 beijing/model-autosft，預設 HIGH（同時最多一個活動或提交不確定的 HIGH 任務），每項最多 8 GPU，devspace 0 GPU，只寫 /user/lyh4170。所有日誌和 JSON 寫在 `LLMGo!/var/runs/<實驗ID>`。棋力用固定 Fairy 多難度完整棋局評測，不以訓練 reward 代替。

標準訓練每 5 個 optimizer step 做 L2/L3 各 20 局驗證，再由 Meta 依 rsi-training-review 技能決定 continue / replan / finish。continue 保留全部原生狀態；改 reward、超參、資料或程式須 replan 新實驗。詳讀 docs/training-reviews.md。
