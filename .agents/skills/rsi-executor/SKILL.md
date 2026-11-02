---
name: rsi-executor
description: 依 LLMGo! 研究提案提交、監控和收集 Chess-R1 GRPO 或固定評測任務。
---

讀 `docs/rsi-operations.md` 和 `configs/rsi/site.beijing.json`。檢查受保護 manifest SHA、候選源碼快照、模型身份與資源上限，預覽任務後再提交。只用 beijing/model-autosft/NORMAL，每項最多 8 GPU，devspace 0 GPU，只寫 /user/lyh4170；不停止其他任務。GPU 任務每 5–10 分鐘輪詢，CPU 可較頻繁。提交響應不確定時以 description、intent 和任務 ID 對帳。

研究輸出位於 `LLMGo!/var/runs/<campaign>-<job>`。GRPO 配方、日誌、原生 checkpoint 和模型位於其 `grpo/`，Worker 的 `result.json` 和 `submission.json` 位於根目錄。正式評測每局必有完整 `events.jsonl`、summary、完整性收據；基礎設施錯誤不能計入勝負。新 campaign 必須從新配置建立，不熱改舊凍結快照。
