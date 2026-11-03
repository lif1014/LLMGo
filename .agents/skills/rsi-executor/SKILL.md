---
name: rsi-executor
description: 依 LLMGo! 研究提案提交、監控和收集 Chess-R1 GRPO 或固定評測任務。
---

讀 `docs/rsi-operations.md` 和 `configs/rsi/site.beijing.json`。檢查受保護 manifest SHA、候選源碼快照、模型身份與資源上限，預覽任務後再提交。只用 beijing/model-autosft，預設 HIGH（同時最多一個活動或提交不確定的 HIGH 任務），每項最多 8 GPU，devspace 0 GPU，只寫 /user/lyh4170；不停止其他任務。GPU 任務每 600 秒輪詢，CPU 可較頻繁。提交響應不確定時以 description、intent 和任務 ID 對帳。

研究輸出位於 `LLMGo!/var/runs/<campaign>-<job>`。GRPO 配方、日誌、原生 checkpoint 和模型位於其 `grpo/`，Worker 的 `result.json` 和 `submission.json` 位於根目錄。正式評測每局必有完整 `events.jsonl`、summary、完整性收據；基礎設施錯誤不能計入勝負。新 campaign 必須從新配置建立，不熱改舊凍結快照。

新版 L2/L3 全面板合併為 evaluate_panel 任務，每卡一個 vLLM，40 局共用佇列，同時上限40局；每副本max_num_seqs=40。每檔仍獨立20局，全部完成才回到Meta決策；等待排程與走棋期間只由Python處理，不用LLM輪詢。根目錄games下的ID帶難度／step，完整指標在evaluations/L2、L3或scaling/evaluations/step-N中。原任務的凍結版本不變。
