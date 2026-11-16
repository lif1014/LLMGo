---
name: rsi-executor
description: 依 LLMGo! 研究提案提交、監控和收集 Chess-R1 GRPO 或固定評測任務。
---

讀 `docs/rsi-operations.md` 和實際 site 配置。檢查受保護 manifest SHA、候選源碼快照、模型身份與資源上限，預覽任務後再提交。Phase 2 只做 harness／skills 評測，使用 beijing/model-autosft/NORMAL，或獲授權卡池的 PREEMPTABLE，不使用 HIGH。Phase 1 學生訓練使用 model-autosft/HIGH，同時最多一個活動或提交不確定的 HIGH。每項最多8 GPU，devspace 0 GPU，只寫/user/lyh4170；不停止其他任務。GPU 任務每600秒輪詢，CPU可較頻繁。提交響應不確定時以description、intent和任務ID對帳。

研究輸出位於 `LLMGo!/var/runs/<campaign>-<job>`。GRPO 配方、日誌、原生 checkpoint 和模型位於其 `grpo/`，Worker 的 `result.json` 和 `submission.json` 位於根目錄。正式評測每局必有完整 `events.jsonl`、summary、完整性收據；基礎設施錯誤不能計入勝負。新 campaign 必須從新配置建立，不熱改舊凍結快照。

L2/L3全面板合併為evaluate_panel任務，每檔40局，共80局共用佇列；每卡一個vLLM，每副本max_num_seqs=20，實際並行上限min(80,20×卡數)。各難度分開統計，全部完成才回到Meta決策；等待排程與走棋期間只由Python處理，不用LLM輪詢。根目錄games下的ID帶難度／step，完整指標在evaluations/L2、L3或scaling/evaluations/step-N中。原任務的凍結版本不變。
