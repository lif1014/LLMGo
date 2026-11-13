---
name: rsi-research-operations
description: 定位本研究證據、正確統計棋局、提交RL或開發評測並完成自主交接。
---

先讀 docs/meta-research-operations.md、docs/clean-grpo-pipeline.md 和 context.json。
若 workspace 有 OPERATOR_GUIDANCE.md，先讀其中更正與本次證據路徑。
本階段先提交L2導向RL訓練，不先要求使用者做L2.5；保留每5step的L2/L3各40局正式驗證。
直接讀context.research_evidence（審查時在campaign下）的read_first和index，使用固定stdlib解析／統計報告，不安裝依賴。
評測協議固定，300半步上限不能改成4步，分析展示截取不改對局長度。
需要追查原始證據才按 context.executor.latest.artifact.path → panel.evaluations[].ref.path → metrics.games[].raw_events
定位，不全盤搜尋別的研究。JSONL一行一個JSON；使用專案控制Python，不猜系統python3依賴。
以正式statistics及逐局學生result交叉核對勝和負，score=(wins+.5*draws)/games。
完整讀雙方逐手事件；展示截斷清單不能拿來計算棋局長度或失敗分類。
L2/L2.5/L3已授權，可自行在proposal登記，不能因零勝就宣稱RL無法改善或要求使用者代做。
在 workspace 用 request_evaluation.py 或寫RL proposal，然後執行當前prompt的finish_command。
只在缺少使用者獨有資訊／權限時needs_input，具體寫出缺什麼。工具／分析失敗先修正再繼續。
Meta推理卡不跑學生訓練或正式GPU評測；Controller/Executor負責任務提交與監控。


交接前必須在 workspace 執行（本研究要求先提交訓練）：

```bash
/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python scripts/check_research_proposal.py --proposal proposal.json --require-training
```

`hypothesis` 必須是字串；詳細假設可放 `research_notes`。訓練使用 `jobs[].argv` 字串陣列，
`training_config`、`status: ready_for_execution` 或口頭描述不會提交訓練。
修正檢查器指出的實際問題，讀回檔案並重跑檢查，成功後才執行本次 finish_command。
finish_research.py 與 Controller 共用此格式檢查；格式錯誤會在當次工具呼叫返回，
保持本會話繼續修改，不等 GPU 結束後才發現。檢查不提交任務，也不替你選擇 RL 策略。


可選資料選樣工具：`scripts/select_lichess_data.py` 已支援分類配比與難度範圍，
例如 `--mix fork=1 pin=2 --num-puzzles 30000 --rating-min 800 --rating-max 1600`。
先用 `--list-tags` 查真實分類、`--preview` 查配額是否可滿足；複合分類／各組難度用 `--spec` JSON。
用 `/user/lyh4170/venvs/llmgo-lichess-data-20261103/bin/python` 執行 CPU 選樣。
比例以 puzzle 計算，同題只加入一次，再保留其在訓練池中的全部局面；每輪保留交付檔。
詳見 `docs/lichess-data.md`。**不一定需要用這個腳本**：可以保留預設資料，也可以自行寫選樣程式。
是否使用、分類配比及難度由你根據本研究敗局自主決定，不是強制課程。
只修改實驗配方的 train_file，不改預設資料、固定學生 prompt、non-thinking 或正式評測協議。


2026-11-13 現行啟動規則：Meta 使用 Qwen3.8-27B thinking，128K context／64K最大輸出／56K compaction，
學生仍是固定 Qwen3.5-4B non-thinking。所有新任務（含CPU維護、Meta、學生訓練／評測）
只可在 model-new、model-claw-tool、model-codeagent2 使用 PREEMPTABLE；禁止 model-autosft，
此規則取代本文任何舊 HIGH／model-autosft 指示。新研究由使用者明確啟動，不恢復舊研究。
詳見 [Qwen3.8 操作說明](../../../docs/qwen38-meta.md)。
訓練審查交接必須包含當前context的 training_review.request_id、action、reason、selected_step；
執行 check_research_proposal.py 檢查後才交接，格式錯誤留在本回合修正。
