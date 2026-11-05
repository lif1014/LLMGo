---
name: rsi-phase2
description: 在 Phase 2 固定 checkpoint 上自主研究 harness、追加提示與任意 skills，閱讀敗局並交付評測候選。
---

先讀 context.json、docs/phase2-prompt-research.md 與持久 Memory。Phase 2 的原始模型可能是 base 或 Phase 1 checkpoint；本 campaign 的來源見 model_origin。這是獨立研究，不套用 Phase 1 的空 skills／禁止 harness 規則。

自由修改 harness.py、skills 與相關 CPU 分析程式。原始 Chess-R1 prompt 由 default_messages 傳入，在此基礎加入內容；可以自行載入 skills/ 下的 Markdown、JSON、Python 或其他資源。四參數介面及完整走棋歷史介面見 docs/phase2-prompt-research.md。

禁止任何訓練或權重更新，包括 RL、SFT、LoRA、微調與蒸餾；違反則本階段成果判 0 分。此規則以 prompt 約定，不靠程式偵測。學生使用指定 checkpoint、non-thinking、boxed SAN，正式裁判和計分不改。

逐手檢查敗局的 FEN、提示、原始回答、解析、fallback 和對手走法。每輪交付候選、改動摘要、失敗模式假設與評測證據。預設 jobs=[]；evaluation 的 levels=[2,3]、tracks=[player] 為各40局完整面板，可用 scripts/request_evaluation.py --full-panel 產生提案。

decision=continue 繼續；decision=finish 交付；final_candidate=best 選先前觀測最佳完整候選。執行本輪 request 指定的 finish_research.py 後才算交接，不直接手動提交重複 GPU 任務。最終產物是 final-candidate.json 所指的 checkpoint 與完整 harness/skills source，不產生新訓練 checkpoint。DeepSeek non-thinking 留作使用者獨立最終測試。
