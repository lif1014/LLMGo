# GRPO 共用程式修復 · 2026-11-04

本次以遠端 live 的逐檔 SHA 為基準，核對主研究 `chess-grpo-v1` 第 1–6 輪的失敗紀錄及第 6 輪候選源碼，把通用修復整理回主倉庫。資料選樣程式、研究配方、模型與過去結果不搬入共用預設。

| 問題與來源 | 共用程式行為 |
| --- | --- |
| 第 1 輪：supervisor 先寫 progress.json，首次訓練卻因沒有 identity.json 被拒絕 | 只允許既有配方、鎖、logs 和 progress；無 identity 的 checkpoint／更新等產物仍拒收 |
| 第 2 輪：單分片匯出混入 base 的兩分片索引，vLLM 找不到權重 | 不複製 base 的 safetensors／PyTorch 分片索引，保留 merger 產生的索引與權重；不帶入上個模型的匯出收據 |
| 第 4 輪：只缺 2 組仍生成完整 1024 回覆批次，補採樣時 vLLM 失效 | 補採樣限制為缺少組數加 8 組備用，不超過 loader 批次；此改動減少工作量，不宣稱已證明所有 EngineDeadError 的根因 |
| 第 5 輪：recipe=8192，但訓練入口 assert response=16384 | 移除固定等值 assertion；以共用函式檢查整數範圍，讓入口、verl、上下文和截斷過濾使用相同上限，且 python -O 不能略過驗證 |
| 候選修好了 exporter，但分段 supervisor 仍呼叫受保護舊 exporter | 新版分段流程呼叫候選任務凍結的 exporter；棋局裁判與驗證協議仍由受保護源碼執行 |

`max_response_tokens` 支援 512–16384，預設 16384；`max_prompt_tokens` 支援 256–4096，預設 1024。達到回答上限仍整組（8 回覆）丟棄；最多 10 次生成仍無法補齊則失敗，不使用部分組更新。降低長度須建立新實驗，不能改已凍結配方或沿用另一配方的 optimizer 狀態。

回歸測試覆蓋實際 complete_groups callback 的動態長度、整組剔除、1024→80 回覆的少量補採樣及 UID 對齊；也覆蓋啟動身份、單／多分片索引保留、匯出程式選擇、參數提前失敗與固定 non-thinking 協議。CPU 測試不等同於新一輪 GPU 訓練或棋力驗證。

## 已有研究如何採用

live、可編輯 workspace、候選快照及受保護 campaign source 是不同版本。發布共用檔案不會熱改正在跑的任務，也不會自動更新已有 Controller 的 Python 模組。新 campaign 可直接從新版 live 建立快照。

現行 `chess-grpo-v1` 第 6 輪繼續原任務。Meta 在後續可編輯研究副本中可合入訓練檔案及配套測試，重新測試並按正常提案凍結；不要覆寫活動 workspace。分段 supervisor 的 exporter 選擇及提前驗證屬受保護框架修復，須在無活動任務、無未完成研究 workspace 的正常邊界由 operator upgrade 採用。不能直接修改歷史快照、hash 或成績，也不能將 live 發布當成現行 campaign 已升級。

部署前後的 SHA、備份、遠端隔離測試與回執保存在 `/user/lyh4170/LLMGo!/var/migrations/20261104/grpo-base-fixes/`，網頁可讀回執在 campaign 的 `operator-audits/`。更新過程不重啟服務、不停止任務、不提交 GPU 驗收。
