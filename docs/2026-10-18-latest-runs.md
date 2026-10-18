# 最新兩局 random 原因與修正

分析基準為 GitHub `a9b3355`，直接讀取兩局完整 `events.jsonl`，包括每次請求的原始
SSE、token IDs、usage、prompt、skills、工具回傳與最後走子。原始日誌與既有成績不回寫。
原始檔：[eaa2](../runs/harness_runs/20261018-153738-eaa2/events.jsonl)、
[0a3b](../runs/harness_runs/20261018-153740-0a3b/events.jsonl)。

| run | 已落子的模型手數 | model / random | 已結束請求 | 正常 stop 卻誤拒 | 真正逾時 | 未結束請求 |
|---|---:|---:|---:|---:|---:|---:|
| `20261018-153738-eaa2` | 2 | 0 / 2 | 5 | 5 | 0 | 1 |
| `20261018-153740-0a3b` | 3 | 0 / 3 | 6 | 5 | 1 | 1 |

兩局均沒有 `summary.json` 或 `game_finished`，只是尚未結束的日誌快照，不能計算勝率或 Elo。

## 主要原因：文字通道丟失了思考結束標記

10 次正常 `stop` 回應都有合法最終 UCI，但文字通道只有 `content`，沒有獨立 reasoning，
也沒有可見的 `</think>`。例如第一次回應尾端是長篇思考後另起一段的 `e2e4`。
舊 Harness 為避免從未完成思考猜步，將其判為 `incomplete_thinking`，重試後仍失敗，最後隨機走。

原始 SSE 實際包含：

```json
{"delta": {}, "token_ids": [248069], "finish_reason": null}
```

本次實際 Qwen3.5 tokenizer 中，`248069` 就是 `</think>`；`248046` 是 `<|im_end|>`。
雖然請求已設定 `skip_special_tokens=false` 和 `include_reasoning=true`，服務仍把控制 token
從文字 delta 中省略。這兩個參數本身不足以修復此服務的輸出。從完整生成 IDs 解碼，
10 次都可恢復 `</think> → 合法 UCI → <|im_end|>`，並通過原局面的合法步集合驗證。

修正使用**模型自身 tokenizer 的控制 token 映射**，不在程式寫死上述 ID：

- 串流遇到獨立的隱藏結束 token，即補回實際生成的分隔，使最終回答能立即出現在網頁。
- 若結束 token 與一般文字打包，解碼生成 IDs；只有重新解碼文字與收到的文字吻合才恢復分隔。
- 原始 HTTP/SSE 保持不變。請求另存 `boundary_source`、`boundary_error` 和 tokenizer 指紋。
- 沒有明確分隔、tokenizer 不吻合、請求出錯或生成未完成時，繼續拒收，不搜尋思考中的 UCI／JSON。

`eval_harness.py --model-path ...` 與網頁既有 `model_path` 現在也用於解碼；沒有改動伺服器路徑。
只用 `--base-url` 的呼叫也應附現有 `--model-path`，或設定 `LLMGO_TOKENIZER`。
Arena 同樣可附 `--model-path`。

## 真正逾時與 skills

`0a3b` 的 `4:white:2` 在 540 秒時仍反覆比較同樣走法，生成 15,916 tokens，沒有結束標記、
EOS 或 finish_reason，這次拒收是正確的。但它前一次 `4:white:1` 已正常選出 `e2e4`，
修復解析後會直接採納前一次答案，避免進入這次多餘的重試。

每題最多只有兩張 skills，初始 prompt 的 usage 為 619 tokens；工具最長耗時分別約 40 ms
和 18 ms，模型生成中位速度約 28–29 tokens/s。現有證據不支持「工具數量或 skills 長度」
是這些正常答案被拒收的原因。

思考內容仍暴露了提示誤解：模型把對局當作有唯一勝著的題目、把 skills 當作答案提示，
並把「未列出風險」當成安全。因此本次另做以下精簡：

- 特定開局卡已涵蓋發展原則時，不再同時加入通用 development 卡；戰術卡仍可按條件觸發。
- 提示明確說明一般對局不保證有強制勝著，skills 是一般原則。
- 候選觀察明確說明：未列出的步並未證明安全，負的一步子力差可能有後續回吃。

初始 prompt 本文使用相同 tokenizer 計數，由 609 降至 536 tokens；這不等於已證明棋力改善。
thinking 仍開啟，總額度 32768、540 秒／1800 秒、200 半步及 UCI Fairy-Stockfish 對手均保留。

## 網頁統計為何顯示 31/67

`31/67、forced 2、random 36、request 69、1691151 tokens、timeout 34` 可完整重現為
原先**所有已存歷史摘要的匯總**，並非上述任一單局。舊頁面同時有兩個問題：

1. 全部對局匯總直接顯示在列表上方，容易被理解為選中單局。
2. 沒有局末摘要的對局，切換時不會清除之前的單局統計；延遲回應亦可能覆蓋新選擇。

現在單局區顯示選中 game ID；切換立即清除舊資料，沒有局末摘要時從該局精簡事件計算
暫時統計，標明「未完對局／日誌快照」。走子分母只計已落子，請求可包含未結束者，
token 合計只計已結束請求：兩局分別為 32,034 與 39,945，均有精確 usage／IDs。
歷史匯總的 1,691,151 含舊版字節估算，不能當成這兩局的真實 token 數。

全部對局匯總移入明確標示的折疊區；請求版本與 game ID 檢查避免舊回應覆蓋新選擇，
也不會在刷新列表時自動跳離使用者選中的對局。

## 驗證與重跑

完整原始 SSE 的離線重播結果為：10/10 正常完成答案恢復，1 次未完成回應仍拒收。
亦測試即時顯示、打包分隔、錯誤 tokenizer 拒收、未完單局統計與選局競態。
逐請求結果與原始檔 SHA-256 見 [重播紀錄](../runs/analysis/20261018-two-run-replay.json)。

這是同一批已生成輸出的解析回歸，沒有新跑 Qwen 或改寫歷史 random 成 model；
修復後對手回應與後續局面會改變，實際勝率及 Elo 仍須在伺服器重新跑完整對局。

更新後重啟評測程序與網頁服務即可使用客戶端修正，既有模型服務可繼續使用。
PostTrainBench owner 在 RSI 輪次之間重新執行 `package_seal.py` 同步 baseline。
