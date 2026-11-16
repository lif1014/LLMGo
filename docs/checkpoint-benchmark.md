# Checkpoint 對戰與表現分

此為使用者獨立測試，不向 Meta-Agent 回傳 DeepSeek 成績。每項 1 H800、model-new／PREEMPTABLE；使用者明確授權一次提交六項。主研究的 HIGH allocation、Controller 與其他任務不變。

配置：`configs/evaluation/checkpoint-elo-20261106.json`。base 累積 0 step、j000003 step-5 累積 5、j000007 step-5 累積 10、j000007 step-10 為累積 15 的分支、j000008 step-5 累積 15、j000010 step-5 累積 20。依權重父子關係重建，不重寫舊 manifest 的 lineage_steps；j000003 servable-v1 為原權重的匯出包裝修復，不另算一次 RL。

每模型 40 局、白黑各 20，一卡一個 TP=1 vLLM，最多40局同時進行，max_num_seqs=40。沿用現有最終測試的同步排程與 PIECEWISE CUDA Graph；Qwen／DeepSeek 都 non-thinking、原始 prompt、最後一個 boxed 合法 SAN、16384 tokens、300 半步、五次重複判和。學生與對手請求無客戶端期限。新測試打完整40局，不混入過去不同解析版本的棋局；中斷後可依下述身份核對機制補測。

DeepSeek HTTP 307／308 的同站重新導向會保留 POST、JSON 與 request ID。沒有可跟隨地址或暫時網路／429／5xx 錯誤時，同一請求最多嘗試8次，退避2、4、8、16、30、30、30秒；不設單次請求期限，也不向另一站轉送登入憑證。重試耗盡時，該局記為技術中斷，不當敗局；其餘對局繼續收集。`final-test/progress.json` 即時保存 completed、failed、remaining 與錯誤，單局收據位於 `game_errors/`。未取得完整40局不發布最終得分。

停止舊任務後，可在新的 run 使用 `scripts/run_final_test.py --resume-from <舊run>/final-test`。框架先核對模型SHA、prompt資源、對手、解析規則、局數與裁判協議，再重放已完成棋譜核對終局，將有效對局及原始事件複製至新run；中斷或尚未完成的局從初始局面重打。`resume.json` 記錄來源與復用局號，舊run保留。這是評測恢復，不是另做一批40局；更新live程式不會熱改其他執行中的凍結任務。

統計 `S=(W+0.5D)/(W+D+L)`，四捨五入到整數百分比，再使用 [FIDE Rating Regulations §8.1.1](https://handbook.fide.com/chapter/B022024) 的 dp 表：表現分＝1800＋dp。65% 為 1910，50% 為 1800，全勝為 2600、全負為 1000；兩端的 ±800 是名義約定，不代表精確測得的實力差。不是官方 FIDE Elo，不用 K-factor，不估計區間。未完成整批或模型／協議身份不符時不發布表現分。

所有模型、日誌及 JSON 在 `/user/lyh4170/LLMGo!`：

- `var/benchmarks/<batch>/`：共用凍結 source、benchmark-plan、六項提交 request／dryrun／intent／created、衍生 results。
- `var/runs/<batch>-<checkpoint>/`：plan、submission、worker.log、progress、failure、final-test 原始棋局及結果、performance。
- `var/runs/chess-checkpoint-benchmark/benchmark.json`：網頁用衍生對照表；不改研究狀態及 Memory。

入口 `scripts/run_checkpoint_benchmark.py launch --config ...` 準備六份模型身份、建立共用快照、逐項提交並持久化 receipt；只在明確 launch 時提交。`watch --batch ...` 只接管已提交項目，每 600 秒查一次平台；期間每 30 秒讀共享盤結果、更新網頁目錄。提交結果不確定時保留 intent，停止自動重提。`collect --batch ...` 只從現有結果更新對照表。GPU worker 使用同一快照的 `worker --batch ... --entry ...`。

GPU 主程序使用配置中的絕對 `worker_python`（與此前成功 DeepSeek 評測相同的 control venv），不用 site 中可能只是 `python3` 的通用值。提交前以該解譯器、實際凍結 source、無 CPU 測試依賴 overlay 的環境，驗證 `chess` 及完整最終測試模組可匯入，保存 `worker-import-check.log`。首批 `chess-elo-20261106T014000` 因裸 `python3` 缺 chess、在對局前失敗；保留所有舊任務與產物，新批次使用獨立 source、run 與 runtime 路徑。

網頁保留六項進度、勝和負、得分率、Teacher=1800 的表現分及逐局復盤入口。現有 DeepSeek 表格與「學生 · 對局與復盤」照常可看原始回答及雙方隨機後備走子。沒有新增曲線。

第 11 輪的額外對戰使用 `configs/evaluation/checkpoint-elo-r11-20261106.json`，兩項皆為 model-new／PREEMPTABLE／1 H800、40 局／40 並行，沿用同一對手與棋局協議。j000012 step-5／step-10 是從 j000008 step-5 分支續訓而來，累積 GRPO step 分別為 20／25，不是接在 j000010 後面。其目錄為 `var/benchmarks/chess-elo-r11-20261106T043000/`，衍生目錄為 `var/runs/chess-checkpoint-benchmark-r11/benchmark.json`，保留原六 checkpoint 的對照表。
