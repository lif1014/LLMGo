# Checkpoint 對戰與表現分

此為使用者獨立測試，不向 Meta-Agent 回傳 DeepSeek 成績。每項 1 H800、model-new／PREEMPTABLE；使用者明確授權一次提交六項。主研究的 HIGH allocation、Controller 與其他任務不變。

配置：`configs/evaluation/checkpoint-elo-20261106.json`。base 累積 0 step、j000003 step-5 累積 5、j000007 step-5 累積 10、j000007 step-10 為累積 15 的分支、j000008 step-5 累積 15、j000010 step-5 累積 20。依權重父子關係重建，不重寫舊 manifest 的 lineage_steps；j000003 servable-v1 為原權重的匯出包裝修復，不另算一次 RL。

每模型 40 局、白黑各 20，一卡一個 TP=1 vLLM，40 局同時進行，max_num_seqs=40。沿用現有最終測試的同步排程與 PIECEWISE CUDA Graph；Qwen／DeepSeek 都 non-thinking、原始 prompt、最後一個 boxed 合法 SAN、16384 tokens、300 半步、五次重複判和。學生與對手請求無客戶端期限。每次重新打完整 40 局，不混入過去不同解析版本的棋局。

統計 `S=(W+0.5D)/(W+D+L)`，四捨五入到整數百分比，再使用 [FIDE Rating Regulations §8.1.1](https://handbook.fide.com/chapter/B022024) 的 dp 表：表現分＝1800＋dp。65% 為 1910，50% 為 1800，全勝為 2600、全負為 1000；兩端的 ±800 是名義約定，不代表精確測得的實力差。不是官方 FIDE Elo，不用 K-factor，不估計區間。未完成整批或模型／協議身份不符時不發布表現分。

所有模型、日誌及 JSON 在 `/user/lyh4170/LLMGo!`：

- `var/benchmarks/<batch>/`：共用凍結 source、benchmark-plan、六項提交 request／dryrun／intent／created、衍生 results。
- `var/runs/<batch>-<checkpoint>/`：plan、submission、worker.log、progress、failure、final-test 原始棋局及結果、performance。
- `var/runs/chess-checkpoint-benchmark/benchmark.json`：網頁用衍生對照表；不改研究狀態及 Memory。

入口 `scripts/run_checkpoint_benchmark.py launch --config ...` 準備六份模型身份、建立共用快照、逐項提交並持久化 receipt；只在明確 launch 時提交。`watch --batch ...` 只接管已提交項目，每 600 秒查一次平台；期間每 30 秒讀共享盤結果、更新網頁目錄。提交結果不確定時保留 intent，停止自動重提。`collect --batch ...` 只從現有結果更新對照表。GPU worker 使用同一快照的 `worker --batch ... --entry ...`。

GPU 主程序使用配置中的絕對 `worker_python`（與此前成功 DeepSeek 評測相同的 control venv），不用 site 中可能只是 `python3` 的通用值。提交前以該解譯器、實際凍結 source、無 CPU 測試依賴 overlay 的環境，驗證 `chess` 及完整最終測試模組可匯入，保存 `worker-import-check.log`。首批 `chess-elo-20261106T014000` 因裸 `python3` 缺 chess、在對局前失敗；保留所有舊任務與產物，新批次使用獨立 source、run 與 runtime 路徑。

網頁保留六項進度、勝和負、得分率、Teacher=1800 的表現分及逐局復盤入口。現有 DeepSeek 表格與「學生 · 對局與復盤」照常可看原始回答及雙方隨機後備走子。沒有新增曲線。
