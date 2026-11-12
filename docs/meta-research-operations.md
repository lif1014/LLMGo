# Meta-Agent 研究操作手冊

這是操作介面說明，沒有其他研究的成績、配方或結論。適用 Phase 1：只做 RL，
可研究資料、reward、超參與 RL 方法；初始學生、Chess-R1 prompt、non-thinking 固定，禁 SFT。
Meta 自己開 thinking。學生 skills.json=[]，不能新增 harness。

## 本階段的優先任務

使用者明確要求先圍繞 L2 敗局提出並執行 RL 訓練，先交付第一個訓練方案；
資料、reward、超參與 RL 方法由你自主決定。首次訓練前不以L2.5診斷取代訓練。
L2/L3每5 step各40局的正式驗證保留，不能改對手、局數、300半步上限或判和規則。
不能因L3基線零勝就宣稱RL無法改善。

## Controller 已提供的固定資料介面

`context.research_evidence`（訓練審查時在 `context.campaign.research_evidence`）直接提供：

- `read_first`：可直接閱讀的正式勝和負摘要。
- `index`：已核驗的metrics、完整逐手報告與固定Python路徑。
- `reports[].report`：每局真實半步數、勝負、終局原因及 timeline 路徑。
- 每局timeline含所有雙方走子及FEN；decision_events用byte offset/length定位完整prompt、回答與fallback。

這些檔案由 Controller 的 `research_evidence.py` 在呼叫Meta前產生，標準庫解析普通JSONL，
核對正式metrics、逐局收據及棋譜hash。直接讀取，不自行尋找棋譜或重寫長度統計。
正式成績只取metrics.statistics；衍生分析不能覆蓋它。標準 `loss_length_bins_plies`
使用完整對局半步分組，顯示前四步不改其結果。

## 開始時讀什麼

在 Controller 指定的 workspace 工作，先讀 `context.json`、本文件、
`.agents/skills/rsi-memory/SKILL.md`；若有 `OPERATOR_GUIDANCE.md` 也先讀它。
context.state.evidence_root 是本研究 campaign，context.state.working 是目前父模型。
context.executor.latest.artifact.path 是最近任務的產物索引；直接解析該路徑，
不要從整個 /user/lyh4170 搜尋其他研究。context.handoff 是操作員／前輪的交接。
不要讀其他 campaign、checkpoint 或研究結論；共享磁碟可見不代表它們是本次輸入。

CPU 指令用專案的控制 Python：

```bash
CONTROL_PY=/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python
"$CONTROL_PY" -c 'import sys, chess; print(sys.executable)'
```

系統 `python3` 不保證安裝 chess、pyarrow 或訓練依賴。先以需要的 interpreter
檢查 import。本階段不要安裝依賴或自行處理環境；固定分析器只用Python標準庫。若確實缺少額外研究工具，清楚回報所需用途。
不需要 lz4 解壓普通 `.jsonl`；不要猜壓縮格式或自行造不存在的工具名稱。

## 結果路徑與欄位

| 檔案 | 讀取方式 |
|---|---|
| campaign/jobs/JOB/executor/report.json | status/platform_status，artifact.path 指向已核驗產物 |
| RUN/panel.json | evaluations 陣列，每項的 level、ref.path 指向該難度 metrics.json |
| RUN/evaluations/L2/metrics.json（亦有 L3／L2.5） | statistics 包含 wins/draws/losses/games/score；games 是逐局收據 |
| metrics.games[i].raw_events | 該局完整 events.jsonl 的絕對路徑；不要自行拼猜 games/L3/ |
| games/GAME/summary.json | 該局學生角度 result：1 勝、0 和、-1 負；plies 是雙方半步總數 |
| games/GAME/events.jsonl | UTF-8，一行一個 JSON event，不是 JSON 陣列、不是壓縮檔 |

若目前 metrics 的 statistics 結構不同，先印 keys 並讀實際 schema，不猜數值位置。
`score=(wins+0.5*draws)/games`；勝率是 wins/games，與得分率不同。
wins+draws+losses 必須等於 games。不要把百分比、和棋或黑方獲勝誤當學生勝局。

以下程式只讀當前 context 指向的已完成評測，並從逐局收據交叉核對：

```python
import json
from pathlib import Path
from collections import Counter
context = json.loads(Path('context.json').read_text())
panel_path = Path(context['executor']['latest']['artifact']['path'])
panel = json.loads(panel_path.read_text())
for row in panel['evaluations']:
    metrics = json.loads(Path(row['ref']['path']).read_text())
    print('level', row['level'], 'statistics', metrics['statistics'])
    games = metrics['games']
    counts = Counter(g['diagnostics']['result'] for g in games)
    wins, draws, losses = counts[1], counts[0], counts[-1]
    assert wins + draws + losses == len(games)
    stats = metrics['statistics']
    assert (wins, draws, losses, len(games)) == tuple(stats[k] for k in ('wins','draws','losses','games'))
    print(dict(games=len(games), wins=wins, draws=draws, losses=losses,
               win_rate=wins/len(games), score=(wins+.5*draws)/len(games)))
```

這個例子用於 evaluate_panel；若最近 artifact 是訓練結果，按 `docs/clean-grpo-pipeline.md`
讀 scaling/curve.json 與各 step 評測。不要把索引檔本身當逐局分數。

## 完整復盤與分析程式

先看一局的實際事件 keys，再寫解析器。對每盤敗局，讀完所有行，保留雙方走法：
`student_move`、`opponent_move`、`forced_move`；另外關聯 `model_request_started`、
`model_response`、`fallback`、`episode_finished`。依 ply 排序、核對 FEN 與終局，
分清模型選擇、強制唯一合法步與隨機後備。詳細走子不等於只列學生開局幾步。

- 棋局長度取 summary.plies／最後事件 ply，或實際走子事件的最大 ply。
  不用 JSONL 行數（同一步有多種事件），也不用學生決策數代替雙方半步。
- 保留完整 moves；展示前幾步用另一個變數。不能用已截成四步的清單判斷是否四步內輸棋。
- 分類先在一局短局、一局長局核對，再批次分析。空欄位／空清單不能默認為零步敗局。
- 分開「觀察到的事實」「假設」「下一次可推翻假設的實驗」。
  基線零勝不能證明 RL 無法改善；尚未訓練不能宣稱訓練失敗。
- 保存分析程式、統計與逐局證據引用。修正錯誤保留原輸出，不改正式棋局／metrics。

## 自主提交，而非把可做的工作交回使用者

L2、L2.5、L3 都已獲准作開發驗證；本階段先提交L2導向RL訓練。你自行決定是否需要診斷、資料選擇或訓練，
不需要為已授權的操作再取得使用者同意。先讀腳本 `--help`，再在 workspace 更新提案。
首次RL訓練後，例如你判斷需要 L2.5：

```bash
"$CONTROL_PY" scripts/request_evaluation.py --level L2.5 --gpus 4 \
  --hypothesis '填入你基於當前敗局提出的具體診斷假設'
```

這只寫 proposal.json，不啟動推理／訓練。單獨診斷提案的 jobs 應為 []；
該工具會保留原有 jobs，先核對不要意外帶入其他命令。`--full-panel` 請求 L2/L3 各40局。
寫好提案後，用本次 prompt 提供的 `finish_research.py --request ...` 命令交接：

```text
<本次 finish_command> --status ready --proposal proposal.json --summary '操作、證據與提案內容'
```

必須使用本次 request 的真實路徑，不能重用上次的完成命令。交接後不要再修改副本；
若修改，重新交接。Controller/Executor 會凍結、提交和監控；Meta 不自行再提交一份 cctl job。
不要直接執行 evaluate.py、run_grpo.py 或 GPU 訓練來代替登記提案；Meta 推理卡用於 Meta。

`needs_input` 只用於缺少使用者才能提供的資料、權限或不可自行解決的阻礙，
summary 必須寫明具體問題。低分、零勝、需要再測一檔、可修的程式錯誤、找不到已給路徑，
都不是必須讓使用者接手的理由。失敗先讀證據、修正或提出新的可執行方案。

## 訓練交付與監管

訓練提案範本與欄位見 `docs/clean-grpo-pipeline.md`。每輪交付資料選擇程式／查詢、
ID 清單、分類統計與敗局假設；沿用預設資料也記錄。不要捏造已執行、已選樣或已提高分數。
每5個 GRPO step 保存完整 checkpoint，完成 L2/L3 各40局後，Meta 決定 continue/replan/finish。
模型交付不是強制達到固定步數；保留全部中間點及原始分數。DeepSeek 最終測試留給使用者。
所有產物寫本研究的遠端 workspace/campaign/var/runs，沿用既有網頁與 Executor。


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
