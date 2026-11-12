# 新研究的 GRPO 操作介面

本文件只描述共用介面，不提供其他研究的結果或成功配方。
先讀 docs/meta-research-operations.md；先以L2敗局為目標提交RL訓練，研究假設與RL策略自行提出。
評測協議不可改：L2/L3各40局、300半步、原有判和規則；不能改成4步。

## 資產與預設

- 初始學生：assets/models/Qwen3.5-4B（固定；本研究可續訓其派生 checkpoint）。
- 預設資料：assets/datasets/chess-r1/boxed-nonthinking-v2/train.parquet。
- 可選完整 Lichess 池：assets/datasets/lichess/20261103/boxed-nonthinking-v2；先核對 verification.json。
- Reward model：assets/models/searchless-270M/270M；上游介面在 assets/vendor/Chess-R1-830c0597。
- 配方模板：configs/grpo/default.json。不要從其他研究複製配方／checkpoint／結論。

Phase1只允許 RL，禁止 SFT。Student prompt、non-thinking、初始模型固定，skills.json=[]、無harness。
可改資料、reward、RL 方法與超參。預設每題8回答、rollout_max_num_seqs=384、回覆上限16384、
KL係數0.05、entropy係數0；其他設定以本副本實際 default.json 與解析後配置為準。
回覆截斷的同題組，預設整組拒收並補樣；修改訓練行為需在自己的提案明確說明，不能改正式裁判。

## 訓練提案

在 workspace 建立獨立配方，例如複製 default.json 為 configs/grpo/experiment.json，
再按自己的假設修改。不是要求使用以下任何超參或策略。可執行提案的介面範例：

```json
{
  "hypothesis": "填入有證據、可用實驗推翻的 RL 假設",
  "evidence_refs": [],
  "jobs": [{
    "label": "本輪 RL 實驗",
    "argv": ["{python}", "scripts/run_grpo.py", "--output", "{output}",
             "--model", "{model}", "--template", "configs/grpo/experiment.json"],
    "resources": {"gpu_count": 8, "cpu": 64, "memory_gib": 512}
  }],
  "evaluation": {"levels": [2, 3], "tracks": ["player"]},
  "recommend": false
}
```

占位符 {python}/{model}/{output} 等由 Worker 替換，不在 Meta shell 執行此訓練。
受保護 Worker 會辨識標準入口並接管 reviewed training；直接跑 run_grpo.py 會拒絕
training_review=true，不能藉關掉審查避開。用本次 finish_command 交接後由 Controller/Executor 提交。
受保護 source、manifest、已完成結果與初始模型不能覆寫；修改只發生在註冊 workspace。

CPU 的資料檢查使用有對應依賴的專案 interpreter。先測 import chess,pyarrow,zstandard；
可查看 site 的 worker_venv 或 requirements-rsi-data.txt，缺依賴時建本研究獨立環境。
沿用資料可使用 scripts/report_training_data.py，選樣使用 scripts/select_lichess_data.py；
先讀 --help，交付選樣程式／查詢、selected_ids.txt、selected_positions.txt、stats.json 與假設。
不要求 random seed 或算力預算比較。

## Step、驗證與續訓

Worker 預設每5個 GRPO global_step 保存 native checkpoint、匯出HF並完成L2/L3各40局。
global_step 不是每一個 minibatch 更新。完整 checkpoint 包括 optimizer/RNG/dataloader。
驗證後透過 .agents/skills/rsi-training-review/SKILL.md 交接 continue/replan/finish：
continue 保持凍結配方原地繼續；replan 在自己的副本改下一實驗；finish 選定已評測模型。
改配方不能偽裝成原實驗的精確續訓。不得把不同權重／協議或部分棋局混為同一分數。

L2/L3各40局，白黑各20；可自行提出L2.5診斷。棋步上限300半步，五次重複自動和棋。
每卡一個vLLM副本，評測max_num_seqs=20；學生訓練／驗證預設model-autosft/HIGH、最多8卡。
Meta一卡的四池排程由框架管理，見 docs/meta-execution.md，不自行提交推理卡。

## 產物位置

| 路徑（相對本實驗 RUN） | 內容 |
|---|---|
| grpo/recipe.json、resolved-config.json | 凍結配方與實際配置，未必代表訓練已完成 |
| grpo/progress.json、updates/、sampling/ | 當前階段、已完成更新及採樣診斷 |
| grpo/checkpoints/、native-checkpoints/ | checkpoint收據與可續訓原生檔案 |
| scaling/models/step-N/ | 該step匯出的HF模型 |
| scaling/curve.json | 原始step分數、父系與checkpoint身份 |
| scaling/evaluations/step-N/L2、L3 | 各step正式開發驗證與完整棋譜 |
| scaling/reviews/step-N/ | 本次審查request、decision、ack |
| result.json、submission.json | Worker結果與交付模型；不是 grpo/result.json |

RUN 由 manifest.output 決定，通常是 var/runs/<campaign>-<job>。
新 campaign 先測原始學生基線，再研究；基線不產生 RL checkpoint。
Web 從共享產物更新，等待結果不會反覆呼叫 Meta。
