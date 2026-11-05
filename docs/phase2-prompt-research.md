# Phase 2：固定 checkpoint 的 harness／skills 研究

Phase 2 的 scope 是 `phase2_prompt_only`，與預設的 Phase 1 `phase1_rl_only` 分開。學生可從 base Qwen3.5-4B 或使用者選定的 Phase 1 HF checkpoint 開始。這個起點在本研究中保持不變，研究輸出是「checkpoint 引用＋harness／skills」，不產生新訓練權重。

Meta-Agent 可自由編寫 `harness.py`、追加提示、任意 skills、輔助程式及資源。禁止任何訓練或權重更新，包括 RL、SFT、LoRA、微調與蒸餾；違反則本階段成果判 0 分。這些研究限制只透過 [完整 Meta-Agent prompt](phase2-meta-prompt.md) 告知，不實作訓練偵測、命令攔截或自動判罰。checkpoint、non-thinking、原始 Chess-R1 prompt 基礎、boxed SAN 和裁判不變也屬於 Agent 應遵守的研究約定。既有 Controller／Executor 的交接、快照、資源和結果紀錄機制繼續使用。

## 建立與啟動

以下命令供日後將此分支部署到遠端後，在自己的 0 GPU devspace 使用。本次開發只在本機驗證，沒有部署或提交任務。

從 base 建立新研究：

```bash
cd '/user/lyh4170/LLMGo!'
export RSI_PYTHON=/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python
"$RSI_PYTHON" scripts/rsi.py init chess-harness-base-v1 --scope phase2_prompt_only
bash scripts/start_rsi.sh --campaign chess-harness-base-v1
```

從 Phase 1 成果建立**另一個**新研究（將路徑替換為實際導出的 HF 模型目錄）：

```bash
"$RSI_PYTHON" scripts/rsi.py init chess-harness-p1-v1 \
  --scope phase2_prompt_only \
  --checkpoint '/user/lyh4170/LLMGo!/var/runs/<phase1-experiment>/scaling/models/step-10' \
  --max-model-len 32768
bash scripts/start_rsi.sh --campaign chess-harness-p1-v1
```

`--checkpoint` 要傳包含 `config.json`、tokenizer 和模型權重的 HF 目錄，不是原生 verl 分片 checkpoint。省略時使用 site 的 `base_model`。Phase 1 不使用此旗標。`--max-model-len` 是此 Phase 2 campaign 的推理上下文長度，預設32768，涵蓋提示、歷史和最多16384 tokens 的回答；它會一併傳入開發評測和最終測試。Agent 自行安排 skills 長度，無額外 skill schema、卡片數或文字長度限制，實際仍受模型上下文容量約束。

首次只做指定起始模型的**原始 Chess-R1 prompt、無新增 harness**基線：L2、L3 各40局。完成後產生第一輪 workspace，附上可自由改寫的 [harness 範例](../examples/phase2/harness.py) 和一份簡單 skill。這份 starter 的額外提示不混入第0輪基線。

初始化後可沿用 launcher 的 `--status`、`--logs`、`--resume` 及研究暫停／恢復命令，campaign 名稱必須對應 Phase 2。沿用現有網頁時不加 `--web`。Controller 與外部 Executor 仍是獨立程序；任務未返回結果時不持續呼叫 Meta-Agent，排程輪詢沿用600秒政策。

## 獨立性與目錄

| 路徑 | 內容 |
| --- | --- |
| `var/campaigns/<phase2-id>/` | 本研究狀態、Memory、jobs、Agent trace、完整候選快照 |
| `var/meta/phase2/<phase2-id>/` | 獨立 CODEX_HOME、Phase 2 AGENTS.md、Meta sessions |
| `var/workspaces/<phase2-id>/round-NNNNNN/` | 每輪可編輯 harness／skills／程式與 context |
| `var/runs/<phase2-id>-<job-id>/` | 評測、日誌、JSON、完整逐手棋局 |
| `var/campaigns/<phase2-id>/prompt-curve.json` | 原始基線與各候選 L2／L3 分數和觀測最佳值 |
| `var/campaigns/<phase2-id>/final-candidate.json` | 最終模型引用與完整 harness／skills 快照 |

初始化時只從 site 指定的 CODEX_HOME 複製 provider 設定及登入檔，留在忽略的私有 `var` 中；不複製 Phase 1 AGENTS、研究會話或 Memory。新目錄使用 Phase 2 專用指示。CLI 載入 campaign 時會自動使用這個獨立 home。Phase 1 的 checkpoint 以引用方式使用，Phase 1 狀態與證據不會併入 Phase 2。

## Harness 介面

在 workspace 根目錄放 `harness.py`：

```python
def build_messages(fen, legal_san, skills, default_messages):
    result = [dict(message) for message in default_messages]
    result[-1]['content'] += '\nCheck threats and undefended pieces before choosing.'
    return result
```

`default_messages` 是 `src/llmgo/training/chess_r1.py` 產生的原始 system／user 訊息，包含 FEN、合法 SAN 和 boxed SAN 規約。可在此基礎追加 skills 與提示。回傳供聊天 API 使用的 messages。`skills` 是選填 `skills.json` 的原始 JSON 值，沒有指定 schema；也可完全不使用它，自行用 `Path(__file__).parent` 載入 `skills/` 下任何資源、Python helpers 或其他檔案。所有必要資源放進 workspace 的普通來源檔案，使候選快照能保存完整實作；大型 `assets/` 和運行用 `var/` 不會複製成候選程式。import workspace 內的 helper 模組可直接使用。

需要完整走棋歷史時，改實作下面的函式；存在時優先使用它：

```python
def build_messages_with_history(request):
    # request: fen, legal_san, skills, default_messages, research_scope,
    #          initial_fen, moves_uci, moves_san
    result = [dict(message) for message in request['default_messages']]
    result[-1]['content'] += '\nGame history: ' + ' '.join(request['moves_san'])
    return result
```

歷史包含從 `initial_fen` 到當前局面雙方每一手的 UCI／SAN。每手以 CPU 子程序載入 harness，工作目錄是凍結 source；stdout 是 JSON 交接通道，除錯輸出寫 stderr。Phase 2 沒有另外的 harness 執行秒數上限。harness 不存在時使用原始 prompt；`track=model` 和初始 baseline 也使用原始 prompt，供固定模型對照。

## 每輪交接與終止

先閱讀 `context.json`、Memory 及敗局的全部 `events.jsonl`，還原每一步 FEN、對手步、學生提示與原文、SAN 解析、fallback 和終局原因。寫下假設後修改 harness／skills，再交付提案：

```json
{
  "hypothesis": "根據敗局，追加受攻擊棋子檢查能減少無補償失子",
  "evidence_refs": ["<loss-events-path>"],
  "jobs": [],
  "evaluation": {"levels": [2, 3], "tracks": ["player"]},
  "decision": "continue"
}
```

`python scripts/request_evaluation.py --full-panel` 可更新評測計畫；然後按本輪 `request.json` 的命令執行 `finish_research.py` 才完成交接。預設 `jobs=[]`，透過既有 Executor 評測候選，沒有 RL step／每5step訓練審查。也可以只測 L2、L2.5 或 L3 作診斷，只有 L2 和 L3 都齊備才得到完整比較分數。

每版完整驗證為 L2、L3 各40局、白黑各20局；共80局共享佇列，每卡一個 vLLM、最多同時20個請求。五次重複局面自動和棋，上限300半步（150回合），學生固定 non-thinking、boxed SAN，沿用正式判定和日誌。

評測結束後再叫醒 Agent，讓它檢查結果、繼續修訂，或決定交付。交付有兩種用法：

- `decision=finish`、省略 `final_candidate`：此提案的候選評測完成後交付本版。
- `decision=finish`、`final_candidate=best`：交付此前完整面板中觀測最佳的完整候選，包含其原始 source／skills；分數相同時保留先前最佳。

若讀完上一輪結果便決定結束，可設 `jobs=[]`、`evaluation=false`、`decision=finish`、`final_candidate=best`，經正常交接即可直接交付先前最佳，不需重跑另一輪。最初 baseline 也可成為最佳候選。省略 `decision` 時繼續研究，沒有固定輪數或研究時間上限。

每輪保存改動摘要、敗局假設、完整候選與評測證據。網頁依候選版本顯示 L2／L3 各自勝／和／負、得分率，以及原始分數與觀測最佳曲線。測量會波動；最佳值包絡與自適應選優本身不能證明實際能力單調或可擴展提升。DeepSeek 成績不回流作下一輪調整。

## 最終交付與 DeepSeek 測試

`final-candidate.json` 包含 candidate ID、模型路徑／雜湊、model_origin、完整 source 的路徑和檔案雜湊、skills 引用、開發驗證引用及推理上下文長度。harness 和任意 skills／helper 資源在所指向的凍結 source 裡。`final_checkpoint` 只保留相容的模型引用，不代表訓練了新 checkpoint。

日後由使用者授權，在獨立1 GPU任務執行：

```bash
python scripts/run_final_test.py \
  --final-candidate '/user/lyh4170/LLMGo!/var/campaigns/chess-harness-base-v1/final-candidate.json' \
  --output '/user/lyh4170/LLMGo!/var/runs/<unique-final-run-id>/final-test' \
  --inference-python /user/lyh4170/venvs/llmgo-rsi-20261022-inference/bin/python \
  --games 20 --parallel-games 20 \
  --opponent-model deepseek-v4.1-flash
```

此命令同時載入選中的模型與 harness／skills，使用 bundle 的 Phase 2 scope 和 context；不會把 harness 略過。對手為 provider 所提供的 DeepSeek Flash non-thinking，模型別名與 API 憑證設定沿用 [RSI 操作](rsi-operations.md)。基線／Phase 1 的既有 `--model`、`--candidate` 用法保留；若手動傳入 Phase 2 來源，也需 `--scope phase2_prompt_only`。

本機測試使用假的排程／推理服務、真實 harness 子程序與 Controller／Worker／網頁 API，驗證整個候選交接流程；它不代表已完成真實 GPU 推理、棋力提升或 DeepSeek 測試。

## 本機驗證紀錄（2026-11-05）

完整 pytest 套件194項通過；之後修正首次 starter 僅從凍結來源建立、保留 Agent 主動移除 harness 的選擇，最終 Phase 2 專項13項通過（包含新增案例）。測試涵蓋 base／Phase 1 checkpoint 初始化、獨立 Meta home 與真實假 app-server 協議、自由 skills／helper import／完整走棋歷史、兩輪80局的模擬評測與選最佳、無新評測直接交付、最終測試 bundle 與實際 harness 渲染、vLLM context 傳遞及網頁 API。另通過 JavaScript 語法與 DOM 渲染／切換 campaign 的本機 smoke check。沒有連接真實 Meta provider、GPU 或遠端排程。
