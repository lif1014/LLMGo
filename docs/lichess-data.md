# Optional full Lichess training pool

The default remains `assets/datasets/chess-r1/boxed-nonthinking-v2/train.parquet`. The Meta-Agent decides whether to retain it, select from the full pool, or write its own selection/mixing code. RL algorithms, hyperparameters, reward and curriculum may change. The initial Qwen3.5-4B checkpoint, fixed Chess-R1 prompt and non-thinking mode stay fixed. Continuing a checkpoint descended from the initial model is allowed. No data-selection random seed or compute-budget comparison is required.

## Files

All paths below are relative to `/user/lyh4170/LLMGo!`.

Use the dedicated CPU interpreter `/user/lyh4170/venvs/llmgo-lichess-data-20261103/bin/python` for the preparation/selection/report commands below. Its dependencies are listed in `requirements-rsi-data.txt`; the training and control environments are not modified.

| Path | Contents |
| --- | --- |
| `assets/datasets/lichess/20261103/raw/lichess_db_puzzle.csv.zst` | Complete public Lichess puzzle snapshot, before our filtering |
| `assets/datasets/lichess/20261103/raw/lichess_db_puzzle.csv.zst.json` | Download URL, HTTP metadata, bytes and SHA-256 |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/train.parquet` | Training-only pool using the exact existing Arrow schema |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/index.parquet` | Aligned metadata for selection; `row_number` locates each training row |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/manifest.json` | Source/reference/held-out identities, processing counts and file hashes |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/verification.json` | Independent full scan; require `status: passed` before using the pool |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/stats.json` | Position/puzzle counts, rating bands, side and category distributions |
| `var/campaigns/chess-grpo-v1/operator-audits/lichess-pool-result.json` | Build receipt, including the preparation log's run-specific path |

The official source is [Lichess Puzzle Database](https://database.lichess.org/#puzzles), downloaded from [CSV.zst](https://database.lichess.org/lichess_db_puzzle.csv.zst), under CC0. This is the full published puzzle collection, which Lichess itself has generated and curated. It is not the database of all chess games. The snapshot and original compressed source are retained.

The retrieved raw snapshot contains 6,193,530 puzzles in 309,751,173 compressed bytes (SHA-256 `7b0d235fa40cfd7fdb7df1fec56a96b3b6ee1496b32ed2eb05be90d0ebf94253`). Independent local decompression checked the complete source row count. Prepared training counts and final exclusions are recorded in the pool manifest and verification receipt.

The downloader respects the process `http_proxy`/`https_proxy` settings, or explicit `--proxy`. The 2026-11-03 remote probe reached USTC's package index/wheel and hf-mirror through `http://whitelist-proxy.cybertron.svc.cluster.local:7891`, while the official Lichess address failed TLS. This snapshot therefore uses the official file downloaded locally and uploaded with full-file SHA verification. `HF_ENDPOINT=https://hf-mirror.com` affects Hugging Face clients; it does not redirect the official Lichess URL. Probe evidence is in `var/campaigns/chess-grpo-v1/operator-audits/network-proxy-probe.json`.

The current filtered dataset has 99,571 train positions, 2,291 dev positions and 23,243 test positions. It is a Chess-R1-derived dataset, not a verified exact copy of the paper's 19.2k training subset. The [paper](https://arxiv.org/html/2507.00726v1) and [public preprocessing code](https://github.com/krafton-ai/Chess-R1/tree/830c059702e7b2d51ddd923a8c9a577c3f6d06d5/data_preprocess) describe different sample counts.

## Conversion and isolation

`scripts/prepare_lichess_pool.py` streams the compressed CSV. Each raw FEN precedes the opponent's setup move, so it applies `Moves[0]` first. It emits only solver turns at move indices 1, 3, 5, etc. SAN labels and the legal SAN list are computed from the board before the solution move. It validates the complete trajectory before emitting any rows from that puzzle. Full PGN history is unnecessary for the fixed FEN prompt and is not fabricated.

The existing dev/test files remain held out. The converter removes every matching puzzle ID, and removes an entire puzzle if any solver position matches a held-out position. It then removes duplicate training positions. Position identity retains pieces, side to move, castling rights and legally relevant en passant; move clocks are ignored to catch equivalent positions. Malformed trajectories are recorded in `invalid.jsonl`. The new pool contains only `train.parquet`, with no new dev/test split.

The five training columns are unchanged: `data_source`, `prompt`, `ability`, `reward_model`, `extra_info`. `reward_model.ground_truth` stays a JSON string with `board_fen`, `next_move_san`, `legal_moves_san`, `id`, `rating`. The student sees the same two non-thinking chat messages and boxed SAN answer protocol as before. Puzzle solutions remain reward/diagnostic metadata, not supervised training targets or student prompt content.

The separate index contains `sample_id` (`PuzzleId:solver_ply`), puzzle ID, rating, rating deviation, popularity, number of plays, themes, opening tags, game URL, daily-puzzle date, side, legal-move count, solution length and position identity. Themes and opening tags are inherited **puzzle-level metadata**, not newly annotated labels for each intermediate position. They do not enter the student prompt. Theme counts are multi-label and need not sum to the number of rows.

Verification scans every output row for held-out IDs/positions, duplicate positions, the fixed prompt, protocol and schema; it also regenerates legal SAN on a sample. Original dev/test hashes must match throughout preparation. The source pool is immutable after publication; create another snapshot or experiment directory for subsequent work.

## Meta-Agent selection and delivery

Read losing games' entire `events.jsonl` before forming a data-selection hypothesis. For example, missed forks can motivate a subset containing `fork`; endgame losses can motivate endgame categories. Rating is human puzzle difficulty, so test its usefulness for this student rather than treating it as the student's Elo.

```bash
python scripts/select_lichess_data.py \
  --output /user/lyh4170/LLMGo!/var/runs/<experiment-id>/data-selection \
  --rating-min 1000 --rating-max 1800 --theme fork \
  --hypothesis 'Prior losing games missed forks; test more fork positions' \
  --evidence /user/lyh4170/LLMGo!/var/runs/<prior-run>/games/<game>/events.jsonl
```

Repeated `--theme` values require all themes; repeated `--opening` values match any opening. `--puzzle-ids` accepts a whitespace-separated puzzle ID file; `--sample-ids` accepts precise `PuzzleId:solver_ply` IDs. `--max-positions` is optional and takes the first matching positions in source order. There is no random sampling or seed option. The Meta-Agent can generate an ID list with its own query/program and pass it to the tool.

The selector writes `train.parquet` and `index.parquet`, plus the four required deliveries:

1. Selection program or query: its own source file and/or `selection.json` query.
2. Selected IDs: `selected_ids.txt` (puzzles) and `selected_positions.txt` (individual positions).
3. Category statistics: `stats.json`.
4. Selection hypothesis from losing-game evidence: `selection.json` records `hypothesis` and `evidence`. If no games exist yet, explicitly describe it as an initial hypothesis.

Copy the default recipe to the experiment, set that copy's `train_file` to the selected `train.parquet`, and pass it through `scripts/run_grpo.py --template <recipe>`. Keep the default template's `train_file` unchanged. Changing a recipe or reward creates a new experiment; continue only matching checkpoints through the existing resume protocol. If mixing sources, preserve held-out exclusions and deduplicate equivalent positions.

When retaining the default filtered data, record the same deliveries without copying its training rows:

```bash
python scripts/report_training_data.py \
  --train /user/lyh4170/LLMGo!/assets/datasets/chess-r1/boxed-nonthinking-v2/train.parquet \
  --output /user/lyh4170/LLMGo!/var/runs/<experiment-id>/data-selection \
  --hypothesis 'Retain the existing data for this experiment; explain the evidence'
```

The old filtered data lacks themes/opening tags, so this report gives rating and side distributions and marks the missing classifications explicitly. Development validation remains L2/L2.5/L3; DeepSeek Flash non-thinking remains the separate held-out final opponent.


## Optional category mixtures (no custom selection code required)

The helper is **optional**. The Meta-Agent may keep the default filtered dataset,
use this helper, or write its own query/selector. No mixture or curriculum below
is mandatory, and an available tool is not evidence that selection has run.
Use the fixed CPU data interpreter; this command does not submit GPU work.

```bash
DATA_PY=/user/lyh4170/venvs/llmgo-lichess-data-20261103/bin/python
# Read-only catalog: real theme/opening names, puzzle counts and rating bands.
"$DATA_PY" scripts/select_lichess_data.py --list-tags --rating-min 800 --rating-max 1600
# Read-only feasibility check, including shortages caused by overlapping tags.
"$DATA_PY" scripts/select_lichess_data.py \
  --mix fork=1 pin=2 --num-puzzles 30000 \
  --rating-min 800 --rating-max 1600 --preview
# Export only after choosing the research hypothesis and evidence.
"$DATA_PY" scripts/select_lichess_data.py \
  --mix fork=1 pin=2 --num-puzzles 30000 \
  --rating-min 800 --rating-max 1600 \
  --output /user/lyh4170/LLMGo!/var/runs/<experiment-id>/data-selection \
  --hypothesis 'Test more forks/pins based on the documented losing games' \
  --evidence /user/lyh4170/LLMGo!/var/runs/<prior-run>/games/<game>/events.jsonl
```

`fork=1 pin=2` requests 10,000 and 20,000 **puzzles**, not positions. Fractional
weights are accepted. Integer quotas use largest remainders; ties follow group
order. Each selected puzzle retains **all solver positions available in the
training-only pool**; the pool may already have removed duplicate positions.
Position ratios can therefore differ from puzzle ratios. Sampling has no seed
option: retain actual IDs to reproduce a selection. Within each exact tag/filter
eligibility combination, puzzles are sampled uniformly without replacement.

A puzzle can have multiple tags. The allocator assigns it to exactly one group,
while retaining all original metadata; a shared puzzle cannot fill two quotas.
Allocation accounts for overlap, so a feasible mixture is not rejected just
because an earlier group consumed a shared puzzle. If the requested quotas cannot
be filled, preview/export returns `status: insufficient` and exit code 2, with
per-group eligible counts and shortages. Export preserves `selection.json` and
`stats.json` but writes **no training parquet**; use a new output directory for
an adjusted selection. It never silently changes ratios, duplicates puzzles or
adds other categories. Preview reads index metadata only; export also checks the
existing full-pool identity before copying training rows.

For different difficulty bands or compound categories, use `--spec FILE.json`:

```json
{
  "num_puzzles": 30000,
  "groups": [
    {"name": "easy-forks", "weight": 1,
     "themes_any": ["fork"], "rating_min": 800, "rating_max": 1400},
    {"name": "pins", "weight": 2,
     "themes_all": ["pin"], "themes_exclude": ["mateIn1"],
     "rating_min": 1200, "rating_max": 1800}
  ]
}
```

Each group supports `themes_all`, `themes_any`, `themes_exclude`, `openings_any`,
`rating_min`, `rating_max`. If no tag filter is specified, the group name is the
theme to match. An explicit empty tag list matches all tags. Global CLI difficulty,
`--theme` (all) and `--opening` (any) further restrict every group. JSON may set
global `rating_min`/`rating_max`; corresponding CLI bounds take precedence.
CLI `--num-puzzles` overrides the JSON total. Lichess ratings describe human puzzle
difficulty, not this student's Elo; L2/L3 are opponents, not native puzzle tags.

Mixtures produce the usual delivery files and `selected_groups.json` (puzzle ID
→ quota group). `stats.json.groups` includes target/eligible/assigned puzzle counts,
shortfalls and expanded position counts. `themes`/`opening_tags` count positions;
`puzzle_themes`/`puzzle_opening_tags` count puzzles. These tag statistics are
multi-label and may sum to more than the dataset size. `selection.json` records
the normalized query, source identity, hypothesis, evidence and output hashes.
Read-only catalog counts use puzzles; its `positions` field reports the expanded
size before group filtering. A shortage assignment is one maximum feasible
allocation, not a claim that a particular group deserves priority.

Set only your experiment recipe's `train_file` to the resulting `train.parquet`.
The default recipe, initial checkpoint, fixed student prompt, non-thinking mode,
reward interface and held-out evaluation protocol do not change merely by using
this selector. Never claim a text-only selection plan was applied without checking
the output receipt and the submitted recipe's actual `train_file`.

## 舊 thinking 副本

`assets/datasets/chess-r1/boxed-v2` 已按使用者要求移除。逐列核對 train/dev/test 共 125,105 筆，和 boxed-nonthinking-v2 的唯一差異是 system prompt：舊版要求 `<think>`，目前只要求 boxed SAN。局面、答案、合法步、rating、ID、split 和 schema 完全相同，沒有額外的教師思考標注。刪除前後 non-thinking 三份 parquet 的 SHA 必須相同；回執在遠端 `var/migrations/20261103/remove-thinking-dataset/`。完整 Lichess 訓練池另外保留，不受此清理影響。
