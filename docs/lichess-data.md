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
| `var/runs/lichess-pool-20261103-offline/prepare.log` | Conversion and verification log |

The official source is [Lichess Puzzle Database](https://database.lichess.org/#puzzles), downloaded from [CSV.zst](https://database.lichess.org/lichess_db_puzzle.csv.zst), under CC0. This is the full published puzzle collection, which Lichess itself has generated and curated. It is not the database of all chess games. The snapshot and original compressed source are retained.

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
