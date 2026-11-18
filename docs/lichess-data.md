# Lichess training pool: independent selection for new studies

New RL studies receive only the verified, training-only Lichess pool. No training subset is preselected: configs/grpo/default.json has train_file=null. The Meta-Agent must independently choose data and export a training parquet before submitting training. The selection helper is optional; its own query/program is equally valid. Do not import another study's selections, recipes or conclusions. Later rounds may reuse this study's own selection with a recorded reason. RL algorithms, hyperparameters, reward and curriculum may change. The initial Qwen3.5-4B checkpoint, fixed Chess-R1 prompt and non-thinking mode stay fixed. Continuing this study's derived checkpoint is allowed. No selection seed or compute-budget comparison is required.

## Files

Asset paths below are relative to the prepared research source. The selector also accepts an explicit --pool path. The prepared source exposes training-only pool files; the unpartitioned raw source and held-out parquet files are kept outside this research input.

Use the dedicated CPU interpreter `/user/lyh4170/venvs/llmgo-lichess-data-20261103/bin/python` for the preparation/selection/report commands below. Its dependencies are listed in `requirements-rsi-data.txt`; the training and control environments are not modified.

| Path | Contents |
| --- | --- |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/train.parquet` | Training-only pool using the exact existing Arrow schema |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/index.parquet` | Aligned metadata for selection; `row_number` locates each training row |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/manifest.json` | Source identity, processing counts and file hashes; the research view omits historical reference/held-out paths |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/verification.json` | Independent full scan; require `status: passed` before using the pool |
| `assets/datasets/lichess/20261103/boxed-nonthinking-v2/stats.json` | Position/puzzle counts, rating bands, side and category distributions |

The official source is [Lichess Puzzle Database](https://database.lichess.org/#puzzles), downloaded from [CSV.zst](https://database.lichess.org/lichess_db_puzzle.csv.zst), under CC0. This is the full published puzzle collection, which Lichess itself has generated and curated. It is not the database of all chess games. The snapshot and original compressed source are retained.

The retrieved raw snapshot contains 6,193,530 puzzles in 309,751,173 compressed bytes (SHA-256 `7b0d235fa40cfd7fdb7df1fec56a96b3b6ee1496b32ed2eb05be90d0ebf94253`). Independent local decompression checked the complete source row count. Prepared training counts and final exclusions are recorded in the pool manifest and verification receipt.

The verified training-only pool contains **6,142,016 puzzles / 14,124,539 positions**. Start selection from this pool; the operator already removed reserved puzzle IDs, overlapping solver positions and duplicate positions. No tactical or difficulty subset is supplied as a starting curriculum.

## Conversion and isolation

The operator-side converter streams the compressed CSV. Each raw FEN precedes the opponent's setup move, so it applies `Moves[0]` first. It emits only solver turns at move indices 1, 3, 5, etc. SAN labels and the legal SAN list are computed from the board before the solution move. It validates the complete trajectory before emitting any rows from that puzzle. Full PGN history is unnecessary for the fixed FEN prompt and is not fabricated.

Reserved dev/test puzzles remain held out; those parquet files are not supplied to the Meta-Agent. The converter removes every matching puzzle ID, and removes an entire puzzle if any solver position matches a held-out position. It then removes duplicate training positions. Position identity retains pieces, side to move, castling rights and legally relevant en passant; move clocks are ignored to catch equivalent positions. Malformed trajectories are recorded in `invalid.jsonl`. The new pool contains only `train.parquet`, with no new dev/test split.

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

Copy the default recipe to the experiment, set that copy's `train_file` to the selected `train.parquet`, and pass it through `scripts/run_grpo.py --template <recipe>`. Keep the default template's `train_file` unset; selection belongs to the experiment recipe. Formal reviewed validation uses L2/L3, with no puzzle validation split. verl still constructs an unused validation loader from the selected training file while its puzzle validation is disabled. Changing a recipe or reward creates a new experiment; continue only matching checkpoints through the existing resume protocol. If mixing sources, preserve held-out exclusions and deduplicate equivalent positions.

For subsequent reuse of this study's selected subset, scripts/report_training_data.py can record its IDs and statistics without copying training rows again. Record why reuse follows from the current evidence. Development validation remains L2/L3; DeepSeek Flash non-thinking remains the separate held-out final opponent.

## Optional category mixtures (no custom selection code required)

The helper is **optional**. The Meta-Agent must select data from the supplied pool,
using this helper or its own query/selector. No mixture or curriculum below
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
