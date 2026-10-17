# Chess skills

These cards are original short rules inferred from public game scores, not a
runtime opening book or engine policy. `skills.py` selects at most three cards
by stage and board conditions. Only the selected cards enter the prompt; the
web tool library exposes every card, trigger, counterexample and source.

`master_games.pgn` retains the historical game scores (without copied modern
annotations). Each card's evidence links a game ID, the zero-based number of
plies before the move, and SAN. Tests replay every game and verify the evidence.

Sources read for this revision:

- Morphy, Paris consultation, 1858: https://lichess.org/study/LteGa5JT/cFv9dmED
  Development, coordinated rooks, and a forcing finish. The sacrifice is a
  counterexample to a blanket prohibition on sacrificing material.
- Rotlewi–Rubinstein, Lodz, 1907: https://lichess.org/study/swGPAS0z/JHGjKfwb
  Coordinated bishops/rooks and concrete calculation before accepting material.
- Capablanca–Tartakower, New York, 1924: https://lichess.org/study/M3mZMe8l/MshiKf1G
  Active king and rook supporting passed pawns. This does not imply that every
  rook exchange or king advance is sound.

RSI may edit these files and the retriever in its run-local harness submission.
Add triggers, counterexamples, reproducible evidence and held-out regression
positions when adding a card. A plausible rule is not proof of an Elo gain.

## Early opening knowledge

Six additional cards cover White's central start, Black's central reply, e4/e5
development, White against the Sicilian and Caro-Kann, and d4/d5 development.
They include short illustrative SAN lines checked against the Lichess opening
dataset: https://github.com/lichess-org/chess-openings (b.tsv, c.tsv, d.tsv).
Each card records its source; `opening_examples` retains testable notation.
The prompt carries the example as part of its prose rule, never a compulsory move.

These cards enter only piece/move questions, during the first six full moves
(initial central-pawn advice only the first three), while both kings and queens
remain on their home squares and the moving side is not in check. Broad pawn
structures select the card; Sicilian/Caro-Kann recapture structures are included.
No TSV, FEN-to-move table, engine, or opening continuation lookup is used at runtime.
The same three-card limit applies. Check, forcing-reply and capture reminders
have higher priority; an example line cannot authorize ignoring an actual threat.
After an unfamiliar reply, use the actual legal options and general development
principles. The examples cover a small starting repertoire, not all openings.
