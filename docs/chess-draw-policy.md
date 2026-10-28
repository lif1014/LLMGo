# Chess draw policy

Repetition does not end a game: threefold claims and automatic fivefold
repetition are both disabled, including further repetitions. Playing, move previews, training episodes,
curriculum loading and the LC0 terminal-value shortcut use the same rule in
`llmgo.env.rules.chess_outcome`.

Stalemate, insufficient material, the existing fifty-move claim policy, the
seventy-five-move rule and the 200-ply protocol cap remain. Repeated positions
can still reach a fifty-move draw if neither side captures or moves a pawn.
The fifty-move claim is a standard rule under [FIDE Laws of Chess, Article
9.3](https://handbook.fide.com/chapter/e012023). This pipeline automatically
adjudicates eligible claims, including a declared next move that reaches the
threshold; it does not ask either player to submit a claim.

The regression fixture comes from task 1380000, PPO batch 4,
`train-00000004-0001`. The old policy ended after `47.Kb2` because Black could
declare `47...Rb3+` and claim threefold repetition. The new policy continues
both before and after that third occurrence, and also after five or more
occurrences. This policy supersedes the earlier threefold-only update.

Deploy updated live files without changing active research workspaces, frozen
sources, manifests or historical results. Existing workers keep their original
rules. Meta-Agent must merge the updated files into its next editable research
workspace before freezing a new training proposal. An existing campaign's
protected evaluation source requires the normal migration at an idle boundary;
updating live or candidate files does not migrate that frozen referee. Results
under different rule versions need explicit provenance and are not directly
comparable. Do not interrupt an active Meta turn or running job for this update.
