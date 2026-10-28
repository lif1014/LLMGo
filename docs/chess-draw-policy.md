# Chess draw policy

Threefold repetition does not end a game, including when a player could claim
it by declaring the next move. Playing, move previews, training episodes,
curriculum loading and the LC0 terminal-value shortcut use the same rule in
`llmgo.env.rules.chess_outcome`.

Fivefold repetition, stalemate, insufficient material, the existing fifty-move
claim policy, the seventy-five-move rule and the 200-ply protocol cap remain.
This change does not remove every repetition-related draw.

The regression fixture comes from task 1380000, PPO batch 4,
`train-00000004-0001`. The old policy ended after `47.Kb2` because Black could
declare `47...Rb3+` and claim threefold repetition. The new policy continues
both before and after that third occurrence.

Deploy updated live files without changing active research workspaces, frozen
sources, manifests or historical results. Existing workers keep their original
rules. Meta-Agent must merge the updated files into its next editable research
workspace before freezing a new training proposal. An existing campaign's
protected evaluation source requires the normal migration at an idle boundary;
updating live or candidate files does not migrate that frozen referee. Results
under different rule versions need explicit provenance and are not directly
comparable. Do not interrupt an active Meta turn or running job for this update.
