# Bounded tactical observations

The Harness reports limited rule checks. A move with no reported warning is not
certified safe or recommended: a fork, a delayed material loss or a longer mating
attack can remain. A warning is a fact to inspect, not a forbidden move. All legal
moves remain available. This wording change adds no search, engine move selection,
candidate pruning, reward changes or PPO changes.

The live Harness states these limits even when no observation line is emitted.
Its existing skill cards also distinguish short-horizon observations from a
complete tactical assessment.

## Existing r2 research variants

The round-13 r2 Harness has an additional `safe moves` / `avoid` summary and a
`material-safety` skill in both `skills.json` and the packaged cards. Updating live
alone does not replace that inherited research code. The accompanying
[r2 patch](patches/r2-tactical-observations.patch) removes the endorsement in all
four places while retaining the existing checks, budgets, ordering and skill
selection. Internal `kind='safe'` remains a compatibility key; it is not printed
as a safety claim.

Only apply it in a **new editable research workspace**, before normal proposal
registration and source freezing:

```bash
git apply --check '/user/lyh4170/LLMGo!/docs/patches/r2-tactical-observations.patch'
git apply '/user/lyh4170/LLMGo!/docs/patches/r2-tactical-observations.patch'
```

Review differences first if the check fails. Do not overwrite an independently
modified Harness. Never apply it to a registered candidate, worker source, an old
submitted workspace, manifest, active experiment or historical result. In
particular, tasks **1395785 and 1395788** retain their original prompts and skills
throughout their current comparison. Adoption must be recorded as a new source
identity; old scores continue to describe the old prompts.
