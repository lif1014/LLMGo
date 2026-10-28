# Optional reference KL stopping

`training.reference_kl_stop` defaults to `0`, which disables stopping at a
finite reference KL threshold. In particular, a measured KL above `0.1` no
longer stops a run using the default recipe. This applies before and after
optimizer updates.

KL measurement and the warning threshold (`reference_kl_warning=0.05`) remain.
The independent KL loss penalty (`kl_beta=0.05`) also remains, as do rollout
quality, actor/rollout synchronization and finite-update checks.

Meta-Agent is authorized to decide whether to re-enable KL stopping in a new
experiment, without asking the user again. For example:

```json
{"training": {"reference_kl_stop": 0.1}}
```

Any positive threshold enables the existing gate; the warning threshold must
not exceed it. Set `reference_kl_stop` explicitly to `0` to keep it disabled
when inheriting a recipe that previously specified `0.1`. Updating framework
defaults does not override explicit configuration.

Apply this default and record the actual choice in the next editable research
workspace and recipe. Do not change an active worker, its frozen source,
manifest or existing checkpoint identity. Changing code or configuration
starts a new experiment; it does not retroactively resume an incompatible
optimizer checkpoint. Earlier dated reports describing a default `0.1` limit
are historical and are superseded by this policy.
