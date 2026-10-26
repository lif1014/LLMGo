#!/usr/bin/env python3
"""Replay trusted, own saved PPO batches without changing training artifacts."""
import argparse
import gc
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.contracts import atomic_json, file_hash, utc
from llmgo.training.advantages import guarded_advantages


def check(paths, penalty=1.0):
    import torch
    torch.set_num_threads(1)
    rows = []
    for path in paths:
        # Pickle is needed for verl's NumPy metadata. Only pass trusted own batches.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        tensors, metadata = saved["tensors"], saved["non_tensor"]
        rewards, mask = tensors["token_level_rewards"], tensors["response_mask"]
        original_rewards = rewards.clone()
        advantages, _, diag = guarded_advantages(rewards, mask, metadata["failure_reason"], penalty=penalty)
        assert torch.equal(rewards, original_rewards), "Replay changed raw TD rewards"
        active = mask.bool()
        invalid = torch.tensor([bool(r) for r in metadata["failure_reason"]]) & active.any(-1)
        old_positive = int(((tensors["advantages"] > 0) & active & invalid[:,None]).any(-1).sum())
        assert not bool(((advantages >= 0) & active & invalid[:,None]).any()), "Invalid answer received nonnegative advantage"
        row = {key:value for key,value in diag.items() if key != "samples"}
        row.update(path=str(path), step=int(saved["step"]), old_invalid_positive=old_positive,
                   raw_td_unchanged=True)
        rows.append(row)
        del saved, tensors, metadata, rewards, mask, original_rewards, advantages, active, invalid
        gc.collect()
    if not rows:
        raise ValueError("No saved batches matched")
    return {"ok":True, "utc":utc(), "batch_count":len(rows),
            "sample_count":sum(r["valid"]+r["invalid"] for r in rows),
            "invalid":sum(r["invalid"] for r in rows),
            "old_invalid_positive":sum(r["old_invalid_positive"] for r in rows),
            "new_invalid_positive":sum(r["invalid_positive_advantages"] for r in rows),
            "guard_source_sha256":file_hash(Path(__file__).resolve().parents[1]/"src/llmgo/training/advantages.py"),
            "read_only_inputs":True, "batches":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", action="append", required=True, type=Path,
                        help="Own trusted training/batches directory; may repeat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--penalty", type=float, default=1.0)
    args = parser.parse_args()
    paths = [p for folder in args.batches for p in sorted(folder.glob("step-*.pt"))]
    result = check(paths, args.penalty)
    atomic_json(args.output, result)
    print({k:v for k,v in result.items() if k != "batches"}, flush=True)


if __name__ == "__main__":
    main()
