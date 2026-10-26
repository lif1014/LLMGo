#!/usr/bin/env python3
"""Normalize an already-exported checkpoint so the pinned evaluator can serve it.

Round-5 job j000016 produced healthy training artifacts, but its HF export could not be
loaded by the evaluation image: `ValueError: Tokenizer class TokenizersBackend does not
exist or is not currently imported` (job j000018, 0/20 games). Weights are fine; only the
tokenizer side carries transformers-5.x naming. This tool rebuilds the auxiliary file set
from the base model after proving both vocabularies are identical, then reports whether
the pinned inference interpreter can load the result.

    python scripts/repair_export.py --source <export> --target <new dir> --base <base model>

Round-7 added a second, independent defect: `verl.model_merger` also writes parameter
names in a transformers-5.4 layout (`model.language_model.language_model.language_model.*`
and `model.language_model.visual.*`) that vLLM cannot resolve, so the engine aborted with
`Following weights were not initialized from checkpoint` (jobs j000019/j000020). Those
names are remapped onto the base checkpoint naming here, streaming the tensor payload.

The source export is never modified: weights are linked (hardlink when possible) into the
target directory, the remap replaces the target's own shard, and the report is written next
to it. `--check-only` prints the name verdict for an existing directory.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.contracts import atomic_json, model_identity, read_json
from llmgo.training.export_weights import normalize_export_weights, verify_weight_names
from llmgo.training.verl_ppo import normalize_export_for_inference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="exported HF directory to repair")
    parser.add_argument("--target", help="new directory to write (required unless --check-only)")
    parser.add_argument("--base", required=True, help="base model with a known-good tokenizer")
    parser.add_argument("--report", help="where to write the JSON report")
    parser.add_argument("--checker", action="append", default=[],
        help="interpreter of the pinned inference stack (repeatable)")
    parser.add_argument("--check-only", action="store_true",
        help="report the parameter-name verdict for --source without writing anything")
    args = parser.parse_args(argv)
    if args.check_only:
        report = verify_weight_names(args.source, args.base)
        print(json.dumps(report))
        return 0 if report["ok"] else 1
    if not args.target:
        parser.error("--target is required unless --check-only is given")
    source, target, base = Path(args.source), Path(args.target), Path(args.base)
    if not (source/"config.json").is_file():
        parser.error(f"{source} is not an HF export")
    if target.exists():
        parser.error(f"{target} already exists; verify it instead of overwriting")
    target.mkdir(parents=True)
    linked, copied = {}, []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        destination = target/path.name
        if path.suffix in {".safetensors", ".bin"}:
            try:
                os.link(path, destination)
                linked[path.name] = str(path)
                continue
            except OSError:
                pass
        shutil.copyfile(path, destination)
        copied.append(path.name)
    weights = normalize_export_weights(target, base)
    normalization = normalize_export_for_inference(target, base, args.checker)
    report = {"source": str(source), "target": str(target),
        "source_identity": model_identity(source), "target_identity": model_identity(target),
        "linked_weights": linked, "copied_files": copied, "weights": weights,
        "normalization": normalization,
        "source_tokenizer_config": read_json(source/"tokenizer_config.json")}
    atomic_json(Path(args.report) if args.report else target.parent/(target.name+".repair.json"), report)
    verdict = normalization["dynamic_verification"]
    print(json.dumps({"ok": verdict.get("ok", verdict.get("static_ok")), "verdict": verdict,
        "vocabulary_size": normalization["vocabulary_size"], "removed": normalization["removed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
