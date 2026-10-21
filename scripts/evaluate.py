#!/usr/bin/env python3
"""Protected multi-level chess evaluation. Run inside a GPU job, not devspace."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.artifacts import source_snapshot
from llmgo.rsi.contracts import REPO, atomic_json, digest, file_hash, load_site, model_identity, protocol
from llmgo.rsi.evaluator import engine_identity, evaluate
from llmgo.rsi.ladder import evaluation_plan, panel_summary, opponent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", default=str(REPO/"configs/rsi/site.beijing.json"))
    p.add_argument("--model-path", required=True)
    p.add_argument("--candidate-root", type=Path, help="Editable research checkout; ignored for model track")
    p.add_argument("--track", choices=("player", "model", "both"), default="player")
    p.add_argument("--levels", default="1,2,3,4,5,6,7,8", help="Subset is a probe, never a full score")
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if args.batch_index < 0:
        p.error("batch-index must be nonnegative")
    plan = evaluation_plan({"levels":[int(x) for x in args.levels.split(",")],
                            "tracks":["player", "model"] if args.track=="both" else [args.track]})
    site = load_site(args.site)
    from llmgo.rsi.contracts import inside
    output = inside(site["user_root"], args.output)
    if output.exists():
        raise FileExistsError("Use a new output directory; preserve previous evidence")
    output.mkdir(parents=True)
    protected = source_snapshot(REPO, output/"protected-source")
    candidate = source_snapshot(args.candidate_root or REPO, output/"candidate-source")
    cards = Path(candidate["root"])/"src/llmgo/harness/skills/cards.json"
    if (Path(candidate["root"])/"skills.json").is_file():
        cards = Path(candidate["root"])/"skills.json"
    identity = model_identity(args.model_path)
    rows = []
    for track in plan["tracks"]:
        for level in plan["levels"]:
            spec = opponent(level)
            folder = output/f"{track}-L{level}"
            payload = {"site":site, "source":protected, "candidate_source":candidate, "reference_source":protected,
                "job_id":f"probe-{track}-L{level}", "candidate_id":candidate["sha256"][:12], "track":track,
                "output":str(folder), "model_path":str(Path(args.model_path).resolve()),
                "expected_model_sha256":identity["sha256"], "skills_path":str(cards) if cards.is_file() else None, "skills_sha256":file_hash(cards) if cards.is_file() else None,
                "protocol":protocol(spec, site["engine"]), "engine_identity":engine_identity(site["engine"],spec),
                "seed_schedule":[100000*level+100*args.batch_index+i for i in range(10)],
                "comparison_id":f"panel-v2:L{level}:b{args.batch_index}"}
            folder.mkdir()
            atomic_json(folder/"evaluation-input.json",payload)
            metrics_path = evaluate(payload)
            metrics = json.loads(metrics_path.read_text())
            rows.append({"track":track, "level":level, "statistics":metrics["statistics"], "path":str(metrics_path)})
            atomic_json(output/"panel.json", {"plan":plan,"panel":panel_summary(rows),"results":rows,
                                               "complete_requested":len(rows)==len(plan["levels"])*len(plan["tracks"])})
    print(json.dumps({"output":str(output/"panel.json"),"panel":panel_summary(rows)},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
