"""Prepare an evaluation proposal without running GPU work in the Meta-Agent."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from .contracts import atomic_json, digest, inside, read_json
from .ladder import evaluation_plan


def level_number(value):
    try:
        level = int(str(value).upper().removeprefix("L"))
    except ValueError:
        raise argparse.ArgumentTypeError("Use a difficulty from L1 to L8") from None
    if level not in range(1,9):
        raise argparse.ArgumentTypeError("Use a difficulty from L1 to L8")
    return level


def request_evaluation(workspace, *, level=None, full_panel=False, track="player",
                       proposal="proposal.json", hypothesis=None, batch_index=None, gpus=8):
    if (level is None and not full_panel) or (level is not None and full_panel):
        raise ValueError("Specify exactly one of level or full_panel")
    if level is not None and (type(level) is not int or level not in range(1,9)):
        raise ValueError("level must be an integer from 1 to 8")
    if type(gpus) is not int or not 1 <= gpus <= 8:
        raise ValueError("gpus must be an integer from 1 to 8")
    workspace = Path(workspace).resolve()
    path = inside(workspace, workspace/proposal)
    if path.parent != workspace:
        raise ValueError("The proposal must be in the registered workspace root")
    data = read_json(path) if path.exists() else {"jobs":[], "recommend":False}
    if not isinstance(data, dict):
        raise ValueError("Expected a proposal JSON object")
    if hypothesis is not None:
        if not isinstance(hypothesis,str) or not hypothesis.strip():
            raise ValueError("hypothesis must be nonempty")
        data["hypothesis"] = hypothesis
    elif "hypothesis" not in data:
        data["hypothesis"] = f"Measure the current candidate against {'L1-L8' if full_panel else 'L'+str(level)}"
    if batch_index is not None:
        if type(batch_index) is not int or batch_index < 0:
            raise ValueError("batch_index must be nonnegative")
        data["evaluation_batch"] = batch_index
    tracks = ["player","model"] if track == "both" else [track]
    data["evaluation"] = evaluation_plan({"levels":list(range(1,9)) if full_panel else [level], "tracks":tracks})
    data["evaluation_gpus"] = gpus
    digest(data)
    atomic_json(path, data)
    return {"status":"ready", "summary":f"Evaluation requested: {data['evaluation']}", "proposal_path":str(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--level",type=level_number,help="One difficulty, e.g. L2 or 2")
    target.add_argument("--full-panel",action="store_true",help="All L1-L8 for a complete score")
    parser.add_argument("--track",choices=("player","model","both"),default="player")
    parser.add_argument("--proposal",default="proposal.json")
    parser.add_argument("--hypothesis")
    parser.add_argument("--batch-index",type=int)
    parser.add_argument("--gpus",type=int,default=8,help="Parallel Qwen replicas, 1-8; default 8")
    args = parser.parse_args(argv)
    try:
        result = request_evaluation(Path.cwd(),level=args.level,full_panel=args.full_panel,
            track=args.track,proposal=args.proposal,hypothesis=args.hypothesis,batch_index=args.batch_index,gpus=args.gpus)
    except (ValueError,OSError) as error:
        parser.error(str(error))
    print(json.dumps(result,ensure_ascii=False))
    return 0
