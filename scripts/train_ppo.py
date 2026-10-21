#!/usr/bin/env python3
"""Optional starting recipe, editable by Meta-Agent. Not an RSI method restriction."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.contracts import atomic_json, read_json
from llmgo.training.ppo import train, training_config
from llmgo.training.rewards import Lc0Reward, validate_reward
from llmgo.training.rollout import load_curriculum


def main():
    output = Path(os.environ["LLMGO_OUTPUT"])
    candidate = Path(os.environ["LLMGO_CANDIDATE"])
    site = read_json(os.environ["LLMGO_SITE"])["payload"]["site"]
    recipe = read_json(candidate/"ppo.json")
    config = training_config(recipe.get("training",{}))
    reward = validate_reward(recipe.get("reward",{}))
    provider = Lc0Reward(site,reward)
    def progress(stage,message,**extra):
        atomic_json(output/"progress.json", {"stage":stage,"message":message,**extra})
    resume = None
    prior = os.environ.get("LLMGO_RESUME_FROM")
    if prior:
        checkpoints = sorted((Path(prior)/"training").glob("checkpoint-*/state.json"))
        if checkpoints:
            resume = checkpoints[-1].parent
    try:
        model = train(os.environ["LLMGO_MODEL"], load_curriculum(candidate/"curriculum.jsonl"),
                      read_json(candidate/"skills.json"), provider,reward,config,output/"training",
                      resume=resume,progress=progress)
    finally:
        provider.close()
    atomic_json(output/"submission.json", {"model_path":str(model),"method":"initial hf_ppo/lc0 recipe"})


if __name__ == "__main__":
    main()
