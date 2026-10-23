#!/usr/bin/env python3
"""Editable verl PPO + LC0 research recipe, launched by the registered Worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.contracts import atomic_json, read_json


def main():
    manifest = read_json(os.environ["LLMGO_SITE"])["payload"]
    site = manifest["site"]
    prefix = Path(site.get("verl_venv", "/user/lyh4170/venvs/llmgo-verl-ppo-tf54-20261024"))
    if Path(sys.prefix).resolve() != prefix.resolve():
        repo = Path(__file__).resolve().parents[1]
        subprocess.run([sys._base_executable,str(repo/"scripts/bootstrap_verl.py"),"--prefix",str(prefix)],check=True)
        os.execve(str(prefix/"bin/python"),[str(prefix/"bin/python"),str(Path(__file__).resolve())],os.environ.copy())
    from llmgo.training.verl_ppo import train, training_config
    from llmgo.training.rewards import validate_reward
    from llmgo.training.rollout import load_curriculum
    output = Path(os.environ["LLMGO_OUTPUT"])
    candidate = Path(os.environ["LLMGO_CANDIDATE"])
    recipe = read_json(candidate/"ppo.json")
    gpus = manifest.get("resources",{}).get("gpu_count",site["gpu_count"])
    config = training_config(recipe.get("training",{}),gpus)
    reward = validate_reward(recipe.get("reward",{}))
    atomic_json(output/"progress.json", {"stage":"ppo_starting","message":"启动 verl PPO + LC0", "allocated_gpu_count":gpus})
    model = train(os.environ["LLMGO_MODEL"],load_curriculum(candidate/"curriculum.jsonl"),
        read_json(candidate/"skills.json"),site,reward,config,output/"training",
        allocated_gpus=gpus,resume_from=os.environ.get("LLMGO_RESUME_FROM") or None)
    atomic_json(output/"submission.json", {"model_path":str(model),"method":"verl_ppo/lc0_td0", "sft_enabled":False})


if __name__ == "__main__":
    main()
