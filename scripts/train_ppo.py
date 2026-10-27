#!/usr/bin/env python3
"""Editable verl PPO + LC0 research recipe, launched by the registered Worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from llmgo.rsi.contracts import atomic_json, read_json


def exec_into_verl(prefix, script, argv):
    """Re-exec this entry point inside the pinned verl venv, forwarding every argument.

    Round 4 (job j000014) dropped the argument vector here, so the submitted
    `--recipe ppo-control-sync.json --control` job silently ran ppo.json as an
    ordinary training run: the control that was supposed to isolate the serving and
    weight-sync path never executed and its learning_rate was 2e-6, not 0.
    """
    interpreter = str(prefix/"bin/python")
    os.execve(interpreter, [interpreter, str(script), *argv], os.environ.copy())


def main():
    manifest = read_json(os.environ["LLMGO_SITE"])["payload"]
    site = manifest["site"]
    prefix = Path(site.get("verl_venv", "/user/lyh4170/venvs/llmgo-verl-ppo-tf54-20261024"))
    if Path(sys.prefix).resolve() != prefix.resolve():
        repo = Path(__file__).resolve().parents[1]
        subprocess.run([sys._base_executable,str(repo/"scripts/bootstrap_verl.py"),"--prefix",str(prefix)],check=True)
        exec_into_verl(prefix, Path(__file__).resolve(), sys.argv[1:])
    from llmgo.training.verl_ppo import recipe_override, train, training_config
    from llmgo.training.lc0 import teacher_config
    output = Path(os.environ["LLMGO_OUTPUT"])
    candidate = Path(os.environ["LLMGO_CANDIDATE"])
    recipe_path, control = recipe_override(candidate/"ppo.json", sys.argv[1:])
    recipe = read_json(recipe_path)
    # A no-op (learning_rate=0) run is a diagnostic by construction: it must never be
    # able to publish itself as the candidate the next evaluation would measure.
    control_reason = "flag"
    if not control and recipe.get("training",{}).get("learning_rate") == 0:
        control, control_reason = True, "learning_rate=0"
    atomic_json(output/"recipe.json", {"recipe_path":str(recipe_path),"control":control,
        "control_reason":control_reason if control else None,"argv":sys.argv[1:],
        "recipe":recipe})
    gpus = manifest.get("resources",{}).get("gpu_count",site["gpu_count"])
    config = training_config(recipe.get("training",{}),gpus)
    if set(recipe)-{"training","teacher"}:
        raise ValueError("Full-game recipe accepts training/teacher; migrate old reward/curriculum recipe")
    reward = teacher_config(recipe.get("teacher",{}))
    atomic_json(output/"progress.json", {"stage":"ppo_starting","message":"启动 verl PPO + LC0", "allocated_gpu_count":gpus})
    model = train(os.environ["LLMGO_MODEL"],read_json(candidate/"skills.json"),site,reward,config,output/"training",
        allocated_gpus=gpus,resume_from=os.environ.get("LLMGO_RESUME_FROM") or None)
    if control:
        # A diagnostic recipe (for example learning_rate=0) must never become the
        # candidate model that the next evaluation measures.
        atomic_json(output/"control-result.json", {"model_path":str(model),"control":True,
            "note":"Diagnostic export; submission.json intentionally not written."})
    else:
        result = read_json(output/"training/training-result.json")
        atomic_json(output/"submission.json", {"model_path":str(model),
            "method":"verl_ppo/"+config["mode"],"loss_agg_mode":config["loss_agg_mode"],
            "selected_checkpoint_step":result.get("selected_checkpoint_step"),
            "internal_validation":result["internal_validation"],"validation":result["validation"],
            "stopped_early":result.get("stopped_early",False),"sft_enabled":False})


if __name__ == "__main__":
    main()
