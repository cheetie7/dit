"""Export advantages from a sparse or dense IQL checkpoint, then train IQL-FM."""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def merge(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def prepare_policy_config(settings, repo, run, critic):
    policy = merge(json.loads((repo / "configs/iql_fm.json").read_text()), settings["policy"])
    policy["training"]["objective"] = "iql_fm"
    policy["dataset"] = deepcopy(critic["config"]["dataset"])
    policy["dataset"]["root"] = settings["data"]["root"]
    horizon = critic["model_config"]["action_horizon"]
    policy["policy"]["dit"]["action_horizon"] = horizon
    policy["policy"]["n_action_steps"] = min(8, horizon)
    policy["advantage_weighting"].update(
        labels_path=str(run / "advantages.jsonl"),
        metadata_path=str(run / "advantages.metadata.json"),
    )
    policy["sim_eval"]["python"] = settings["runtime"]["python"]
    policy["sim_eval"]["mapping"] = str(repo / "configs/eval_sim_block_insertion.json")
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    settings = json.loads(args.config.read_text())
    runtime = settings["runtime"]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S-%f}-job{os.environ.get('SLURM_JOB_ID', 'local')}"
    run = Path(runtime["output_root"]).resolve() / run_id
    run.mkdir(parents=True)
    import torch

    checkpoint = Path(settings["critic_checkpoint"]).resolve()
    critic = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mode = critic["data_contract"].get("reward_mode", "sparse")
    module = {"sparse": "awbc_training.iql", "dense": "awbc_training.iql_dense"}[mode]
    policy = prepare_policy_config(settings, repo, run, critic)
    del critic
    for name, values in (("server", settings), ("policy", policy)):
        (run / f"{name}.json").write_text(json.dumps(values, indent=2) + "\n")
    environment = os.environ.copy()
    environment["TORCH_HOME"] = runtime["pretrained_cache"]
    environment["MPLCONFIGDIR"] = str(run / "matplotlib")
    export = [sys.executable, "-m", module, "export", "--checkpoint", str(checkpoint),
              "--output", str(run / "advantages.jsonl"),
              "--set", "dataset.root=" + settings["data"]["root"],
              "--set", "labels_path=" + settings["data"]["labels_path"]]
    if mode == "dense":
        export.extend(["--set", "progress_path=" + settings["data"]["progress_path"]])
    commands = [
        export,
        [sys.executable, "-m", "awbc_training.train", "--config", str(run / "policy.json"),
         "--run-dir", str(run / "policy")],
    ]
    print(f"Run directory: {run}", flush=True)
    for command in commands:
        print("Executing " + " ".join(command), flush=True)
        subprocess.run(command, cwd=repo, env=environment, check=True)


if __name__ == "__main__":
    main()
