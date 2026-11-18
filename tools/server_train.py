"""Prepare immutable run configs and execute dense critic -> export -> policy."""

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


def prepare_configs(settings, repo, run):
    data = settings["data"]
    dataset = {**data["dataset"], "root": data["root"]}
    critic = merge(json.loads((repo / "configs/iql_dense_critic.json").read_text()), settings["critic"])
    critic.update(dataset=dataset, labels_path=data["labels_path"], progress_path=data["progress_path"],
                  action_horizon=data["action_horizon"])
    policy = merge(json.loads((repo / "configs/iql_dense_awbc.json").read_text()), settings["policy"])
    policy["dataset"] = deepcopy(dataset)
    policy["policy"]["dit"]["action_horizon"] = data["action_horizon"]
    policy["policy"]["n_action_steps"] = min(8, data["action_horizon"])
    policy["advantage_weighting"].update(
        labels_path=str(run / "critic/advantages.jsonl"),
        metadata_path=str(run / "critic/advantages.metadata.json"),
    )
    policy["dense_awbc"]["progress_path"] = data["progress_path"]
    policy["sim_eval"]["python"] = settings["runtime"]["python"]
    policy["sim_eval"]["mapping"] = str(repo / "configs/eval_sim_block_insertion.json")
    return critic, policy


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
    critic, policy = prepare_configs(settings, repo, run)
    for name, values in (("server", settings), ("critic", critic), ("policy", policy)):
        (run / f"{name}.json").write_text(json.dumps(values, indent=2) + "\n")
    environment = os.environ.copy()
    environment["TORCH_HOME"] = runtime["pretrained_cache"]
    environment["MPLCONFIGDIR"] = str(run / "matplotlib")
    commands = [
        [sys.executable, "-m", "awbc_training.iql_dense", "train", "--config", str(run / "critic.json"),
         "--run-dir", str(run / "critic")],
        [sys.executable, "-m", "awbc_training.iql_dense", "export",
         "--checkpoint", str(run / "critic/checkpoints/latest_checkpoint.pt"),
         "--output", str(run / "critic/advantages.jsonl")],
        [sys.executable, "-m", "awbc_training.train", "--config", str(run / "policy.json"),
         "--run-dir", str(run / "policy")],
    ]
    print(f"Run directory: {run}", flush=True)
    for command in commands:
        print("Executing " + " ".join(command), flush=True)
        subprocess.run(command, cwd=repo, env=environment, check=True)


if __name__ == "__main__":
    main()
