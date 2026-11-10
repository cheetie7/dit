"""Train an FM/AWR-FM policy from JSON configuration and optional CLI overrides."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import Subset

from awbc_models import FlowMatchingPolicy, FlowMatchingPolicyConfig
from awbc_data.dataset import ROBOMETER_CHUNK_SIZE
from awbc_data.inspect import tensor_report
from .config import ExperimentConfig, apply_overrides
from .data import prepare_datasets
from .runtime import resolve_device, seed_everything
from .trainer import Trainer
from .advantage import prepare_advantage_datasets


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    state = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if args.config:
        config = ExperimentConfig.from_json(args.config, args.overrides)
    elif state is not None:
        config = ExperimentConfig.from_dict(apply_overrides(state["config"], args.overrides))
    else:
        config = ExperimentConfig.from_dict(apply_overrides({}, args.overrides))
    source = (str(args.config.resolve()) if args.config else
              f"{args.resume.resolve()} (checkpoint config)" if args.resume else "package defaults")
    print(f"Config source: {source}", flush=True)
    print(f"Trainer source: {Path(__file__).with_name('trainer.py').resolve()}", flush=True)
    if config.training.num_steps is not None:
        print(f"Training budget: steps={config.training.num_steps}; num_epochs is ignored", flush=True)
    else:
        print(f"Training budget: epochs={config.training.num_epochs}; training.num_steps=null", flush=True)
    seed_everything(config.training.seed)
    info = json.loads((Path(config.dataset.root) / "meta/info.json").read_text())
    policy_config = (FlowMatchingPolicyConfig.from_dict(state["policy_config"]) if state is not None
                     else config.policy_configuration(info))
    horizon = policy_config.dit.action_horizon
    # Checkpoints made before configurable windows always used 10-step dataset chunks.
    data_horizon = state["split"].get("action_horizon", ROBOMETER_CHUNK_SIZE) if state is not None else horizon
    train, val, stats, manifest = prepare_datasets(config.dataset, policy_config.image_preprocess,
                                                   action_horizon=data_horizon)
    weighting, advantage_report = None, None
    if config.training.objective == "awr_fm":
        train, val, weighting, advantage_report = prepare_advantage_datasets(
            train, val, config.advantage_weighting, horizon)
        manifest["advantage_fingerprint"] = advantage_report["fingerprint"]
        manifest["retained_train_samples"] = len(train)
        manifest["retained_val_samples"] = len(val)
    if config.training.overfit_one_batch:
        count = min(config.training.batch_size, len(train))
        train = Subset(train, list(range(count))*config.training.overfit_batches)
        manifest["overfit_one_batch_size"] = count
        manifest["overfit_batches"] = config.training.overfit_batches
    if args.resume:
        directory = args.resume.resolve().parent.parent
    else:
        directory = args.run_dir or Path(config.output_root) / f"{config.experiment_name}-{timestamp()}"
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "normalization_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        (directory / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if advantage_report is not None:
            (directory / "advantage_provenance.json").write_text(json.dumps(advantage_report, indent=2) + "\n")
            (directory / "advantage_stats.json").write_text(json.dumps(advantage_report["statistics"], indent=2) + "\n")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        metadata = {"git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                    "torch_version": str(torch.__version__), "split_seed": config.dataset.split_seed,
                    "source_sha256": {f"{package}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                                      for package in ("awbc_data", "awbc_models", "awbc_training")
                                      for p in files(package).iterdir() if p.name.endswith(".py")}}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    config.to_json(directory / (f"resume_config_{timestamp()}.json" if args.resume else "config.json"))
    policy_config.normalization_stats_path = str((directory / "normalization_stats.json").resolve())
    if not args.resume:
        policy_config.to_json(directory / "policy_config.json")
    # Resume weights already contain the backbone; initializing it needs no ImageNet download.
    if state is not None:
        policy_config.vision.pretrained = False
    model = FlowMatchingPolicy(policy_config, stats=stats)
    trainer = Trainer(config, model, train, val, stats, manifest, directory,
                      resolve_device(config.training.device), resume_state=state, weighting=weighting)
    (directory / (f"resume_optimizer_{timestamp()}.json" if args.resume else "optimizer.json")).write_text(
        json.dumps(trainer.summary, indent=2) + "\n")
    if not args.resume:
        batch = next(iter(trainer.loader(train, batches=[list(range(min(len(train), config.training.batch_size)))])))
        tensors = {key: value for key, value in batch.items() if isinstance(value, (dict, torch.Tensor))}
        (directory / "batch_report.json").write_text(json.dumps(tensor_report(tensors), indent=2) + "\n")
        delta = (batch["video_timestamp"]-batch["video_target_timestamp"]).abs().max().item()
        assert delta <= 0.5/info["fps"]+1e-3
    try:
        result = trainer.fit()
        (directory / (f"resume_result_{timestamp()}.json" if args.resume else "result.json")).write_text(
            json.dumps(result, indent=2) + "\n")
        print(f"Experiment directory: {directory}")
    finally:
        trainer.logger.close()


if __name__ == "__main__":
    main()
