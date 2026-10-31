"""Bounded synthetic-only FM/AWR acceptance. No RoboMeter labels or robot data are fabricated."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from awbc_data.advantage import AdvantageDataset, PrecomputedAdvantageProvider, window_metadata
from awbc_models import (FlowMatchingPolicy, FlowMatchingPolicyConfig, VisualEncoderConfig,
                         ObservationEncoderConfig, ConditionalDiTConfig, AdvantageNormalizer, AdvantageWeighting)
from .config import ExperimentConfig
from .runtime import seed_everything
from .trainer import Trainer
from .train import timestamp


class SyntheticWindows(Dataset):
    def __init__(self, episode_ids):
        self.episode_ids = episode_ids
        self.samples = [(e, start, 3 if start == 20 else 10) for e in episode_ids for start in (0, 10, 20)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode, start, length = self.samples[index]
        return {"observation": {"images": torch.full((1, 3, 32, 48), episode/4+start/40),
                                "state": torch.linspace(-1, 1, 8)},
                "action": torch.full((10, 3), episode/10+start/50),
                "action_is_pad": torch.arange(10) >= length, "episode_index": torch.tensor(episode),
                "frame_index": torch.tensor(start)}


def main():
    directory = Path("outputs") / f"stage8_synthetic-{timestamp()}"
    directory.mkdir(parents=True, exist_ok=False)
    raw_train, raw_val = SyntheticWindows([0, 1]), SyntheticWindows([2, 3])
    metadata = {"source": "synthetic_acceptance_only", "model_version": "artificial-v1", "semantics": "advantage",
                "advantage_definition": "Deterministic artificial scalars; not robot quality or RoboMeter output",
                "sampling_frequency": 30, "window_length": 10, "training_trajectory_ids": [],
                "evaluation_trajectory_ids": [0, 1, 2, 3], "provenance_reviewed": True}
    rows = [{**window_metadata(e, t, length), "advantage": float(index-2) if e < 2 else 100.+index}
            for index, (e, t, length) in enumerate(raw_train.samples+raw_val.samples)]
    (directory / "synthetic_labels.json").write_text(json.dumps(rows, indent=2)+"\n")
    (directory / "synthetic_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    provider = PrecomputedAdvantageProvider.from_file(directory / "synthetic_labels.json", metadata)
    coverage = provider.validate_windows(rows, 30)
    train, val = AdvantageDataset(raw_train, provider), AdvantageDataset(raw_val, provider)
    stats = {"action": {"min": [-1.]*3, "max": [1.]*3}, "observation.state": {"min": [-1.]*8, "max": [1.]*8}}
    manifest = {"synthetic_only": True, "train_episodes": [0, 1], "val_episodes": [2, 3],
                "advantage_fingerprint": provider.fingerprint}
    results = {}
    for mode in ("fm", "awr_fm"):
        seed_everything(42)
        config = ExperimentConfig()
        config.training = replace(config.training, objective=mode, num_epochs=1, batch_size=2, num_workers=0,
                                    gradient_accumulation_steps=1, max_updates=3)
        config.amp.enabled = False; config.logging.use_wandb = False; config.logging.log_interval = 1
        config.scheduler.warmup_steps = 0; config.validation.batch_size = 2
        config.advantage_weighting.enabled = mode == "awr_fm"
        config.advantage_weighting.beta = 0.5
        policy_config = FlowMatchingPolicyConfig(
            vision=VisualEncoderConfig(d_model=32, pretrained=False, num_cameras=1),
            observation=ObservationEncoderConfig(state_dim=8, hidden_dim=32, num_layers=1, num_heads=4, ffn_dim=64, dropout=0),
            dit=ConditionalDiTConfig(action_dim=3, action_horizon=10, hidden_dim=32, num_layers=1, num_heads=4,
                                      ff_dim=64, dropout=0, timestep_embedding_dim=16, timestep_hidden_dim=32),
            camera_names=("synthetic_camera",))
        policy = FlowMatchingPolicy(policy_config, stats)
        weighting = AdvantageWeighting(AdvantageNormalizer.fit(train.advantages), beta=0.5) if mode == "awr_fm" else None
        run_dir = directory / mode; run_dir.mkdir()
        config.to_json(run_dir / "config.json")
        policy_config.to_json(run_dir / "policy_config.json")
        trainer = Trainer(config, policy, train, val, stats, manifest, run_dir, torch.device("cpu"), weighting=weighting)
        result = trainer.fit()
        validation = trainer.validate()
        assert validation == trainer.validate()
        assert result["global_step"] == result["ema_updates"] == 3
        assert all(p.grad is None for p in policy.vision_encoder.backbone.parameters())
        records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
        results[mode] = {**result, "last_epoch": records[-1], "checkpoint": str(run_dir / "checkpoints/latest_checkpoint.pt")}
        trainer.logger.close()
    report = {"synthetic_only": True, "real_awr_training_started": False, "coverage": coverage,
              "source": metadata, "train_advantage_mean": train.advantages.mean().item(),
              "train_advantage_std": train.advantages.std(correction=0).item(), "modes": results,
              "run_directory": str(directory.resolve())}
    output = Path("artifacts/awbc_awr"); output.mkdir(parents=True, exist_ok=True)
    (output / "synthetic_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"run_directory": str(directory), "awr_statistics": results["awr_fm"]["last_epoch"]}, indent=2))


if __name__ == "__main__":
    main()
