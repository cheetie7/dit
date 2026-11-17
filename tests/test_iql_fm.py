from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from awbc_data import ImagePreprocessConfig
from awbc_data.advantage import sample_metadata
from awbc_models import FlowMatchingObjective, FlowMatchingPolicy
from awbc_models.advantage_weighting import AdvantageNormalizer, AdvantageWeighting
from awbc_training.advantage import prepare_iql_advantage_dataset
from awbc_training.config import DatasetConfig, ExperimentConfig
from awbc_training.data import prepare_datasets
from awbc_training.inference import load_policy
from awbc_training.train import main
from awbc_training.trainer import Trainer
from test_advantage_weighting import tiny_policy
from test_awbc_data import root
from test_training import ToyDataset, ToyPolicy


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class LabeledToyDataset(ToyDataset):
    def __init__(self, advantages):
        super().__init__(len(advantages))
        self.advantages = torch.tensor(advantages, dtype=torch.float32)

    def __getitem__(self, index):
        return {**super().__getitem__(index), "advantage": self.advantages[index],
                "trajectory_id": torch.tensor(0)}


class ToyFlowPolicy(ToyPolicy):
    def forward(self, batch, sample_weight=None, **pair):
        prediction = self.dit(batch["action"][:, 0]).unsqueeze(1)
        return FlowMatchingObjective().compute_loss(
            prediction, batch["target"].unsqueeze(1), sample_weight=sample_weight
        )


def one_update(tmp_path, name, advantages, batch_size, accumulation, objective="iql_fm"):
    torch.manual_seed(10)
    dataset = LabeledToyDataset(advantages)
    policy = ToyFlowPolicy()
    config = ExperimentConfig.from_json(CONFIGS / "iql_fm.json")
    config.training = replace(config.training, objective=objective, num_steps=1, batch_size=batch_size,
                              num_workers=0, gradient_accumulation_steps=accumulation, gradient_clip_norm=100.)
    config.amp.enabled = config.ema.enabled = config.logging.progress_bar = False
    config.validation.update_interval = config.checkpoint.update_interval = 0
    directory = tmp_path / name
    directory.mkdir()
    weighting = AdvantageWeighting(AdvantageNormalizer("none")) if objective == "iql_fm" else None
    trainer = Trainer(config, policy, dataset, dataset, {}, {}, directory, torch.device("cpu"), weighting=weighting)
    batch = torch.utils.data.default_collate([dataset[i] for i in range(len(dataset))])
    residual = policy.dit(batch["action"][:, 0]).squeeze(1) - batch["target"].squeeze(1)
    weights = [min(10., max(0.1, math.exp(a))) for a in advantages] if weighting else [1.] * len(advantages)
    manual_loss = sum(w * error.item()**2 for w, error in zip(weights, residual)) / sum(weights)
    manual_gradient = (2. * torch.tensor(weights) * residual.detach()) / sum(weights)
    expected = (manual_gradient @ batch["action"][:, 0], manual_gradient.sum().reshape(1))

    gradients = []
    optimizer_step = trainer.optimizer.step

    def capture_gradient(*args, **kwargs):
        gradients.extend(parameter.grad.detach().clone().flatten() for parameter in policy.parameters())
        return optimizer_step(*args, **kwargs)

    try:
        with patch.object(trainer.optimizer, "step", side_effect=capture_gradient):
            trainer.fit()
    finally:
        trainer.logger.close()
    metrics = trainer.last_update_metrics
    assert metrics["train/loss"] == pytest.approx(manual_loss, abs=1e-7)
    for actual, target in zip(gradients, expected, strict=True):
        torch.testing.assert_close(actual, target.flatten())
    return metrics, gradients


def test_uniform_weights_have_fm_loss_and_gradients(tmp_path):
    scores = [0.5] * 4
    fm, fm_grad = one_update(tmp_path, "fm", scores, 4, 1, "fm")
    iql, iql_grad = one_update(tmp_path, "iql", scores, 4, 1)
    assert iql["train/weighted_loss"] == pytest.approx(fm["train/loss"])
    assert iql["train/unweighted_loss"] == fm["train/loss"]
    for actual, expected in zip(iql_grad, fm_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_weighted_loss_and_gradient_accumulation_match_large_batch(tmp_path):
    scores = [-4., 0., 1., 4.]
    large, large_grad = one_update(tmp_path, "large", scores, 4, 1)
    micro, micro_grad = one_update(tmp_path, "micro", scores, 2, 2)
    assert micro["train/weighted_loss"] == pytest.approx(large["train/weighted_loss"])
    assert micro["weight/clip_fraction"] == 0.5
    assert micro["weight/effective_sample_size"] == pytest.approx(large["weight/effective_sample_size"])
    for actual, expected in zip(micro_grad, large_grad, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.fixture
def critic_export(root, tmp_path):
    # Four full windows and the one-action tail [12,13).
    config = DatasetConfig(root=str(root), stratify_key=None, train_stride=3)
    train, val, _, split = prepare_datasets(config, ImagePreprocessConfig(size=(24, 32)), action_horizon=4)
    rows = []
    for index in range(len(train)):
        row = sample_metadata(train, index, 4)
        row.pop("sample_id")
        rows.append({**row, "valid_length": row["end_timestep"] - row["start_timestep"], "advantage": index - 2.})
    metadata = {
        "checkpoint": {"path": "/critic/latest_checkpoint.pt", "sha256": "a" * 64, "global_step": 5},
        "split": split, "data_contract": {"action_horizon": 4, "train_stride": 3},
        "advantage_definition": "min(target_q1, target_q2) - v", "record_count": len(rows),
    }
    settings = ExperimentConfig.from_json(CONFIGS / "iql_fm.json").advantage_weighting
    settings.labels_path = str(tmp_path / "advantages.jsonl")
    settings.metadata_path = str(tmp_path / "advantages.metadata.json")
    write_export(settings, rows[::-1], metadata)
    yield config, train, val, split, rows, metadata, settings
    train.dataset.close()
    val.dataset.close()


def write_export(settings, rows, metadata):
    Path(settings.labels_path).write_text("".join(json.dumps(row) + "\n" for row in rows))
    Path(settings.metadata_path).write_text(json.dumps(metadata))


def test_iql_joins_training_windows_including_short_tail(critic_export):
    _, train, val, split, rows, metadata, settings = critic_export
    settings.normalization = "zscore"
    weighted, weighting, report = prepare_iql_advantage_dataset(train, settings, 4, split)
    assert weighted.advantages.tolist() == [row["advantage"] for row in rows]
    assert weighted[0]["action_is_pad"].tolist() == [False] * 4
    tail = weighted[-1]
    assert tail["action_is_pad"].tolist() == [False, True, True, True]
    assert (tail["start_timestep"].item(), tail["end_timestep"].item(), tail["valid_length"].item()) == (12, 13, 1)
    assert not val[0]["advantage_valid"]
    assert report["critic_checkpoint"] == metadata["checkpoint"]
    assert report["statistics"]["count"] == len(train)
    assert report["statistics"]["training_trajectory_ids"] == split["train_episodes"]
    assert weighting.normalizer.mean.item() == pytest.approx(0.)
    assert weighting.normalizer.std.item() == pytest.approx(math.sqrt(2))
    assert not weighting(weighted.advantages.clone().requires_grad_()).requires_grad


@pytest.mark.parametrize(("failure", "message"), [
    ("missing", "Missing advantage"),
    ("duplicate", "Duplicate"),
    ("nonfinite", "Nonfinite advantage"),
    ("horizon", "action_horizon=5.*action_horizon=4.*re-export"),
    ("row_horizon", "Label horizon 5.*metadata horizon 4.*re-export"),
    ("stride", "train_stride=4.*train_stride=3.*re-export"),
    ("valid_length", "valid_length=2.*expected 1"),
    ("end_timestep", "end_timestep=14.*expected 13"),
    ("split", "split.*train_episodes"),
    ("val_label", "Unexpected labels"),
])
def test_iql_rejects_inconsistent_exports(critic_export, failure, message):
    _, train, val, split, rows, metadata, settings = critic_export
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(rows[0])
    elif failure == "nonfinite":
        rows[0]["advantage"] = float("nan")
    elif failure == "horizon":
        metadata["data_contract"]["action_horizon"] = 5
    elif failure == "row_horizon":
        rows[0]["action_horizon"] = 5
    elif failure == "stride":
        metadata["data_contract"]["train_stride"] = 4
    elif failure == "valid_length":
        rows[-1]["valid_length"] = 2
    elif failure == "end_timestep":
        rows[-1]["end_timestep"] = 14
    elif failure == "split":
        metadata["split"] = {**split, "train_episodes": split["val_episodes"]}
    elif failure == "val_label":
        rows.append({**sample_metadata(val, 0, 4), "valid_length": 4, "advantage": 1.})
    write_export(settings, rows, metadata)
    with pytest.raises(ValueError, match=message):
        prepare_iql_advantage_dataset(train, settings, 4, split)


def test_iql_cli_validation_resume_and_inference(critic_export, tmp_path, monkeypatch):
    data_config, _, _, split, rows, metadata, settings = critic_export
    for objective in ("fm", "awr_fm", "iql_fm"):
        assert ExperimentConfig.from_json(CONFIGS / f"{objective}.json").training.objective == objective
    config = ExperimentConfig.from_json(CONFIGS / "iql_fm.json")
    config.dataset, config.advantage_weighting = data_config, settings
    config.training = replace(config.training, num_steps=2, max_updates=1, batch_size=3, num_workers=0, device="cpu")
    config.amp.enabled = config.ema.enabled = config.logging.progress_bar = False
    config.scheduler.warmup_steps = 0
    config.validation.update_interval = 1
    policy, _ = tiny_policy()
    config.policy = asdict(policy.config)
    for key in ("camera_names", "normalization_stats_path"):
        config.policy.pop(key)
    config.policy["dit"].pop("action_dim")
    config.policy["dit"]["action_horizon"] = config.policy["n_action_steps"] = 4
    config.policy["image_preprocess"]["size"] = [24, 32]
    path = tmp_path / "train.json"
    config.to_json(path)
    output = tmp_path / "policy"
    observations = []
    original_forward = FlowMatchingPolicy.forward

    def track_validation(self, batch, *args, **kwargs):
        if not self.training:
            observations.append((set(batch["episode_index"].tolist()), batch["advantage_valid"].any().item(),
                                 kwargs.get("sample_weight")))
        return original_forward(self, batch, *args, **kwargs)

    monkeypatch.setattr(FlowMatchingPolicy, "forward", track_validation)
    monkeypatch.setattr("sys.argv", ["awbc-train", "--config", str(path), "--run-dir", str(output)])
    main()
    assert observations == [(set(split["val_episodes"]), False, None)]
    checkpoint = output / "checkpoints/latest_checkpoint.pt"
    provenance = json.loads((output / "advantage_provenance.json").read_text())
    first = torch.load(checkpoint, weights_only=False)
    assert provenance["fingerprint"] == first["split"]["advantage_fingerprint"]
    assert provenance["weight_configuration"]["normalization"] == "none"
    assert provenance["critic_checkpoint"] == metadata["checkpoint"]
    assert math.isfinite(first["last_validation"]["val/loss"])
    resume = ["awbc-train", "--resume", str(checkpoint), "--set", "training.max_updates=null"]
    monkeypatch.setattr("sys.argv", resume + ["--set", "advantage_weighting.beta=2.0"])
    with pytest.raises(ValueError, match="original advantage normalization/weighting"):
        main()
    rows[0]["advantage"] += 0.25
    write_export(settings, rows, metadata)
    monkeypatch.setattr("sys.argv", resume)
    with pytest.raises(ValueError, match="IQL Advantage label fingerprint"):
        main()
    rows[0]["advantage"] -= 0.25
    new_metadata = {**metadata, "checkpoint": {**metadata["checkpoint"], "sha256": "b" * 64}}
    write_export(settings, rows, new_metadata)
    with pytest.raises(ValueError, match="IQL Advantage critic checkpoint"):
        main()
    write_export(settings, rows, metadata)
    main()
    final = torch.load(checkpoint, weights_only=False)
    assert final["global_step"] == 2
    Path(settings.labels_path).unlink()
    Path(settings.metadata_path).unlink()
    loaded = load_policy(checkpoint, device="cpu")
    assert not loaded.training and loaded.config.dit.action_horizon == 4
    torch.testing.assert_close(loaded.dit.velocity_head.weight, final["policy"]["dit.velocity_head.weight"])
