from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Subset

from awbc_data import ImagePreprocessConfig
from awbc_data.advantage import sample_metadata
from awbc_models.dense_awbc_weighting import DenseAWBCWeighting
from awbc_training.advantage import prepare_dense_awbc_dataset
from awbc_training.config import DatasetConfig, DenseAWBCConfig, ExperimentConfig
from awbc_training.data import prepare_datasets
from awbc_training.train import main
from awbc_training.trainer import Trainer
from test_advantage_weighting import tiny_policy
from test_awbc_data import root
from test_iql_dense import jsonl, progress_row
from test_iql_fm import LabeledToyDataset, ToyFlowPolicy


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/iql_dense_awbc.json"


def test_threshold_cap_and_pure_critic():
    config = DenseAWBCConfig(beta_q=2., critic_weight_cap=2., robometer_threshold=0.125)
    deltas = [-0.25, -0.125, 0.0625, 0.125, 0.25, 0.25]
    scores = [0., 0., 0., 0., 0., 3.]
    weighting = DenseAWBCWeighting(scores, deltas, config)
    assert weighting.progress_gate.tolist() == [-1., 0., 0., 0., 1., 1.]
    assert weighting.above_threshold.tolist() == [True, False, False, False, True, True]
    assert weighting.critic_clipped.tolist() == [False] * 5 + [True]
    raw = torch.tensor([math.exp(-1), 1., 1., 1., math.e, 2 * math.e], dtype=torch.float64)
    actual = weighting.normalized_weights(list(range(6)), 1.)
    torch.testing.assert_close(actual, raw / raw.sum())
    assert actual[-1] / actual[1] > config.critic_weight_cap
    pure_config = replace(config, robometer_scale_start=0.)
    pure = DenseAWBCWeighting(scores, deltas, pure_config)
    expected = torch.tensor([1., 1., 1., 1., 1., 2.], dtype=torch.float64) / 7
    torch.testing.assert_close(pure.normalized_weights(list(range(6)), pure.robometer_scale(0)), expected)


@pytest.mark.parametrize(("change", "field"), [
    ({"robometer_threshold": 0}, "robometer_threshold"),
    ({"robometer_threshold": float("inf")}, "robometer_threshold"),
    ({"critic_weight_cap": -1}, "critic_weight_cap"),
    ({"critic_weight_cap": float("nan")}, "critic_weight_cap"),
    ({"robometer_decay_enabled": True, "robometer_decay_steps": 0}, "robometer_decay_steps"),
])
def test_weight_config_validation(change, field):
    with pytest.raises(ValueError, match=field):
        DenseAWBCWeighting([0.], [0.], DenseAWBCConfig(**change))


def test_scale_schedule():
    config = ExperimentConfig.from_json(CONFIG_PATH).dense_awbc
    fixed = DenseAWBCWeighting([0.], [0.], config)
    assert [fixed.robometer_scale(k) for k in (0, 1, 10_000)] == [1., 1., 1.]
    decayed = DenseAWBCWeighting([0.], [0.], replace(config, robometer_scale_start=2.,
                                 robometer_decay_enabled=True, robometer_decay_steps=4))
    assert [decayed.robometer_scale(k) for k in (0, 1, 2, 4, 10)] == [2., 1.5, 1., 0., 0.]


@pytest.mark.parametrize(("threshold", "scale"), [(0.02, 1.), (0.000001, 1.), (0.02, 0.)])
def test_one_update_large_batch_matches_accumulated_gradients(tmp_path, threshold, scale):
    scores = [-0.1, 0.2, 0.5, -0.3, 0.25]
    deltas = [0.1, -0.3, 0.02, 0., -0.01]
    full = LabeledToyDataset(scores)
    # Mimic the trainer's overfit subset mapping, including reorder and repetition.
    order = [0, 4, 3, 4, 2]
    dataset = Subset(full, order)

    def run(name, batch_size, accumulation):
        torch.manual_seed(13)
        policy = ToyFlowPolicy()
        cfg = ExperimentConfig.from_json(CONFIG_PATH)
        cfg.dense_awbc.robometer_threshold = threshold
        cfg.dense_awbc.robometer_scale_start = scale
        cfg.training = replace(cfg.training, num_steps=1, num_workers=0, batch_size=batch_size,
                               gradient_accumulation_steps=accumulation, gradient_clip_norm=100.)
        cfg.amp.enabled = cfg.ema.enabled = cfg.logging.progress_bar = False
        cfg.validation.update_interval = cfg.checkpoint.update_interval = 0
        weighting = DenseAWBCWeighting(scores, deltas, cfg.dense_awbc)
        directory = tmp_path / name
        directory.mkdir()
        trainer = Trainer(cfg, policy, dataset, dataset, {}, {}, directory, torch.device("cpu"), weighting=weighting)
        batch = torch.utils.data.default_collate(list(dataset))
        residual = policy.dit(batch["action"][:, 0]).squeeze(1) - batch["target"].squeeze(1)
        log_weights = torch.tensor([
            min(scores[i], math.log(10.)) + scale * math.copysign(max(abs(deltas[i]) / threshold - 1, 0), deltas[i])
            for i in order
        ], dtype=torch.float64)
        expected = (log_weights.softmax(0).to(residual) * residual.square()).sum()
        expected_grads = torch.autograd.grad(expected, tuple(policy.parameters()))
        seen = []
        optimizer_step = trainer.optimizer.step

        def capture():
            seen.extend(parameter.grad.clone() for parameter in policy.parameters())
            return optimizer_step()

        with patch.object(trainer.optimizer, "step", side_effect=capture):
            trainer.fit()
        trainer.logger.close()
        for actual, target in zip(seen, expected_grads, strict=True):
            torch.testing.assert_close(actual, target)
        assert trainer.last_update_metrics["train/weighted_loss"] == pytest.approx(expected.item())
        assert trainer.last_update_metrics["robometer/scale"] == scale
        assert trainer.last_update_metrics["robometer/above_threshold_fraction"] > 0
        assert "weight/normalized_p90" in trainer.last_update_metrics
        return seen

    for large, micro in zip(run("large", 5, 1), run("micro", 2, 3), strict=True):
        torch.testing.assert_close(large, micro)


def make_export(root, tmp_path, stride):
    cfg = ExperimentConfig.from_json(CONFIG_PATH)
    cfg.dataset = DatasetConfig(root=str(root), stratify_key=None, train_stride=stride)
    cfg.dense_awbc.robometer_threshold = 0.1
    train, val, _, split = prepare_datasets(cfg.dataset, ImagePreprocessConfig(size=(24, 32)), action_horizon=5)
    rows = []
    for index in range(len(train)):
        row = sample_metadata(train, index, 5)
        row.pop("sample_id")
        rows.append({**row, "valid_length": row["end_timestep"] - row["start_timestep"], "advantage": 0.})
    cfg.advantage_weighting.labels_path = str(jsonl(tmp_path / "critic.jsonl", rows[::-1]))
    metadata = tmp_path / "critic.metadata.json"
    metadata.write_text(json.dumps({
        "checkpoint": {"sha256": "a" * 64, "global_step": 2},
        "data_contract": {"reward_mode": "dense", "action_horizon": 5, "train_stride": stride},
        "split": split, "record_count": len(rows), "advantage_definition": "min(target_q1, target_q2) - v",
    }))
    cfg.advantage_weighting.metadata_path = str(metadata)
    progress = [progress_row(eid, frame, raw) for eid in (0, 1)
                for frame, raw in ((0, 0.2), (10, 0.7), (13, 0.4))]
    cfg.dense_awbc.progress_path = str(jsonl(tmp_path / "progress.jsonl", progress))
    return cfg, train, val, split, progress


@pytest.mark.parametrize("stride", [1, 2])
def test_window_aligned_progress_uses_raw_endpoints(root, tmp_path, stride):
    cfg, train, _, split, _ = make_export(root, tmp_path, stride)
    dataset, weighting, report = prepare_dense_awbc_dataset(train, cfg.advantage_weighting, cfg.dense_awbc, 5, split)
    assert len(dataset) == len(train)
    assert dataset.records[0]["end_timestep"] == 5
    assert weighting.progress_gate[0].item() == pytest.approx(1.5)
    tail = dataset.records[12 // stride]
    assert (tail["start_timestep"], tail["end_timestep"], tail["valid_length"]) == (12, 13, 1)
    assert weighting.progress_gate[12 // stride].item() == pytest.approx(0., abs=1e-12)
    # Endpoints 10 and 13 are both original labels, giving a negative increment of 0.3.
    assert weighting.progress_gate[10 // stride].item() == pytest.approx(-2.)
    assert report["progress_sha256"]
    metadata = Path(cfg.advantage_weighting.metadata_path)
    sparse = json.loads(metadata.read_text())
    sparse["data_contract"]["reward_mode"] = "sparse"
    metadata.write_text(json.dumps(sparse))
    with pytest.raises(ValueError, match="dense IQL critic"):
        prepare_dense_awbc_dataset(train, cfg.advantage_weighting, cfg.dense_awbc, 5, split)


def test_cli_decay_continues_on_resume_and_checks_progress(root, tmp_path, monkeypatch):
    cfg, _, _, _, progress = make_export(root, tmp_path, 2)
    cfg.dense_awbc.robometer_decay_enabled = True
    cfg.dense_awbc.robometer_decay_steps = 2
    cfg.training = replace(cfg.training, num_steps=3, max_updates=1, batch_size=2, num_workers=0, device="cpu")
    cfg.amp.enabled = cfg.ema.enabled = cfg.logging.progress_bar = False
    cfg.logging.log_interval = 1
    cfg.validation.update_interval = 1
    model, _ = tiny_policy()
    cfg.policy = asdict(model.config)
    for key in ("camera_names", "normalization_stats_path"):
        cfg.policy.pop(key)
    cfg.policy["dit"].pop("action_dim")
    cfg.policy["dit"]["action_horizon"] = cfg.policy["n_action_steps"] = 5
    cfg.policy["image_preprocess"]["size"] = [24, 32]
    path = tmp_path / "config.json"
    cfg.to_json(path)
    run = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["awbc-train", "--config", str(path), "--run-dir", str(run)])
    main()

    checkpoint = run / "checkpoints/latest_checkpoint.pt"
    first = torch.load(checkpoint, weights_only=False)
    assert first["last_update_metrics"]["robometer/scale"] == 1.
    assert math.isfinite(first["last_validation"]["val/loss"])
    resume = ["awbc-train", "--resume", str(checkpoint), "--set", "training.max_updates=null"]
    monkeypatch.setattr("sys.argv", resume + ["--set", "dense_awbc.beta_q=2"])
    with pytest.raises(ValueError, match="Dense AWBC weight configuration"):
        main()
    monkeypatch.setattr("sys.argv", resume)
    progress_path = Path(cfg.dense_awbc.progress_path)
    jsonl(progress_path, [{**progress[0], "progress_raw": 0.19}, *progress[1:]])
    with pytest.raises(ValueError, match="RoboMeter progress fingerprint"):
        main()
    jsonl(progress_path, progress)
    main()
    final = torch.load(checkpoint, weights_only=False)
    assert final["global_step"] == 3
    logs = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    scales = {log["global_step"]: log["robometer/scale"] for log in logs if "robometer/scale" in log}
    assert scales == {1: 1., 2: 0.5, 3: 0.}
