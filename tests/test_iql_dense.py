import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from awbc_data import ChunkDataset, ImagePreprocessConfig, RLChunkDataset
from awbc_data.iql_dense import DenseRLChunkDataset, load_progress
from awbc_training.advantage import prepare_iql_advantage_dataset
from awbc_training.config import AdvantageWeightingConfig, DatasetConfig
from awbc_training.data import prepare_datasets
from awbc_training.iql import main as sparse_main, td_target
from awbc_training.iql_dense import main as dense_main
from test_awbc_data import root


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def progress_row(episode, frame, raw):
    return {"episode_index": episode, "relative_frame_index": frame,
            "progress_raw": raw, "progress_used": 1.0}


@pytest.fixture
def trajectory(tmp_path, monkeypatch):
    frames = np.arange(48)
    values = frames.astype(np.float32)[:, None]
    source = SimpleNamespace(has_advantage=False, rows={
        0: {"frame_index": frames, "index": frames, "action": values, "timestamp": frames / 25.0}
    })
    stats = {"action": {"min": [0.], "max": [46.]}}

    def observe(self, episode, frame):
        return {"observation": {"state": torch.tensor([frame], dtype=torch.float32)}}

    monkeypatch.setattr(ChunkDataset, "read_observation", observe)
    rows = [progress_row(0, f, p) for f, p in [(47, 0.3), (20, 0.6), (0, 0.2), (40, 0.4)]]

    def build(stride, *, success=True, end_type="terminated", scale=1.0):
        chunks = ChunkDataset(source, [0], stats, action_horizon=10, stride=stride)
        labels = jsonl(tmp_path / "end.jsonl", [
            dict(episode_index=0, success=success, end_type=end_type, end_frame_index=47)
        ])
        base = RLChunkDataset(chunks, labels, gamma=0.9)
        path = jsonl(tmp_path / "progress.jsonl", rows)
        potentials, _ = load_progress(path, source)
        return base, DenseRLChunkDataset(base, potentials, shaping_scale=scale)

    return source, rows, build


def test_linear_progress_preserves_raw_anchor_values(trajectory, tmp_path):
    source, rows, _ = trajectory
    path = jsonl(tmp_path / "progress.jsonl", rows)
    potentials, fingerprint = load_progress(path, source)
    assert fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()
    values = potentials[0]
    assert values[5] == pytest.approx(0.3)
    assert values[30] == pytest.approx(0.5)
    assert values[45] == pytest.approx(0.4 + (0.3 - 0.4) * 5 / 7)
    assert [values[f] for f in (0, 20, 40, 47)] == [0.2, 0.6, 0.4, 0.3]


@pytest.mark.parametrize(("change", "message"), [
    ("duplicate", "Duplicate progress for episode 0, frame 47"),
    ("nonfinite", "Nonfinite progress for episode 0, frame 20"),
    ("out_of_bounds", r"episode 0, frame 48 is outside \[0, 47\]"),
    ("missing_first", r"Missing progress for episode 0, boundary frames \[0\]"),
    ("missing_last", r"Missing progress for episode 0, boundary frames \[47\]"),
])
def test_invalid_progress(trajectory, tmp_path, change, message):
    source, rows, _ = trajectory
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "nonfinite":
        rows[1]["progress_raw"] = float("inf")
    elif change == "out_of_bounds":
        rows[0]["relative_frame_index"] = 48
    elif change == "missing_first":
        rows.pop(2)
    elif change == "missing_last":
        rows.pop(0)
    with pytest.raises(ValueError, match=message):
        load_progress(jsonl(tmp_path / "progress.jsonl", rows), source)


@pytest.mark.parametrize("stride", [1, 2, 20])
def test_sliding_windows_and_short_tail(trajectory, stride):
    _, _, build = trajectory
    base, dense = build(stride, scale=0.5)
    sample = dense[20 // stride]
    assert sample["frame_index"] == 20 and sample["next_frame_index"] == 30
    assert sample["observation"]["state"] == 20 and sample["next_observation"]["state"] == 30
    assert sample["reward_dense"].item() == pytest.approx(0.9 * 0.5 - 0.6)
    assert sample["reward_dense"] < 0
    tail = dense[40 // stride]
    assert tail["next_frame_index"] == 47 and tail["valid_length"] == 7
    assert tail["action_is_pad"].tolist() == [False] * 7 + [True] * 3
    assert tail["discount"].item() == pytest.approx(0.9**7)
    assert tail["reward_sparse"].item() == pytest.approx(0.9**6)
    expected = 0.9 * 0.3 - 0.4
    assert tail["reward_dense"].item() == pytest.approx(expected)
    assert tail["reward"].item() == pytest.approx(0.9**6 + 0.5 * expected)
    assert dense.policy_dataset is base.policy_dataset


@pytest.mark.parametrize(("end_type", "mask"), [("terminated", 0), ("truncated", 1)])
def test_failure_and_truncation_keep_base_flags(trajectory, end_type, mask):
    _, _, build = trajectory
    base, dense = build(2, success=False, end_type=end_type)
    sample = dense[20]
    assert sample["terminated"] == (end_type == "terminated")
    assert sample["truncated"] == (end_type == "truncated")
    assert sample["reward_sparse"] == base[20]["reward"] == 0
    assert sample["bootstrap_mask"] == base[20]["bootstrap_mask"] == mask
    assert sample["reward"].item() == pytest.approx(0.9 * 0.3 - 0.4)


def test_zero_shaping_matches_sparse_batch_and_td_target(trajectory):
    _, _, build = trajectory
    base, dense = build(2, scale=0)
    indices = [20, 3, 20]
    sparse, shaped = [next(iter(DataLoader(Subset(dataset, indices), batch_size=3))) for dataset in (base, dense)]
    for key, value in sparse.items():
        if isinstance(value, dict):
            for field in value:
                assert torch.equal(value[field], shaped[key][field])
        else:
            assert torch.equal(value, shaped[key])
    value_fn = lambda obs: obs["state"].squeeze(-1)
    assert torch.equal(td_target(sparse, value_fn), td_target(shaped, value_fn))
    assert shaped["reward_dense"].abs().sum() > 0


def test_dense_cli_checkpoint_and_iql_fm_export(root, tmp_path):
    ends = jsonl(tmp_path / "ends.jsonl", [
        dict(episode_index=e, success=e == 0, end_type="terminated", end_frame_index=13) for e in range(2)
    ])
    raw_rows = [progress_row(e, frame, value) for e in range(2) for frame, value in [(0, 0.2), (13, 0.5)]]
    progress = jsonl(tmp_path / "progress.jsonl", raw_rows)
    data_settings = {"root": str(root), "stratify_key": None, "train_stride": 2}
    image_preprocess = {"size": [24, 32]}
    common = {
        "dataset": data_settings, "image_preprocess": image_preprocess,
        "labels_path": str(ends), "action_horizon": 4, "num_steps": 1, "batch_size": 2,
        "pretrained": False, "hidden_dims": [8], "device": "cpu",
        "logging": {"use_wandb": False, "progress_bar": False},
    }
    config = tmp_path / "dense.json"
    config.write_text(json.dumps({**common, "progress_path": str(progress), "shaping_scale": 0.5}))
    run = tmp_path / "dense"
    dense_main(["train", "--config", str(config), "--run-dir", str(run)])
    checkpoint = run / "checkpoints/latest_checkpoint.pt"
    state = torch.load(checkpoint, weights_only=False)
    contract = state["data_contract"]
    assert contract["reward_mode"] == "dense"
    assert contract["reward_definition"] == "discounted_sparse_terminal_success_plus_single_gamma_potential_difference_v1"
    assert contract["progress_interpolation"] == "linear_between_raw_labeled_frames"
    assert contract["progress_labels_sha256"] == hashlib.sha256(progress.read_bytes()).hexdigest()
    assert (contract["shaping_scale"], contract["gamma"], contract["action_horizon"], contract["train_stride"]) == (
        0.5, 0.99, 4, 2
    )
    for phase in ("train", "val"):
        log = (run / "metrics.jsonl").read_text()
        assert all(f'"{phase}/{metric}_mean"' in log for metric in ("reward", "reward_sparse", "reward_dense"))
    with pytest.raises(ValueError, match="shaping_scale"):
        dense_main(["train", "--resume", str(checkpoint), "--set", "shaping_scale=1.0"])
    jsonl(progress, [{**raw_rows[0], "progress_raw": 0.1}, *raw_rows[1:]])
    with pytest.raises(ValueError, match="progress_labels_sha256"):
        dense_main(["train", "--resume", str(checkpoint)])
    jsonl(progress, raw_rows)
    dense_main(["train", "--resume", str(checkpoint), "--set", "num_steps=2"])
    assert torch.load(checkpoint, weights_only=False)["global_step"] == 2
    output = run / "advantages.jsonl"
    dense_main(["export", "--checkpoint", str(checkpoint), "--output", str(output)])
    metadata_path = output.with_suffix(".metadata.json")
    assert json.loads(metadata_path.read_text())["data_contract"]["reward_mode"] == "dense"
    train, _, _, split = prepare_datasets(
        DatasetConfig(**data_settings), ImagePreprocessConfig(**image_preprocess), action_horizon=4
    )
    settings = AdvantageWeightingConfig(enabled=True, labels_path=str(output), metadata_path=str(metadata_path))
    weighted, _, _ = prepare_iql_advantage_dataset(train, settings, 4, split)
    assert len(weighted) == len(train)
    assert torch.isfinite(weighted.advantages).all()

    config.write_text(json.dumps(common))
    sparse_run = tmp_path / "sparse"
    sparse_main(["train", "--config", str(config), "--run-dir", str(sparse_run)])
    sparse_checkpoint = sparse_run / "checkpoints/latest_checkpoint.pt"
    sparse_state = torch.load(sparse_checkpoint, weights_only=False)
    assert sparse_state["data_contract"].pop("reward_mode") == "sparse"
    torch.save(sparse_state, sparse_checkpoint)  # Legacy sparse checkpoints predate reward_mode.
    sparse_main(["export", "--checkpoint", str(sparse_checkpoint), "--output", str(sparse_run / "advantages.jsonl")])
    with pytest.raises(ValueError, match="reward_mode"):
        dense_main(["train", "--resume", str(sparse_checkpoint)])
