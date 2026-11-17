import json

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from awbc_data import ImagePreprocessConfig, RLChunkDataset
from awbc_data.images import IMAGENET_MEAN, IMAGENET_STD
from awbc_training.config import DatasetConfig
from awbc_training.data import prepare_datasets
from test_awbc_data import root


@pytest.fixture
def splits(root):
    config = DatasetConfig(root=str(root), stratify_key=None, train_stride=1)
    train, val, stats, manifest = prepare_datasets(
        config, ImagePreprocessConfig(size=(12, 16)), action_horizon=4
    )
    yield train, val, stats, manifest
    train.dataset.close()
    val.dataset.close()


def write_labels(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def label(episode, *, success=True, end_type="terminated"):
    return dict(episode_index=episode, success=success, end_type=end_type, end_frame_index=13)


def assert_observation(sample, key, chunks, episode, frame):
    observation = sample[key]
    image = observation["images"][0] * IMAGENET_STD + IMAGENET_MEAN
    assert image.mean().item() * 255 == pytest.approx((episode * 20 + frame) * 6, abs=4)
    state = chunks.denormalize("observation.state", observation["state"])
    torch.testing.assert_close(state, torch.full((8,), float(episode * 100 + frame)))


def test_overlapping_windows_use_horizon_for_next_observation(splits, tmp_path):
    train, _, _, manifest = splits
    episode, = manifest["train_episodes"]
    path = write_labels(tmp_path / "labels.jsonl", label(episode))
    dataset = RLChunkDataset(train, path, gamma=0.9)
    for start in (0, 1):
        sample = dataset[start]
        assert sample["frame_index"] == start
        assert sample["next_frame_index"] == start + 4
        assert sample["valid_length"] == 4
        assert not sample["action_is_pad"].any()
        assert sample["reward"] == 0
        assert sample["discount"].item() == pytest.approx(0.9**4)
        assert sample["bootstrap_mask"] == 1
        assert not sample["terminated"] and not sample["truncated"]
        assert_observation(sample, "observation", train.dataset, episode, start)
        assert_observation(sample, "next_observation", train.dataset, episode, start + 4)


def test_success_tail_and_dataloader(splits, tmp_path):
    train, _, _, manifest = splits
    episode, = manifest["train_episodes"]
    path = write_labels(tmp_path / "labels.jsonl", label(episode))
    dataset = RLChunkDataset(Subset(train, [0, 10]), path, gamma=0.9)
    tail = dataset[1]
    assert tail["next_frame_index"] == 13
    assert_observation(tail, "next_observation", train.dataset, episode, 13)
    assert tail["reward"].item() == pytest.approx(0.81)
    assert tail["discount"].item() == pytest.approx(0.729)
    torch.testing.assert_close(tail["action"], train[10]["action"])
    torch.testing.assert_close(tail["action"][2], tail["action"][3])
    for field in ("reward", "discount", "bootstrap_mask"):
        assert tail[field].shape == () and tail[field].dtype == torch.float32

    batch = next(iter(DataLoader(dataset, batch_size=2)))
    for key in ("observation", "next_observation"):
        assert batch[key]["images"].shape == (2, 1, 3, 12, 16)
        assert batch[key]["state"].shape == (2, 8)
        assert batch[key]["images"].dtype == batch[key]["state"].dtype == torch.float32
    assert batch["action"].shape == (2, 4, 8) and batch["action"].dtype == torch.float32
    assert batch["action_is_pad"].dtype == torch.bool
    assert batch["action_is_pad"].tolist() == [[False] * 4, [False, False, False, True]]
    assert batch["valid_length"].tolist() == [4, 3]
    assert batch["episode_index"].tolist() == [episode, episode]
    assert batch["frame_index"].tolist() == [0, 10]
    assert batch["next_frame_index"].tolist() == [4, 13]
    for field in ("episode_index", "frame_index", "next_frame_index", "valid_length"):
        assert batch[field].shape == (2,) and batch[field].dtype == torch.int64
    for field in ("reward", "discount", "bootstrap_mask"):
        assert batch[field].shape == (2,) and batch[field].dtype == torch.float32
    assert batch["bootstrap_mask"].tolist() == [1., 0.]
    assert batch["terminated"].dtype == batch["truncated"].dtype == torch.bool
    assert batch["terminated"].tolist() == [False, True]
    assert batch["truncated"].tolist() == [False, False]
    target = batch["reward"] + batch["discount"] * batch["bootstrap_mask"] * 2
    torch.testing.assert_close(target, torch.tensor([2 * 0.9**4, 0.81]))


@pytest.mark.parametrize("end_type,success,mask", [
    ("terminated", False, 0.),
    ("truncated", False, 1.),
    ("truncated", True, 1.),
])
def test_end_flags_and_rewards(splits, tmp_path, end_type, success, mask):
    train, _, _, manifest = splits
    episode, = manifest["train_episodes"]
    path = write_labels(tmp_path / "labels.jsonl", label(episode, end_type=end_type, success=success))
    dataset = RLChunkDataset(train, path, gamma=1.)
    for start in (8, 9, 10):
        sample = dataset[start]
        reaches_end = start >= 9
        assert sample["reward"] == 0
        assert sample["terminated"].item() == (reaches_end and end_type == "terminated")
        assert sample["truncated"].item() == (reaches_end and end_type == "truncated")
        assert sample["bootstrap_mask"] == (mask if reaches_end else 1.)
        assert sample["discount"] == 1.


def test_splits_and_nested_subset_keep_window_identity(splits, tmp_path):
    train, val, stats, manifest = splits
    train_episode, = manifest["train_episodes"]
    val_episode, = manifest["val_episodes"]
    assert train_episode != val_episode
    assert train.dataset.stats is val.dataset.stats is stats
    path = write_labels(tmp_path / "labels.jsonl", label(val_episode), label(train_episode))
    subset = Subset(Subset(train, [12, 3, 0]), [1, 0, 1, 2])
    train_rl = RLChunkDataset(subset, path, gamma=0.9)
    val_rl = RLChunkDataset(val, path, gamma=0.9)
    assert [sample["frame_index"].item() for sample in train_rl] == [3, 12, 3, 0]
    for dataset, policy_dataset, episode, chunks in (
        (train_rl, subset, train_episode, train.dataset),
        (val_rl, val, val_episode, val.dataset),
    ):
        assert len(dataset) == len(policy_dataset)
        for index in range(len(dataset)):
            sample, policy_sample = dataset[index], policy_dataset[index]
            assert sample["episode_index"] == policy_sample["episode_index"] == episode
            assert sample["frame_index"] == policy_sample["frame_index"]
            assert sample["next_frame_index"] <= 13
            assert_observation(sample, "next_observation", chunks, episode, sample["next_frame_index"].item())


@pytest.mark.parametrize("updates,match", [
    ({"episode_index": 100}, "Unknown episode_index 100"),
    ({"episode_index": True}, "episode_index must be an integer"),
    ({"success": 1}, "success must be a boolean"),
    ({"end_type": "done"}, "end_type"),
    ({"end_type": None}, "end_type"),
    ({"end_frame_index": 12}, "end_frame_index"),
    ({"end_frame_index": 13.0}, "end_frame_index"),
    ({"end_frame_index": None}, "end_frame_index"),
])
def test_invalid_episode_label(splits, tmp_path, updates, match):
    train, _, _, manifest = splits
    episode, = manifest["train_episodes"]
    path = write_labels(tmp_path / "labels.jsonl", {**label(episode), **updates})
    with pytest.raises(ValueError, match=match):
        RLChunkDataset(train, path, gamma=0.9)


def test_missing_and_duplicate_labels(splits, tmp_path):
    train, _, _, manifest = splits
    episode, = manifest["train_episodes"]
    path = tmp_path / "labels.jsonl"
    with pytest.raises(ValueError, match=rf"Missing labels for episodes \[{episode}\]"):
        RLChunkDataset(train, write_labels(path), gamma=0.9)
    with pytest.raises(ValueError, match=f"Duplicate label for episode {episode}"):
        RLChunkDataset(train, write_labels(path, label(episode), label(episode)), gamma=0.9)


@pytest.mark.parametrize("gamma", [0, -0.1, 1.01, float("nan"), float("inf"), True, "0.9"])
def test_invalid_gamma(splits, tmp_path, gamma):
    train, _, _, _ = splits
    with pytest.raises(ValueError, match="gamma"):
        RLChunkDataset(train, tmp_path / "labels.jsonl", gamma=gamma)
