"""Episode-labeled transitions over the policy's existing action windows."""

import json
from numbers import Real
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from .dataset import ChunkDataset


def _read_episode_labels(path, chunks):
    labels = {}
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode = row.get("episode_index") if isinstance(row, dict) else None
            if type(episode) is not int:
                raise ValueError(f"Line {line_number}: episode_index must be an integer, got {episode!r}")
            if episode not in chunks.source.rows:
                raise ValueError(f"Unknown episode_index {episode} in episode labels")
            if episode in labels:
                raise ValueError(f"Duplicate label for episode {episode}")
            if type(row.get("success")) is not bool:
                raise ValueError(f"Episode {episode}: success must be a boolean")
            end_type = row.get("end_type")
            if end_type not in ("terminated", "truncated"):
                raise ValueError(f"Episode {episode}: end_type must be 'terminated' or 'truncated', got {end_type!r}")
            end_frame = row.get("end_frame_index")
            expected = len(chunks.source.rows[episode]["frame_index"]) - 1
            if type(end_frame) is not int or end_frame != expected:
                raise ValueError(
                    f"Episode {episode}: end_frame_index must identify the post-action observation "
                    f"at frame {expected}, got {end_frame!r}"
                )
            labels[episode] = (row["success"], end_type, end_frame)
    missing = sorted(set(chunks.episode_ids) - labels.keys())
    if missing:
        raise ValueError(f"Missing labels for episodes {missing}: expected success, end_type and end_frame_index")
    return labels


class RLChunkDataset(Dataset):
    """Wrap policy windows with sparse discounted success rewards.

    The caller's JSONL supplies the saved episode IDs and explicitly verified
    post-action end frames. A label is required for each episode in the split.
    """

    def __init__(self, dataset, labels_path: str | Path, *, gamma: float):
        if isinstance(gamma, bool) or not isinstance(gamma, Real) or not 0 < gamma <= 1:
            raise ValueError("gamma must satisfy 0 < gamma <= 1")
        self.dataset = dataset
        self.gamma = float(gamma)
        chunks = dataset
        while isinstance(chunks, Subset):
            chunks = chunks.dataset
        if not isinstance(chunks, ChunkDataset):
            raise TypeError("RLChunkDataset requires a ChunkDataset or a Subset of it")
        self.chunks = chunks
        self.labels = _read_episode_labels(labels_path, chunks)

    def __len__(self):
        return len(self.dataset)

    @property
    def policy_dataset(self):
        return self.dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        episode = sample["episode_index"].item()
        length = sample["valid_length"].item()
        next_frame = sample["frame_index"].item() + length
        success, end_type, end_frame = self.labels[episode]
        terminated = next_frame == end_frame and end_type == "terminated"
        truncated = next_frame == end_frame and end_type == "truncated"
        next_observation = self.chunks.read_observation(episode, next_frame)["observation"]
        return {
            **sample,
            "next_observation": next_observation,
            "next_frame_index": torch.tensor(next_frame),
            "reward": torch.tensor(self.gamma ** (length - 1) if terminated and success else 0., dtype=torch.float32),
            "discount": torch.tensor(self.gamma ** length, dtype=torch.float32),
            "bootstrap_mask": torch.tensor(0. if terminated else 1., dtype=torch.float32),
            "terminated": torch.tensor(terminated),
            "truncated": torch.tensor(truncated),
        }
