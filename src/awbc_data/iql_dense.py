"""Interpolate raw RoboMeter progress and shape existing IQL window rewards."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def load_progress(path, source):
    """Return per-episode frame potentials and the fingerprint of the input JSONL."""
    content = Path(path).read_bytes()
    points = {episode: {} for episode in source.rows}
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        episode = row["episode_index"]
        frame = row["relative_frame_index"]  # LeRobot's episode-local frame_index.
        progress = row["progress_raw"]
        if type(episode) is not int or episode not in points:
            raise ValueError(f"Unknown progress episode_index {episode!r}")
        last = len(source.rows[episode]["frame_index"]) - 1
        if type(frame) is not int or not 0 <= frame <= last:
            raise ValueError(f"Progress episode {episode}, frame {frame!r} is outside [0, {last}]")
        if frame in points[episode]:
            raise ValueError(f"Duplicate progress for episode {episode}, frame {frame}")
        if not math.isfinite(progress):
            raise ValueError(f"Nonfinite progress for episode {episode}, frame {frame}")
        points[episode][frame] = progress

    potentials = {}
    for episode, labels in points.items():
        last = len(source.rows[episode]["frame_index"]) - 1
        missing = [frame for frame in (0, last) if frame not in labels]
        if missing:
            raise ValueError(f"Missing progress for episode {episode}, boundary frames {missing}")
        frames, scores = zip(*sorted(labels.items()))
        potentials[episode] = np.interp(np.arange(last + 1), frames, scores)
    return potentials, hashlib.sha256(content).hexdigest()


class DenseRLChunkDataset(Dataset):
    def __init__(self, base, potentials, *, shaping_scale=1.0):
        self.base = base
        self.policy_dataset = base.policy_dataset
        self.potentials = potentials
        self.shaping_scale = shaping_scale

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        progress = self.potentials[sample["episode_index"].item()]
        sparse = sample["reward"]
        start = sparse.new_tensor(progress[sample["frame_index"].item()])
        end = sparse.new_tensor(progress[sample["next_frame_index"].item()])
        dense = self.base.gamma * end - start
        reward = sparse if self.shaping_scale == 0 else sparse + self.shaping_scale * dense
        return {**sample, "reward_sparse": sparse, "reward_dense": dense, "reward": reward}
