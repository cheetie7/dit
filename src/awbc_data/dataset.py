"""Read numeric Parquet and timestamp-aligned RGB video without importing LeRobot."""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .images import IMAGENET_MEAN, IMAGENET_STD, ImagePreprocessConfig, ImagePreprocessor
LABELS = ("annotation.advantage", "annotation.advantage_valid",
          "annotation.action_chunk_start", "annotation.action_chunk_valid_length")


class DatasetSource:
    """Small numeric columns stay in RAM; shared MP4 files are decoded lazily."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError("Only LeRobot v3.0 is supported")
        self.fps = float(self.info["fps"])
        self.cameras = tuple(k for k, v in self.info["features"].items() if v["dtype"] == "video")
        self.episodes = {}
        for path in sorted((self.root / "meta/episodes").rglob("*.parquet")):
            for ep in pq.read_table(path).to_pylist():
                eid = int(ep["episode_index"])
                self.episodes[eid] = ep
        self.rows = {}
        # Group by episode IDs rather than assuming one file per episode or
        # global dataset indices equal file-local row offsets.
        pieces = {}
        required = ["index", "episode_index", "frame_index", "timestamp", "observation.state", "action"]
        labeled_files = []
        for path in sorted((self.root / "data").rglob("*.parquet")):
            names = pq.read_schema(path).names
            present = [k in names for k in LABELS]
            if any(present) and not all(present):
                raise ValueError(f"Incomplete annotation columns in {path}")
            labeled_files.append(all(present))
            table = pq.read_table(path, columns=required + (list(LABELS) if all(present) else []))
            for row in table.to_pylist():
                pieces.setdefault(int(row["episode_index"]), []).append(row)
        if not labeled_files or len(set(labeled_files)) != 1:
            raise ValueError("Missing data or inconsistent annotation schemas")
        self.has_advantage = labeled_files[0]
        for eid, records in pieces.items():
            records.sort(key=lambda r: r["frame_index"])
            ep = self.episodes[eid]
            n = int(ep["length"])
            data = {k: np.asarray([r[k] for r in records]) for k in records[0]}
            if n < 2 or len(records) != n or not np.array_equal(data["frame_index"], np.arange(n)):
                raise ValueError(f"Invalid frame sequence in episode {eid}")
            for key in ("observation.state", "action"):
                data[key] = data[key].astype(np.float32)
            self.rows[eid] = data

    def statistics(self, episode_ids):
        """Training frames only, no duplicate chunk rows or padded actions."""
        result = {"train_episode_ids": list(episode_ids), "normalization": "min_max"}
        for key in ("observation.state", "action"):
            values = np.concatenate([self.rows[e][key][:-1] if key == "action" else self.rows[e][key]
                                     for e in episode_ids]).astype(np.float64)
            result[key] = {"count": len(values), **{k: v.tolist() for k, v in {
                "min": values.min(0), "max": values.max(0),
                "mean": values.mean(0), "std": values.std(0)}.items()}}
        return result


class ChunkDataset(Dataset):
    """One observation at t, action[t:t+10]; True in action_is_pad means ignore."""

    def __init__(self, source: DatasetSource, episode_ids, stats, image_size=(240, 320),
                 require_advantage=False, image_preprocess=None):
        self.source = source
        self.episode_ids = tuple(int(e) for e in episode_ids)
        self.stats = stats
        self.image_preprocessor = ImagePreprocessor(image_preprocess or ImagePreprocessConfig(size=tuple(image_size)))
        self.image_size = tuple(self.image_preprocessor.config.size)
        if require_advantage and not source.has_advantage:
            raise ValueError("AWBC requires an annotated dataset; raw data has no advantage labels")
        self.samples = []
        for eid in self.episode_ids:
            data = source.rows[eid]
            n = len(data["index"])
            starts = np.arange(0, n - 1, 10)
            if source.has_advantage:
                actual = np.flatnonzero(data[LABELS[2]])
                if not np.array_equal(actual, starts):
                    raise ValueError(f"Annotated starts disagree with 10-step chunks in episode {eid}")
                for t in starts:
                    length = min(10, n - 1 - t)
                    span = slice(t, t + length)
                    if (not data[LABELS[1]][span].all()
                        or not np.all(data[LABELS[3]][span] == length)
                        or not np.isfinite(data[LABELS[0]][span]).all()
                        or not np.all(data[LABELS[0]][span] == data[LABELS[0]][t])):
                        raise ValueError(f"Invalid advantage chunk at episode {eid}, frame {t}")
            self.samples.extend((eid, int(t), min(10, n - 1 - int(t))) for t in starts)
        self._containers = OrderedDict()
        self._pid = os.getpid()

    def __len__(self):
        return len(self.samples)

    def normalize(self, key, values):
        lo = torch.tensor(self.stats[key]["min"], dtype=torch.float32)
        hi = torch.tensor(self.stats[key]["max"], dtype=torch.float32)
        span = hi - lo
        return torch.where(span > 1e-8, 2 * (values - lo) / span.clamp_min(1e-8) - 1,
                           torch.zeros_like(values))

    def denormalize(self, key, values):
        lo = torch.tensor(self.stats[key]["min"], dtype=torch.float32)
        hi = torch.tensor(self.stats[key]["max"], dtype=torch.float32)
        return (values + 1) * (hi - lo) / 2 + lo

    def _image(self, eid, t, camera):
        if os.getpid() != self._pid:
            self.close()
            self._pid = os.getpid()
        ep = self.source.episodes[eid]
        prefix = f"videos/{camera}"
        path = self.source.root / self.source.info["video_path"].format(
            video_key=camera, chunk_index=ep[f"{prefix}/chunk_index"],
            file_index=ep[f"{prefix}/file_index"])
        if path not in self._containers:
            self._containers[path] = av.open(str(path))
            if len(self._containers) > 6:
                self._containers.popitem(last=False)[1].close()
        self._containers.move_to_end(path)
        container = self._containers[path]
        stream = container.streams.video[0]
        timestamps = self.source.rows[eid]["timestamp"]
        target = float(ep[f"{prefix}/from_timestamp"]) + float(timestamps[t] - timestamps[0])
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        best, error = None, float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            seconds = float(frame.pts * stream.time_base)
            distance = abs(seconds - target)
            if distance < error:
                best, error = frame, distance
            if seconds >= target:
                break
        if best is None or error > 0.5 / self.source.fps + 1e-3:
            raise ValueError(f"Video alignment failed: {path}, target={target}, error={error}")
        image = self.image_preprocessor(best.to_image())
        return image, float(best.pts * stream.time_base), target

    def __getitem__(self, idx):
        eid, t, length = self.samples[idx]
        data = self.source.rows[eid]
        decoded = [self._image(eid, t, camera) for camera in self.source.cameras]
        actions = torch.from_numpy(data["action"][t:t + length].copy())
        actions = torch.cat((actions, actions[-1:].repeat(10 - length, 1)))
        return {
            "observation": {"images": torch.stack([x[0] for x in decoded]),
                            "state": self.normalize("observation.state", torch.from_numpy(data["observation.state"][t].copy()))},
            "action": self.normalize("action", actions),
            "action_is_pad": torch.arange(10) >= length,
            "advantage": torch.tensor(float(data[LABELS[0]][t]) if self.source.has_advantage else 0, dtype=torch.float32),
            "advantage_valid": torch.tensor(self.source.has_advantage),
            "episode_index": torch.tensor(eid), "frame_index": torch.tensor(t),
            "timestamp": torch.tensor(float(data["timestamp"][t]), dtype=torch.float64),
            "video_timestamp": torch.tensor([x[1] for x in decoded], dtype=torch.float64),
            "video_target_timestamp": torch.tensor([x[2] for x in decoded], dtype=torch.float64),
            "valid_length": torch.tensor(length),
        }

    def close(self):
        for container in self._containers.values():
            container.close()
        self._containers.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_containers"] = OrderedDict()  # Spawn workers must open their own handles.
        return state


def build_datasets(root, val_fraction=0.1, shuffle_episodes=False, seed=0,
                   image_size=(240, 320), require_advantage=False, image_preprocess=None):
    """Sequential last-10% holdout by default, matching the dataset report."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    source = DatasetSource(root)
    ids = sorted(source.episodes)
    if len(ids) < 2:
        raise ValueError("At least two episodes are required for train/validation split")
    if shuffle_episodes:
        np.random.default_rng(seed).shuffle(ids)
    n_val = min(len(ids) - 1, max(1, math.ceil(len(ids) * val_fraction)))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    stats = source.statistics(train_ids)
    kwargs = dict(stats=stats, image_size=image_size, require_advantage=require_advantage,
                  image_preprocess=image_preprocess)
    return ChunkDataset(source, train_ids, **kwargs), ChunkDataset(source, val_ids, **kwargs)
