from collections import defaultdict
import math

import numpy as np
from torch.utils.data import Subset

from awbc_data import ChunkDataset, DatasetSource
from awbc_data.dataset import ROBOMETER_CHUNK_SIZE


def prepare_datasets(config, image_preprocess, *, action_horizon=ROBOMETER_CHUNK_SIZE):
    source = DatasetSource(config.root)
    groups = defaultdict(list)
    for eid in sorted(source.episodes):
        value = source.episodes[eid][config.stratify_key] if config.stratify_key else "all"
        key = tuple(value) if isinstance(value, list) else (value,)
        groups[key].append(eid)
    rng = np.random.default_rng(config.split_seed)
    train_ids, val_ids = [], []
    for ids in groups.values():
        rng.shuffle(ids)
        n_val = min(len(ids)-1, max(1, math.ceil(len(ids)*config.val_fraction)))
        if n_val:
            train_ids.extend(ids[:-n_val])
            val_ids.extend(ids[-n_val:])
        else:
            train_ids.extend(ids)  # A singleton condition cannot occur in both sets.
    stats = source.statistics(train_ids)
    train = ChunkDataset(source, train_ids, stats, image_preprocess=image_preprocess,
                         action_horizon=action_horizon, stride=config.train_stride)
    val = ChunkDataset(source, val_ids, stats, image_preprocess=image_preprocess, action_horizon=action_horizon)
    assert not set(train_ids) & set(val_ids)
    for dataset in (train, val):
        for eid, start, length in dataset.samples:
            assert start + length <= len(source.rows[eid]["action"])-1
    manifest = {"train_episodes": train_ids, "val_episodes": val_ids,
                "train_chunks": len(train), "val_chunks": len(val), "split_seed": config.split_seed,
                "stratify_key": config.stratify_key, "groups": {str(k): len(v) for k, v in groups.items()},
                "statistics_episode_ids": stats["train_episode_ids"],
                "normalization": "min_max, train trajectories only", "augmentation": "none",
                "windows_cross_episode": False, "episode_overlap": False}
    if action_horizon != ROBOMETER_CHUNK_SIZE:
        manifest["action_horizon"] = action_horizon
    if train.stride != action_horizon:
        manifest["train_stride"] = train.stride
    selected = []
    for dataset, limit in ((train, config.max_train_samples), (val, config.max_val_samples)):
        indices = list(range(len(dataset)))
        if limit is not None:
            indices = rng.permutation(len(dataset))[:limit].tolist()
        selected.append(Subset(dataset, indices))
    manifest["train_sample_indices"] = selected[0].indices
    manifest["val_sample_indices"] = selected[1].indices
    return *selected, stats, manifest
