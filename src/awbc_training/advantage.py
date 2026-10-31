import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from awbc_data.advantage import AdvantageDataset, PrecomputedAdvantageProvider, sample_metadata
from awbc_models.advantage_weighting import AdvantageNormalizer, AdvantageWeighting


def base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def advantage_values(dataset):
    if isinstance(dataset, Subset):
        return advantage_values(dataset.dataset)[dataset.indices]
    return dataset.advantages


def prepare_advantage_datasets(train, val, config, horizon):
    if not config.enabled:
        raise ValueError("awr_fm requires advantage_weighting.enabled=true")
    source = base_dataset(train).source
    metadata_path = Path(config.metadata_path) if config.metadata_path else source.root / "labeling/advantage_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if not metadata.get("provenance_reviewed") or not metadata.get("model_version"):
        raise ValueError("Real AWR-FM requires reviewed provenance and an explicit model_version; no labels are fabricated")
    provider = (PrecomputedAdvantageProvider.from_file(config.labels_path, metadata) if config.labels_path
                else PrecomputedAdvantageProvider.from_robometer_dataset(source, metadata))
    full_train, full_val = base_dataset(train), base_dataset(val)
    windows = [sample_metadata(dataset, index, horizon) for dataset in (full_train, full_val) for index in range(len(dataset))]
    coverage = provider.validate_windows(windows, source.fps, config.missing_label_policy)
    fit_records = [provider.get_record(sample_metadata(full_train, index, horizon))
                   for index in range(len(full_train))
                   if (full_train.samples[index][0], full_train.samples[index][1], horizon) in provider.records]
    if not fit_records:
        raise ValueError("No labeled training windows remain after the missing-label policy")
    values = torch.tensor([row["advantage"] for row in fit_records], dtype=torch.float64)
    ids = sorted(row["sample_id"] for row in fit_records)
    stats = {"mean": values.mean().item(), "std": values.std(correction=0).item(), "count": len(values),
             "training_trajectory_ids": list(full_train.episode_ids), "provider_fingerprint": provider.fingerprint,
             "sample_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    if config.statistics_path:
        loaded = json.loads(Path(config.statistics_path).read_text())
        if loaded["provider_fingerprint"] != provider.fingerprint or set(loaded["training_trajectory_ids"]) != set(full_train.episode_ids):
            raise ValueError("Advantage statistics do not belong to this label version and training split")
        stats = loaded
    weighting = AdvantageWeighting(AdvantageNormalizer(config.normalization, stats["mean"], stats["std"]),
                                   config.beta, config.min_weight, config.max_weight, config.detach_weights)
    report = {**provider.provenance_report(full_train.episode_ids, full_val.episode_ids),
              "coverage": coverage, "statistics": stats, "normalization_mode": config.normalization}
    return AdvantageDataset(train, provider, horizon, config.missing_label_policy), val, weighting, report


def training_forward(policy, batch, weighting=None, **pair):
    if weighting is None:
        output = policy(batch, **pair)
    else:
        weights = weighting(batch["advantage"])
        output = policy(batch, sample_weight=weights, **pair)
        output["advantage"] = batch["advantage"]
    output["unweighted_loss"] = output["per_sample_loss"].mean()
    output["weighted_loss"] = output["loss"]
    return output
