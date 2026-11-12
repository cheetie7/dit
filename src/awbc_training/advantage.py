import hashlib
import json
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import Subset

from awbc_data.advantage import AdvantageDataset, PrecomputedAdvantageProvider, sample_metadata, window_metadata
from awbc_data.dataset import ROBOMETER_CHUNK_SIZE, action_windows
from awbc_models.advantage_weighting import AdvantageNormalizer, AdvantageWeighting


def base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def advantage_values(dataset):
    if isinstance(dataset, Subset):
        return advantage_values(dataset.dataset)[dataset.indices]
    return dataset.advantages


def load_advantage_metadata(source, config):
    """Read an explicit sidecar or the existing, already-scored RoboMeter export."""
    path = Path(config.metadata_path) if config.metadata_path else source.root / "labeling/advantage_metadata.json"
    if config.metadata_path or path.is_file():
        return json.loads(path.read_text()), None
    summary_path = source.root / "labeling/summary.json"
    labels_path = source.root / "labeling/chunk_advantages.jsonl"
    if config.labels_path or not source.has_advantage or not summary_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"Advantage metadata not found: {path}. Provide advantage_weighting.metadata_path; "
                                "automatic metadata loading requires an annotated RoboMeter export with summary.json and chunk_advantages.jsonl")
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("The RoboMeter export contains no chunk labels")
    for row in rows:
        divisor = row['advantage_divisor']
        if divisor not in (1, 2) or divisor != row['chunks_in_group']:
            raise ValueError(f"Unexpected RoboMeter group definition at {row['chunk_id']}")
        expected = (row['progress_end']-row['progress_start'])/divisor
        if not math.isclose(row['advantage'], expected, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Stored advantage does not match the declared progress-increment formula at {row['chunk_id']}")
        if row.get('terminal_progress_forced') and row['progress_end'] != 1.:
            raise ValueError(f"Forced terminal progress must be 1 at {row['chunk_id']}")
    horizons = {row['target_chunk_length'] for row in rows}
    if len(horizons) != 1:
        raise ValueError("RoboMeter export mixes different action horizons")
    metadata = {
        'source': 'robometer', 'semantics': 'progress_increment_proxy',
        'model_name': Path(summary['model_path']).name, 'model_path_recorded': summary['model_path'],
        'model_version': summary.get('model_version'), 'checkpoint_hash': summary.get('checkpoint_hash'),
        'label_generation_date': summary.get('label_generation_date'),
        'training_trajectory_ids': summary.get('training_trajectory_ids'),
        'evaluation_trajectory_ids': sorted({row['episode_index'] for row in rows}),
        'window_length': horizons.pop(), 'sampling_frequency': source.fps,
        'advantage_definition': 'Stored advantage=(progress_end-progress_start)/chunks_in_group; '
                                'paired 10-step chunks share the group delta; singleton tail receives the full delta. '
                                'Terminal progress is 1 where terminal_progress_forced=true; negatives are retained. '
                                'Progress-increment proxy, not Q-V.',
        'metadata_origin': 'labeling/summary.json + labeling/chunk_advantages.jsonl',
        'provenance_reviewed': False,
    }
    return metadata, labels_path


def sum_robometer_labels(provider, windows, horizon, *, train_stride=None):
    """Combine validated, consecutive base chunks into one label per policy window."""
    records = []
    for window in windows:
        episode = window["trajectory_id"]
        starts = range(window["start_timestep"], window["end_timestep"], ROBOMETER_CHUNK_SIZE)
        components = [provider.records.get((episode, start, ROBOMETER_CHUNK_SIZE)) for start in starts]
        if all(row is not None for row in components):
            records.append({**window, "advantage": math.fsum(float(row["advantage"]) for row in components)})
    metadata = {
        **provider.metadata,
        "window_length": horizon,
        "advantage_definition": (
            "Sum the stored advantages of consecutive 10-step chunks covering real actions in the policy window. "
            f"Base label: {provider.metadata['advantage_definition']}"
        ),
        "aggregation": {
            "method": "sum",
            "base_window_length": ROBOMETER_CHUNK_SIZE,
            "base_provider_fingerprint": provider.fingerprint,
            "window_stride": horizon if train_stride is None else train_stride,
        },
    }
    if train_stride is not None and train_stride != horizon:
        metadata["aggregation"]["validation_window_stride"] = horizon
    return PrecomputedAdvantageProvider(records, metadata)


def prepare_advantage_datasets(train, val, config, horizon):
    if not config.enabled:
        raise ValueError("awr_fm requires advantage_weighting.enabled=true")
    if horizon <= 0 or horizon % ROBOMETER_CHUNK_SIZE:
        raise ValueError("AWR-FM action_horizon must be a positive multiple of 10")
    full_train, full_val = base_dataset(train), base_dataset(val)
    source = full_train.source
    metadata, exported_labels = load_advantage_metadata(source, config)
    train_stride = getattr(full_train, "stride", horizon)
    robometer_base = (metadata["source"], metadata["semantics"], metadata["window_length"]) == (
        "robometer", "progress_increment_proxy", ROBOMETER_CHUNK_SIZE
    )
    if robometer_base and train_stride % ROBOMETER_CHUNK_SIZE:
        raise ValueError("AWR-FM with 10-step RoboMeter labels requires train_stride to be a multiple of 10")
    if not metadata.get('model_version') or metadata.get('training_trajectory_ids') is None:
        logging.getLogger(__name__).warning('Advantage provenance is incomplete: exact model version or estimator training range '
                                          'is unknown; preserving unknown fields in advantage_provenance.json')
    provider = (PrecomputedAdvantageProvider.from_file(config.labels_path, metadata) if config.labels_path
                else PrecomputedAdvantageProvider.from_robometer_dataset(source, metadata))
    if any(getattr(dataset, "action_horizon", horizon) != horizon for dataset in (full_train, full_val)):
        raise ValueError("Dataset window horizon must match the AWR-FM policy horizon")
    windows = [sample_metadata(dataset, index, horizon) for dataset in (full_train, full_val) for index in range(len(dataset))]
    base_windows = windows
    aggregate = metadata["window_length"] != horizon or (robometer_base and train_stride != horizon)
    if aggregate:
        if not robometer_base:
            raise ValueError("Label horizon must match policy horizon unless summing 10-step RoboMeter progress-increment labels")
        base_windows = [
            window_metadata(episode, start, length)
            for episode in (*full_train.episode_ids, *full_val.episode_ids)
            for start, length in action_windows(len(source.rows[episode]["action"]) - 1, ROBOMETER_CHUNK_SIZE)
        ]
    coverage = provider.validate_windows(base_windows, source.fps, config.missing_label_policy)
    if exported_labels is not None:
        exported = PrecomputedAdvantageProvider.from_file(exported_labels, metadata)
        exported.validate_windows(base_windows, source.fps, config.missing_label_policy)
        if set(exported.records) != set(provider.records) or any(
                not math.isclose(row['advantage'], exported.records[key]['advantage'], rel_tol=1e-6, abs_tol=1e-8)
                for key, row in provider.records.items()):
            raise ValueError('RoboMeter JSONL labels do not match Parquet annotations')
    if aggregate:
        provider = sum_robometer_labels(provider, windows, horizon, train_stride=train_stride)
        coverage = {**provider.validate_windows(windows, source.fps, config.missing_label_policy),
                    "base": coverage}
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
