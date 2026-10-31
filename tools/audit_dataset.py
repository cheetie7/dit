"""Read-only inspection of the downloaded RoboMeter export; no training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from awbc_data.advantage import AdvantageDataset, PrecomputedAdvantageProvider, sample_metadata
from awbc_data.inspect import tensor_report, visualize
from awbc_models.advantage_weighting import AdvantageNormalizer, AdvantageWeighting
from awbc_models.flow_matching_objective import FlowMatchingObjective
from awbc_data.images import ImagePreprocessConfig
from awbc_training.config import DatasetConfig
from awbc_training.data import prepare_datasets


def describe(values):
    values = np.asarray(values)
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()), "median": float(np.median(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/data_audit'))
    args = parser.parse_args()
    root, original, output = args.root, args.original, args.output
    output.mkdir(parents=True, exist_ok=True)
    chunks = [json.loads(line) for line in (root/'labeling/chunk_advantages.jsonl').read_text().splitlines()]
    boundaries = {row['boundary_id']: row for row in
                  (json.loads(line) for line in (root/'labeling/boundary_progress.jsonl').read_text().splitlines())}
    summary = json.loads((root/'labeling/summary.json').read_text())
    raw = pq.read_table(original/'data/chunk-000/file-000.parquet')
    labeled = pq.read_table(root/'data/chunk-000/file-000.parquet')
    preserved = {name: raw[name].equals(labeled[name]) for name in raw.column_names}
    assert all(preserved.values())
    hashes = json.loads((root/'labeling/video_sha256.json').read_text())
    video_checks = {}
    for name, expected in hashes.items():
        with (root/name).open('rb') as stream:
            video_checks[name] = hashlib.file_digest(stream, 'sha256').hexdigest() == expected
    assert all(video_checks.values())

    train, val, stats, split = prepare_datasets(DatasetConfig(root=str(root)), ImagePreprocessConfig())
    source = train.dataset.source
    provenance = {"source": "robometer", "semantics": "progress_increment_proxy", "model_name": "Robometer-4B",
                  "model_path_recorded": summary['model_path'], "model_version": None, "checkpoint_hash": None,
                  "label_generation_date": None, "training_trajectory_ids": None,
                  "evaluation_trajectory_ids": sorted(source.episodes), "window_length": 10, "sampling_frequency": source.fps,
                  "advantage_definition": "(progress_end-progress_start)/chunks_in_group; two 10-step chunks share a boundary delta, singleton tail receives full delta; terminal progress forced to 1",
                  "provenance_reviewed": False,
                  "review_status": "Data/formula checks passed; exact checkpoint identity and estimator training range remain unknown"}
    parquet_provider = PrecomputedAdvantageProvider.from_robometer_dataset(source, provenance)
    table_provider = PrecomputedAdvantageProvider.from_file(root/'labeling/chunk_advantages.jsonl', provenance)
    windows = [sample_metadata(ds, i) for ds in (train, val) for i in range(len(ds))]
    coverage = parquet_provider.validate_windows(windows, source.fps)
    table_provider.validate_windows(windows, source.fps)
    for window in windows:
        assert np.isclose(parquet_provider.get_advantage(window), table_provider.get_advantage(window), rtol=1e-6, atol=1e-8)
    for row in chunks:
        start, end = boundaries[row['start_boundary_id']], boundaries[row['end_boundary_id']]
        assert start['episode_index'] == end['episode_index'] == row['episode_index']
        assert row['relative_end_frame']-row['relative_start_frame'] == row['valid_length']
        assert row['advantage_divisor'] == row['chunks_in_group']
        assert np.isclose(row['progress_start'], start['progress_used'])
        assert np.isclose(row['progress_end'], end['progress_used'])
        assert np.isclose(row['advantage'], (end['progress_used']-start['progress_used'])/row['advantage_divisor'])
    for episode, data in source.rows.items():
        assert not data['annotation.advantage_valid'][-1]
        assert not data['annotation.action_chunk_start'][-1]
        assert data['frame_index'][-1] == source.episodes[episode]['length']-1
    labeled_train = AdvantageDataset(train, parquet_provider)
    normalizer = AdvantageNormalizer.fit(labeled_train.advantages)
    weighting = AdvantageWeighting(normalizer)
    weights = weighting(labeled_train.advantages)
    flags = weighting.clip_flags(labeled_train.advantages)
    by_key = {(r['episode_index'],r['relative_start_frame']): r for r in chunks}
    forced = torch.tensor([by_key[(r['trajectory_id'],r['start_timestep'])]['terminal_progress_forced']
                           for r in labeled_train.records])
    train_ids = set(split['train_episodes'])
    def subgroup(predicate):
        selected = [r['advantage'] for r in chunks if predicate(r)]
        return describe(selected)
    tail = next(i for i,r in enumerate(labeled_train.records) if r['end_timestep']-r['start_timestep'] == 2)
    full = next(i for i,r in enumerate(labeled_train.records) if r['end_timestep']-r['start_timestep'] == 10)
    batch = next(iter(DataLoader(Subset(labeled_train,[full,tail]), batch_size=2)))
    for i in range(2):
        length = int(batch['valid_length'][i])
        assert int((~batch['action_is_pad'][i]).sum()) == length
        assert int(batch['end_timestep'][i]-batch['start_timestep'][i]) == length
    delta = (batch['video_timestamp']-batch['video_target_timestamp']).abs().max().item()
    assert delta <= 0.5/source.fps+1e-3
    prediction = torch.ones_like(batch['action'])
    prediction[batch['action_is_pad']] = 1e6
    masked = FlowMatchingObjective().compute_loss(prediction, torch.zeros_like(prediction), batch['action_is_pad'],
                                                weighting(batch['advantage']))
    torch.testing.assert_close(masked['per_sample_loss'], torch.ones(2))
    visualize(batch, train.dataset, output/'batch.png')
    tensors = {k:v for k,v in batch.items() if isinstance(v,(dict,torch.Tensor))}
    report = {"root": str(root), "inspection_only": True, "training_started": False,
              "episodes": len(source.episodes), "frames": labeled.num_rows, "fps": source.fps,
              "camera_order": list(source.cameras), "chunks": len(chunks), "boundaries": len(boundaries),
              "valid_transitions": sum(r['valid_length'] for r in chunks),
              "partial_chunks": sum(r['is_partial'] for r in chunks),
              "valid_length_histogram": dict(sorted(Counter(r['valid_length'] for r in chunks).items())),
              "original_columns_preserved": preserved, "video_sha256_matches_export": video_checks,
              "coverage": coverage, "jsonl_and_parquet_labels_agree": True, "boundary_formula_matches": True,
              "terminal_boundaries_forced": sum(r['terminal_progress_forced'] for r in boundaries.values()),
              "paired_chunks": sum(r['advantage_divisor']==2 for r in chunks),
              "singleton_chunks": sum(r['advantage_divisor']==1 for r in chunks),
              "negative_chunks": sum(r['advantage']<0 for r in chunks),
              "advantage_all": describe([r['advantage'] for r in chunks]),
              "advantage_forced_terminal_groups": subgroup(lambda r:r['terminal_progress_forced']),
              "advantage_other_groups": subgroup(lambda r:not r['terminal_progress_forced']),
              "split": {k:v for k,v in split.items() if not k.endswith('sample_indices')},
              "train_advantage_stats": {"mean": normalizer.mean.item(), "std": normalizer.std.item(), "count": len(labeled_train)},
              "example_weight_config": {"normalization":"zscore","beta":1,"min_weight":0.1,"max_weight":10},
              "train_weight_stats": {**describe(weights.numpy()), "clip_fraction": flags.float().mean().item(),
                  "ess": float(weights.sum().square()/weights.square().sum()),
                  "ess_ratio": float(weights.sum().square()/weights.square().sum()/len(weights)),
                  "forced_terminal_group_sample_fraction": forced.float().mean().item(),
                  "forced_terminal_group_weight_fraction": float(weights[forced].sum()/weights.sum())},
              "batch": tensor_report(tensors), "batch_valid_lengths": batch['valid_length'].tolist(),
              "real_batch_mask_test": {"padding_error":1e6,"expected_valid_mse":1.,
                                       "per_sample_loss":masked['per_sample_loss'].tolist(),"weighted_loss":masked['loss'].item()},
              "max_video_time_error_seconds": delta,
              "provenance": parquet_provider.provenance_report(train_ids,split['val_episodes'])}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'observed_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('split','batch','provenance')},indent=2))
    print('split:', {k:split[k] for k in ['train_chunks','val_chunks']}, 'episodes:',len(train_ids),len(split['val_episodes']))
    print('batch shapes:',{k:v['shape'] for k,v in report['batch'].items()})


if __name__ == '__main__':
    main()
