from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from awbc_data.advantage import (AdvantageDataset, PrecomputedAdvantageProvider, window_metadata)
from awbc_models import (FlowMatchingPolicy, FlowMatchingPolicyConfig, VisualEncoderConfig,
                         ObservationEncoderConfig, ConditionalDiTConfig, FlowMatchingObjective)
from awbc_models.advantage_weighting import AdvantageNormalizer, AdvantageWeighting
from awbc_training.advantage import prepare_advantage_datasets, training_forward
from awbc_training.advantage_analysis import summarize_rows, write_analysis
from awbc_training.config import ExperimentConfig, AdvantageWeightingConfig
from awbc_training.trainer import Trainer
from test_awbc_data import root


class SyntheticWindows(Dataset):
    def __init__(self, episode=0):
        self.samples = [(episode, 0, 10), (episode, 10, 3)]
        self.episode_ids = (episode,)
        self.source = SimpleNamespace(fps=30)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode, start, length = self.samples[index]
        return {"observation": {"images": torch.full((1, 3, 32, 48), (episode+start)/20),
                                "state": torch.linspace(-1, 1, 8)},
                "action": torch.full((10, 3), episode+start/100), "action_is_pad": torch.arange(10) >= length,
                "episode_index": torch.tensor(episode), "frame_index": torch.tensor(start)}


def metadata():
    return {"source": "synthetic", "semantics": "advantage", "model_version": "unit-test-v1",
            "advantage_definition": "Artificial scalars for unit tests only; not robot quality labels",
            "sampling_frequency": 30, "window_length": 10, "provenance_reviewed": True,
            "training_trajectory_ids": [], "evaluation_trajectory_ids": [0, 1]}


def records():
    return [{**window_metadata(e, t, length), "advantage": value}
            for e, t, length, value in [(0, 0, 10, -2.), (0, 10, 3, -1.), (1, 0, 10, 100.), (1, 10, 3, 200.)]]


def provider():
    return PrecomputedAdvantageProvider(records(), metadata())


def test_labels_join_by_window_not_row_order_and_follow_shuffle():
    source = SyntheticWindows()
    a = PrecomputedAdvantageProvider(records()[::-1], metadata())
    assert a.fingerprint == provider().fingerprint
    dataset = AdvantageDataset(source, a)
    batches = DataLoader(dataset, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(4))
    for batch in batches:
        start = int(batch["start_timestep"][0])
        assert batch["advantage"][0] == {-0: -2., 10: -1.}[start]
        assert batch["action"][0, 0, 0] == pytest.approx(start/100)
        assert batch["end_timestep"][0] == start+(10 if start == 0 else 3)
    expected = [window_metadata(e, t, n) for e in (0, 1) for _, t, n in SyntheticWindows(e).samples]
    assert a.validate_windows(expected, 30)["label_count"] == 4


@pytest.mark.parametrize("duplicate_key", [False, True])
def test_duplicate_sample_id_and_window_key_are_rejected(duplicate_key):
    rows = records()
    if duplicate_key:
        rows.append({**rows[0], "sample_id": "different-id"})
    else:
        rows[1]["sample_id"] = rows[0]["sample_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        PrecomputedAdvantageProvider(rows, metadata())


def test_missing_extra_horizon_end_and_frequency_must_match():
    expected = records()
    missing = PrecomputedAdvantageProvider(records()[:-1], metadata())
    with pytest.raises(ValueError, match="Missing"):
        missing.validate_windows(expected, 30)
    assert missing.validate_windows(expected, 30, "drop")["missing_windows"] == 1
    dropped = AdvantageDataset(SyntheticWindows(1), missing, missing_policy="drop")
    assert len(dropped) == 1
    wrong_end = records()
    wrong_end[1]["end_timestep"] = 20
    with pytest.raises(ValueError, match="end_timestep"):
        PrecomputedAdvantageProvider(wrong_end, metadata()).validate_windows(expected, 30)
    wrong_horizon = [{**r, "action_horizon": 32} for r in records()]
    with pytest.raises(ValueError, match="horizon"):
        PrecomputedAdvantageProvider(wrong_horizon, {**metadata(), "window_length": 32}).validate_windows(expected, 30)
    with pytest.raises(ValueError, match="sampling_frequency"):
        provider().validate_windows(expected, 20)
    wrong_trajectory = records()
    wrong_trajectory[0]["trajectory_id"] = 99
    with pytest.raises(ValueError, match="Unexpected"):
        PrecomputedAdvantageProvider(wrong_trajectory, metadata()).validate_windows(expected, 30)
    fractional = records(); fractional[0]["start_timestep"] = 0.5
    with pytest.raises(ValueError, match="Unexpected"):
        PrecomputedAdvantageProvider(fractional, metadata()).validate_windows(expected, 30)


@pytest.mark.parametrize("suffix", ["json", "jsonl", "csv", "parquet"])
def test_precomputed_storage_formats(tmp_path, suffix):
    import pyarrow as pa
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    path = tmp_path / f"labels.{suffix}"
    if suffix == "json": path.write_text(json.dumps(records()))
    elif suffix == "jsonl": path.write_text("\n".join(json.dumps(row) for row in records()))
    elif suffix == "csv": pcsv.write_csv(pa.Table.from_pylist(records()), path)
    else: pq.write_table(pa.Table.from_pylist(records()), path)
    assert PrecomputedAdvantageProvider.from_file(path, metadata()).get_advantage(records()[1]) == -1


def test_original_robometer_chunk_file_adapter(tmp_path):
    path = tmp_path / "chunk_advantages.jsonl"
    rows = [{"chunk_id": r["sample_id"], "episode_index": r["trajectory_id"],
             "relative_start_frame": r["start_timestep"], "relative_end_frame": r["end_timestep"],
             "target_chunk_length": 10, "advantage": r["advantage"]} for r in records()]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    adapted = PrecomputedAdvantageProvider.from_file(path, {**metadata(), "source": "robometer", "semantics": "progress_increment_proxy"})
    assert adapted.get_advantage(records()[0]) == -2


def test_robometer_annotated_dataset_adapter_reads_only_chunk_starts():
    source = SimpleNamespace(has_advantage=True, rows={})
    for episode in (0, 1):
        advantage = torch.full((14,), float("nan"))
        starts = torch.zeros(14, dtype=torch.bool)
        valid = torch.zeros(14, dtype=torch.bool)
        lengths = torch.zeros(14, dtype=torch.int64)
        for row in records():
            if row["trajectory_id"] == episode:
                start = row["start_timestep"]
                advantage[start] = row["advantage"]
                starts[start] = valid[start] = True
                lengths[start] = row["end_timestep"]-start
        source.rows[episode] = {"action": torch.zeros(14, 3),
            "annotation.advantage": advantage, "annotation.action_chunk_start": starts,
            "annotation.advantage_valid": valid, "annotation.action_chunk_valid_length": lengths}
    adapted = PrecomputedAdvantageProvider.from_robometer_dataset(source, metadata())
    assert adapted.validate_windows(records(), 30)["label_count"] == 4
    assert adapted.get_record(records()[1])["end_timestep"] == 13
    assert adapted.get_advantage(records()[1]) == -1
    source.rows[0]["annotation.advantage_valid"][10] = False
    with pytest.raises(ValueError, match="Invalid annotated chunk"):
        PrecomputedAdvantageProvider.from_robometer_dataset(source, metadata())


def test_raw_scores_rejected_and_provenance_overlap_is_explicit():
    with pytest.raises(ValueError, match="Raw reward"):
        PrecomputedAdvantageProvider(records(), {**metadata(), "semantics": "progress_score"})
    p = PrecomputedAdvantageProvider(records(), {**metadata(), "training_trajectory_ids": [0]})
    report = p.provenance_report([0], [1])
    assert report["policy_train_overlap_with_estimator_train"] == [0]
    p = PrecomputedAdvantageProvider(records(), {**metadata(), "training_trajectory_ids": None})
    assert p.provenance_report([0], [1])["policy_train_overlap_with_estimator_train"] is None
    invalid = records(); invalid[0]["advantage"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        PrecomputedAdvantageProvider(invalid, metadata())


@pytest.mark.parametrize("mode", ["none", "center", "zscore"])
def test_normalization_manual_values_and_reload(tmp_path, mode):
    values = torch.tensor([-2., -1., 0., 1., 2.])
    normalizer = AdvantageNormalizer.fit(values, mode)
    expected = values if mode in ("none", "center") else values/(values.std(correction=0)+1e-8)
    torch.testing.assert_close(normalizer(values), expected)
    shifted = AdvantageNormalizer.fit(values+3, mode)
    expected_shifted = values+3 if mode == "none" else expected
    torch.testing.assert_close(shifted(values+3), expected_shifted)
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"mean": shifted.mean.item(), "std": shifted.std.item()}))
    loaded = AdvantageNormalizer.from_json(path, mode)
    torch.testing.assert_close(loaded(values+3), shifted(values+3))
    torch.testing.assert_close(AdvantageNormalizer.fit(torch.ones(3))(torch.ones(3)), torch.zeros(3))


def test_train_only_normalization_not_validation_and_saved_statistics(tmp_path):
    labels, meta = tmp_path / "labels.json", tmp_path / "metadata.json"
    labels.write_text(json.dumps(records())); meta.write_text(json.dumps(metadata()))
    cfg = AdvantageWeightingConfig(enabled=True, labels_path=str(labels), metadata_path=str(meta))
    train, val, weighting, report = prepare_advantage_datasets(Subset(SyntheticWindows(), [0]),
        Subset(SyntheticWindows(1), [0]), cfg, 10)
    assert report["statistics"]["count"] == 2  # Full train split, before diagnostic subsetting.
    assert weighting.normalizer.mean == -1.5 and weighting.normalizer.std == 0.5
    assert len(train) == len(val) == 1
    path = tmp_path / "advantage_stats.json"
    path.write_text(json.dumps(report["statistics"]))
    cfg.statistics_path = str(path)
    _, _, loaded, _ = prepare_advantage_datasets(Subset(SyntheticWindows(), [0]), Subset(SyntheticWindows(1), [0]), cfg, 10)
    torch.testing.assert_close(weighting.normalizer.mean, loaded.normalizer.mean)
    labels.write_text(json.dumps(records()[:-1]))
    cfg.statistics_path = None; cfg.missing_label_policy = "drop"
    train, val, _, report = prepare_advantage_datasets(Subset(SyntheticWindows(), [0, 1]),
        Subset(SyntheticWindows(1), [0, 1]), cfg, 10)
    assert len(train) == len(val) == 2 and report["coverage"]["missing_windows"] == 1


def test_weight_transform_monotonic_temperature_clipping_and_detachment():
    a = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
    normalizer = AdvantageNormalizer("none")
    broad = AdvantageWeighting(normalizer, beta=1., min_weight=None, max_weight=None)(a)
    sharp = AdvantageWeighting(normalizer, beta=0.5, min_weight=None, max_weight=None)(a)
    assert (broad.diff() > 0).all() and sharp[-1]/sharp[0] > broad[-1]/broad[0]
    assert not broad.requires_grad
    clipped = AdvantageWeighting(normalizer, beta=0.1)(a)
    assert clipped.min() >= 0.1-1e-6 and clipped.max() <= 10+1e-5
    assert torch.isfinite(clipped).all() and (clipped > 0).all()
    differentiable = AdvantageWeighting(normalizer, detach_weights=False)(a)
    differentiable.sum().backward()
    assert a.grad.abs().sum() > 0
    for beta in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="beta"):
            AdvantageWeighting(normalizer, beta=beta)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        AdvantageWeighting(normalizer, min_weight=None, max_weight=None)(torch.tensor([1e6]))


def test_existing_fm_loss_manual_weighting_equal_weights_and_padding():
    objective = FlowMatchingObjective()
    target = torch.zeros(3, 10, 2)
    prediction = torch.tensor([1., 2., 3.]).sqrt()[:, None, None].expand_as(target).clone()
    mask = torch.zeros(3, 10, dtype=torch.bool); mask[1, 3:] = True
    prediction[1, 3:] = 1e20
    weighted = objective.compute_loss(prediction, target, mask, torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(weighted["loss"], torch.tensor(14/6))
    plain = objective.compute_loss(prediction, target, mask)
    equal = objective.compute_loss(prediction, target, mask, torch.ones(3))
    torch.testing.assert_close(plain["loss"], equal["loss"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="all-padding"):
        objective.compute_loss(prediction, target, torch.ones_like(mask))


def tiny_policy():
    config = FlowMatchingPolicyConfig(
        vision=VisualEncoderConfig(d_model=32, pretrained=False, num_cameras=1),
        observation=ObservationEncoderConfig(state_dim=8, hidden_dim=32, num_layers=1, num_heads=4, ffn_dim=64, dropout=0),
        dit=ConditionalDiTConfig(action_dim=3, action_horizon=10, hidden_dim=32, num_layers=1, num_heads=4,
                                 ff_dim=64, dropout=0, timestep_embedding_dim=16, timestep_hidden_dim=32),
        camera_names=("synthetic_camera",), n_action_steps=8)
    stats = {"action": {"min": [-1.]*3, "max": [1.]*3}, "observation.state": {"min": [-1.]*8, "max": [1.]*8}}
    return FlowMatchingPolicy(config, stats), stats


def test_complete_synthetic_awr_gradient_path_and_detached_weights():
    policy, _ = tiny_policy()
    with torch.no_grad():
        policy.dit.velocity_head.weight.copy_(torch.linspace(-0.1, 0.1, 96).reshape(3, 32))
        policy.dit.blocks[0].adaLN_modulation[-1].bias[5*32:6*32] = 0.2
    batch = next(iter(DataLoader(AdvantageDataset(SyntheticWindows(), provider()), batch_size=2)))
    batch["advantage"] = batch["advantage"].requires_grad_()
    output = training_forward(policy, batch, AdvantageWeighting(AdvantageNormalizer("none")))
    assert output["unweighted_loss"] == output["per_sample_loss"].mean()
    output["loss"].backward()
    assert batch["advantage"].grad is None and not output["sample_weight"].requires_grad
    assert policy.dit.velocity_head.weight.grad.abs().sum() > 0
    assert policy.observation_encoder.state_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.vision_encoder.backbone.parameters())
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


@pytest.mark.parametrize("mode", ["fm", "awr_fm"])
def test_training_modes_checkpoint_and_unweighted_validation(tmp_path, mode):
    directory = tmp_path / mode; directory.mkdir()
    p = provider()
    train, val = AdvantageDataset(SyntheticWindows(), p), AdvantageDataset(SyntheticWindows(1), p)
    policy, stats = tiny_policy()
    cfg = ExperimentConfig()
    cfg.training = replace(cfg.training, objective=mode, num_epochs=2, batch_size=1, num_workers=0,
                            gradient_accumulation_steps=2, max_updates=2)
    cfg.amp.enabled = False; cfg.logging.use_wandb = False; cfg.logging.log_interval = 1
    cfg.scheduler.warmup_steps = 0; cfg.validation.batch_size = 2
    cfg.advantage_weighting.enabled = mode == "awr_fm"
    weighting = AdvantageWeighting(AdvantageNormalizer.fit(train.advantages)) if mode == "awr_fm" else None
    run = Trainer(cfg, policy, train, val, stats, {"synthetic": True}, directory, torch.device("cpu"), weighting=weighting)
    result = run.fit()
    assert result["global_step"] == 2 and run.ema.num_updates == 2
    calls = []
    handle = policy.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs.get("sample_weight")), with_kwargs=True)
    before = weighting.normalizer.state_dict() if weighting else None
    ema_updates = run.ema.num_updates
    assert run.validate() == run.validate()
    assert all(weight is None for weight in calls) and run.ema.num_updates == ema_updates
    if weighting:
        for key, value in before.items(): torch.testing.assert_close(value, weighting.normalizer.state_dict()[key])
    state = torch.load(directory / "checkpoints/latest_checkpoint.pt", weights_only=False)
    assert state["advantage_weighting"] is not None if weighting else state["advantage_weighting"] is None
    assert (directory / "metrics.jsonl").exists()
    handle.remove(); run.logger.close()


def test_ess_and_analysis_outputs(tmp_path):
    rows = [{"trajectory_id": i%2, "advantage": float(i-2), "weight": float(i+1), "loss": float(i+1), "clipped": i==0} for i in range(5)]
    summary = summarize_rows(rows, min_weight=1.)
    assert summary["weight/effective_sample_size"] == pytest.approx(15**2/55)
    assert summary["weight/ess_ratio"] == pytest.approx(15**2/55/5)
    assert summary["weight/clip_fraction"] == 0.2
    write_analysis(rows, tmp_path, min_weight=1.)
    assert (tmp_path / "distributions.png").exists()
    assert len(json.loads((tmp_path / "analysis.json").read_text())["loss_by_advantage_quantile"]) == 4


def test_weighted_accumulation_matches_large_batch_and_resumes(tmp_path):
    from test_training import ToyDataset, ToyPolicy

    class WeightedToyDataset(ToyDataset):
        def __init__(self):
            super().__init__(5)
            self.advantages = torch.linspace(-2, 2, 5)
        def __getitem__(self, index):
            return {**super().__getitem__(index), "advantage": self.advantages[index], "trajectory_id": torch.tensor(index//2)}

    class WeightedToyPolicy(ToyPolicy):
        def forward(self, batch, sample_weight=None, **pair):
            prediction = self.dit(batch["action"][:, 0]).unsqueeze(1)
            return FlowMatchingObjective().compute_loss(prediction, batch["target"].unsqueeze(1), sample_weight=sample_weight)

    def run(name, size, accumulation, max_updates=None, state=None):
        torch.manual_seed(12)
        cfg = ExperimentConfig()
        cfg.training = replace(cfg.training, objective="awr_fm", num_epochs=1, batch_size=size,
                                num_workers=0, gradient_accumulation_steps=accumulation, max_updates=max_updates)
        cfg.logging.use_wandb = False; cfg.amp.enabled = False; cfg.scheduler.warmup_steps = 0
        cfg.advantage_weighting.enabled = True
        directory = tmp_path / name; directory.mkdir()
        weighting = AdvantageWeighting(AdvantageNormalizer("none"))
        trainer = Trainer(cfg, WeightedToyPolicy(), WeightedToyDataset(), WeightedToyDataset(),
                          {"synthetic": True}, {"split": "synthetic"}, directory, torch.device("cpu"),
                          weighting=weighting, resume_state=state)
        trainer.fit()
        trainer.logger.close()
        return trainer

    large, micro = run("large", 4, 1), run("micro", 2, 2)
    for a, b in zip(large.policy.parameters(), micro.policy.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    first = run("first", 2, 2, max_updates=1)
    state = torch.load(first.directory / "checkpoints/latest_checkpoint.pt", weights_only=False)
    assert len(state["epoch_awr_rows"]) == 4
    resumed = run("resumed", 2, 2, state=state)
    for a, b in zip(micro.policy.parameters(), resumed.policy.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert resumed.ema.num_updates == 2


def test_new_default_horizon_ten_preserves_saved_sixteen_configuration():
    config = FlowMatchingPolicyConfig()
    assert config.dit.action_horizon == 10
    saved = asdict(config)
    saved["dit"]["action_horizon"] = 16
    restored = FlowMatchingPolicyConfig.from_dict(saved)
    assert restored.dit.action_horizon == 16


def write_legacy_export(root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    labels = []
    for path in (root/'data').rglob('*.parquet'):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            start, episode = row['frame_index'], row['episode_index']
            row.update({'annotation.advantage': -0.02 if episode == 0 else 0.1,
                        'annotation.advantage_valid': start < 13,
                        'annotation.action_chunk_start': start in (0, 10),
                        'annotation.action_chunk_valid_length': 10 if start < 10 else 3})
        pq.write_table(pa.Table.from_pylist(rows), path)
    for episode in (0, 1):
        value = -0.02 if episode == 0 else 0.1
        for start, length in ((0, 10), (10, 3)):
            labels.append({'chunk_id': f'ep{episode}_c{start}', 'episode_index': episode,
                           'relative_start_frame': start, 'relative_end_frame': start+length,
                           'target_chunk_length': 10, 'advantage': value, 'advantage_valid': True,
                           'chunks_in_group': 2, 'advantage_divisor': 2,
                           'progress_start': 0.5, 'progress_end': 0.5+2*value,
                           'terminal_progress_forced': False})
    folder = root/'labeling'
    folder.mkdir()
    (folder/'summary.json').write_text(json.dumps({'model_path': '/synthetic/Robometer-test-fixture'}))
    (folder/'chunk_advantages.jsonl').write_text('\n'.join(json.dumps(row) for row in labels)+'\n')
    return labels


def test_existing_robometer_export_needs_no_new_metadata_file(root, caplog):
    from awbc_training.data import prepare_datasets
    from awbc_training.config import DatasetConfig
    from awbc_training.advantage import load_advantage_metadata
    from awbc_data.images import ImagePreprocessConfig
    write_legacy_export(root)
    train, val, _, _ = prepare_datasets(DatasetConfig(root=str(root), stratify_key=None), ImagePreprocessConfig())
    config = AdvantageWeightingConfig(enabled=True)
    bound, original_val, weighting, report = prepare_advantage_datasets(train, val, config, 10)
    assert len(bound) == len(original_val) == 2
    assert original_val is val and report['coverage']['missing_windows'] == 0
    assert report['metadata']['source'] == 'robometer'
    assert report['metadata']['model_version'] is None
    assert report['metadata']['training_trajectory_ids'] is None
    assert report['metadata']['provenance_reviewed'] is False
    assert not report['estimator_training_range_known']
    assert torch.isfinite(weighting(bound.advantages)).all()
    assert 'provenance is incomplete' in caplog.text
    config.metadata_path = str(root/'explicit_missing_metadata.json')
    with pytest.raises(FileNotFoundError, match='explicit_missing_metadata'):
        load_advantage_metadata(train.dataset.source, config)


@pytest.mark.parametrize('horizon', [10, 20])
@pytest.mark.parametrize('corruption', ['formula', 'parquet'])
def test_legacy_metadata_loading_rejects_inconsistent_labels(root, corruption, horizon):
    from awbc_training.data import prepare_datasets
    from awbc_training.config import DatasetConfig
    from awbc_data.images import ImagePreprocessConfig
    labels = write_legacy_export(root)
    labels[0]['advantage'] += 0.1
    if corruption == 'parquet':
        labels[0]['progress_end'] += 0.2  # Formula still matches, but annotation values do not.
    (root/'labeling/chunk_advantages.jsonl').write_text('\n'.join(json.dumps(row) for row in labels)+'\n')
    train, val, _, _ = prepare_datasets(DatasetConfig(root=str(root), stratify_key=None), ImagePreprocessConfig(),
                                       action_horizon=horizon)
    with pytest.raises(ValueError, match='formula' if corruption == 'formula' else 'Parquet annotations'):
        prepare_advantage_datasets(train, val, AdvantageWeightingConfig(enabled=True), horizon)


@pytest.mark.parametrize("horizon", [20, 30])
def test_larger_awr_windows_sum_real_chunks_and_refit_train_stats(root, horizon):
    from awbc_data.images import ImagePreprocessConfig
    from awbc_training.config import DatasetConfig
    from awbc_training.data import prepare_datasets

    labels = write_legacy_export(root)
    dataset_config = DatasetConfig(root=str(root), stratify_key=None)
    preprocess = ImagePreprocessConfig(size=(24, 32))
    config = AdvantageWeightingConfig(enabled=True)
    base_train, base_val, base_stats, base_manifest = prepare_datasets(dataset_config, preprocess)
    _, _, _, base_report = prepare_advantage_datasets(base_train, base_val, config, 10)
    train, val, stats, manifest = prepare_datasets(dataset_config, preprocess, action_horizon=horizon)
    bound, original_val, weighting, report = prepare_advantage_datasets(train, val, config, horizon)

    assert stats == base_stats and manifest["train_episodes"] == base_manifest["train_episodes"]
    assert manifest["action_horizon"] == horizon
    assert len(bound) == len(val) == 1 and original_val is val
    batch = next(iter(DataLoader(bound, batch_size=1)))
    episode = batch["trajectory_id"].item()
    expected = sum(row["advantage"] for row in labels if row["episode_index"] == episode)
    assert batch["action_horizon"].item() == horizon
    assert batch["start_timestep"].item() == 0 and batch["end_timestep"].item() == 13
    assert batch["advantage_valid"].item() and batch["advantage"].item() == pytest.approx(expected)
    assert batch["action"].shape == (1, horizon, 8)
    assert batch["action_is_pad"][0].tolist() == [False] * 13 + [True] * (horizon - 13)
    raw = train.dataset.denormalize("action", batch["action"][0])[:, 0]
    torch.testing.assert_close(raw, episode * 100 + torch.arange(horizon, dtype=torch.float32).clamp_max(12))
    assert report["statistics"]["count"] == 1 and report["statistics"]["mean"] == pytest.approx(expected)
    assert report["coverage"]["base"]["label_count"] == 4 and report["coverage"]["label_count"] == 2
    assert report["metadata"]["aggregation"]["base_provider_fingerprint"] == base_report["fingerprint"]
    assert torch.isfinite(weighting(batch["advantage"])).all()

    old_stats = root / "base_advantage_stats.json"
    old_stats.write_text(json.dumps(base_report["statistics"]))
    config.statistics_path = str(old_stats)
    with pytest.raises(ValueError, match="statistics do not belong"):
        prepare_advantage_datasets(train, val, config, horizon)
    train.dataset.close()


def test_aggregation_uses_each_base_chunk_once_and_never_sums_missing_components():
    from awbc_training.advantage import sum_robometer_labels

    chunks = [
        {**window_metadata(0, start, length), "advantage": value}
        for start, length, value in [(0, 10, 0.5), (10, 10, -0.25), (20, 10, 0.125), (30, 4, -0.5)]
    ]
    target = [window_metadata(0, 0, 30, 30), window_metadata(0, 30, 4, 30)]
    provenance = {**metadata(), "source": "robometer", "semantics": "progress_increment_proxy"}
    complete = sum_robometer_labels(PrecomputedAdvantageProvider(chunks[::-1], provenance), target, 30)
    assert [complete.get_advantage(window) for window in target] == [0.375, -0.5]
    missing = sum_robometer_labels(PrecomputedAdvantageProvider(chunks[1:], provenance), target, 30)
    assert missing.validate_windows(target, 30, "drop")["missing_windows"] == 1
    with pytest.raises(ValueError, match="Missing advantage"):
        missing.validate_windows(target, 30)
