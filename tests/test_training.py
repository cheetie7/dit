from copy import deepcopy
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from awbc_models import ConditionalDiTConfig, FlowMatchingPolicyConfig
from awbc_training.checkpoint import save_checkpoint
from awbc_training.config import ExperimentConfig, AMPConfig
from awbc_training.ema import EMA
from awbc_training.optim import build_optimizer
from awbc_training.runtime import Precision, restore_rng, rng_state, seed_everything
from awbc_training.trainer import Trainer
from awbc_training.optim import build_scheduler
from awbc_training.data import prepare_datasets
from awbc_training.config import DatasetConfig
from awbc_data import ImagePreprocessConfig
from test_awbc_data import root


class ToyDataset(Dataset):
    def __init__(self, n=9):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return {"action": torch.tensor([[(index+1)/10, 1.]]), "target": torch.tensor([(index+1)/5])}


class ToyPolicy(nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.config = FlowMatchingPolicyConfig(dit=ConditionalDiTConfig(action_dim=2, action_horizon=1), n_action_steps=1)
        self.dit = nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.1 if stochastic else 0.)
        self.stochastic = stochastic

    def forward(self, batch, *, noise=None, timestep=None):
        prediction = self.dit(self.dropout(batch["action"][:, 0]))
        if self.stochastic:
            random_input = torch.rand_like(prediction) if noise is None else noise[:, 0, :1]
            prediction = prediction + random_input*0.05
        per_sample = (prediction-batch["target"]).float().square().mean(1)
        return {"loss": per_sample.mean(), "per_sample_loss": per_sample}


def trainer(tmp_path, name, *, batch_size=2, accumulation=2, max_updates=2, epochs=2,
            state=None, stochastic=False, n=9, num_steps=None):
    directory = tmp_path / name
    directory.mkdir()
    config = ExperimentConfig()
    config.training = replace(config.training, num_epochs=epochs, num_steps=num_steps, batch_size=batch_size, num_workers=0,
                              gradient_accumulation_steps=accumulation, max_updates=max_updates,
                              gradient_clip_norm=100., device="cpu")
    config.amp.enabled = False
    config.scheduler.warmup_steps = 0
    config.validation.batch_size = 3
    config.logging.use_wandb = False
    config.logging.log_interval = 1
    model = ToyPolicy(stochastic=stochastic)
    return Trainer(config, model, ToyDataset(n), ToyDataset(4), {"train": True}, {"split": "same"},
                   directory, torch.device("cpu"), resume_state=state)


def test_accumulation_matches_larger_batch_and_incomplete_window(tmp_path):
    seed_everything(4)
    large = trainer(tmp_path, "large", batch_size=4, accumulation=1, n=5, epochs=1, max_updates=None)
    large.fit()
    seed_everything(4)
    micro = trainer(tmp_path, "micro", batch_size=2, accumulation=2, n=5, epochs=1, max_updates=None)
    micro.fit()
    assert large.global_step == micro.global_step == 2
    assert micro.ema.num_updates == 2 and micro.scheduler.last_epoch == 2
    for a, b in zip(large.policy.parameters(), micro.policy.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    large.logger.close()
    micro.logger.close()


def test_mid_epoch_resume_restores_optimizer_scheduler_ema_rng_and_data_cursor(tmp_path):
    seed_everything(9)
    whole = trainer(tmp_path, "whole", max_updates=4, stochastic=True)
    whole.fit()
    expected = deepcopy(whole.policy.state_dict())
    expected_ema = deepcopy(whole.ema.state_dict())
    whole.logger.close()
    seed_everything(9)
    first = trainer(tmp_path, "first", max_updates=2, stochastic=True)
    first.fit()
    assert first.batch_in_epoch == 4 and first.epoch == 0
    checkpoint = torch.load(first.directory / "checkpoints/latest_checkpoint.pt", weights_only=False)
    assert checkpoint["global_step"] == 2
    fixed_batch = {"action": torch.ones(2, 1, 2), "target": torch.ones(2, 1)}
    fixed_noise = torch.zeros(2, 1, 2)
    before_restore = first.policy.eval()(fixed_batch, noise=fixed_noise)["per_sample_loss"].detach()
    first.logger.close()
    random.random(); np.random.rand(); torch.rand(5)
    resumed = trainer(tmp_path, "resumed", state=checkpoint, max_updates=4, stochastic=True)
    assert resumed.optimizer.state and resumed.scheduler.last_epoch == 2 and resumed.ema.num_updates == 2
    torch.testing.assert_close(torch.get_rng_state(), checkpoint["rng_state"]["torch"])
    after_restore = resumed.policy.eval()(fixed_batch, noise=fixed_noise)["per_sample_loss"].detach()
    torch.testing.assert_close(before_restore, after_restore, rtol=0, atol=0)
    resumed.fit()
    assert resumed.global_step == 4 and resumed.ema.num_updates == 4 and resumed.scheduler.last_epoch == 4
    for name, value in expected.items():
        torch.testing.assert_close(value, resumed.policy.state_dict()[name], rtol=0, atol=0)
    for name, value in expected_ema["policy"].items():
        torch.testing.assert_close(value, resumed.ema.policy.state_dict()[name], rtol=0, atol=0)
    assert not list((resumed.directory / "checkpoints").glob("*best*"))
    resumed.logger.close()


def test_validation_is_repeatable_shares_conditions_and_preserves_training_rng(tmp_path):
    run = trainer(tmp_path, "validation", stochastic=True)
    before = torch.get_rng_state().clone()
    a, b = run.validate(), run.validate()
    assert a == b and a["val/loss"] == a["val/ema_loss"]
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert run.global_step == run.ema.num_updates == 0
    run.logger.close()


def test_ema_math_frozen_parameters_buffers_and_restore():
    model = nn.Linear(2, 1)
    model.weight.requires_grad_(False)
    model.register_buffer("counter", torch.tensor(0))
    with torch.no_grad():
        model.weight.fill_(1.)
        model.bias.fill_(1.)
    ema = EMA(model, 0.5)
    with torch.no_grad():
        model.weight.fill_(3.)
        model.bias.fill_(5.)
        model.counter.fill_(7)
    ema.update(model)
    torch.testing.assert_close(ema.policy.weight, torch.full_like(model.weight, 3.))
    torch.testing.assert_close(ema.policy.bias, torch.full_like(model.bias, 3.))
    assert ema.policy.counter == 7 and ema.num_updates == 1
    restored = EMA(model, 0.9)
    restored.load_state_dict(ema.state_dict())
    assert restored.decay == 0.5 and restored.num_updates == 1
    torch.testing.assert_close(restored.policy.bias, ema.policy.bias)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_available_cpu_precisions_forward_backward_and_scaler(dtype):
    precision = Precision(AMPConfig(enabled=True, dtype=dtype), torch.device("cpu"))
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with precision.autocast():
        loss = model(torch.full((2, 2), 0.01)).float().square().mean()*0.01
    precision.scaler.scale(loss).backward()
    precision.scaler.unscale_(optimizer)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    precision.scaler.step(optimizer)
    precision.scaler.update()
    assert torch.isfinite(loss)
    assert precision.scaler.is_enabled() == (dtype == "float16")
    if dtype == "float16":
        restored = Precision(AMPConfig(enabled=True, dtype=dtype), torch.device("cpu"))
        restored.scaler.load_state_dict(precision.scaler.state_dict())
        assert restored.scaler.state_dict() == precision.scaler.state_dict()


def test_optimizer_groups_frozen_backbone_independent_lr_and_unique_parameters():
    model = nn.Module()
    model.vision_encoder = nn.Module()
    model.vision_encoder.backbone = nn.Linear(2, 2)
    model.vision_encoder.input_proj = nn.Linear(2, 2)
    model.observation_encoder = nn.LayerNorm(2)
    model.dit = nn.Linear(2, 2)
    cfg = ExperimentConfig().optimizer
    optimizer = build_optimizer(model, cfg)
    assert {g["lr"] for g in optimizer.param_groups if g["name"].startswith("backbone")} == {cfg.backbone_learning_rate}
    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    model.vision_encoder.backbone.requires_grad_(False)
    optimizer = build_optimizer(model, cfg)
    assert not any(g["name"].startswith("backbone") for g in optimizer.param_groups)
    assert all(p.requires_grad for g in optimizer.param_groups for p in g["params"])


def test_checkpoint_atomic_retention(tmp_path):
    for epoch in range(1, 5):
        save_checkpoint({"epoch": epoch}, tmp_path, epoch_number=epoch, keep_last_n=2)
    assert [p.name for p in sorted(tmp_path.glob("epoch_*.pt"))] == ["epoch_0003.pt", "epoch_0004.pt"]
    assert torch.load(tmp_path / "latest_checkpoint.pt", weights_only=False)["epoch"] == 4
    assert not list(tmp_path.glob("*.tmp"))


def test_trajectory_split_train_only_statistics_and_video_alignment(root):
    train, val, stats, manifest = prepare_datasets(
        DatasetConfig(root=str(root), val_fraction=0.5, stratify_key=None), ImagePreprocessConfig(size=(12, 16)))
    a, b = train.dataset, val.dataset
    assert not set(a.episode_ids) & set(b.episode_ids)
    assert a.stats is b.stats
    assert stats == a.source.statistics(a.episode_ids)
    assert manifest["train_chunks"] == manifest["val_chunks"] == 2
    for dataset in (a, b):
        for eid, start, length in dataset.samples:
            assert start+length <= dataset.source.episodes[eid]["length"]-1
        sample = dataset[0]
        torch.testing.assert_close(sample["video_timestamp"], sample["video_target_timestamp"])
        assert (~sample["action_is_pad"]).sum() == sample["valid_length"]
        dataset.close()


def test_warmup_cosine_update_schedule():
    cfg = ExperimentConfig()
    cfg.scheduler.warmup_steps = 2
    model = nn.Linear(1, 1)
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps=6)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(cfg.optimizer.learning_rate/2)
    rates = []
    for _ in range(6):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[1] == pytest.approx(cfg.optimizer.learning_rate)
    assert rates[-1] == pytest.approx(cfg.optimizer.learning_rate*cfg.scheduler.min_lr_ratio)
    assert scheduler.last_epoch == 6


def test_skipped_scaled_update_does_not_step_scheduler_or_ema(tmp_path, monkeypatch):
    bars = record_progress(monkeypatch)
    class SkipOnceScaler:
        def __init__(self):
            self.value, self.first = 2., True
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def get_scale(self): return self.value
        def is_enabled(self): return True
        def step(self, optimizer):
            if not self.first: optimizer.step()
        def update(self):
            if self.first: self.value, self.first = 1., False
        def state_dict(self): return {"value": self.value}

    run = trainer(tmp_path, "skip", max_updates=2, n=8)
    run.precision.scaler = SkipOnceScaler()
    run.fit()
    assert run.global_step == run.scheduler.last_epoch == run.ema.num_updates == 2
    assert bars[0].n == 2
    run.logger.close()


def record_progress(monkeypatch):
    import awbc_training.trainer as module
    original, bars = module.tqdm, []
    def tracked(*args, **kwargs):
        kwargs['disable'] = False
        bar = original(*args, **kwargs)
        if kwargs.get('desc') == 'Training':
            bars.append(bar)
        return bar
    monkeypatch.setattr(module, 'tqdm', tracked)
    return bars


def test_step_budget_overrides_epochs_and_resumes_progress_and_schedule(tmp_path, monkeypatch):
    bars = record_progress(monkeypatch)
    seed_everything(9)
    whole = trainer(tmp_path, 'steps_whole', num_steps=5, epochs=1, max_updates=None, n=5, stochastic=True)
    whole.fit()
    assert whole.total_steps == whole.global_step == whole.scheduler.last_epoch == whole.ema.num_updates == 5
    assert whole.epoch > 1 and bars[0].total == bars[0].n == 5
    seed_everything(9)
    first = trainer(tmp_path, 'steps_first', num_steps=5, epochs=1, max_updates=3, n=5, stochastic=True)
    first.fit()
    state = torch.load(first.directory/'checkpoints/latest_checkpoint.pt', weights_only=False)
    resumed = trainer(tmp_path, 'steps_resumed', num_steps=5, epochs=1, max_updates=None, n=5,
                      stochastic=True, state=state)
    resumed.fit()
    assert bars[-1].initial == 3 and bars[-1].n == bars[-1].total == 5
    for a, b in zip(whole.policy.parameters(), resumed.policy.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for run in (whole, first, resumed):
        run.logger.close()


def test_wandb_failure_keeps_local_logging(tmp_path, monkeypatch):
    from awbc_training.logging import ExperimentLogger
    import sys

    def unavailable(**kwargs):
        raise RuntimeError("offline test service unavailable")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=unavailable))
    log = ExperimentLogger(tmp_path, ExperimentConfig())
    log.log({"global_step": 0, "train/loss": 1.})
    log.close()
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["train/loss"] == 1.


def test_cli_overrides_use_existing_configuration_style(tmp_path):
    config = ExperimentConfig()
    path = tmp_path / "config.json"
    config.to_json(path)
    loaded = ExperimentConfig.from_json(path, ["training.batch_size=3", "optimizer.learning_rate=0.0002",
                                              "logging.use_wandb=false", "policy.vision.backbone_mode=layer4"])
    assert loaded.training.batch_size == 3 and loaded.optimizer.learning_rate == 2e-4
    assert not loaded.logging.use_wandb
    assert loaded.policy["vision"]["backbone_mode"] == "layer4"
