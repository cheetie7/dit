"""Train an IQL critic and export raw advantages for its training windows."""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import ClassVar

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from awbc_data import ImagePreprocessConfig, RLChunkDataset
from awbc_data.advantage import sample_metadata, window_key
from awbc_models.iql_critic import IQLCritic, IQLCriticConfig
from .checkpoint import save_checkpoint
from .config import DatasetConfig, LoggingConfig, apply_overrides
from .data import prepare_datasets
from .logging import ExperimentLogger
from .runtime import move_batch, resolve_device, restore_rng, rng_state, seed_everything


COUNT_METRICS = ("nonzero_reward_windows", "terminated_windows", "truncated_windows")
MEAN_METRICS = ("q_loss", "v_loss", "q1_mean", "q2_mean", "v_mean", "td_target_mean", "q_abs_diff")
DENSE_REWARD_FIELDS = ("reward_sparse", "reward_dense", "reward")


@dataclass
class IQLConfig:
    reward_mode: ClassVar[str] = "sparse"
    labels_path: str
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    image_preprocess: ImagePreprocessConfig = field(default_factory=ImagePreprocessConfig)
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig(use_wandb=False, wandb_project="iql-critic"))
    action_horizon: int = 10
    gamma: float = 0.99
    expectile: float = 0.7
    target_update_rate: float = 0.005
    q_learning_rate: float = 3e-4
    v_learning_rate: float = 3e-4
    hidden_dims: tuple[int, ...] = (256, 256)
    pretrained: bool = True
    seed: int = 42
    num_steps: int = 10_000
    batch_size: int = 16
    num_workers: int = 0
    device: str = "auto"
    validation_interval: int = 1000
    checkpoint_interval: int = 5000
    output_root: str = "outputs"

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, factory in (("dataset", DatasetConfig), ("image_preprocess", ImagePreprocessConfig),
                             ("logging", LoggingConfig)):
            if key in values:
                values[key] = factory(**values[key])
        config = cls(**values)
        config.dataset.root = str(Path(config.dataset.root).resolve())
        config.labels_path = str(Path(config.labels_path).resolve())
        config.hidden_dims = tuple(config.hidden_dims)
        if not 0 < config.expectile < 1 or not 0 < config.target_update_rate <= 1:
            raise ValueError("Expected 0 < expectile < 1 and 0 < target_update_rate <= 1")
        if config.num_steps < 1 or config.batch_size < 1:
            raise ValueError("num_steps and batch_size must be positive")
        return config

    def prepare_rewards(self, train, val):
        return train, val, {}


def expectile_loss(residual, expectile):
    return (torch.where(residual > 0, expectile, 1 - expectile) * residual.square()).mean()


@torch.no_grad()
def td_target(batch, next_value):
    target = batch["reward"].reshape(-1).clone()
    bootstrap = batch["bootstrap_mask"].reshape(-1).bool()
    if bootstrap.any():
        observation = {key: value[bootstrap] for key, value in batch["next_observation"].items()}
        target[bootstrap] += batch["discount"].reshape(-1)[bootstrap] * next_value(observation).reshape(-1)
    return target


def _metrics(batch, value, q1, q2, target, v_loss, q_loss):
    reward_means = ({f"{key}_mean": batch[key].mean().item() for key in DENSE_REWARD_FIELDS}
                    if "reward_dense" in batch else {})
    return dict(
        v_loss=v_loss.item(), q_loss=q_loss.item(), v_mean=value.mean().item(),
        q1_mean=q1.mean().item(), q2_mean=q2.mean().item(),
        td_target_mean=target.mean().item(), q_abs_diff=(q1 - q2).abs().mean().item(),
        nonzero_reward_windows=int(batch["reward"].count_nonzero()),
        terminated_windows=int(batch["terminated"].count_nonzero()),
        truncated_windows=int(batch["truncated"].count_nonzero()),
        **reward_means,
    )


class _MetricTotals:
    def __init__(self):
        self.totals = {}
        self.size = 0

    def add(self, metrics, size):
        for key, metric in metrics.items():
            self.totals[key] = self.totals.get(key, 0) + (metric if key in COUNT_METRICS else metric * size)
        self.size += size

    def result(self, phase):
        return {f"{phase}/{key}": value if key in COUNT_METRICS else value / self.size
                for key, value in self.totals.items()}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _window_metadata(dataset, horizon):
    return [sample_metadata(dataset, index, horizon) for index in range(len(dataset))]


def _prepare(config):
    train, val, stats, split = prepare_datasets(
        config.dataset, config.image_preprocess, action_horizon=config.action_horizon
    )
    source = train.dataset.source
    features = source.info["features"]
    model_config = IQLCriticConfig(
        len(source.cameras), features["observation.state"]["shape"][0], features["action"]["shape"][0],
        config.action_horizon, config.hidden_dims, config.pretrained,
    )
    contract = {
        "action_horizon": config.action_horizon,
        "train_stride": train.dataset.stride, "val_stride": val.dataset.stride,
        "gamma": config.gamma, "camera_names": source.cameras,
        "reward_mode": config.reward_mode,
        "reward_definition": "discounted_sparse_terminal_success_v1",
        "labels_sha256": _sha256(config.labels_path),
        "windows_sha256": {
            name: hashlib.sha256(json.dumps(_window_metadata(dataset, config.action_horizon)).encode()).hexdigest()
            for name, dataset in (("train", train), ("val", val))
        },
        "image_preprocess": json.loads(json.dumps(asdict(config.image_preprocess))),
    }
    train_rl = RLChunkDataset(train, config.labels_path, gamma=config.gamma)
    val_rl = RLChunkDataset(val, config.labels_path, gamma=config.gamma)
    train_rl, val_rl, reward_contract = config.prepare_rewards(train_rl, val_rl)
    return train_rl, val_rl, stats, split, {**contract, **reward_contract}, model_config


def _check_reward_mode(state, mode):
    saved = state["data_contract"].get("reward_mode", "sparse")
    if saved != mode:
        raise ValueError(f"Checkpoint reward_mode={saved!r} differs from this entry point's reward_mode={mode!r}")


def _check_data(state, stats, split, contract, model_config):
    saved_contract = {"reward_mode": "sparse", **state["data_contract"]}
    for key, value in contract.items():
        if saved_contract.get(key) != value:
            raise ValueError(f"Dataset {key} differs from the checkpoint")
    if state["split"] != split or state["normalization_stats"] != stats:
        raise ValueError("Dataset split or normalization statistics differ from the checkpoint")
    if state["model_config"] != asdict(model_config):
        raise ValueError("Critic architecture or initialization configuration differs from the checkpoint")


class IQLTrainer:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model.to(device=device, dtype=torch.float32)
        self.device = device
        self.v_optimizer = torch.optim.Adam(self.model.v.parameters(), lr=config.v_learning_rate)
        self.q_optimizer = torch.optim.Adam(self.model.q.parameters(), lr=config.q_learning_rate)
        self.global_step = self.epoch = self.batch_in_epoch = 0

    def state_dict(self):
        return {
            "critic": self.model.state_dict(), "v_optimizer": self.v_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(), "global_step": self.global_step,
            "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch, "rng_state": rng_state(),
        }

    def load_state_dict(self, state):
        for key in ("seed", "batch_size", "expectile", "target_update_rate", "v_learning_rate", "q_learning_rate"):
            if state["config"][key] != getattr(self.config, key):
                raise ValueError(f"Resume must keep the checkpoint's {key}")
        self.model.load_state_dict(state["critic"])
        self.v_optimizer.load_state_dict(state["v_optimizer"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.batch_in_epoch = state["batch_in_epoch"]
        restore_rng(state["rng_state"])

    def value_loss(self, features, action, mask):
        with torch.no_grad():
            q1, q2 = self.model.action_values(features, action, mask, target=True)
        value = self.model.v(features)
        return value, expectile_loss(torch.minimum(q1, q2) - value, self.config.expectile)

    def q_loss(self, batch, features, action, mask):
        target = td_target(batch, lambda obs: self.model.v(self.model.encode(obs)))
        q1, q2 = self.model.action_values(features, action, mask)
        loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        return q1, q2, target, loss

    def step(self, batch):
        self.model.train()
        features = self.model.encode(batch["observation"])
        inputs = features, batch["action"], batch["action_is_pad"]

        self.v_optimizer.zero_grad(set_to_none=True)
        value, v_loss = self.value_loss(*inputs)
        v_loss.backward()
        self.v_optimizer.step()

        self.q_optimizer.zero_grad(set_to_none=True)
        q1, q2, target, q_loss = self.q_loss(batch, *inputs)
        q_loss.backward()
        self.q_optimizer.step()

        self.model.update_target(self.config.target_update_rate)
        self.global_step += 1
        return _metrics(batch, value, q1, q2, target, v_loss, q_loss)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        totals = _MetricTotals()
        for batch in loader:
            batch = move_batch(batch, self.device)
            inputs = self.model.encode(batch["observation"]), batch["action"], batch["action_is_pad"]
            value, v_loss = self.value_loss(*inputs)
            q1, q2, target, q_loss = self.q_loss(batch, *inputs)
            totals.add(_metrics(batch, value, q1, q2, target, v_loss, q_loss), len(value))
        return totals.result("val")


def _train(config, directory, state):
    seed_everything(config.seed)
    train, val, stats, split, contract, model_config = _prepare(config)
    if not len(train):
        raise ValueError("The training split contains no policy windows")
    device = resolve_device(config.device)
    if state is not None:
        _check_data(state, stats, split, contract, model_config)
    model = IQLCritic(model_config if state is None else replace(model_config, pretrained=False))
    trainer = IQLTrainer(config, model, device)
    if state is not None:
        trainer.load_state_dict(state)
        if config.num_steps < trainer.global_step:
            raise ValueError("num_steps precedes the checkpoint's training step")

    directory.mkdir(parents=True, exist_ok=state is not None)
    metrics = (*MEAN_METRICS, *COUNT_METRICS)
    if config.reward_mode == "dense":
        metrics += tuple(f"{key}_mean" for key in DENSE_REWARD_FIELDS)
    columns = [f"{phase}/{key}" for phase in ("train", "val") for key in metrics]
    logger = ExperimentLogger(directory, config, resume=state is not None, extra_columns=columns)
    checkpoint_dir = directory / "checkpoints"
    metadata = {"config": asdict(config), "split": split, "normalization_stats": stats,
                "data_contract": contract, "model_config": asdict(model_config)}
    val_loader = DataLoader(val, batch_size=config.batch_size, num_workers=config.num_workers,
                            generator=torch.Generator().manual_seed(config.seed))
    totals = _MetricTotals()
    try:
        while trainer.global_step < config.num_steps:
            generator = torch.Generator().manual_seed(config.seed + trainer.epoch)
            order = torch.randperm(len(train), generator=generator).tolist()
            batches = [order[i:i + config.batch_size] for i in range(0, len(order), config.batch_size)]
            start = trainer.batch_in_epoch
            remaining = config.num_steps - trainer.global_step
            loader = DataLoader(train, batch_sampler=batches[start:start + remaining], generator=generator,
                                num_workers=config.num_workers)
            for batch in loader:
                epoch = trainer.epoch
                metrics = trainer.step(move_batch(batch, device))
                totals.add(metrics, batch["action"].shape[0])
                trainer.batch_in_epoch += 1
                if trainer.batch_in_epoch == len(batches):
                    trainer.epoch += 1
                    trainer.batch_in_epoch = 0
                step = trainer.global_step
                final = step == config.num_steps
                save = final or (config.checkpoint_interval and step % config.checkpoint_interval == 0)
                log = config.logging.log_interval and step % config.logging.log_interval == 0
                if step == 1 or save or log or trainer.batch_in_epoch == 0:
                    logger.log({"epoch": epoch, "global_step": step, **totals.result("train")})
                    totals = _MetricTotals()
                if len(val) and (final or (config.validation_interval and step % config.validation_interval == 0)):
                    logger.log({"epoch": epoch, "global_step": step, **trainer.validate(val_loader)})
                if save:
                    save_checkpoint({**metadata, **trainer.state_dict()}, checkpoint_dir)
        if not (checkpoint_dir / "latest_checkpoint.pt").exists():
            save_checkpoint({**metadata, **trainer.state_dict()}, checkpoint_dir)
    finally:
        logger.close()
    return checkpoint_dir / "latest_checkpoint.pt"


@torch.no_grad()
def export_advantages(checkpoint, output, device=None, *, config_type=IQLConfig):
    checkpoint, output = Path(checkpoint), Path(output)
    metadata_path = output.with_suffix(".metadata.json")
    if checkpoint.resolve() in (output.resolve(), metadata_path.resolve()):
        raise ValueError("Advantage export paths must differ from the checkpoint")
    checkpoint_hash = _sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _check_reward_mode(state, config_type.reward_mode)
    config = config_type.from_dict(state["config"])
    train, _, stats, split, contract, model_config = _prepare(config)
    _check_data(state, stats, split, contract, model_config)
    device = resolve_device(device or config.device)
    model = IQLCritic(replace(model_config, pretrained=False)).to(device).eval()
    model.load_state_dict(state["critic"])

    # Advantage only needs the current observation. Keep the wrapped policy subset's order.
    policy_train = train.policy_dataset
    windows = _window_metadata(policy_train, config.action_horizon)
    expected = {window_key(row): row["end_timestep"] for row in windows}
    if len(expected) != len(windows):
        raise ValueError("Training window keys are duplicated")
    loader = DataLoader(policy_train, batch_size=config.batch_size, num_workers=config.num_workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    seen = set()
    try:
        with temporary.open("w") as stream:
            for batch in tqdm(loader, desc="Exporting training advantages", disable=not config.logging.progress_bar):
                values = model.advantage(move_batch(batch, device))
                if not torch.isfinite(values).all():
                    raise FloatingPointError("Critic produced non-finite advantages")
                for episode, start, length, advantage in zip(
                    batch["episode_index"].tolist(), batch["frame_index"].tolist(),
                    batch["valid_length"].tolist(), values.tolist(), strict=True,
                ):
                    key = (episode, start, config.action_horizon)
                    end = start + length
                    if key in seen or expected.get(key) != end:
                        raise ValueError(f"Unexpected or duplicate training window {key} ending at {end}")
                    seen.add(key)
                    row = dict(trajectory_id=episode, start_timestep=start, end_timestep=end,
                               action_horizon=config.action_horizon, valid_length=length, advantage=advantage)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        if len(seen) != len(expected):
            raise ValueError(f"Export covered {len(seen)} of {len(expected)} training windows")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_hash,
                       "global_step": state["global_step"]},
        "split": split, "data_contract": contract, "reward_labels_path": config.labels_path,
        "record_count": len(seen), "advantage_definition": "min(target_q1, target_q2) - v",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None, *, config_type=IQLConfig, default_config="configs/iql_critic.json"):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path)
    train.add_argument("--set", dest="overrides", action="append", default=[])
    train.add_argument("--resume", type=Path)
    train.add_argument("--run-dir", type=Path)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--device")
    args = parser.parse_args(argv)
    if args.command == "export":
        metadata = export_advantages(args.checkpoint, args.output, args.device, config_type=config_type)
        print(f"Exported {metadata['record_count']} windows to {args.output}")
        return
    state = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if state is not None:
        _check_reward_mode(state, config_type.reward_mode)
    config_path = args.config or Path(default_config)
    values = deepcopy(state["config"]) if state is not None and args.config is None else json.loads(config_path.read_text())
    config = config_type.from_dict(apply_overrides(values, args.overrides))
    if args.run_dir:
        directory = args.run_dir
    elif args.resume:
        directory = args.resume.resolve().parent.parent
    else:
        name = "iql-dense-critic" if config.reward_mode == "dense" else "iql-critic"
        directory = Path(config.output_root) / f"{name}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    print(f"Critic checkpoint: {_train(config, directory, state)}")


if __name__ == "__main__":
    main()
