from dataclasses import asdict, dataclass, field, replace
import json
from importlib.resources import files
from pathlib import Path

from awbc_models import FlowMatchingPolicyConfig


@dataclass
class DatasetConfig:
    root: str = "data/block_insertion_real_combined_v30_advantage"
    val_fraction: float = 0.1
    split_seed: int = 42
    stratify_key: str | None = "tasks"
    max_train_samples: int | None = None
    max_val_samples: int | None = None


@dataclass
class TrainingConfig:
    objective: str = "fm"
    seed: int = 42
    num_epochs: int = 200
    num_steps: int | None = None  # Optimizer-update budget; overrides num_epochs when set.
    batch_size: int = 16
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    device: str = "auto"
    max_updates: int | None = None  # Diagnostic stop only; does not change the LR schedule.
    overfit_one_batch: bool = False
    overfit_batches: int = 24


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-4
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    no_decay_bias_and_norm: bool = True


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01


@dataclass
class AMPConfig:
    enabled: bool = True
    dtype: str = "bfloat16"


@dataclass
class EMAConfig:
    enabled: bool = True
    decay: float = 0.999


@dataclass
class ValidationConfig:
    interval: int = 1
    update_interval: int = 0  # Positive values override the epoch interval.
    batch_size: int = 16
    seed: int = 1234
    use_ema: bool = True


@dataclass
class CheckpointConfig:
    save_interval: int = 10
    keep_last_n: int = 3
    update_interval: int = 0  # Positive values override save_interval (epochs).


@dataclass
class LoggingConfig:
    use_wandb: bool = True
    wandb_mode: str = "offline"
    wandb_project: str = "dit-flow-baseline"
    log_interval: int = 10
    progress_bar: bool = True


@dataclass
class AdvantageWeightingConfig:
    enabled: bool = False
    normalization: str = "zscore"
    beta: float = 1.0
    min_weight: float | None = 0.1
    max_weight: float | None = 10.0
    detach_weights: bool = True
    labels_path: str | None = None
    metadata_path: str | None = None
    statistics_path: str | None = None
    missing_label_policy: str = "error"


@dataclass
class ExperimentConfig:
    experiment_name: str = "flow_matching_baseline"
    output_root: str = "outputs"
    policy_config: str | None = None  # None uses the model configuration shipped with the package.
    policy: dict = field(default_factory=dict)  # Overrides existing Policy dataclasses.
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    advantage_weighting: AdvantageWeightingConfig = field(default_factory=AdvantageWeightingConfig)

    @classmethod
    def from_dict(cls, value):
        sections = {"dataset": DatasetConfig, "training": TrainingConfig, "optimizer": OptimizerConfig,
                    "scheduler": SchedulerConfig, "amp": AMPConfig, "ema": EMAConfig,
                    "validation": ValidationConfig, "checkpoint": CheckpointConfig, "logging": LoggingConfig,
                    "advantage_weighting": AdvantageWeightingConfig}
        return cls(**{key: sections[key](**item) if key in sections else item for key, item in value.items()})

    @classmethod
    def from_json(cls, path, overrides=()):
        value = json.loads(Path(path).read_text())
        return cls.from_dict(apply_overrides(value, overrides))

    def policy_configuration(self, info):
        path = self.policy_config or files("awbc_models").joinpath("configs/flow_matching_policy.json")
        base = FlowMatchingPolicyConfig.from_json(path, dataset_info=info)
        updates = {}
        for key, value in self.policy.items():
            original = getattr(base, key)
            updates[key] = replace(original, **value) if hasattr(original, "__dataclass_fields__") else value
        return replace(base, **updates)

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


def apply_overrides(value, overrides):
    for override in overrides:
        key, text = override.split("=", 1)
        try:
            replacement = json.loads(text)
        except json.JSONDecodeError:
            replacement = text
        target = value
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = replacement
    return value
