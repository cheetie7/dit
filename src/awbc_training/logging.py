import csv
from dataclasses import asdict
import json
import logging


class ExperimentLogger:
    def __init__(self, directory, config, resume=False):
        self.logger = logging.getLogger(str(directory))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handlers = [logging.StreamHandler(), logging.FileHandler(directory / "train.log")]
        for handler in self.handlers:
            self.logger.addHandler(handler)
        self.jsonl = (directory / "metrics.jsonl").open("a")
        csv_path = directory / "metrics.csv"
        has_header = csv_path.exists() and csv_path.stat().st_size > 0
        self.csv_file = csv_path.open("a", newline="")
        columns = ["epoch", "global_step", "train/loss", "train/gradient_norm", "train/learning_rate",
                   "val/loss", "val/ema_loss", "system/gpu_memory", "system/epoch_time",
                   "system/step_time", "system/data_time", "train/skipped_update"]
        columns += ["train/unweighted_loss", "train/weighted_loss", "advantage/mean", "advantage/std", "advantage/min",
                    "advantage/max", "advantage/median", "weight/mean", "weight/std", "weight/min", "weight/max",
                    "weight/clip_fraction", "weight/effective_sample_size", "weight/ess_ratio",
                    "weight/largest_trajectory_mass_fraction", "weight/minimum_fraction"]
        self.csv = csv.DictWriter(self.csv_file, fieldnames=columns, extrasaction="ignore")
        if not has_header:
            self.csv.writeheader()
        self.wandb = None
        if config.logging.use_wandb:
            try:
                import wandb
                self.wandb = wandb.init(project=config.logging.wandb_project, name=directory.name,
                                        id=directory.name, resume="allow" if resume else None,
                                        mode=config.logging.wandb_mode, dir=str(directory),
                                        config=asdict(config))
            except Exception as error:
                self.logger.warning("WandB unavailable; using local logs: %s", error)

    def log(self, metrics):
        self.jsonl.write(json.dumps(metrics, allow_nan=False) + "\n")
        self.jsonl.flush()
        self.csv.writerow(metrics)
        self.csv_file.flush()
        self.logger.info(json.dumps(metrics, allow_nan=False))
        if self.wandb is not None:
            try:
                self.wandb.log(metrics, step=metrics["global_step"])
            except Exception as error:
                self.logger.warning("WandB logging stopped; local logging continues: %s", error)
                self.wandb = None

    def close(self):
        if self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception as error:
                self.logger.warning("WandB finish failed: %s", error)
        self.jsonl.close()
        self.csv_file.close()
        for handler in self.handlers:
            handler.close()
            self.logger.removeHandler(handler)
