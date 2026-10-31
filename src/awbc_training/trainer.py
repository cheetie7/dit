from dataclasses import asdict
import math
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from awbc_models.euler import synchronize
from .checkpoint import save_checkpoint
from .ema import EMA
from .logging import ExperimentLogger
from .optim import build_optimizer, build_scheduler, optimizer_summary
from .runtime import Precision, move_batch, restore_rng, rng_state
from .advantage import advantage_values, training_forward
from .advantage_analysis import summarize_rows, write_analysis


class Trainer:
    def __init__(self, config, policy, train_dataset, val_dataset, stats, manifest, directory, device,
                 resume_state=None, weighting=None):
        self.config, self.policy = config, policy.to(device)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.stats, self.manifest, self.directory, self.device = stats, manifest, directory, device
        self.optimizer = build_optimizer(self.policy, config.optimizer)
        self.num_batches = math.ceil(len(train_dataset)/config.training.batch_size)
        self.updates_per_epoch = math.ceil(self.num_batches/config.training.gradient_accumulation_steps)
        self.total_steps = (config.training.num_steps if config.training.num_steps is not None
                            else self.updates_per_epoch*config.training.num_epochs)
        if self.total_steps <= 0:
            raise ValueError("The training update budget must be positive")
        self.scheduler = build_scheduler(self.optimizer, config.scheduler, self.total_steps)
        self.precision = Precision(config.amp, device)
        self.ema = EMA(self.policy, config.ema.decay) if config.ema.enabled else None
        self.epoch = self.batch_in_epoch = self.global_step = 0
        self.epoch_loss_sum = self.epoch_samples = self.epoch_seconds = 0
        self.fixed_pair = None
        self.last_update_metrics = {}
        self.last_validation_step = -1
        self.last_validation = {}
        if config.training.objective not in ("fm", "awr_fm"):
            raise ValueError("training.objective must be fm or awr_fm")
        self.weighting = weighting.to(device) if weighting is not None and config.training.objective == "awr_fm" else None
        if config.training.objective == "awr_fm" and self.weighting is None:
            raise ValueError("awr_fm requires aligned labels and an AdvantageWeighting module")
        self.dataset_weights = self.weighting(advantage_values(train_dataset).to(device)).detach().cpu() if self.weighting else None
        self.epoch_awr_rows = []
        self.epoch_weighted_sum = self.epoch_weight_sum = 0.
        self.logger = ExperimentLogger(directory, config, resume=resume_state is not None)
        if resume_state is not None:
            self.load_state(resume_state)
        self.summary = optimizer_summary(self.policy, self.optimizer)
        self.logger.logger.info("Parameters and optimizer: %s", self.summary)
        self.logger.logger.info("Device=%s AMP=%s scaler=%s planned updates=%s", device,
                                config.amp.dtype if self.precision.enabled else "float32",
                                self.precision.scaler.is_enabled(), self.total_steps)
        for name, steps, epochs in (("Validation", config.validation.update_interval, config.validation.interval),
                                    ("Periodic checkpoint", config.checkpoint.update_interval, config.checkpoint.save_interval)):
            frequency = f"every {steps} optimizer updates" if steps else f"every {epochs} epochs" if epochs else "disabled"
            self.logger.logger.info("%s schedule: %s", name, frequency)

    def epoch_batches(self):
        if self.config.training.overfit_one_batch:
            order = list(range(len(self.train_dataset)))
        else:
            order = torch.randperm(len(self.train_dataset), generator=torch.Generator().manual_seed(
                self.config.training.seed+self.epoch)).tolist()
        size = self.config.training.batch_size
        return [order[i:i+size] for i in range(0, len(order), size)]

    def loader(self, dataset, *, batches=None, validation=False):
        kwargs = dict(num_workers=self.config.training.num_workers, pin_memory=self.device.type == "cuda",
                      generator=torch.Generator().manual_seed(self.config.training.seed+self.epoch))
        if batches is not None:
            return DataLoader(dataset, batch_sampler=batches, **kwargs)
        return DataLoader(dataset, batch_size=self.config.validation.batch_size, shuffle=False, **kwargs)

    def training_pair(self, batch, generator):
        action = batch["action"]
        shape = (action.shape[0], self.policy.config.dit.action_horizon, self.policy.config.dit.action_dim)
        return {"noise": torch.randn(shape, generator=generator, dtype=action.dtype).to(self.device),
                "timestep": torch.rand(shape[0], generator=generator).to(self.device)}

    @torch.inference_mode()
    def validate(self):
        policies = {"val/loss": self.policy}
        if self.ema is not None and self.config.validation.use_ema:
            policies["val/ema_loss"] = self.ema.policy
        for policy in policies.values():
            policy.eval()
        sums = {name: 0. for name in policies}
        count = 0
        generator = torch.Generator().manual_seed(self.config.validation.seed)
        loader = self.loader(self.val_dataset, validation=True)
        for batch in tqdm(loader, desc=f"Validation at step {self.global_step}", unit="batch", leave=False,
                          dynamic_ncols=True, disable=not self.config.logging.progress_bar):
            batch = move_batch(batch, self.device)
            pair = self.training_pair(batch, generator)
            for name, policy in policies.items():
                with self.precision.autocast():
                    output = policy(batch, **pair)
                sums[name] += output["per_sample_loss"].float().sum().item()
            count += batch["action"].shape[0]
        result = {name: total/count for name, total in sums.items()}
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("Validation produced NaN or Inf")
        return result

    def state_dict(self):
        return {"epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch, "global_step": self.global_step,
                "policy": self.policy.state_dict(), "ema": self.ema.state_dict() if self.ema else None,
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "grad_scaler": self.precision.scaler.state_dict(), "config": asdict(self.config),
                "normalization_stats": self.stats, "split": self.manifest,
                "policy_config": asdict(self.policy.config), "rng_state": rng_state(),
                "epoch_loss_sum": self.epoch_loss_sum, "epoch_samples": self.epoch_samples,
                "epoch_seconds": self.epoch_seconds, "fixed_pair": self.fixed_pair,
                "total_steps": self.total_steps, "last_update_metrics": self.last_update_metrics,
                "advantage_weighting": self.weighting.state_dict() if self.weighting else None,
                "epoch_awr_rows": self.epoch_awr_rows, "epoch_weighted_sum": self.epoch_weighted_sum,
                "epoch_weight_sum": self.epoch_weight_sum,
                "last_validation_step": self.last_validation_step, "last_validation": self.last_validation}

    def load_state(self, state):
        if state["normalization_stats"] != self.stats or state["split"] != self.manifest:
            raise ValueError("Resume dataset split/statistics differ from the checkpoint")
        if state["total_steps"] != self.total_steps:
            raise ValueError("Resume must retain the original total optimizer-update budget")
        original = state["config"]["training"]
        if original.get("objective", "fm") != self.config.training.objective:
            raise ValueError("Resume cannot change the training objective; start a separate comparison run")
        if state["config"].get("advantage_weighting", {}) != asdict(self.config.advantage_weighting) and self.weighting:
            raise ValueError("Resume must retain the original advantage normalization/weighting configuration")
        if any(original[key] != getattr(self.config.training, key)
               for key in ("seed", "batch_size", "gradient_accumulation_steps")):
            raise ValueError("Resume must retain seed, batch_size and gradient_accumulation_steps to preserve the data cursor")
        self.policy.load_state_dict(state["policy"])
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.precision.scaler.load_state_dict(state["grad_scaler"])
        if self.weighting is not None:
            self.weighting.load_state_dict(state["advantage_weighting"])
        self.epoch_awr_rows = state.get("epoch_awr_rows", [])
        self.epoch_weighted_sum = state.get("epoch_weighted_sum", 0.)
        self.epoch_weight_sum = state.get("epoch_weight_sum", 0.)
        self.last_validation_step = state.get("last_validation_step", -1)
        self.last_validation = state.get("last_validation", {})
        for name in ("epoch", "batch_in_epoch", "global_step", "epoch_loss_sum", "epoch_samples", "epoch_seconds", "fixed_pair", "last_update_metrics"):
            setattr(self, name, state[name])
        restore_rng(state["rng_state"])

    def save(self, periodic=False, *, step_periodic=False):
        config = self.config.checkpoint
        save_checkpoint(self.state_dict(), self.directory / "checkpoints",
                        epoch_number=self.epoch if periodic else None, keep_last_n=config.keep_last_n,
                        step_number=self.global_step if step_periodic else None)

    def _validate_and_log(self):
        self.last_validation = self.validate()
        self.last_validation_step = self.global_step
        self.logger.log({"epoch": self.epoch, "global_step": self.global_step, **self.last_validation})
        self.policy.train()

    def fit(self):
        with logging_redirect_tqdm(loggers=[self.logger.logger]), tqdm(
                total=self.total_steps, initial=self.global_step, desc="Training", unit="update",
                dynamic_ncols=True, disable=not self.config.logging.progress_bar) as progress:
            return self._fit(progress)

    def _fit(self, progress):
        config = self.config.training
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        parameters = [p for p in self.policy.parameters() if p.requires_grad]
        while (self.global_step < self.total_steps if config.num_steps is not None else self.epoch < config.num_epochs):
            if config.max_updates is not None and self.global_step >= config.max_updates:
                break
            self.policy.train()
            batches = self.epoch_batches()
            start_batch = self.batch_in_epoch
            loader = self.loader(self.train_dataset, batches=batches[start_batch:])
            end = perf_counter()
            window_loss = window_samples = window_data_time = window_batch_time = 0
            window_weighted_sum = window_weight_sum = 0.
            window_awr_rows = []
            step_checkpoint_due = False
            for index, batch in enumerate(loader, start=start_batch):
                data_time = perf_counter()-end
                start = perf_counter()
                batch = move_batch(batch, self.device)
                window_start = index//config.gradient_accumulation_steps*config.gradient_accumulation_steps
                window_end = min(window_start+config.gradient_accumulation_steps, len(batches))
                window_count = sum(len(indices) for indices in batches[window_start:window_end])
                pair = {}
                if config.overfit_one_batch:
                    if self.fixed_pair is None:
                        self.fixed_pair = self.training_pair(batch, torch.Generator().manual_seed(config.seed+9999))
                    pair = self.fixed_pair
                with self.precision.autocast():
                    output = training_forward(self.policy, batch, self.weighting, **pair)
                    loss = output["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at update {self.global_step}")
                count = batch["action"].shape[0]
                if self.weighting is None:
                    scaled_loss = loss*count/window_count
                    weight_mass = count
                    weighted_sum = output["per_sample_loss"].detach().sum().item()
                else:
                    window_indices = [item for indices in batches[window_start:window_end] for item in indices]
                    total_mass = self.dataset_weights[window_indices].sum().item()
                    weight_mass = output["sample_weight"].sum().item()
                    scaled_loss = loss*weight_mass/total_mass
                    weighted_sum = (output["sample_weight"]*output["per_sample_loss"].detach()).sum().item()
                    rows = [{"trajectory_id": int(e), "advantage": float(a), "weight": float(w), "loss": float(l), "clipped": bool(c)}
                            for e, a, w, l, c in zip(batch["trajectory_id"].tolist(), batch["advantage"].tolist(),
                                                   output["sample_weight"].tolist(), output["per_sample_loss"].detach().tolist(),
                                                   self.weighting.clip_flags(batch["advantage"]).tolist())]
                    self.epoch_awr_rows.extend(rows)
                    window_awr_rows.extend(rows)
                self.precision.scaler.scale(scaled_loss).backward()
                loss_sum = output["per_sample_loss"].detach().sum().item()
                window_loss += loss_sum
                window_weighted_sum += weighted_sum
                window_weight_sum += weight_mass
                self.epoch_weighted_sum += weighted_sum
                self.epoch_weight_sum += weight_mass
                window_samples += count
                self.epoch_loss_sum += loss_sum
                self.epoch_samples += count
                self.batch_in_epoch = index+1
                if index+1 == window_end:
                    scaler = self.precision.scaler
                    scaler.unscale_(self.optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
                    if not scaler.is_enabled() and not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite gradient norm")
                    previous_scale = scaler.get_scale()
                    learning_rates = {group["name"]: group["lr"] for group in self.optimizer.param_groups}
                    scaler.step(self.optimizer)
                    scaler.update()
                    updated = scaler.get_scale() >= previous_scale
                    if updated:
                        self.global_step += 1
                        self.scheduler.step()
                        if self.ema is not None:
                            self.ema.update(self.policy)
                    self.optimizer.zero_grad(set_to_none=True)
                synchronize(self.device)
                batch_time = perf_counter()-start
                self.epoch_seconds += batch_time+data_time
                window_data_time += data_time
                window_batch_time += batch_time
                if index+1 == window_end:
                    metrics = {"epoch": self.epoch, "global_step": self.global_step,
                               "train/loss": window_weighted_sum/window_weight_sum if self.weighting else window_loss/window_samples,
                               "train/gradient_norm": norm.item() if torch.isfinite(norm) else None,
                               "train/learning_rate": max(learning_rates.values()), "train/group_lrs": learning_rates,
                               "train/skipped_update": int(not updated),
                               "system/data_time": window_data_time, "system/step_time": window_batch_time,
                               "system/gpu_memory": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0}
                    if self.weighting:
                        metrics.update({"train/unweighted_loss": window_loss/window_samples,
                                        "train/weighted_loss": window_weighted_sum/window_weight_sum,
                                        **summarize_rows(window_awr_rows, self.weighting.min_weight)})
                    if self.global_step % self.config.logging.log_interval == 0 or not updated:
                        self.logger.log(metrics)
                    self.last_update_metrics = metrics
                    progress.set_postfix(epoch=self.epoch+1, loss=f"{metrics['train/loss']:.4f}",
                                         lr=f"{metrics['train/learning_rate']:.2e}", refresh=False)
                    if updated:
                        progress.update(1)
                    window_loss = window_samples = window_data_time = window_batch_time = 0
                    window_weighted_sum = window_weight_sum = 0.
                    window_awr_rows = []
                    validation_interval = self.config.validation.update_interval
                    if updated and validation_interval and self.global_step % validation_interval == 0:
                        self._validate_and_log()
                    stop_requested = ((config.max_updates is not None and self.global_step >= config.max_updates)
                                      or (config.num_steps is not None and self.global_step >= self.total_steps))
                    checkpoint_interval = self.config.checkpoint.update_interval
                    step_checkpoint_due = bool(updated and checkpoint_interval and self.global_step % checkpoint_interval == 0)
                    # At epoch boundaries or stopping points, save after advancing the data cursor below.
                    if step_checkpoint_due and not stop_requested and index+1 < len(batches):
                        self.save(step_periodic=True)
                    if stop_requested:
                        break
                end = perf_counter()
            completed = self.batch_in_epoch == len(batches)
            epoch_metrics = {**self.last_update_metrics, "epoch": self.epoch, "global_step": self.global_step,
                             "train/loss": self.epoch_loss_sum/self.epoch_samples,
                             "system/epoch_time": self.epoch_seconds}
            if self.weighting:
                epoch_metrics.update({"train/loss": self.epoch_weighted_sum/self.epoch_weight_sum,
                                      "train/weighted_loss": self.epoch_weighted_sum/self.epoch_weight_sum,
                                      "train/unweighted_loss": self.epoch_loss_sum/self.epoch_samples,
                                      **summarize_rows(self.epoch_awr_rows, self.weighting.min_weight)})
                write_analysis(self.epoch_awr_rows, self.directory / "advantage_analysis" / f"epoch_{self.epoch:04d}_step_{self.global_step}",
                               self.weighting.min_weight)
            stopping = ((config.max_updates is not None and self.global_step >= config.max_updates)
                        or (config.num_steps is not None and self.global_step >= self.total_steps))
            finished = stopping or (completed and config.num_steps is None and self.epoch+1 >= config.num_epochs)
            validation = self.config.validation
            epoch_validation_due = (not validation.update_interval and validation.interval and completed
                                    and (self.epoch+1) % validation.interval == 0)
            final_validation_due = finished and (validation.update_interval or validation.interval)
            if (epoch_validation_due or final_validation_due) and self.last_validation_step != self.global_step:
                self._validate_and_log()
            if self.last_validation_step == self.global_step:
                epoch_metrics.update(self.last_validation)
            self.logger.log(epoch_metrics)
            if completed:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.epoch_loss_sum = self.epoch_samples = self.epoch_seconds = 0
                self.epoch_awr_rows = []
                self.epoch_weighted_sum = self.epoch_weight_sum = 0.
            checkpoint = self.config.checkpoint
            periodic = (not checkpoint.update_interval and completed and checkpoint.save_interval
                        and self.epoch % checkpoint.save_interval == 0)
            if not checkpoint.update_interval or step_checkpoint_due or finished:
                self.save(periodic=periodic, step_periodic=step_checkpoint_due)
        return {"global_step": self.global_step, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                "validation": self.last_validation, "ema_updates": self.ema.num_updates if self.ema else 0}
