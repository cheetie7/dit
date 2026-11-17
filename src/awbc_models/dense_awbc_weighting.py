"""Fuse static critic scores and RoboMeter progress at a policy optimizer update."""

import math

import torch
from torch import nn


class DenseAWBCWeighting(nn.Module):
    min_weight = None

    def __init__(self, advantages, progress_deltas, config):
        super().__init__()
        for name in ("critic_weight_cap", "robometer_threshold"):
            value = getattr(config, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(config.beta_q) or not math.isfinite(config.robometer_scale_start):
            raise ValueError("beta_q and robometer_scale_start must be finite")
        if config.robometer_decay_enabled and not config.robometer_decay_steps > 0:
            raise ValueError("robometer_decay_steps must be positive when decay is enabled")
        self.config = config
        # Keep these static vectors on CPU in float64, including on MPS runs.
        log_q = config.beta_q * torch.as_tensor(advantages, dtype=torch.float64, device="cpu").detach()
        cap = math.log(config.critic_weight_cap)
        self.log_q = log_q.clamp_max(cap)
        self.critic_clipped = log_q > cap
        delta = torch.as_tensor(progress_deltas, dtype=torch.float64, device="cpu").detach()
        self.above_threshold = delta.abs() > config.robometer_threshold
        self.progress_gate = delta.sign() * (delta.abs() / config.robometer_threshold - 1).clamp_min(0)

    def robometer_scale(self, step):
        scale = self.config.robometer_scale_start
        if self.config.robometer_decay_enabled:
            scale *= max(1 - step / self.config.robometer_decay_steps, 0)
        return scale

    def normalized_weights(self, indices, scale):
        log_weights = self.log_q[indices]
        if scale != 0:
            log_weights = log_weights + scale * self.progress_gate[indices]
        weights = torch.softmax(log_weights, dim=0)
        if not torch.isfinite(weights).all():
            raise FloatingPointError("Nonfinite fused weights; review critic scores, progress threshold and scale")
        return weights
