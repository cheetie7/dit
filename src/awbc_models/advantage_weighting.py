"""Train-fitted scalar normalization and log-space advantage weighting."""
import json
import math
from pathlib import Path

import torch
from torch import nn


class AdvantageNormalizer(nn.Module):
    def __init__(self, mode="zscore", mean=0., std=1., eps=1e-8):
        super().__init__()
        if mode not in ("none", "center", "zscore"):
            raise ValueError("normalization must be none, center or zscore")
        self.mode, self.eps = mode, eps
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(float(std), dtype=torch.float32))

    @classmethod
    def fit(cls, advantage, mode="zscore", eps=1e-8):
        values = torch.as_tensor(advantage, dtype=torch.float64)
        return cls(mode, values.mean().item(), values.std(correction=0).item(), eps)

    @classmethod
    def from_json(cls, path, mode="zscore", eps=1e-8):
        stats = json.loads(Path(path).read_text())
        return cls(mode, stats["mean"], stats["std"], eps)

    def forward(self, advantage):
        advantage = advantage.float()
        if self.mode == "none":
            return advantage
        centered = advantage-self.mean
        return centered if self.mode == "center" else centered/(self.std+self.eps)


class AdvantageWeighting(nn.Module):
    def __init__(self, normalizer, beta=1., min_weight=0.1, max_weight=10., detach_weights=True):
        super().__init__()
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive")
        if (min_weight is not None and min_weight <= 0) or (max_weight is not None and max_weight <= 0):
            raise ValueError("Weight clipping limits must be positive or None")
        if min_weight is not None and max_weight is not None and min_weight > max_weight:
            raise ValueError("min_weight cannot exceed max_weight")
        self.normalizer, self.beta = normalizer, beta
        self.min_weight, self.max_weight, self.detach_weights = min_weight, max_weight, detach_weights

    def log_weights(self, advantage):
        if advantage.ndim != 1:
            raise ValueError("advantage must be [B], one scalar per window")
        return self.normalizer(advantage.detach() if self.detach_weights else advantage)/self.beta

    def clip_flags(self, advantage):
        log_weights = self.log_weights(advantage)
        flags = torch.zeros_like(log_weights, dtype=torch.bool)
        if self.min_weight is not None:
            flags |= log_weights < math.log(self.min_weight)
        if self.max_weight is not None:
            flags |= log_weights > math.log(self.max_weight)
        return flags

    def forward(self, advantage):
        log_weights = self.log_weights(advantage)
        if self.min_weight is not None:
            log_weights = log_weights.clamp_min(math.log(self.min_weight))
        if self.max_weight is not None:
            log_weights = log_weights.clamp_max(math.log(self.max_weight))
        weights = log_weights.exp()
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise FloatingPointError("Advantage produced nonfinite or zero weights; review labels, beta or clipping")
        return weights
