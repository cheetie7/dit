"""Tensor normalization using the exact train-only Dataset min/max convention."""
import torch
from torch import nn


class MinMaxNormalizer(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.register_buffer("minimum", torch.tensor(stats["min"], dtype=torch.float32))
        self.register_buffer("maximum", torch.tensor(stats["max"], dtype=torch.float32))

    def normalize(self, value):
        lo, hi = self.minimum, self.maximum
        span = hi-lo
        return torch.where(span > 1e-8, 2*(value-lo)/span.clamp_min(1e-8)-1, torch.zeros_like(value))

    def denormalize(self, value):
        lo, hi = self.minimum, self.maximum
        return (value+1)*(hi-lo)/2+lo
