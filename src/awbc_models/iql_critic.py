"""Frozen visual features and independent IQL value and action-value heads."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .vision import ResNet18Backbone


@dataclass
class IQLCriticConfig:
    num_cameras: int
    state_dim: int
    action_dim: int
    action_horizon: int
    hidden_dims: tuple[int, ...] = (256, 256)
    pretrained: bool = True


def _head(input_dim, hidden_dims):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, 1), nn.Flatten(0, 1))


class IQLCritic(nn.Module):
    def __init__(self, config: IQLCriticConfig):
        super().__init__()
        self.config = config
        self.backbone = ResNet18Backbone(pretrained=config.pretrained, mode="frozen").eval()
        obs_dim = config.num_cameras * self.backbone.num_channels * 4 + config.state_dim
        q_dim = obs_dim + config.action_horizon * (config.action_dim + 1)
        self.v = _head(obs_dim, config.hidden_dims)
        self.q = nn.ModuleList(_head(q_dim, config.hidden_dims) for _ in range(2))
        self.target_q = deepcopy(self.q).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.target_q.eval()
        return self

    @torch.no_grad()
    def encode(self, observation):
        images = observation["images"]
        maps = self.backbone(images.flatten(0, 1))
        pooled = F.adaptive_avg_pool2d(maps, (2, 2)).reshape(images.shape[0], -1)
        return torch.cat((pooled, observation["state"]), dim=1)

    def action_values(self, features, action, padding, *, target=False):
        masked_action = action.masked_fill(padding.unsqueeze(-1), 0)
        inputs = torch.cat((features, masked_action.flatten(1), padding.to(features.dtype)), dim=1)
        return tuple(head(inputs) for head in (self.target_q if target else self.q))

    @torch.no_grad()
    def update_target(self, rate):
        for target, online in zip(self.target_q.parameters(), self.q.parameters()):
            target.lerp_(online, rate)

    @torch.no_grad()
    def advantage(self, batch):
        features = self.encode(batch["observation"])
        q1, q2 = self.action_values(features, batch["action"], batch["action_is_pad"], target=True)
        return torch.minimum(q1, q2) - self.v(features)
