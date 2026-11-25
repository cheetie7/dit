"""IQL heads with shared frozen or separate trainable visual encoders."""

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
    vision_encoder_mode: str = "shared_frozen"

    def __post_init__(self):
        if self.vision_encoder_mode not in ("shared_frozen", "separate_trainable"):
            raise ValueError(f"Unknown vision_encoder_mode: {self.vision_encoder_mode!r}")


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
        if config.vision_encoder_mode == "shared_frozen":
            # Keep the original state_dict keys for old checkpoints.
            self.backbone = ResNet18Backbone(pretrained=config.pretrained, mode="frozen").eval()
        else:
            self.q_backbone = ResNet18Backbone(pretrained=config.pretrained, mode="all")
            self.v_backbone = ResNet18Backbone(pretrained=config.pretrained, mode="all")
            self.target_backbone = deepcopy(self.q_backbone).requires_grad_(False).eval()
        obs_dim = config.num_cameras * ResNet18Backbone.num_channels * 4 + config.state_dim
        q_dim = obs_dim + config.action_horizon * (config.action_dim + 1)
        self.v = _head(obs_dim, config.hidden_dims)
        self.q = nn.ModuleList(_head(q_dim, config.hidden_dims) for _ in range(2))
        self.target_q = deepcopy(self.q).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.vision_encoder_mode == "shared_frozen":
            self.backbone.eval()
        else:
            self.target_backbone.eval()
        self.target_q.eval()
        return self

    def encode(self, observation, *, branch="q", target=False):
        frozen = self.config.vision_encoder_mode == "shared_frozen"
        if frozen:
            backbone = self.backbone
        elif target:
            backbone = self.target_backbone
        else:
            backbone = self.v_backbone if branch == "v" else self.q_backbone
        images = observation["images"]
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen and not target):
            maps = backbone(images.flatten(0, 1))
        pooled = F.adaptive_avg_pool2d(maps, (2, 2)).reshape(images.shape[0], -1)
        return torch.cat((pooled, observation["state"]), dim=1)

    def action_values(self, features, action, padding, *, target=False):
        masked_action = action.masked_fill(padding.unsqueeze(-1), 0)
        inputs = torch.cat((features, masked_action.flatten(1), padding.to(features.dtype)), dim=1)
        return tuple(head(inputs) for head in (self.target_q if target else self.q))

    def value_parameters(self):
        yield from self.v.parameters()
        if self.config.vision_encoder_mode == "separate_trainable":
            yield from self.v_backbone.parameters()

    def q_parameters(self):
        yield from self.q.parameters()
        if self.config.vision_encoder_mode == "separate_trainable":
            yield from self.q_backbone.parameters()

    @torch.no_grad()
    def update_target(self, rate):
        for target, online in zip(self.target_q.parameters(), self.q.parameters()):
            target.lerp_(online, rate)
        if self.config.vision_encoder_mode == "separate_trainable":
            for target, online in zip(self.target_backbone.parameters(), self.q_backbone.parameters()):
                target.lerp_(online, rate)
            # FrozenBatchNorm buffers are fixed statistics, not learned parameters.
            for target, online in zip(self.target_backbone.buffers(), self.q_backbone.buffers()):
                target.copy_(online)

    @torch.no_grad()
    def advantage(self, batch):
        features = self.encode(batch["observation"], target=True)
        q1, q2 = self.action_values(features, batch["action"], batch["action_is_pad"], target=True)
        return torch.minimum(q1, q2) - self.v(self.encode(batch["observation"], branch="v"))
