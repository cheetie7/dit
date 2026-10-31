"""Conditional action DiT network only; no sampling, objective, or ODE integration.

AdaLN-Zero reference: https://github.com/facebookresearch/DiT/blob/main/models.py
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn


@dataclass
class ConditionalDiTConfig:
    action_dim: int | None = None
    action_horizon: int = 10
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    timestep_embedding_dim: int = 128
    timestep_hidden_dim: int = 256
    max_period: float = 10000.0
    norm_eps: float = 1e-6

    @classmethod
    def from_json(cls, path, action_dim=None):
        values = json.loads(Path(path).read_text())
        if action_dim is not None:
            values["action_dim"] = action_dim
        return cls(**values)


def sinusoidal_embedding(timestep, embedding_dim=128, max_period=10000):
    """Continuous scalar times -> Fourier features; no integer cast or sampling."""
    half = embedding_dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(
        half, device=timestep.device, dtype=torch.float32) / half)
    angles = timestep.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, config: ConditionalDiTConfig):
        super().__init__()
        self.embedding_dim = config.timestep_embedding_dim
        self.max_period = config.max_period
        self.mlp = nn.Sequential(
            nn.Linear(config.timestep_embedding_dim, config.timestep_hidden_dim), nn.SiLU(),
            nn.Linear(config.timestep_hidden_dim, config.hidden_dim))

    def forward(self, timestep):
        embedding = sinusoidal_embedding(timestep, self.embedding_dim, self.max_period)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ConditionalDiTBlock(nn.Module):
    """Timestep-modulated self-attention, observation cross-attention, then FFN."""

    def __init__(self, config: ConditionalDiTConfig):
        super().__init__()
        d = config.hidden_dim
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=config.norm_eps)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=config.norm_eps)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False, eps=config.norm_eps)
        self.self_attn = nn.MultiheadAttention(d, config.num_heads, dropout=config.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, config.num_heads, dropout=config.dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, config.ff_dim), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(config.ff_dim, d), nn.Dropout(config.dropout))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 9*d))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, timestep_condition, observation_memory, observation_pos=None,
                observation_padding_mask=None):
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(timestep_condition).chunk(9, dim=-1))
        h = modulate(self.norm1(x), shift_sa, scale_sa)
        self_out = self.self_attn(h, h, h, need_weights=False)[0]
        x = x + gate_sa.unsqueeze(1) * self_out
        query = modulate(self.norm2(x), shift_ca, scale_ca)
        key = observation_memory if observation_pos is None else observation_memory + observation_pos
        cross_out = self.cross_attn(query=query, key=key, value=observation_memory,
                                    key_padding_mask=observation_padding_mask, need_weights=False)[0]
        x = x + gate_ca.unsqueeze(1) * cross_out
        h = modulate(self.norm3(x), shift_mlp, scale_mlp)
        return x + gate_mlp.unsqueeze(1) * self.mlp(h)


class DiTFinalLayer(nn.Module):
    def __init__(self, config: ConditionalDiTConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_dim, elementwise_affine=False, eps=config.norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_dim, 2*config.hidden_dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, timestep_condition):
        shift, scale = self.adaLN_modulation(timestep_condition).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class ConditionalDiT(nn.Module):
    """Externally supplied [B,H,A] input and time -> predicted velocity [B,H,A]."""

    def __init__(self, config: ConditionalDiTConfig):
        super().__init__()
        if config.action_dim is None:
            raise ValueError("Set action_dim from Dataset metadata or configuration")
        self.config = config
        self.action_proj = nn.Linear(config.action_dim, config.hidden_dim)
        self.action_pos_embed = nn.Parameter(torch.zeros(1, config.action_horizon, config.hidden_dim))
        self.timestep_embedder = TimestepEmbedder(config)
        self.blocks = nn.ModuleList([ConditionalDiTBlock(config) for _ in range(config.num_layers)])
        self.final_layer = DiTFinalLayer(config)
        self.velocity_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for linear in (self.timestep_embedder.mlp[0], self.timestep_embedder.mlp[2]):
            nn.init.normal_(linear.weight, std=0.02)
        for module in (*self.blocks, self.final_layer):
            nn.init.zeros_(module.adaLN_modulation[-1].weight)
            nn.init.zeros_(module.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def forward(self, noisy_action, timestep, observation_memory, observation_pos=None,
                observation_padding_mask=None):
        x = self.action_proj(noisy_action) + self.action_pos_embed
        condition = self.timestep_embedder(timestep)
        for block in self.blocks:
            x = block(x, condition, observation_memory, observation_pos, observation_padding_mask)
        return self.velocity_head(self.final_layer(x, condition))

    def parameter_counts(self):
        counts = {name: sum(p.numel() for p in module.parameters()) for name, module in self.named_children()}
        counts["action_pos_embed"] = self.action_pos_embed.numel()
        counts["total"] = sum(counts.values())
        return counts
