"""ACT/DETR-style visual-state fusion, returning memory for a future policy.

Reference: https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ObservationEncoderConfig:
    state_dim: int | None = None  # Set from Dataset metadata; not a robot-specific default.
    use_state: bool = True
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1


class ObservationEncoderLayer(nn.Module):
    """Post-norm DETR encoder layer; position is used only in attention Q/K."""

    def __init__(self, config: ObservationEncoderConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ffn_dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, config.hidden_dim))
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ffn_dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, tokens, pos, padding_mask=None):
        query = key = tokens + pos
        attended = self.self_attn(query, key, value=tokens,
                                  key_padding_mask=padding_mask, need_weights=False)[0]
        tokens = self.norm1(tokens + self.attention_dropout(attended))
        return self.norm2(tokens + self.ffn_dropout(self.ffn(tokens)))


class ObservationEncoder(nn.Module):
    """[state, visual tokens] -> fused memory, positions, and True-means-ignore mask."""

    def __init__(self, config: ObservationEncoderConfig):
        super().__init__()
        self.config = config
        if config.use_state:
            if config.state_dim is None:
                raise ValueError("Set state_dim from Dataset metadata when use_state=True")
            self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
            self.state_pos_embed = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        else:
            self.state_projection = None
            self.register_parameter("state_pos_embed", None)
        self.layers = nn.ModuleList([ObservationEncoderLayer(config) for _ in range(config.num_layers)])
        # ACT initializes Transformer matrix parameters with Xavier uniform.
        for parameter in self.layers.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def prepare_tokens(self, visual_tokens, visual_pos, state=None, visual_padding_mask=None):
        """Expose the pre-attention sequence for diagnostics; no input is modified."""
        tokens, pos, mask = visual_tokens, visual_pos, visual_padding_mask
        if self.config.use_state:
            if state is None:
                raise ValueError("state is required when use_state=True")
            b = visual_tokens.shape[0]
            state_token = self.state_projection(state).unsqueeze(1)
            tokens = torch.cat((state_token, visual_tokens), dim=1)
            state_pos = self.state_pos_embed.to(visual_pos.dtype).expand(b, -1, -1)
            pos = torch.cat((state_pos, visual_pos), dim=1)
            if mask is not None:
                state_mask = torch.zeros(b, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat((state_mask, mask), dim=1)
        return {"observation_tokens": tokens, "observation_pos": pos,
                "observation_padding_mask": mask}

    def forward(self, visual_tokens, visual_pos, state=None, visual_padding_mask=None):
        prepared = self.prepare_tokens(visual_tokens, visual_pos, state, visual_padding_mask)
        memory = prepared["observation_tokens"]
        pos = prepared["observation_pos"]
        mask = prepared["observation_padding_mask"]
        for layer in self.layers:
            memory = layer(memory, pos, mask)
        return {"observation_memory": memory, "observation_pos": pos,
                "observation_padding_mask": mask}
