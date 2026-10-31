"""ACT-style spatial visual features, without a policy or observation Transformer.

References: https://github.com/tonyzhaozh/act/tree/main/detr/models
Position encodings remain separate from content, as in ACT's attention interface.
"""
from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


@dataclass
class VisualEncoderConfig:
    d_model: int = 256
    pretrained: bool = True
    backbone_mode: Literal["frozen", "layer4", "all"] = "frozen"
    num_cameras: int = 3


class ResNet18Backbone(nn.Module):
    """ImageNet ResNet-18 through layer4: [B,3,H,W] -> [B,512,Hf,Wf]."""

    num_channels = 512

    def __init__(self, pretrained=True, mode="frozen"):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        # IntermediateLayerGetter drops avgpool and fc entirely.
        self.body = IntermediateLayerGetter(resnet, return_layers={"layer4": "feature_map"})
        self.set_trainable(mode)

    def set_trainable(self, mode):
        prefixes = {"frozen": (), "layer4": ("layer4.",), "all": ("",)}[mode]
        for name, parameter in self.body.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        self.mode = mode

    def forward(self, image):
        return self.body(image)["feature_map"]


class SinePositionEncoding2D(nn.Module):
    """ACT/DETR sine-cosine coordinates, normalized per axis to 2*pi.

    Channels are [y sin/cos, x sin/cos]; coordinates start at 1, as in ACT.
    """

    def __init__(self, d_model=256, temperature=10000):
        super().__init__()
        if d_model <= 0 or d_model % 4:
            raise ValueError("d_model must be a positive multiple of 4 for 2D sin/cos encoding")
        self.num_pos_feats = d_model // 2
        self.temperature = temperature

    def forward(self, feature_map):
        b, _, h, w = feature_map.shape
        y = torch.arange(1, h + 1, device=feature_map.device, dtype=torch.float32)
        x = torch.arange(1, w + 1, device=feature_map.device, dtype=torch.float32)
        y = y / (h + 1e-6) * (2 * math.pi)
        x = x / (w + 1e-6) * (2 * math.pi)
        dim = torch.arange(self.num_pos_feats, device=feature_map.device, dtype=torch.float32)
        frequencies = self.temperature ** (2 * torch.div(dim, 2, rounding_mode="floor") / self.num_pos_feats)

        def encode(coordinates):
            angles = coordinates[:, None] / frequencies
            return torch.stack((angles[:, 0::2].sin(), angles[:, 1::2].cos()), dim=-1).flatten(1)

        pos_y = encode(y)[:, None, :].expand(h, w, -1)
        pos_x = encode(x)[None, :, :].expand(h, w, -1)
        position = torch.cat((pos_y, pos_x), dim=-1).permute(2, 0, 1)
        return position.unsqueeze(0).expand(b, -1, -1, -1).to(feature_map.dtype)


class ACTVisualEncoder(nn.Module):
    """ResNet-18 -> 1x1 projection -> spatial content and position tokens.

    Inputs must already be RGB float images with ImageNet normalization.
    Content and position are separate. Camera identity is added to visual_pos.
    """

    def __init__(self, config: VisualEncoderConfig | None = None):
        super().__init__()
        self.config = config or VisualEncoderConfig()
        self.position_encoding = SinePositionEncoding2D(self.config.d_model)
        self.backbone = ResNet18Backbone(self.config.pretrained, self.config.backbone_mode)
        self.input_proj = nn.Conv2d(512, self.config.d_model, kernel_size=1)
        self.camera_embedding = nn.Embedding(self.config.num_cameras, self.config.d_model)
        nn.init.normal_(self.camera_embedding.weight, std=0.02)

    def forward_features(self, image):
        """Diagnostic single-camera maps; image is [B,3,H,W]."""
        feature_map = self.backbone(image)
        projected_map = self.input_proj(feature_map)
        position_map = self.position_encoding(projected_map)
        return {
            "feature_map": feature_map,
            "projected_map": projected_map,
            "position_map": position_map,
        }

    def forward(self, image, camera_index=0):
        """Accept a single view [B,3,H,W] or camera stack [B,N,3,H,W]."""
        if image.ndim == 5:
            return self.encode_cameras(image)
        maps = self.forward_features(image)
        tokens = maps["projected_map"].flatten(2).transpose(1, 2)
        pos = maps["position_map"].flatten(2).transpose(1, 2)
        return {"visual_tokens": tokens,
                "visual_pos": pos + self.camera_embedding.weight[camera_index]}

    def encode_cameras(self, images):
        """Shared independent extraction; concatenate tokens in (camera,row,col) order."""
        b, n = images.shape[:2]
        # Cameras are separate batch items, never concatenated along RGB or space.
        maps = self.forward_features(images.flatten(0, 1))
        tokens = maps["projected_map"].flatten(2).transpose(1, 2).unflatten(0, (b, n))
        pos = maps["position_map"].flatten(2).transpose(1, 2).unflatten(0, (b, n))
        ids = torch.arange(n, device=images.device)
        pos = pos + self.camera_embedding(ids)[None, :, None, :]
        return {"visual_tokens": tokens.flatten(1, 2), "visual_pos": pos.flatten(1, 2)}

    def parameter_groups(self, lr=1e-4, backbone_lr=1e-5):
        """Call after selecting the backbone mode, then construct the optimizer."""
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": [*self.input_proj.parameters(), *self.camera_embedding.parameters()], "lr": lr}]
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr})
        return groups
