"""Restore a complete training checkpoint for offline inference."""
from dataclasses import replace

import torch

from awbc_models import FlowMatchingPolicy, FlowMatchingPolicyConfig
from .runtime import resolve_device


def load_policy(checkpoint, *, use_ema=False, device="auto"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = FlowMatchingPolicyConfig.from_dict(state["policy_config"])
    # Both raw and EMA checkpoints include ImageNet backbone weights and normalizer buffers.
    config = replace(config, vision=replace(config.vision, pretrained=False))
    policy = FlowMatchingPolicy(config, stats=state["normalization_stats"])
    if use_ema:
        if state["ema"] is None:
            raise ValueError("This checkpoint has no EMA weights; use use_ema=False")
        weights = state["ema"]["policy"]
    else:
        weights = state["policy"]
    policy.load_state_dict(weights)
    return policy.to(resolve_device(device)).eval()
