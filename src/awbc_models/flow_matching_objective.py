"""Linear conditional flow-matching objective with per-sample action masking."""
import torch
from torch import nn


def linear_flow_path(action, noise=None, timestep=None):
    """Noise at t=0, data at t=1; return the path and its constant velocity."""
    if noise is None:
        noise = torch.randn_like(action)
    else:
        noise = noise.to(device=action.device, dtype=action.dtype)
    if timestep is None:
        timestep = torch.rand(action.shape[0], device=action.device)
    else:
        timestep = timestep.to(device=action.device)
    t = timestep.to(action.dtype).view(action.shape[0], 1, 1)
    return {"noise": noise, "timestep": timestep,
            "noisy_action": (1-t)*noise + t*action,
            "target_velocity": action-noise}


class FlowMatchingObjective(nn.Module):
    """Stateless sampling and masked MSE; weights are normalized by their sum."""

    def prepare_training_pair(self, action, noise=None, timestep=None):
        return linear_flow_path(action, noise, timestep)

    def compute_loss(self, predicted_velocity, target_velocity, action_padding_mask=None, sample_weight=None):
        if predicted_velocity.shape != target_velocity.shape:
            raise ValueError(f"Velocity shape {predicted_velocity.shape} must match target shape {target_velocity.shape}")
        # Cast before subtraction/squaring, so half-precision residuals do not overflow.
        error = predicted_velocity.float() - target_velocity.float()
        if action_padding_mask is None:
            valid_steps = torch.ones(target_velocity.shape[:2], dtype=torch.bool, device=target_velocity.device)
        else:
            if action_padding_mask.shape != target_velocity.shape[:2]:
                raise ValueError("action_padding_mask must have shape [B,H]")
            valid_steps = ~action_padding_mask
            # Mask before squaring, including for padded predictions with large values.
            error = error.masked_fill(action_padding_mask.unsqueeze(-1), 0)
        elementwise_loss = error.square()
        valid_action_count = valid_steps.sum(dim=1)
        if (valid_action_count == 0).any():
            raise ValueError("Each sample must contain at least one valid action step; found an all-padding sample")
        denominator = valid_action_count * target_velocity.shape[-1]
        per_sample_loss = elementwise_loss.sum(dim=(1, 2)) / denominator
        if sample_weight is None:
            sample_weight = torch.ones_like(per_sample_loss)
            loss = per_sample_loss.mean()
        else:
            if sample_weight.shape != per_sample_loss.shape:
                raise ValueError("sample_weight must have shape [B]")
            sample_weight = sample_weight.to(device=target_velocity.device, dtype=torch.float32)
            if not torch.isfinite(sample_weight).all() or (sample_weight < 0).any():
                raise ValueError("sample_weight must contain finite nonnegative values")
            weight_sum = sample_weight.sum()
            if weight_sum <= 0:
                raise ValueError("sample_weight must have a positive sum; all-zero weights are invalid")
            if (sample_weight == sample_weight[0]).all():
                loss = per_sample_loss.mean()  # Exact equality for constant positive weights.
            else:
                loss = (sample_weight * per_sample_loss).sum() / weight_sum
        weighted_per_sample_loss = sample_weight * per_sample_loss
        return {"elementwise_loss": elementwise_loss, "per_sample_loss": per_sample_loss, "sample_weight": sample_weight,
                "weighted_per_sample_loss": weighted_per_sample_loss,
                "valid_action_count": valid_action_count, "loss": loss}

    def forward(self, dit, action, observation_memory, observation_pos=None,
                observation_padding_mask=None, action_padding_mask=None,
                sample_weight=None, noise=None, timestep=None):
        path = self.prepare_training_pair(action, noise, timestep)
        prediction = dit(noisy_action=path["noisy_action"], timestep=path["timestep"],
                         observation_memory=observation_memory, observation_pos=observation_pos,
                         observation_padding_mask=observation_padding_mask)
        losses = self.compute_loss(prediction, path["target_velocity"], action_padding_mask, sample_weight)
        return {**path, "predicted_velocity": prediction, **losses}
