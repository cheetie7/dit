import pytest
import torch
from torch import nn

from awbc_models import (ACTVisualEncoder, VisualEncoderConfig, ObservationEncoder, ObservationEncoderConfig,
                         ConditionalDiT, ConditionalDiTConfig, FlowMatchingObjective, linear_flow_path)


class ZeroVelocity(nn.Module):
    def forward(self, noisy_action, **kwargs):
        return torch.zeros_like(noisy_action)


def test_requested_training_pair_shape():
    torch.manual_seed(18)
    action = torch.zeros(4, 16, 7)
    pair = FlowMatchingObjective().prepare_training_pair(action)
    assert pair["noise"].shape == pair["noisy_action"].shape == pair["target_velocity"].shape == (4, 16, 7)
    assert pair["timestep"].shape == (4,)
    assert abs(pair["noise"].mean()) < 0.15
    assert 0.85 < pair["noise"].std() < 1.15


def test_linear_path_endpoints_and_target_direction():
    action = torch.tensor([[[1., 2.]], [[3., 4.]], [[5., 6.]]])
    noise = torch.tensor([[[-1., -2.]], [[-3., -4.]], [[-5., -6.]]])
    time = torch.tensor([0., 1., 0.25])
    path = linear_flow_path(action, noise, time)
    torch.testing.assert_close(path["noisy_action"][0], noise[0])
    torch.testing.assert_close(path["noisy_action"][1], action[1])
    torch.testing.assert_close(path["noisy_action"][2], 0.75*noise[2] + 0.25*action[2])
    torch.testing.assert_close(path["target_velocity"], action-noise)
    other = linear_flow_path(action, noise, 1-time)
    torch.testing.assert_close(other["target_velocity"], path["target_velocity"])
    midpoint = FlowMatchingObjective().prepare_training_pair(action, noise, torch.full((3,), 0.5))
    torch.testing.assert_close(midpoint["noisy_action"], (action+noise)/2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sampling_dtype_distribution_and_reproducibility(dtype):
    action = torch.zeros(256, 16, 8, dtype=dtype)
    torch.manual_seed(33)
    path = linear_flow_path(action)
    torch.manual_seed(33)
    repeated = linear_flow_path(action)
    assert path["noise"].shape == action.shape
    for key in ("noise", "noisy_action", "target_velocity"):
        assert path[key].dtype == dtype and path[key].device == action.device
        assert torch.isfinite(path[key]).all()
        torch.testing.assert_close(path[key], repeated[key], rtol=0, atol=0)
    assert path["timestep"].shape == (256,) and path["timestep"].dtype == torch.float32
    assert ((path["timestep"] >= 0) & (path["timestep"] < 1)).all()
    assert path["timestep"].unique().numel() > 250
    assert abs(path["noise"].float().mean()) < 0.03
    assert abs(path["noise"].float().std()-1) < 0.03


def test_unweighted_and_weighted_per_sample_loss():
    action = torch.tensor([[[1., 2.], [3., 4.]], [[2., 0.], [1., 1.]]])
    objective = FlowMatchingObjective()
    kwargs = dict(dit=ZeroVelocity(), action=action, observation_memory=torch.zeros(2, 3, 4),
                  noise=torch.zeros_like(action), timestep=torch.tensor([0., 1.]))
    plain = objective(**kwargs)
    torch.testing.assert_close(plain["per_sample_loss"], torch.tensor([7.5, 1.5]))
    assert plain["loss"].item() == 4.5
    weighted = objective(**kwargs, sample_weight=torch.tensor([0., 3.]))
    torch.testing.assert_close(weighted["per_sample_loss"], plain["per_sample_loss"])
    torch.testing.assert_close(weighted["weighted_per_sample_loss"], torch.tensor([0., 4.5]))
    assert weighted["loss"].item() == 1.5  # Normalize by the weight sum.
    assert objective(**kwargs, sample_weight=torch.ones(2))["loss"] == plain["loss"]
    assert objective(**kwargs, sample_weight=torch.full((2,), 3.5))["loss"] == plain["loss"]
    mixed = objective(**kwargs, sample_weight=torch.tensor([1., 3.]))
    assert mixed["loss"].item() == 3.
    scaled = objective(**kwargs, sample_weight=torch.tensor([10., 30.]))
    assert scaled["loss"] == mixed["loss"]


@pytest.mark.parametrize("weights", [torch.tensor([-1., 2.]), torch.tensor([float("nan"), 1.]),
                                     torch.tensor([1., float("inf")]), torch.zeros(2)])
def test_invalid_weights_raise_clear_errors(weights):
    with pytest.raises(ValueError, match="sample_weight"):
        FlowMatchingObjective().compute_loss(torch.ones(2, 4, 3), torch.zeros(2, 4, 3), sample_weight=weights)


def test_standalone_loss_zero_prediction_error_and_error_growth():
    objective = FlowMatchingObjective()
    target = torch.linspace(-1, 1, 24).reshape(2, 4, 3)
    exact = objective.compute_loss(target, target)
    error1 = objective.compute_loss(target+1, target)
    error2 = objective.compute_loss(target+2, target)
    assert exact["loss"] == 0
    torch.testing.assert_close(exact["per_sample_loss"], torch.zeros(2))
    torch.testing.assert_close(error1["loss"], torch.tensor(1.))
    torch.testing.assert_close(error2["loss"], torch.tensor(4.))


def test_padding_uses_valid_element_denominator():
    action = torch.tensor([[[1., 2.], [3., 4.]], [[2., 0.], [1., 1.]]])
    mask = torch.tensor([[False, True], [False, False]])
    output = FlowMatchingObjective()(
        ZeroVelocity(), action, torch.zeros(2, 3, 4), action_padding_mask=mask,
        noise=torch.zeros_like(action), timestep=torch.tensor([0.2, 0.8]))
    torch.testing.assert_close(output["per_sample_loss"], torch.tensor([2.5, 1.5]))
    torch.testing.assert_close(output["valid_action_count"], torch.tensor([1, 2]))
    assert output["loss"] == 2.
    assert torch.count_nonzero(output["elementwise_loss"][mask]) == 0


def test_masked_values_and_zero_weight_have_zero_gradient():
    class FixedPrediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.prediction = nn.Parameter(torch.tensor([[[2., 3.], [1e20, 1e20]], [[4., 5.], [6., 7.]]]))

        def forward(self, noisy_action, **kwargs):
            return self.prediction

    model = FixedPrediction()
    action = torch.ones(2, 2, 2)
    output = FlowMatchingObjective()(
        model, action, torch.zeros(2, 3, 4), noise=torch.zeros_like(action),
        timestep=torch.zeros(2), action_padding_mask=torch.tensor([[False, True], [False, False]]),
        sample_weight=torch.tensor([1., 0.]))
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.prediction.grad[0, 0].abs().sum() > 0
    assert torch.count_nonzero(model.prediction.grad[0, 1]) == 0
    assert torch.count_nonzero(model.prediction.grad[1]) == 0
    assert torch.isfinite(model.prediction.grad).all()


@pytest.mark.parametrize("mask", [torch.ones(2, 4, dtype=torch.bool),
                                  torch.tensor([[False, False, False, False], [True, True, True, True]])])
def test_all_padding_sample_is_rejected(mask):
    objective = FlowMatchingObjective()
    with pytest.raises(ValueError, match="all-padding"):
        objective.compute_loss(torch.zeros(2, 4, 3), torch.ones(2, 4, 3), action_padding_mask=mask)


def test_exact_dit_keyword_forwarding_and_observation_gradients():
    class ObservationVelocity(nn.Module):
        def forward(self, **kwargs):
            self.received = kwargs
            return (kwargs["observation_memory"] + kwargs["observation_pos"]).mean(1)[:, None].expand_as(kwargs["noisy_action"])

    model = ObservationVelocity()
    action = torch.ones(2, 4, 3)
    memory = torch.ones(2, 5, 3, requires_grad=True)
    pos = torch.full((2, 5, 3), 0.5, requires_grad=True)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    output = FlowMatchingObjective()(model, action, memory, pos, mask,
                                                noise=torch.zeros_like(action), timestep=torch.tensor([0.25, 0.75]))
    assert model.received["observation_memory"] is memory
    assert model.received["observation_pos"] is pos
    assert model.received["observation_padding_mask"] is mask
    torch.testing.assert_close(model.received["noisy_action"], output["noisy_action"])
    output["loss"].backward()
    assert memory.grad.abs().sum() > 0 and pos.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_dit_low_precision_and_float32_loss(dtype):
    config = ConditionalDiTConfig(action_dim=3, action_horizon=4, hidden_dim=32, num_layers=1,
                                  num_heads=4, ff_dim=64, dropout=0, timestep_embedding_dim=16,
                                  timestep_hidden_dim=32)
    model = ConditionalDiT(config).to(dtype=dtype)
    action = torch.full((2, 4, 3), 300., dtype=dtype)
    memory = torch.zeros(2, 5, 32, dtype=dtype)
    output = FlowMatchingObjective()(model, action, memory, noise=torch.zeros_like(action),
                                                timestep=torch.tensor([0., 1.]))
    assert output["predicted_velocity"].dtype == dtype
    assert output["loss"].dtype == torch.float32 and output["loss"] == 90000
    output["loss"].backward()
    assert model.velocity_head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_real_dit_preserves_upstream_gradient_when_gates_are_open():
    config = ConditionalDiTConfig(action_dim=3, action_horizon=4, hidden_dim=32, num_layers=1,
                                  num_heads=4, ff_dim=64, dropout=0)
    model = ConditionalDiT(config)
    # Test-only nonzero initialization, so AdaLN-Zero cannot hide a detached input.
    with torch.no_grad():
        model.velocity_head.weight.copy_(torch.linspace(-0.1, 0.1, 96).reshape(3, 32))
        model.blocks[0].adaLN_modulation[-1].bias[5*32:6*32] = 0.2
    action = torch.linspace(-1, 1, 24).reshape(2, 4, 3)
    memory = torch.linspace(-1, 1, 320).reshape(2, 5, 32).requires_grad_()
    pos = torch.sin(memory.detach()).requires_grad_()
    output = FlowMatchingObjective()(model, action, memory, pos,
                                                noise=torch.zeros_like(action), timestep=torch.tensor([0.1, 0.9]))
    output["loss"].backward()
    assert memory.grad.abs().sum() > 0 and pos.grad.abs().sum() > 0
    assert torch.isfinite(memory.grad).all() and torch.isfinite(pos.grad).all()


def test_full_pipeline_gradients_after_bounded_diagnostic_updates():
    torch.manual_seed(25)
    vision = ACTVisualEncoder(VisualEncoderConfig(d_model=32, pretrained=False, num_cameras=2)).eval()
    observation = ObservationEncoder(ObservationEncoderConfig(state_dim=8, hidden_dim=32, ffn_dim=64,
                                                              num_layers=1, num_heads=4, dropout=0)).eval()
    dit = ConditionalDiT(ConditionalDiTConfig(action_dim=3, action_horizon=4, hidden_dim=32, num_layers=1,
                                             num_heads=4, ff_dim=64, dropout=0,
                                             timestep_embedding_dim=16, timestep_hidden_dim=32)).eval()
    objective = FlowMatchingObjective()
    # Test-only bounded checks of zero-init opening, not a training API or trainer.
    modules = (vision, observation, dit)
    optimizer = torch.optim.SGD([p for module in modules for p in module.parameters() if p.requires_grad], lr=0.1)
    images = torch.linspace(-2, 2, 2*2*3*64*96).reshape(2, 2, 3, 64, 96)
    state = torch.linspace(-1, 1, 16).reshape(2, 8)
    action = torch.linspace(-1, 1, 24).reshape(2, 4, 3)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        observations = observation(**vision(images), state=state)
        observations["observation_memory"].retain_grad()
        output = objective(dit, action, **observations, noise=torch.zeros_like(action),
                           timestep=torch.tensor([0.2, 0.8]))
        output["loss"].backward()
        for module in modules:
            assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
        assert all(p.grad is None for p in vision.backbone.parameters())
        optimizer.step()
    assert observations["observation_memory"].grad.abs().sum() > 0
    assert observation.state_projection.weight.grad.abs().sum() > 0
    assert observation.state_pos_embed.grad.abs().sum() > 0
    assert observation.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert vision.input_proj.weight.grad.abs().sum() > 0
    assert vision.camera_embedding.weight.grad.abs().sum() > 0
    assert dit.blocks[0].cross_attn.in_proj_weight.grad.abs().sum() > 0


def test_shape_mismatches_cannot_silently_broadcast():
    class WrongShape(nn.Module):
        def forward(self, noisy_action, **kwargs):
            return torch.zeros(2, 1, 3)

    action, memory = torch.zeros(2, 4, 3), torch.zeros(2, 5, 32)
    objective = FlowMatchingObjective()
    with pytest.raises(ValueError, match="Velocity shape"):
        objective(WrongShape(), action, memory)
    with pytest.raises(ValueError, match="sample_weight"):
        objective(ZeroVelocity(), action, memory, sample_weight=torch.ones(2, 1))
    with pytest.raises(ValueError, match="action_padding_mask"):
        objective(ZeroVelocity(), action, memory, action_padding_mask=torch.zeros(4, dtype=torch.bool))
