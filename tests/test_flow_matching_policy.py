from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from awbc_data import ImagePreprocessConfig, ImagePreprocessor
from awbc_models import (ConditionalDiTConfig, FlowMatchingPolicy, FlowMatchingPolicyConfig,
                         MinMaxNormalizer, ObservationEncoderConfig, VisualEncoderConfig, sample_euler)


def config(**kwargs):
    return FlowMatchingPolicyConfig(
        vision=VisualEncoderConfig(d_model=32, pretrained=False, num_cameras=2),
        observation=ObservationEncoderConfig(state_dim=8, hidden_dim=32, num_layers=2,
                                             num_heads=4, ffn_dim=64, dropout=0.1),
        dit=ConditionalDiTConfig(action_dim=3, action_horizon=6, hidden_dim=32, num_layers=2,
                                 num_heads=4, ff_dim=64, dropout=0.1,
                                 timestep_embedding_dim=16, timestep_hidden_dim=32),
        camera_names=("top", "left"), image_preprocess=ImagePreprocessConfig(size=(32, 48)),
        n_action_steps=3, num_inference_steps=4, **kwargs)


@pytest.fixture
def stats():
    return {"action": {"min": [-2., 10., 4.], "max": [2., 20., 4.]},
            "observation.state": {"min": [-2.]*8, "max": [2.]*8}}


@pytest.fixture
def policy(stats):
    torch.manual_seed(16)
    return FlowMatchingPolicy(config(), stats=stats)


@pytest.fixture
def batch():
    return {"observation": {"images": torch.linspace(-2, 2, 2*2*3*32*48).reshape(2, 2, 3, 32, 48),
                            "state": torch.linspace(-1, 1, 16).reshape(2, 8)},
            "action": torch.linspace(-1, 1, 24).reshape(2, 4, 3),
            "action_is_pad": torch.tensor([[False]*4, [False, False, True, True]])}


def open_conditioning(policy):
    """Test-only nonzero head/gates to make dependence and gradients observable."""
    with torch.no_grad():
        policy.dit.velocity_head.weight.copy_(torch.linspace(-0.1, 0.1, 96).reshape(3, 32))
        for block in policy.dit.blocks:
            block.adaLN_modulation[-1].bias[5*32:6*32] = 0.2


class ConstantVelocity(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.times = []
        self.memories = []
        self.modes = []

    def forward(self, noisy_action, timestep, observation_memory, **kwargs):
        self.times.append(timestep.clone())
        self.memories.append(observation_memory)
        self.modes.append((self.training, torch.is_grad_enabled()))
        return torch.full_like(noisy_action, self.value)


@pytest.mark.parametrize("steps", [1, 4, 8, 16])
def test_euler_math_left_endpoints_direction_and_call_count(steps):
    model = ConstantVelocity(0.75)
    memory = torch.zeros(2, 5, 32)
    noise = torch.linspace(-2, 2, 36).reshape(2, 6, 3)
    original = noise.clone()
    timings = {}
    result = sample_euler(model, memory, 6, 3, initial_noise=noise,
                          num_inference_steps=steps, timings=timings)
    torch.testing.assert_close(result, noise+0.75)
    torch.testing.assert_close(noise, original, rtol=0, atol=0)
    assert len(model.times) == steps
    for i, timestep in enumerate(model.times):
        torch.testing.assert_close(timestep, torch.full((2,), i/steps))
    assert all(item is memory for item in model.memories)
    assert model.modes == [(False, False)]*steps
    assert not model.training
    assert len(timings["dit_step_ms"]) == steps and timings["euler_ms"] >= 0
    assert not result.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_euler_dtype_and_generator(dtype):
    memory = torch.zeros(2, 5, 32, dtype=dtype)
    model = ConstantVelocity(0.)
    first = sample_euler(model, memory, 6, 3, generator=torch.Generator().manual_seed(10))
    second = sample_euler(model, memory, 6, 3, generator=torch.Generator().manual_seed(10))
    assert first.dtype == dtype and first.device == memory.device
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_euler_bad_noise_steps_and_nonfinite_output():
    memory = torch.zeros(2, 5, 32)
    with pytest.raises(ValueError, match="initial_noise"):
        sample_euler(ConstantVelocity(0.), memory, 6, 3, initial_noise=torch.zeros(2, 5, 3))
    with pytest.raises(ValueError, match="positive"):
        sample_euler(ConstantVelocity(0.), memory, 6, 3, num_inference_steps=0)
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        sample_euler(ConstantVelocity(float("inf")), memory, 6, 3)


def test_training_forward_mask_adaptation_and_backward(policy, batch):
    output = policy(batch, sample_weight=torch.tensor([1., 2.]))
    assert output["predicted_velocity"].shape == (2, 6, 3)
    torch.testing.assert_close(output["valid_action_count"], torch.tensor([4, 2]))
    assert torch.count_nonzero(output["elementwise_loss"][:, 4:]) == 0
    assert output["per_sample_loss"].shape == (2,) and torch.isfinite(output["loss"])
    expected = (output["per_sample_loss"]*torch.tensor([1., 2.])).sum()/3
    torch.testing.assert_close(output["loss"], expected)
    output["loss"].backward()
    assert policy.dit.velocity_head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.vision_encoder.backbone.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in policy.observation_encoder.parameters() if p.requires_grad)
    with pytest.raises(ValueError, match="exceeds"):
        policy._prepare_action_chunk(torch.zeros(2, 7, 3), None)


def test_training_graph_has_actual_upstream_gradients_when_gates_open(policy, batch):
    open_conditioning(policy)
    policy(batch)["loss"].backward()
    assert policy.observation_encoder.state_projection.weight.grad.abs().sum() > 0
    assert policy.observation_encoder.state_pos_embed.grad.abs().sum() > 0
    assert policy.vision_encoder.input_proj.weight.grad.abs().sum() > 0
    assert policy.vision_encoder.camera_embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.vision_encoder.backbone.parameters())


def test_inference_encodes_once_eval_modes_and_determinism(policy, batch):
    open_conditioning(policy)
    policy.train()
    counts, handles = {"vision": 0, "observation": 0, "dit": 0}, []

    def capture(name):
        def hook(module, args, result):
            counts[name] += 1
            assert not module.training and not torch.is_grad_enabled()
        return hook

    for name, module in (("vision", policy.vision_encoder), ("observation", policy.observation_encoder), ("dit", policy.dit)):
        handles.append(module.register_forward_hook(capture(name)))
    noise = torch.linspace(-1, 1, 36).reshape(2, 6, 3)
    first = policy.predict_action_chunk(batch["observation"], initial_noise=noise, num_inference_steps=8)
    assert counts == {"vision": 1, "observation": 1, "dit": 8}
    assert all(not module.training for module in policy.modules())
    second = policy.predict_action_chunk(batch["observation"], initial_noise=noise, num_inference_steps=8)
    torch.testing.assert_close(first["action_chunk"], second["action_chunk"], rtol=0, atol=0)
    assert first["action_chunk"].shape == first["action_chunk_normalized"].shape == (2, 6, 3)
    assert not first["action_chunk"].requires_grad
    assert len(first["timings"]["dit_step_ms"]) == 8
    assert all(first["timings"][key] >= 0 for key in ("vision_ms", "observation_ms", "euler_ms", "total_ms"))
    for handle in handles:
        handle.remove()


def test_denormalization_once_no_clipping_and_execution_slice(policy, batch):
    noise = torch.full((2, 6, 3), 2.)  # Untrained zero velocity preserves this, without clipping.
    with patch.object(policy.action_normalizer, "denormalize", wraps=policy.action_normalizer.denormalize) as denormalize:
        chunk = policy.predict_action_chunk(batch["observation"], initial_noise=noise)
        assert denormalize.call_count == 1
    torch.testing.assert_close(chunk["action_chunk_normalized"], noise)
    torch.testing.assert_close(chunk["action_chunk"][0, 0], torch.tensor([4., 25., 4.]))
    execution = policy.predict_action(batch["observation"], initial_noise=noise)
    assert execution["action"].shape == (2, 3, 3)
    torch.testing.assert_close(execution["action"], chunk["action_chunk"][:, :3])


def test_policy_generator_and_cpu_mixed_precision(policy, batch):
    open_conditioning(policy)
    first = policy.predict_action_chunk(batch["observation"], generator=torch.Generator().manual_seed(55))
    repeated = policy.predict_action_chunk(batch["observation"], generator=torch.Generator().manual_seed(55))
    torch.testing.assert_close(first["action_chunk"], repeated["action_chunk"], rtol=0, atol=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = policy.predict_action_chunk(batch["observation"], initial_noise=torch.zeros(2, 6, 3))
    assert mixed["action_chunk"].shape == (2, 6, 3)
    assert torch.isfinite(mixed["action_chunk"]).all()


def test_normalizer_matches_dataset_formula_and_constant_dimensions(stats):
    normalizer = MinMaxNormalizer(stats["action"])
    raw = torch.tensor([[[-2., 15., 4.], [2., 20., 4.]]])
    normalized = normalizer.normalize(raw)
    torch.testing.assert_close(normalized, torch.tensor([[[-1., 0., 0.], [1., 1., 0.]]]))
    torch.testing.assert_close(normalizer.denormalize(normalized), raw)


def test_raw_observation_reuses_rgb_preprocessing_state_stats_and_camera_order(policy):
    top = np.full((24, 32, 3), [255, 20, 10], dtype=np.uint8)
    left = np.full((24, 32, 3), [10, 20, 255], dtype=np.uint8)
    raw = {"images": {"left": left, "top": top}, "state": np.ones(8, dtype=np.float32)}
    prepared = policy.prepare_observation(raw)
    preprocess = ImagePreprocessor(policy.config.image_preprocess)
    torch.testing.assert_close(prepared["images"][0, 0], preprocess(top))
    torch.testing.assert_close(prepared["images"][0, 1], preprocess(left))
    torch.testing.assert_close(prepared["state"], torch.full((1, 8), 0.5))
    with patch.object(policy.vision_encoder, "forward", wraps=policy.vision_encoder.forward) as vision:
        policy.encode_observation(prepared)
        assert vision.call_args.args[0] is prepared["images"]


def test_visual_only_policy(stats, batch):
    cfg = config()
    cfg.observation.use_state = False
    cfg.observation.state_dim = None
    model = FlowMatchingPolicy(cfg, stats={"action": stats["action"]})
    observation = {"images": batch["observation"]["images"]}
    encoded = model.encode_observation(observation)
    assert encoded["observation_memory"].shape == (2, 4, 32)  # 2 cameras * (1*2) layer4 grid.
    assert model.state_normalizer is None
    assert model.predict_action(observation)["action"].shape == (2, 3, 3)


def test_missing_stats_and_invalid_execution_length(stats):
    with pytest.raises(ValueError, match="statistics are required"):
        FlowMatchingPolicy(config())
    with pytest.raises(KeyError, match="action"):
        FlowMatchingPolicy(config(), stats={})
    cfg = config()
    with pytest.raises(ValueError, match="n_action_steps"):
        replace(cfg, n_action_steps=7)


def test_state_dict_config_and_normalizer_restore(tmp_path, policy, stats, batch):
    open_conditioning(policy)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats))
    cfg = replace(policy.config, normalization_stats_path="stats.json")
    config_path = tmp_path / "config.json"
    cfg.to_json(config_path)
    checkpoint_path = tmp_path / "policy.pt"
    torch.save(policy.state_dict(), checkpoint_path)
    # Buffers in state_dict must restore the original stats, even if constructor stats differ.
    changed = {**stats, "action": {"min": [-3., 11., 4.], "max": [3., 21., 4.]}}
    stats_path.write_text(json.dumps(changed))
    restored = FlowMatchingPolicy(FlowMatchingPolicyConfig.from_json(config_path))
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    noise = torch.linspace(-1, 1, 36).reshape(2, 6, 3)
    before = policy.predict_action_chunk(batch["observation"], initial_noise=noise)
    after = restored.predict_action_chunk(batch["observation"], initial_noise=noise)
    torch.testing.assert_close(before["action_chunk"], after["action_chunk"], rtol=0, atol=0)


def test_json_derives_actual_dataset_dimensions(tmp_path):
    cfg = config()
    path = tmp_path / "policy.json"
    cfg.to_json(path)
    info = {"features": {"observation.images.top": {"dtype": "video"},
                          "action": {"shape": [8], "dtype": "float32"},
                          "observation.state": {"shape": [8], "dtype": "float32"}}}
    loaded = FlowMatchingPolicyConfig.from_json(path, dataset_info=info)
    assert loaded.dit.action_dim == loaded.observation.state_dim == 8
    assert loaded.vision.num_cameras == 1
    assert loaded.camera_names == ("observation.images.top",)
    del info["features"]["observation.state"]
    assert not FlowMatchingPolicyConfig.from_json(path, dataset_info=info).observation.use_state
