"""Network-only checks; bounded optimizer steps below are diagnostics, not policy training."""
from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest
import torch

from awbc_models import ConditionalDiT, ConditionalDiTBlock, ConditionalDiTConfig, TimestepEmbedder
from awbc_models.dit import modulate, sinusoidal_embedding


def values(shape, phase=0):
    """Deterministic external test tensors; no Gaussian noise sampling."""
    return torch.sin(torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)*0.37 + phase)


@pytest.mark.parametrize("action_dim,horizon", [(7, 16), (8, 10)])
def test_forward_shape_initialization_and_block_identity(action_dim, horizon):
    model = ConditionalDiT(ConditionalDiTConfig(action_dim=action_dim, action_horizon=horizon))
    action, memory, pos = values((4, horizon, action_dim)), values((4, 50, 256)), values((4, 50, 256), 1)
    t = torch.tensor([0., 0.2, 0.65, 1.])
    with torch.no_grad():
        output = model(action, t, memory, pos)
    assert output.shape == (4, horizon, action_dim)
    assert torch.isfinite(output).all() and torch.count_nonzero(output) == 0
    assert model.action_pos_embed.requires_grad and torch.count_nonzero(model.action_pos_embed) == 0
    assert len(model.blocks) == 4
    assert model.blocks[0].self_attn.num_heads == 8
    assert model.blocks[0].mlp[0].out_features == 1024
    for module in (*model.blocks, model.final_layer):
        assert torch.count_nonzero(module.adaLN_modulation[-1].weight) == 0
        assert torch.count_nonzero(module.adaLN_modulation[-1].bias) == 0
    assert torch.count_nonzero(model.velocity_head.weight) == 0
    assert torch.count_nonzero(model.velocity_head.bias) == 0
    x = values((4, horizon, 256))
    condition = model.timestep_embedder(t)
    torch.testing.assert_close(model.blocks[0](x, condition, memory, pos), x, rtol=0, atol=0)
    assert model.parameter_counts()["total"] == sum(p.numel() for p in model.parameters())


def test_continuous_timestep_embedding():
    t = torch.tensor([0., 0.125, 0.5, 1.], requires_grad=True)
    raw = sinusoidal_embedding(t)
    assert raw.shape == (4, 128)
    torch.testing.assert_close(raw[0, :64], torch.ones(64))
    torch.testing.assert_close(raw[0, 64:], torch.zeros(64))
    assert raw[-1, 0].item() == pytest.approx(math.cos(1.))
    assert raw[-1, 64].item() == pytest.approx(math.sin(1.))
    assert not torch.allclose(raw[0], raw[1])
    embedder = TimestepEmbedder(ConditionalDiTConfig())
    condition = embedder(t)
    assert condition.shape == (4, 256) and torch.isfinite(condition).all()
    assert not torch.allclose(condition[0], condition[1])
    condition.sum().backward()
    assert torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    assert sinusoidal_embedding(t.detach(), 127).shape == (4, 127)


def test_modulate_and_attention_qkv_sources():
    config = ConditionalDiTConfig(action_dim=7, hidden_dim=32, num_heads=4, ff_dim=64, dropout=0)
    block = ConditionalDiTBlock(config)
    x, c = values((2, 5, 32)), values((2, 32))
    memory, pos = values((2, 9, 32)), values((2, 9, 32), 2)
    mask = torch.zeros(2, 9, dtype=torch.bool)
    calls = {}

    def capture_self(module, args, kwargs):
        calls["self"] = args

    def capture_cross(module, args, kwargs):
        calls["cross"] = kwargs

    handles = [block.self_attn.register_forward_pre_hook(capture_self, with_kwargs=True),
               block.cross_attn.register_forward_pre_hook(capture_cross, with_kwargs=True)]
    output = block(x, c, memory, pos, mask)
    for tensor in calls["self"]:
        torch.testing.assert_close(tensor, block.norm1(x))
    torch.testing.assert_close(calls["cross"]["query"], block.norm2(x))
    torch.testing.assert_close(calls["cross"]["key"], memory + pos)
    torch.testing.assert_close(calls["cross"]["value"], memory)
    assert calls["cross"]["key_padding_mask"] is mask
    torch.testing.assert_close(output, x)
    for handle in handles:
        handle.remove()
    shift, scale = torch.ones(2, 32), torch.full((2, 32), 2.)
    torch.testing.assert_close(modulate(x, shift, scale), x*3+1)


@pytest.fixture
def opened_model():
    torch.manual_seed(12)
    config = ConditionalDiTConfig(action_dim=7, action_horizon=5, hidden_dim=32,
                                  num_layers=2, num_heads=4, ff_dim=64, dropout=0,
                                  timestep_embedding_dim=16, timestep_hidden_dim=32)
    model = ConditionalDiT(config)
    action, memory, pos = values((2, 5, 7)), values((2, 9, 32)), values((2, 9, 32), 2)
    t, target = torch.tensor([0.2, 0.8]), values((2, 5, 7), 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    records = []
    # Four toy regression updates only to open zero-initialized gates.
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(action, t, memory, pos)
        (prediction-target).square().mean().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        groups = {"head": model.velocity_head, "timestep": model.timestep_embedder,
                  "final_modulation": model.final_layer.adaLN_modulation}
        for index, block in enumerate(model.blocks):
            groups.update({f"block{index}.{name}": getattr(block, name)
                           for name in ("self_attn", "cross_attn", "mlp", "adaLN_modulation")})
        records.append({name: sum(p.grad.abs().sum().item() for p in module.parameters())
                        for name, module in groups.items()})
        optimizer.step()
    model.eval()
    return model, action, t, memory, pos, records


def test_zero_init_gradients_open_after_diagnostic_updates(opened_model):
    model, _, _, _, _, records = opened_model
    assert records[0]["head"] > 0
    assert records[0]["block0.self_attn"] == 0
    assert records[0]["block0.adaLN_modulation"] == 0
    assert records[1]["block0.adaLN_modulation"] > 0
    assert all(value > 0 for value in records[-1].values())
    assert model.action_proj.weight.grad.abs().sum() > 0
    assert model.action_pos_embed.grad.abs().sum() > 0


def test_condition_dependency_after_updates(opened_model):
    model, action, t, memory, pos, _ = opened_model
    with torch.no_grad():
        original = model(action, t, memory, pos)
        changed_memory = model(action, t, memory + values(memory.shape, 4), pos)
        changed_pos = model(action, t, memory, torch.zeros_like(pos))
        changed_time = model(action, 1-t, memory, pos)
        no_pos = model(action, t, memory)
        zero_pos = model(action, t, memory, torch.zeros_like(pos))
    assert not torch.allclose(original, changed_memory)
    assert not torch.allclose(original, changed_pos)
    assert not torch.allclose(original, changed_time)
    torch.testing.assert_close(no_pos, zero_pos)
    assert all(torch.isfinite(v).all() for v in (original, changed_memory, changed_pos, changed_time))


def test_padding_ignores_masked_observations_after_gates_open(opened_model):
    model, action, t, memory, pos, _ = opened_model
    memory, pos = memory.clone().requires_grad_(), pos.clone().requires_grad_()
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[:, -3:] = True
    output = model(action, t, memory, pos, mask)
    changed_memory, changed_pos = memory.detach().clone(), pos.detach().clone()
    changed_memory[:, -3:] += 100
    changed_pos[:, -3:] += 100
    changed = model(action, t, changed_memory, changed_pos, mask)
    torch.testing.assert_close(output, changed)
    output.square().mean().backward()
    assert memory.grad[:, :-3].abs().sum() > 0
    assert pos.grad[:, :-3].abs().sum() > 0
    assert torch.count_nonzero(memory.grad[:, -3:]) == 0
    assert torch.count_nonzero(pos.grad[:, -3:]) == 0
    assert torch.isfinite(output).all()


def test_config_file_controls_architecture(tmp_path):
    from importlib.resources import files
    original = ConditionalDiTConfig.from_json(files("awbc_models").joinpath("configs/conditional_dit.json"), action_dim=8)
    assert original.action_dim == 8 and original.action_horizon == 10
    config = ConditionalDiTConfig(action_dim=5, action_horizon=10, hidden_dim=32, num_layers=1,
                                  num_heads=4, ff_dim=80, dropout=0,
                                  timestep_embedding_dim=16, timestep_hidden_dim=48)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)))
    loaded = ConditionalDiTConfig.from_json(path)
    assert loaded == config
    model = ConditionalDiT(loaded)
    assert len(model.blocks) == 1
    assert model.blocks[0].mlp[0].out_features == 80
    assert model.timestep_embedder.mlp[0].out_features == 48
    assert model(values((2, 10, 5)), torch.tensor([0., 1.]), values((2, 9, 32))).shape == (2, 10, 5)
