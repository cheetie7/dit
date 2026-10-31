import pytest
import torch

from awbc_models import ObservationEncoder, ObservationEncoderConfig


def test_state_projection_concatenation_positions_and_output():
    model = ObservationEncoder(ObservationEncoderConfig(state_dim=8))
    visual = torch.randn(2, 49, 256)
    pos = torch.randn_like(visual)
    state = torch.randn(2, 8)
    state_token = model.state_projection(state).unsqueeze(1)
    assert state_token.shape == (2, 1, 256)
    prepared = model.prepare_tokens(visual, pos, state)
    assert prepared["observation_tokens"].shape == (2, 50, 256)
    torch.testing.assert_close(prepared["observation_tokens"][:, :1], state_token)
    torch.testing.assert_close(prepared["observation_tokens"][:, 1:], visual)
    torch.testing.assert_close(prepared["observation_pos"][:, :1], model.state_pos_embed.expand(2, -1, -1))
    torch.testing.assert_close(prepared["observation_pos"][:, 1:], pos)
    assert model.state_pos_embed.requires_grad
    assert prepared["observation_padding_mask"] is None
    output = model(visual, pos, state)
    assert set(output) == {"observation_memory", "observation_pos", "observation_padding_mask"}
    assert output["observation_memory"].shape == output["observation_pos"].shape == (2, 50, 256)
    assert all(torch.isfinite(v).all() for v in output.values() if v is not None)
    assert len(model.layers) == 2
    assert model.layers[0] is not model.layers[1]
    assert model.layers[0].self_attn.num_heads == 8
    assert model.layers[0].ffn[0].out_features == 1024


def test_each_layer_uses_content_plus_position_for_qk_and_content_for_v():
    model = ObservationEncoder(ObservationEncoderConfig(state_dim=8, dropout=0)).eval()
    visual, pos, state = torch.randn(2, 9, 256), torch.randn(2, 9, 256), torch.randn(2, 8)
    prepared = model.prepare_tokens(visual, pos, state)
    calls, layers = [], []

    def capture_attention(module, args, kwargs):
        calls.append((args[0].detach(), args[1].detach(), kwargs["value"].detach()))

    handles = []
    for layer in model.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(capture_attention, with_kwargs=True))
        handles.append(layer.register_forward_hook(lambda module, args, result: layers.append(result.detach())))
    model(visual, pos, state)
    for index, (query, key, value) in enumerate(calls):
        content = prepared["observation_tokens"] if index == 0 else layers[index-1]
        torch.testing.assert_close(query, content + prepared["observation_pos"])
        torch.testing.assert_close(key, query)
        torch.testing.assert_close(value, content)
    assert len(calls) == 2
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("objective", ["mean", "weighted"])
def test_backward_connectivity_and_meaningful_gradients(objective):
    torch.manual_seed(10)
    model = ObservationEncoder(ObservationEncoderConfig(state_dim=8, dropout=0))
    visual = torch.randn(2, 49, 256, requires_grad=True)
    pos = torch.randn_like(visual, requires_grad=True)
    state = torch.randn(2, 8, requires_grad=True)
    memory = model(visual, pos, state)["observation_memory"]
    loss = memory.mean() if objective == "mean" else (memory * torch.randn_like(memory)).mean()
    loss.backward()
    for parameter in [*model.parameters(), visual, pos, state]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if objective == "weighted":
            assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("n", [49, 240])
def test_visual_only_without_constructing_state_parameters(n):
    model = ObservationEncoder(ObservationEncoderConfig(use_state=False))
    visual, pos = torch.randn(2, n, 256), torch.randn(2, n, 256)
    output = model(visual, pos)
    assert output["observation_memory"].shape == (2, n, 256)
    torch.testing.assert_close(output["observation_pos"], pos)
    assert model.state_projection is None and model.state_pos_embed is None
    assert not any("state" in name for name, _ in model.named_parameters())
    assert output["observation_padding_mask"] is None
    assert torch.isfinite(output["observation_memory"]).all()


@pytest.mark.parametrize("use_state", [True, False])
def test_padding_blocks_masked_content_and_gradients(use_state):
    model = ObservationEncoder(ObservationEncoderConfig(state_dim=8, use_state=use_state,
                                                       hidden_dim=32, ffn_dim=64, dropout=0)).eval()
    visual = torch.randn(2, 9, 32, requires_grad=True)
    pos = torch.randn_like(visual, requires_grad=True)
    state = torch.randn(2, 8) if use_state else None
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[:, -3:] = True
    output = model(visual, pos, state, mask)
    out_mask = output["observation_padding_mask"]
    assert out_mask.shape == (2, 9 + int(use_state))
    torch.testing.assert_close(out_mask[:, int(use_state):], mask)
    assert not out_mask[:, 0].any()
    changed_visual, changed_pos = visual.detach().clone(), pos.detach().clone()
    changed_visual[:, -3:] = torch.randn(2, 3, 32) * 1000
    changed_pos[:, -3:] = torch.randn(2, 3, 32) * 1000
    changed = model(changed_visual, changed_pos, state, mask)["observation_memory"]
    valid = ~out_mask
    memory = output["observation_memory"]
    torch.testing.assert_close(memory[valid], changed[valid])
    (memory[valid] * torch.randn_like(memory[valid])).mean().backward()
    assert visual.grad[:, :-3].abs().sum() > 0
    assert torch.count_nonzero(visual.grad[:, -3:]) == 0
    assert torch.count_nonzero(pos.grad[:, -3:]) == 0


def test_state_changes_visual_memory_and_visual_position_is_used():
    torch.manual_seed(11)
    model = ObservationEncoder(ObservationEncoderConfig(state_dim=8, dropout=0)).eval()
    visual, pos = torch.randn(2, 49, 256), torch.randn(2, 49, 256)
    state = torch.randn(2, 8)
    original = model(visual, pos, state)["observation_memory"]
    changed_state = model(visual, pos, state + 2)["observation_memory"]
    changed_pos = model(visual, torch.zeros_like(pos), state)["observation_memory"]
    assert not torch.allclose(original[:, 1:], changed_state[:, 1:])
    assert not torch.allclose(original, changed_pos)


def test_state_dimension_is_configured_and_missing_state_is_explicit():
    with pytest.raises(ValueError, match="state_dim"):
        ObservationEncoder(ObservationEncoderConfig())
    model = ObservationEncoder(ObservationEncoderConfig(state_dim=5))
    visual = torch.randn(1, 7, 256)
    with pytest.raises(ValueError, match="state is required"):
        model(visual, torch.zeros_like(visual))
    assert model(visual, torch.zeros_like(visual), torch.randn(1, 5))["observation_memory"].shape == (1, 8, 256)
