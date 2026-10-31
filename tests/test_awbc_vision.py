import math

import pytest
import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from awbc_models import ACTVisualEncoder, SinePositionEncoding2D, VisualEncoderConfig


@pytest.mark.parametrize("height,width", [(224, 224), (240, 320), (255, 341)])
def test_dynamic_spatial_grid(height, width):
    model = ACTVisualEncoder(VisualEncoderConfig(pretrained=False))
    with torch.no_grad():
        image = torch.randn(1, 3, height, width)
        maps = model.forward_features(image)
        output = model(image)
    h, w = math.ceil(height / 32), math.ceil(width / 32)
    assert maps["feature_map"].shape == (1, 512, h, w)
    assert maps["projected_map"].shape == maps["position_map"].shape == (1, 256, h, w)
    assert output["visual_tokens"].shape == output["visual_pos"].shape == (1, h*w, 256)
    assert set(output) == {"visual_tokens", "visual_pos"}
    torch.testing.assert_close(output["visual_tokens"][:, w+1], maps["projected_map"][:, :, 1, 1])
    assert all(torch.isfinite(value).all() for value in (*maps.values(), *output.values()))
    assert not any(isinstance(m, (nn.AdaptiveAvgPool2d, nn.Linear, nn.BatchNorm2d)) for m in model.modules())


@pytest.mark.parametrize("mode", ["frozen", "layer4", "all"])
def test_backbone_gradients_and_learning_rates(mode):
    model = ACTVisualEncoder(VisualEncoderConfig(d_model=32, pretrained=False, backbone_mode=mode))
    model.train()
    buffers = {name: value.clone() for name, value in model.backbone.named_buffers()}
    images = torch.randn(2, 3, 64, 96)
    output = model(images)
    (output["visual_tokens"].square().mean() + output["visual_pos"].square().mean()).backward()
    for name, p in model.backbone.body.named_parameters():
        expected = mode == "all" or (mode == "layer4" and name.startswith("layer4."))
        assert p.requires_grad == expected
        assert (p.grad is not None) == expected
        if expected:
            assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert model.input_proj.weight.grad.abs().sum() > 0
    assert model.camera_embedding.weight.grad.abs().sum() > 0
    for name, value in model.backbone.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    assert sum(isinstance(m, FrozenBatchNorm2d) for m in model.modules()) == 20
    groups = model.parameter_groups(lr=1e-3, backbone_lr=1e-5)
    assert [g["lr"] for g in groups] == ([1e-3] if mode == "frozen" else [1e-3, 1e-5])
    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.parameters() if p.requires_grad}
    model.backbone.set_trainable("layer4")
    assert model.backbone.body.layer4[0].conv1.weight.requires_grad
    assert not model.backbone.body.layer3[0].conv1.weight.requires_grad


def test_frozen_batchnorm_uses_saved_statistics_in_train_mode():
    # A batch's mean must never override the stored ImageNet running statistics.
    frozen = FrozenBatchNorm2d(3)
    reference = nn.BatchNorm2d(3).eval()
    with torch.no_grad():
        reference.running_mean.copy_(torch.tensor([1., 2., 3.]))
        reference.running_var.copy_(torch.tensor([2., 3., 4.]))
        reference.weight.copy_(torch.tensor([0.5, 1., 2.]))
        reference.bias.copy_(torch.tensor([-1., 0., 1.]))
    frozen.load_state_dict(reference.state_dict())
    frozen.train()
    image = torch.randn(1, 3, 8, 8) + 10
    torch.testing.assert_close(frozen(image), reference(image))
    torch.testing.assert_close(frozen(image), frozen(torch.cat((image, image + 50)))[0:1])


def test_position_axes_and_act_coordinates():
    encoding = SinePositionEncoding2D(32)
    position = encoding(torch.empty(2, 32, 3, 5))
    torch.testing.assert_close(position[0], position[1])
    # y channels are independent of x; x channels are independent of y.
    torch.testing.assert_close(position[:, :16, :, 0], position[:, :16, :, 4])
    torch.testing.assert_close(position[:, 16:, 0, :], position[:, 16:, 2, :])
    assert position[0, 0, 0, 0].item() == pytest.approx(math.sin(2*math.pi/(3+1e-6)), abs=1e-6)
    assert position[0, 16, 0, 0].item() == pytest.approx(math.sin(2*math.pi/(5+1e-6)), abs=1e-6)
    # ACT uses first coordinate 1 and ends each axis at almost 2*pi.
    assert abs(position[0, 0, -1, 0]) < 1e-5
    assert position[0, 1, -1, 0].item() == pytest.approx(1., abs=1e-6)
    assert not position.requires_grad


@pytest.mark.parametrize("n", [1, 3, 4])
def test_shared_multicamera_concatenates_complete_camera_sequences(n):
    model = ACTVisualEncoder(VisualEncoderConfig(d_model=32, pretrained=False, num_cameras=n)).eval()
    images = torch.randn(2, n, 3, 64, 96)
    with torch.no_grad():
        output = model(images)
        individual = [model(images[:, c], camera_index=c) for c in range(n)]
    for key in output:
        torch.testing.assert_close(output[key], torch.cat([item[key] for item in individual], dim=1))
        assert output[key].shape == (2, n*6, 32)
        assert torch.isfinite(output[key]).all()


def test_camera_identity_only_changes_position():
    model = ACTVisualEncoder(VisualEncoderConfig(d_model=32, pretrained=False)).eval()
    images = torch.randn(2, 1, 3, 64, 96).expand(-1, 3, -1, -1, -1)
    with torch.no_grad():
        output = model(images)
    content = output["visual_tokens"].unflatten(1, (3, 6))
    position = output["visual_pos"].unflatten(1, (3, 6))
    torch.testing.assert_close(content[:, 0], content[:, 1])
    difference = model.camera_embedding.weight[1] - model.camera_embedding.weight[0]
    torch.testing.assert_close(position[:, 1] - position[:, 0], difference.expand(2, 6, 32))
    assert not torch.allclose(position[:, 0], position[:, 1])
