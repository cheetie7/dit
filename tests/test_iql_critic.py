from dataclasses import replace
import json
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from awbc_data import ImagePreprocessConfig
from awbc_models.iql_critic import IQLCritic, IQLCriticConfig
from awbc_training.config import DatasetConfig
from awbc_training.data import prepare_datasets
from awbc_training.iql import IQLConfig, IQLTrainer, expectile_loss, main, td_target
from test_awbc_data import root


def test_expectile_weights_both_signs():
    residuals = torch.tensor([2., -3., 0.], requires_grad=True)
    loss = expectile_loss(residuals, 0.7)
    assert loss.item() == pytest.approx((0.7 * 4 + 0.3 * 9) / 3)
    loss.backward()
    torch.testing.assert_close(residuals.grad, torch.tensor([0.7 * 4, -0.3 * 6, 0.]) / 3)


def test_td_target_uses_window_discount_and_skips_terminal_observations():
    batch = {
        "reward": torch.tensor([[0.], [0.81], [0.]]),
        "discount": torch.tensor([0.9**4, 0.9**3, 0.9**2]),
        "bootstrap_mask": torch.tensor([[1.], [0.], [1.]]),
        "terminated": torch.tensor([False, True, False]),
        "truncated": torch.tensor([False, False, True]),
        "next_observation": {"state": torch.tensor([[2.], [float("nan")], [3.]])},
    }

    def next_value(observation):
        assert not torch.is_grad_enabled()
        torch.testing.assert_close(observation["state"], torch.tensor([[2.], [3.]]))
        return observation["state"]

    target = td_target(batch, next_value)
    torch.testing.assert_close(target, torch.tensor([0.9**4 * 2, 0.81, 0.9**2 * 3]))
    batch["bootstrap_mask"].zero_()
    torch.testing.assert_close(td_target(batch, next_value), batch["reward"].flatten())


def parameters(module):
    return [p.detach().clone() for p in module.parameters()]


def same_parameters(module, previous):
    return all(torch.equal(p, old) for p, old in zip(module.parameters(), previous, strict=True))


def test_q_padding_and_ordered_update():
    torch.manual_seed(7)
    config = IQLConfig(labels_path="", hidden_dims=(12,), pretrained=False, target_update_rate=0.25)
    model = IQLCritic(IQLCriticConfig(2, 2, 3, 4, config.hidden_dims, pretrained=False))
    trainer = IQLTrainer(config, model, torch.device("cpu"))
    observation = {"images": torch.randn(2, 2, 3, 32, 32), "state": torch.randn(2, 2)}
    action = torch.randn(2, 4, 3)
    padding = torch.tensor([[False, False, False, True], [False, False, True, True]])
    batch = {
        "observation": observation, "next_observation": observation, "action": action, "action_is_pad": padding,
        "reward": torch.tensor([0.4, 1.]), "discount": torch.tensor([0.7, 0.6]),
        "bootstrap_mask": torch.tensor([1., 0.]), "terminated": torch.tensor([False, True]),
        "truncated": torch.zeros(2, dtype=torch.bool),
    }
    features = model.encode(observation)
    expected = model.backbone(observation["images"].flatten(0, 1))
    pooled = torch.nn.functional.adaptive_avg_pool2d(expected, (2, 2)).reshape(2, -1)
    torch.testing.assert_close(features, torch.cat((pooled, observation["state"]), dim=1))
    assert not features.requires_grad
    modified = action.masked_fill(padding.unsqueeze(-1), 12345.)
    for target in (False, True):
        for actual, expected in zip(model.action_values(features, action, padding, target=target),
                                    model.action_values(features, modified, padding, target=target), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with patch.object(model.q[0][0], "forward", wraps=model.q[0][0].forward) as first_layer:
        model.action_values(features, torch.zeros_like(action), padding)
        torch.testing.assert_close(first_layer.call_args.args[0][:, -4:], padding.float())

    before_backbone = parameters(model.backbone)
    before_v = parameters(model.v)
    before_q = [parameters(head) for head in model.q]
    before_target = parameters(model.target_q)
    assert same_parameters(model.target_q, [p for head in before_q for p in head])
    step_value, step_q = trainer.v_optimizer.step, trainer.q_optimizer.step

    def after_v():
        step_value()
        assert not same_parameters(model.v, before_v)
        assert all(same_parameters(head, old) for head, old in zip(model.q, before_q, strict=True))
        assert same_parameters(model.target_q, before_target)
        assert all(p.grad is None for p in model.q.parameters())

    def after_q():
        value = parameters(model.v)
        step_q()
        assert same_parameters(model.v, value)
        assert all(not same_parameters(head, old) for head, old in zip(model.q, before_q, strict=True))
        assert same_parameters(model.target_q, before_target)

    with patch.object(trainer.v_optimizer, "step", after_v), patch.object(trainer.q_optimizer, "step", after_q):
        metrics = trainer.step(batch)
    assert trainer.global_step == 1
    assert not model.backbone.training and not model.target_q.training
    assert same_parameters(model.backbone, before_backbone)
    assert all(p.grad is None and not p.requires_grad
               for module in (model.backbone, model.target_q) for p in module.parameters())
    for target, previous, online in zip(model.target_q.parameters(), before_target, model.q.parameters(), strict=True):
        torch.testing.assert_close(target, 0.75 * previous + 0.25 * online)
    expected_target = td_target(batch, lambda obs: model.v(model.encode(obs)))
    assert metrics["td_target_mean"] == pytest.approx(expected_target.mean().item())


@pytest.mark.parametrize(("mode", "feature_mode"), [
    ("shared_frozen", "avg_pool_2x2"),
    ("separate_trainable", "avg_pool_2x2"),
    ("separate_trainable", "spatial_softmax"),
])
def test_cli_train_resume_and_export(root, tmp_path, mode, feature_mode):
    path = root / "episodes.jsonl"
    labels = [dict(episode_index=i, end_type="terminated", success=True, end_frame_index=13) for i in range(2)]
    path.write_text("".join(json.dumps(row) + "\n" for row in labels))
    image_size = (64, 64) if feature_mode == "spatial_softmax" else (24, 32)
    config_path = tmp_path / "iql.json"
    config_path.write_text(json.dumps({
        "dataset": {"root": str(root), "stratify_key": None, "train_stride": 3},
        "labels_path": str(path), "image_preprocess": {"size": image_size},
        "action_horizon": 4, "pretrained": False, "hidden_dims": [16],
        "vision_encoder_mode": mode, "vision_feature_mode": feature_mode,
        "num_steps": 2, "batch_size": 2, "device": "cpu",
        "logging": {"use_wandb": False, "log_interval": 1, "progress_bar": False},
    }))
    directory = tmp_path / "resumed"
    main(["train", "--config", str(config_path), "--run-dir", str(directory)])
    checkpoint = directory / "checkpoints/latest_checkpoint.pt"
    first = torch.load(checkpoint, weights_only=False)
    assert (first["global_step"], first["epoch"], first["batch_in_epoch"]) == (2, 0, 2)
    assert first["model_config"]["vision_encoder_mode"] == mode
    assert first["model_config"]["vision_feature_mode"] == feature_mode
    if feature_mode == "avg_pool_2x2":
        first["config"].pop("vision_feature_mode")
        first["model_config"].pop("vision_feature_mode")
    if mode == "shared_frozen":
        # Old checkpoints lack this option and retain the original backbone keys.
        first["config"].pop("vision_encoder_mode")
        first["model_config"].pop("vision_encoder_mode")
    torch.save(first, checkpoint)
    other_mode = "shared_frozen" if mode == "separate_trainable" else "separate_trainable"
    with pytest.raises(ValueError, match="architecture"):
        main(["train", "--resume", str(checkpoint), "--set", f"vision_encoder_mode={other_mode}",
              "--set", "vision_feature_mode=avg_pool_2x2"])
    if mode == "separate_trainable":
        other_feature = "spatial_softmax" if feature_mode == "avg_pool_2x2" else "avg_pool_2x2"
        with pytest.raises(ValueError, match="architecture"):
            main(["train", "--resume", str(checkpoint), "--set", f"vision_feature_mode={other_feature}"])
    assert first["v_optimizer"]["state"] and first["q_optimizer"]["state"]
    assert first["rng_state"]["torch"].dtype == torch.uint8
    for override, pattern in (("seed=17", "checkpoint's seed"), ("gamma=0.9", "gamma"),
                              ("dataset.train_stride=4", "train_stride"), ("action_horizon=5", "action_horizon")):
        with pytest.raises(ValueError, match=pattern):
            main(["train", "--resume", str(checkpoint), "--set", override])
    labels[0]["success"] = False
    path.write_text("".join(json.dumps(row) + "\n" for row in labels))
    with pytest.raises(ValueError, match="labels_sha256"):
        main(["train", "--resume", str(checkpoint)])
    labels[0]["success"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in labels))
    main(["train", "--resume", str(checkpoint), "--set", "num_steps=5"])
    resumed = torch.load(checkpoint, weights_only=False)
    assert (resumed["global_step"], resumed["epoch"], resumed["batch_in_epoch"]) == (5, 1, 2)
    assert resumed["split"] == first["split"]
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in metrics if "train/q_loss" in row] == [1, 2, 3, 4, 5]
    assert metrics[-1]["val/terminated_windows"] == 1 and metrics[-1]["val/truncated_windows"] == 0

    uninterrupted = tmp_path / "uninterrupted"
    main(["train", "--config", str(config_path), "--run-dir", str(uninterrupted), "--set", "num_steps=5"])
    control = torch.load(uninterrupted / "checkpoints/latest_checkpoint.pt", weights_only=False)
    for key, value in resumed["critic"].items():
        torch.testing.assert_close(value, control["critic"][key], rtol=0, atol=0)

    output = tmp_path / "advantages.jsonl"
    main(["export", "--checkpoint", str(checkpoint), "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["vision_encoder_mode"] == mode
    assert metadata["vision_feature_mode"] == feature_mode
    assert metadata["split"] == resumed["split"]
    assert metadata["checkpoint"]["global_step"] == 5
    assert metadata["data_contract"]["train_stride"] == 3
    assert metadata["data_contract"]["labels_sha256"] == resumed["data_contract"]["labels_sha256"]
    train, _, _, _ = prepare_datasets(DatasetConfig(root=str(root), stratify_key=None, train_stride=3),
                                     ImagePreprocessConfig(size=image_size), action_horizon=4)
    model = IQLCritic(replace(IQLCriticConfig(**resumed["model_config"]), pretrained=False)).eval()
    model.load_state_dict(resumed["critic"])
    batch = next(iter(DataLoader(train, batch_size=len(train))))
    with torch.no_grad():
        features = model.encode(batch["observation"], target=True)
        q1, q2 = model.action_values(features, batch["action"], batch["action_is_pad"], target=True)
        expected = torch.minimum(q1, q2) - model.v(model.encode(batch["observation"], branch="v"))
    keys = [(row["trajectory_id"], row["start_timestep"], row["action_horizon"]) for row in rows]
    assert len(set(keys)) == len(rows) == len(train) == metadata["record_count"]
    assert keys == list(zip(batch["episode_index"].tolist(), batch["frame_index"].tolist(), [4] * len(train)))
    for row, length, value in zip(rows, batch["valid_length"].tolist(), expected, strict=True):
        assert row["valid_length"] == length
        assert row["end_timestep"] == row["start_timestep"] + length
        assert row["advantage"] == pytest.approx(value.item(), abs=1e-6)
    train.dataset.close()


@pytest.mark.parametrize("feature_mode", ["avg_pool_2x2", "spatial_softmax"])
def test_separate_trainable_gradient_isolation_and_target_update(feature_mode):
    from torchvision.ops.misc import FrozenBatchNorm2d

    torch.manual_seed(11)
    config = IQLConfig(labels_path="", pretrained=False, hidden_dims=(8,),
                       vision_encoder_mode="separate_trainable", vision_feature_mode=feature_mode, target_update_rate=0.25)
    model = IQLCritic(IQLCriticConfig(2, 2, 3, 4, config.hidden_dims, pretrained=False,
                                     vision_encoder_mode=config.vision_encoder_mode, vision_feature_mode=feature_mode))
    trainer = IQLTrainer(config, model, torch.device("cpu"))
    model.train()
    observation = {"images": torch.randn(2, 2, 3, 64, 64), "state": torch.randn(2, 2)}
    batch = dict(observation=observation, next_observation=observation,
                 action=torch.randn(2, 4, 3), action_is_pad=torch.tensor([[False]*3+[True], [False]*4]),
                 reward=torch.tensor([0.2, 1.]), discount=torch.tensor([0.8, 0.9]),
                 bootstrap_mask=torch.tensor([1., 0.]), terminated=torch.tensor([False, True]),
                 truncated=torch.tensor([False, False]))
    if feature_mode == "spatial_softmax":
        v_projection_before = parameters(model.v_backbone.projection)
        q_projection_before = parameters(model.q_backbone.projection)
    v_before, q_before = parameters(model.v_backbone), parameters(model.q_backbone)
    target_before = parameters(model.target_backbone)
    head_before = parameters(model.target_q)
    assert same_parameters(model.target_backbone, q_before)
    v_ids = {id(p) for p in model.value_parameters()}
    q_ids = {id(p) for p in model.q_parameters()}
    assert not v_ids & q_ids
    assert v_ids | q_ids == {id(p) for p in model.parameters() if p.requires_grad}
    for backbone in (model.q_backbone, model.v_backbone, model.target_backbone):
        assert any(isinstance(m, FrozenBatchNorm2d) for m in backbone.modules())
    assert all(p.requires_grad for b in (model.q_backbone, model.v_backbone) for p in b.parameters())

    _, loss = trainer.value_loss(observation, batch["action"], batch["action_is_pad"])
    loss.backward()
    assert all(p.grad is not None for p in model.value_parameters())
    assert all(p.grad is None for p in model.q_parameters())
    trainer.v_optimizer.step()
    if feature_mode == "spatial_softmax":
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.v_backbone.projection.parameters())
        assert not same_parameters(model.v_backbone.projection, v_projection_before)
        assert model.v_backbone.backbone.body["conv1"].weight.grad.abs().sum() > 0
    assert not same_parameters(model.v_backbone, v_before)
    assert same_parameters(model.q_backbone, q_before)
    assert same_parameters(model.target_backbone, target_before)
    trainer.v_optimizer.zero_grad(set_to_none=True)
    v_after = parameters(model.v_backbone)
    with patch.object(model.q_backbone, "forward", wraps=model.q_backbone.forward) as encode:
        _, _, _, loss = trainer.q_loss(batch, observation, batch["action"], batch["action_is_pad"])
        loss.backward()
        assert encode.call_count == 1  # Both Q heads consume the same visual features.
    assert all(p.grad is not None for p in model.q_parameters())
    assert all(p.grad is None for p in model.value_parameters())
    trainer.q_optimizer.step()
    if feature_mode == "spatial_softmax":
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.q_backbone.projection.parameters())
        assert not same_parameters(model.q_backbone.projection, q_projection_before)
        assert model.q_backbone.backbone.body["conv1"].weight.grad.abs().sum() > 0
    assert not same_parameters(model.q_backbone, q_before)
    assert same_parameters(model.v_backbone, v_after)
    assert same_parameters(model.target_backbone, target_before)
    assert same_parameters(model.target_q, head_before)
    assert not model.target_backbone.training and not model.target_q.training
    assert all(not p.requires_grad and p.grad is None
               for module in (model.target_backbone, model.target_q) for p in module.parameters())
    model.update_target(0.25)
    for target_module, online_module, previous in ((model.target_backbone, model.q_backbone, target_before),
                                                  (model.target_q, model.q, head_before)):
        for target, online, old in zip(target_module.parameters(), online_module.parameters(), previous, strict=True):
            torch.testing.assert_close(target, 0.75 * old + 0.25 * online)
    for name, buffer in model.target_backbone.named_buffers():
        torch.testing.assert_close(buffer, model.q_backbone.get_buffer(name), rtol=0, atol=0)

    with torch.no_grad():
        features = model.encode(observation)
        changed = batch["action"].masked_fill(batch["action_is_pad"].unsqueeze(-1), 999.)
        for original, padded in zip(model.action_values(features, batch["action"], batch["action_is_pad"]),
                                    model.action_values(features, changed, batch["action_is_pad"]), strict=True):
            torch.testing.assert_close(original, padded, rtol=0, atol=0)


def test_invalid_vision_encoder_mode():
    with pytest.raises(ValueError, match="vision_encoder_mode"):
        IQLCriticConfig(1, 2, 3, 4, vision_encoder_mode="unknown")


def test_spatial_softmax_coordinate_direction_and_uniform_map():
    from awbc_models.iql_critic import SpatialSoftmax

    softmax = SpatialSoftmax()
    torch.testing.assert_close(softmax(torch.zeros(1, 2, 3, 5)), torch.zeros(1, 4), atol=1e-7, rtol=0)
    features = torch.full((1, 2, 3, 5), -80.)
    features[0, 0, 0, 4] = 80.  # Top-right: x=+1, y=-1.
    features[0, 1, 2, 0] = 80.  # Bottom-left: x=-1, y=+1.
    torch.testing.assert_close(softmax(features), torch.tensor([[1., -1., -1., 1.]]))
    # Non-saturated example checks temperature=1 and both coordinate axes.
    logits = torch.tensor([[[[0., 1.], [2., 3.]]]])
    probability = logits.flatten().softmax(0)
    expected = torch.stack(((probability * torch.tensor([-1., 1., -1., 1.])).sum(),
                            (probability * torch.tensor([-1., -1., 1., 1.])).sum()))[None]
    torch.testing.assert_close(softmax(logits), expected)


@pytest.mark.parametrize("num_cameras", [1, 3])
def test_spatial_encoder_expands_rgb_weights_without_changing_resnet(num_cameras):
    from awbc_models.iql_critic import SpatialVisualEncoder
    from awbc_models.vision import ResNet18Backbone

    original = ResNet18Backbone(pretrained=False, mode="all")
    rgb_conv = original.body["conv1"]
    rgb_weight = rgb_conv.weight.detach().clone()
    with patch("awbc_models.iql_critic.ResNet18Backbone", return_value=original) as initialize:
        encoder = SpatialVisualEncoder(num_cameras, pretrained=True)
    initialize.assert_called_once_with(pretrained=True, mode="all")
    expanded = encoder.backbone.body["conv1"]
    assert expanded.in_channels == 3 * num_cameras
    assert (expanded.stride, expanded.padding, expanded.kernel_size) == (
        rgb_conv.stride, rgb_conv.padding, rgb_conv.kernel_size
    )
    torch.testing.assert_close(expanded.weight, rgb_weight.repeat(1, num_cameras, 1, 1) / num_cameras)
    assert encoder.backbone.body["layer4"][-1].conv2.out_channels == 512
    assert isinstance(encoder.projection[0], torch.nn.Linear)
    assert isinstance(encoder.projection[1], torch.nn.LayerNorm)
    assert isinstance(encoder.projection[2], torch.nn.Tanh)


def test_spatial_three_camera_fusion_dimensions_and_advantage():
    model = IQLCritic(IQLCriticConfig(3, 8, 8, 40, pretrained=False,
                                     vision_encoder_mode="separate_trainable",
                                     vision_feature_mode="spatial_softmax")).eval()
    observation = {"images": torch.randn(2, 3, 3, 64, 96), "state": torch.randn(2, 8)}
    action = torch.randn(2, 40, 8)
    padding = torch.zeros(2, 40, dtype=torch.bool)
    assert model.v[0].in_features == 58
    assert all(head[0].in_features == 418 for head in (*model.q, *model.target_q))
    conv = model.q_backbone.backbone.body["conv1"]
    with torch.no_grad(), patch.object(conv, "forward", wraps=conv.forward) as first_conv:
        features = model.encode(observation)
        q1, q2 = model.action_values(features, action, padding)
        assert first_conv.call_count == 1
        stacked = first_conv.call_args.args[0]
        assert stacked.shape == (2, 9, 64, 96)
        for camera in range(3):
            torch.testing.assert_close(stacked[:, camera * 3:(camera + 1) * 3],
                                        observation["images"][:, camera])
        assert features.shape == (2, 58)
        assert features[:, :50].abs().max() <= 1
        torch.testing.assert_close(features[:, 50:], observation["state"])
        assert q1.shape == q2.shape == (2,)
        target_features = model.encode(observation, target=True)
        tq1, tq2 = model.action_values(target_features, action, padding, target=True)
        expected = torch.minimum(tq1, tq2) - model.v(model.encode(observation, branch="v"))
        torch.testing.assert_close(model.advantage(dict(observation=observation, action=action,
                                                      action_is_pad=padding)), expected)


def test_invalid_vision_feature_mode():
    with pytest.raises(ValueError, match="vision_feature_mode"):
        IQLCriticConfig(3, 8, 8, 40, vision_feature_mode="unknown")
    with pytest.raises(ValueError, match="requires vision_encoder_mode=separate_trainable"):
        IQLCriticConfig(3, 8, 8, 40, vision_feature_mode="spatial_softmax")
