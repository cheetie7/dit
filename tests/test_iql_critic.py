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


def test_cli_train_resume_and_export(root, tmp_path):
    path = root / "episodes.jsonl"
    labels = [dict(episode_index=i, end_type="terminated", success=True, end_frame_index=13) for i in range(2)]
    path.write_text("".join(json.dumps(row) + "\n" for row in labels))
    config_path = tmp_path / "iql.json"
    config_path.write_text(json.dumps({
        "dataset": {"root": str(root), "stratify_key": None, "train_stride": 3},
        "labels_path": str(path), "image_preprocess": {"size": [24, 32]},
        "action_horizon": 4, "pretrained": False, "hidden_dims": [16],
        "num_steps": 2, "batch_size": 2, "device": "cpu",
        "logging": {"use_wandb": False, "log_interval": 1, "progress_bar": False},
    }))
    directory = tmp_path / "resumed"
    main(["train", "--config", str(config_path), "--run-dir", str(directory)])
    checkpoint = directory / "checkpoints/latest_checkpoint.pt"
    first = torch.load(checkpoint, weights_only=False)
    assert (first["global_step"], first["epoch"], first["batch_in_epoch"]) == (2, 0, 2)
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
    assert metadata["split"] == resumed["split"]
    assert metadata["checkpoint"]["global_step"] == 5
    assert metadata["data_contract"]["train_stride"] == 3
    assert metadata["data_contract"]["labels_sha256"] == resumed["data_contract"]["labels_sha256"]
    train, _, _, _ = prepare_datasets(DatasetConfig(root=str(root), stratify_key=None, train_stride=3),
                                     ImagePreprocessConfig(size=(24, 32)), action_horizon=4)
    model = IQLCritic(replace(IQLCriticConfig(**resumed["model_config"]), pretrained=False)).eval()
    model.load_state_dict(resumed["critic"])
    batch = next(iter(DataLoader(train, batch_size=len(train))))
    with torch.no_grad():
        features = model.encode(batch["observation"])
        q1, q2 = model.action_values(features, batch["action"], batch["action_is_pad"], target=True)
        expected = torch.minimum(q1, q2) - model.v(features)
    keys = [(row["trajectory_id"], row["start_timestep"], row["action_horizon"]) for row in rows]
    assert len(set(keys)) == len(rows) == len(train) == metadata["record_count"]
    assert keys == list(zip(batch["episode_index"].tolist(), batch["frame_index"].tolist(), [4] * len(train)))
    for row, length, value in zip(rows, batch["valid_length"].tolist(), expected, strict=True):
        assert row["valid_length"] == length
        assert row["end_timestep"] == row["start_timestep"] + length
        assert row["advantage"] == pytest.approx(value.item(), abs=1e-6)
    train.dataset.close()
