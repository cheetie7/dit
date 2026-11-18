from dataclasses import asdict, replace
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from awbc_data import DatasetSource, ImagePreprocessConfig, RLChunkDataset
from awbc_models import FlowMatchingPolicy, FlowMatchingPolicyConfig
from awbc_training.advantage import prepare_iql_advantage_dataset
from awbc_training.config import AdvantageWeightingConfig, DatasetConfig, ExperimentConfig
from awbc_training.data import prepare_datasets
from awbc_training.inference import load_policy
from awbc_training.iql import main as iql_main
from awbc_training.iql_dense import main as dense_main
from awbc_training.train import main as policy_main
from test_advantage_weighting import tiny_policy
from test_awbc_data import root
from test_iql_dense import jsonl, progress_row


INDICES = [7, 0, 5]
SELECTION = {"state_indices": INDICES, "raw_state_dim": 8, "state_dim": 3}
ACTION_INDICES = [4, 1]
ACTION_SELECTION = {"action_indices": ACTION_INDICES, "raw_action_dim": 8, "action_dim": 2}


@pytest.fixture
def distinct_states(root):
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            frame, episode = row["frame_index"], row["episode_index"]
            row["observation.state"] = [10000 * episode + 100 * col + frame**(1 + col % 3)
                                        for col in range(8)]
            row["action"] = [value + 10 * col for col, value in enumerate(row["action"])]
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    return root


@pytest.mark.parametrize(("indices", "message"), [
    ([], "nonempty list"), ("0,1", "nonempty list"), ([0, True], "integers"),
    ([0, 1.0], "integers"), ([2, 2], "duplicates"), ([-1], "raw state dimension 8"),
    ([8], "raw state dimension 8"),
])
def test_invalid_state_indices(root, indices, message):
    with pytest.raises(ValueError, match=message):
        DatasetSource(root, state_indices=indices)


def test_invalid_action_indices(root):
    with pytest.raises(ValueError, match="action_indices.*raw action dimension 8"):
        DatasetSource(root, action_indices=[8])


def policy_settings(config):
    policy, _ = tiny_policy()
    config.policy = asdict(policy.config)
    for key in ("camera_names", "normalization_stats_path", "raw_state_dim", "state_indices",
                "raw_action_dim", "action_indices"):
        config.policy.pop(key)
    config.policy["dit"].pop("action_dim")
    config.policy["dit"]["action_horizon"] = config.policy["n_action_steps"] = 4
    config.policy["image_preprocess"]["size"] = [24, 32]


def test_state_selection_statistics_and_policy_inference(distinct_states):
    data = DatasetConfig(root=str(distinct_states), stratify_key=None, state_indices=INDICES, action_indices=ACTION_INDICES)
    config = ExperimentConfig(dataset=data)
    policy_settings(config)
    full = DatasetSource(distinct_states)
    selected_config = config.policy_configuration(full.info)
    assert selected_config.observation.state_dim == 3
    assert selected_config.raw_state_dim == 8 and selected_config.state_indices == INDICES
    assert selected_config.dit.action_dim == 2
    assert selected_config.raw_action_dim == 8 and selected_config.action_indices == ACTION_INDICES
    direct = FlowMatchingPolicyConfig.from_dict({"state_indices": INDICES, "action_indices": ACTION_INDICES}, full.info)
    assert direct.observation.state_dim == 3
    assert direct.dit.action_dim == 2
    train, val, stats, split = prepare_datasets(data, selected_config.image_preprocess, action_horizon=4)
    full_stats = full.statistics(split["train_episodes"])
    for key in ("min", "max", "mean", "std"):
        assert stats["observation.state"][key] == [full_stats["observation.state"][key][i] for i in INDICES]
    for key in ("min", "max", "mean", "std"):
        assert stats["action"][key] == [full_stats["action"][key][i] for i in ACTION_INDICES]
    assert stats["action"]["count"] == full_stats["action"]["count"]
    assert train.dataset.stats is val.dataset.stats

    end_labels = jsonl(distinct_states / "ends.jsonl", [
        dict(episode_index=e, success=True, end_type="terminated", end_frame_index=13) for e in (0, 1)
    ])
    policy = FlowMatchingPolicy(selected_config, stats).eval()
    for dataset in (train, val):
        sample = dataset[1]
        episode, frame = sample["episode_index"].item(), sample["frame_index"].item()
        raw = full.rows[episode]["observation.state"][frame]
        state = sample["observation"]["state"]
        assert state.shape == (3,)
        assert sample["action"].shape == (4, 2)
        torch.testing.assert_close(dataset.dataset.denormalize("action", sample["action"][0]),
                                   torch.from_numpy(full.rows[episode]["action"][frame, ACTION_INDICES]))
        torch.testing.assert_close(dataset.dataset.denormalize("observation.state", state),
                                   torch.from_numpy(raw[INDICES]))
        images = {name: np.zeros((24, 32, 3), dtype=np.uint8) for name in selected_config.camera_names}
        prepared = policy.prepare_observation({"images": images, "state": raw})
        torch.testing.assert_close(prepared["state"][0], state, rtol=0, atol=0)
        transition = RLChunkDataset(dataset, end_labels, gamma=0.99)[1]
        assert transition["next_frame_index"].item() == frame + 4
        torch.testing.assert_close(transition["observation"]["state"], state)
        next_raw = full.rows[episode]["observation.state"][frame + 4]
        expected_next = policy.prepare_observation({"images": images, "state": next_raw})["state"][0]
        torch.testing.assert_close(transition["next_observation"]["state"], expected_next, rtol=0, atol=0)
        with pytest.raises(ValueError, match="raw state vector of length 8"):
            policy.prepare_observation({"images": images, "state": raw[INDICES]})
    batch = next(iter(DataLoader(train, batch_size=2)))
    assert policy.observation_encoder.state_projection.in_features == 3
    assert torch.isfinite(policy(batch)["loss"])
    assert policy.predict_action_chunk(prepared, num_inference_steps=1)["action_chunk"].shape == (1, 4, 2)


@pytest.mark.parametrize("indices", [INDICES, None])
def test_policy_checkpoint_resume_and_legacy(root, tmp_path, monkeypatch, indices):
    actions = ACTION_INDICES if indices is not None else None
    cfg = ExperimentConfig(dataset=DatasetConfig(root=str(root), stratify_key=None, state_indices=indices,
                                                 action_indices=actions))
    policy_settings(cfg)
    cfg.training = replace(cfg.training, num_steps=2, max_updates=1, batch_size=2, num_workers=0, device="cpu")
    cfg.amp.enabled = cfg.ema.enabled = cfg.logging.use_wandb = cfg.logging.progress_bar = False
    cfg.validation.interval = cfg.validation.update_interval = 0
    path = tmp_path / "policy.json"
    cfg.to_json(path)
    run = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["awbc-train", "--config", str(path), "--run-dir", str(run)])
    policy_main()
    checkpoint = run / "checkpoints/latest_checkpoint.pt"
    state = torch.load(checkpoint, weights_only=False)
    assert state["policy_config"]["state_indices"] == indices
    assert state["policy_config"]["raw_state_dim"] == 8
    assert state["policy_config"]["action_indices"] == actions
    assert state["policy_config"]["raw_action_dim"] == 8
    if indices is None:
        for key in ("state_indices", "raw_state_dim", "action_indices", "raw_action_dim"):
            state["policy_config"].pop(key)
        state["config"]["dataset"].pop("state_indices")
        state["config"]["dataset"].pop("action_indices")
        torch.save(state, checkpoint)
    loaded = load_policy(checkpoint, device="cpu")
    assert loaded.config.state_indices == indices and loaded.config.raw_state_dim == 8
    assert loaded.observation_encoder.state_projection.in_features == (8 if indices is None else 3)
    assert loaded.config.dit.action_dim == (8 if actions is None else 2)
    resume = ["awbc-train", "--resume", str(checkpoint), "--set", "training.max_updates=null"]
    monkeypatch.setattr("sys.argv", resume + ["--set", "dataset.state_indices=[5,0,7]"])
    with pytest.raises(ValueError, match="Resume state_indices"):
        policy_main()
    monkeypatch.setattr("sys.argv", resume + ["--set", "dataset.action_indices=[1,4]"])
    with pytest.raises(ValueError, match="Resume action_indices"):
        policy_main()
    monkeypatch.setattr("sys.argv", resume)
    policy_main()
    assert torch.load(checkpoint, weights_only=False)["global_step"] == 2


@pytest.mark.parametrize(("main", "indices"), [(iql_main, INDICES), (dense_main, INDICES), (iql_main, None)])
def test_critic_checkpoint_and_export(distinct_states, tmp_path, main, indices):
    endings = jsonl(tmp_path / "ends.jsonl", [
        dict(episode_index=e, success=True, end_type="terminated", end_frame_index=13) for e in range(2)
    ])
    settings = {"root": str(distinct_states), "stratify_key": None, "state_indices": indices,
                "action_indices": ACTION_INDICES if indices is not None else None}
    images = {"size": [24, 32]}
    values = {
        "dataset": settings, "image_preprocess": images, "labels_path": str(endings),
        "action_horizon": 4, "num_steps": 1, "batch_size": 2, "pretrained": False,
        "hidden_dims": [8], "device": "cpu", "logging": {"use_wandb": False, "progress_bar": False},
    }
    if main is dense_main:
        values["progress_path"] = str(jsonl(tmp_path / "progress.jsonl", [
            progress_row(e, frame, progress) for e in range(2) for frame, progress in ((0, 0.2), (13, 0.9))
        ]))
    config_path = tmp_path / "critic.json"
    config_path.write_text(json.dumps(values))
    run = tmp_path / "critic"
    main(["train", "--config", str(config_path), "--run-dir", str(run)])
    checkpoint = run / "checkpoints/latest_checkpoint.pt"
    state = torch.load(checkpoint, weights_only=False)
    expected = SELECTION if indices is not None else {"state_indices": None, "raw_state_dim": 8, "state_dim": 8}
    assert state["model_config"]["state_dim"] == expected["state_dim"]
    assert state["data_contract"]["state_selection"] == expected
    expected_actions = ACTION_SELECTION if indices is not None else {
        "action_indices": None, "raw_action_dim": 8, "action_dim": 8,
    }
    assert state["model_config"]["action_dim"] == expected_actions["action_dim"]
    assert state["data_contract"]["action_selection"] == expected_actions
    if indices is None:
        state["data_contract"].pop("state_selection")
        state["data_contract"].pop("action_selection")
        state["config"]["dataset"].pop("state_indices")
        state["config"]["dataset"].pop("action_indices")
        torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="state_selection differs"):
        main(["train", "--resume", str(checkpoint), "--set", "dataset.state_indices=[5,0,7]"])
    with pytest.raises(ValueError, match="action_selection differs"):
        main(["train", "--resume", str(checkpoint), "--set", "dataset.action_indices=[1,4]"])

    output = run / "advantages.jsonl"
    main(["export", "--checkpoint", str(checkpoint), "--output", str(output)])
    metadata_path = output.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["data_contract"]["state_selection"] == expected
    assert metadata["data_contract"]["action_selection"] == expected_actions
    train, _, _, split = prepare_datasets(DatasetConfig(**settings), ImagePreprocessConfig(**images), action_horizon=4)
    labels = AdvantageWeightingConfig(enabled=True, labels_path=str(output), metadata_path=str(metadata_path))
    weighted, _, _ = prepare_iql_advantage_dataset(train, labels, 4, split)
    assert len(weighted) == len(train)
    if indices is not None:
        for critic_state in ({**expected, "state_indices": [5, 0, 7]}, None):
            if critic_state is None:
                metadata["data_contract"].pop("state_selection")
            else:
                metadata["data_contract"]["state_selection"] = critic_state
            metadata_path.write_text(json.dumps(metadata))
            with pytest.raises(ValueError, match="critic state selection"):
                prepare_iql_advantage_dataset(train, labels, 4, split)
        metadata["data_contract"]["state_selection"] = expected
        metadata["data_contract"]["action_selection"] = {**expected_actions, "action_indices": [1, 4]}
        metadata_path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="critic action selection"):
            prepare_iql_advantage_dataset(train, labels, 4, split)
