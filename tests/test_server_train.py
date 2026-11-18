import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import torch

from awbc_training.config import ExperimentConfig

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("server_train", REPO / "tools/server_train.py")
server_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_train)


def settings_with_checkpoint(tmp_path, mode="dense"):
    settings = json.loads((REPO / "configs/server_train.json").read_text())
    checkpoint = tmp_path / "critic.pt"
    dataset = {"root": "/previous/data", "train_stride": 2, "state_indices": [7, 0],
               "action_indices": [4, 1], "split_seed": 32, "max_train_samples": 12}
    torch.save({"config": {"dataset": dataset}, "model_config": {"action_horizon": 4},
                "data_contract": {} if mode == "legacy" else {"reward_mode": mode}}, checkpoint)
    settings["critic_checkpoint"] = str(checkpoint)
    settings["runtime"]["output_root"] = str(tmp_path / "runs")
    return settings, dataset


@pytest.mark.parametrize("mode", ["sparse", "dense", "legacy"])
def test_export_then_bc_uses_checkpoint_windows(tmp_path, monkeypatch, mode):
    settings, dataset = settings_with_checkpoint(tmp_path, mode)
    if mode != "dense":
        settings["data"].pop("progress_path")
    path = tmp_path / "server.json"
    path.write_text(json.dumps(settings))
    monkeypatch.setattr("sys.argv", ["server_train", "--config", str(path)])
    calls = []
    monkeypatch.setattr(server_train.subprocess, "run", lambda command, **kwargs: calls.append(command))
    server_train.main()
    assert len(calls) == 2
    expected_module = "awbc_training.iql_dense" if mode == "dense" else "awbc_training.iql"
    assert calls[0][2:4] == [expected_module, "export"]
    assert any(arg.startswith("progress_path=") for arg in calls[0]) == (mode == "dense")
    assert calls[0][calls[0].index("--checkpoint") + 1] == settings["critic_checkpoint"]
    assert calls[1][2] == "awbc_training.train"
    policy_path = Path(calls[1][calls[1].index("--config") + 1])
    policy = ExperimentConfig.from_dict(json.loads(policy_path.read_text()))
    assert policy.dataset.state_indices == dataset["state_indices"]
    assert policy.dataset.action_indices == dataset["action_indices"]
    assert policy.dataset.train_stride == 2 and policy.dataset.split_seed == 32
    assert policy.dataset.root == settings["data"]["root"]
    assert policy.policy["dit"]["action_horizon"] == policy.policy["n_action_steps"] == 4
    assert policy.advantage_weighting.labels_path == str(policy_path.parent / "advantages.jsonl")
    assert policy.sim_eval.start_step == 10000
    assert policy.training.objective == "iql_fm"
    assert policy.advantage_weighting.normalization == "none"
    assert policy.advantage_weighting.beta == 1.0


def test_failed_export_stops_bc(tmp_path, monkeypatch):
    settings, _ = settings_with_checkpoint(tmp_path)
    path = tmp_path / "server.json"
    path.write_text(json.dumps(settings))
    monkeypatch.setattr("sys.argv", ["server_train", "--config", str(path)])
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(server_train.subprocess, "run", execute)
    with pytest.raises(subprocess.CalledProcessError):
        server_train.main()
    assert len(calls) == 1 and calls[0][3] == "export"
