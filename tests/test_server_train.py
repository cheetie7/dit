import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from awbc_training.config import ExperimentConfig
from awbc_training.iql_dense import IQLDenseConfig


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("server_train", REPO / "tools/server_train.py")
server_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_train)


def test_generated_configs_share_windows_labels_and_simulator(tmp_path):
    settings = json.loads((REPO / "configs/server_train.json").read_text())
    settings["data"].update(action_horizon=4)
    settings["data"]["dataset"]["train_stride"] = 2
    critic, policy = server_train.prepare_configs(settings, REPO, tmp_path)
    dense = IQLDenseConfig.from_dict(critic)
    actor = ExperimentConfig.from_dict(policy)
    assert dense.dataset == actor.dataset
    assert dense.action_horizon == actor.policy["dit"]["action_horizon"] == 4
    assert actor.policy["n_action_steps"] == 4
    assert actor.advantage_weighting.labels_path == str(tmp_path / "critic/advantages.jsonl")
    assert actor.dense_awbc.progress_path == dense.progress_path
    assert actor.sim_eval.python == settings["runtime"]["python"]
    assert actor.sim_eval.enabled and not actor.dense_awbc.robometer_decay_enabled


def test_pipeline_stops_before_policy_when_export_fails(tmp_path, monkeypatch):
    settings = json.loads((REPO / "configs/server_train.json").read_text())
    settings["runtime"]["output_root"] = str(tmp_path / "runs")
    path = tmp_path / "server.json"
    path.write_text(json.dumps(settings))
    monkeypatch.setattr("sys.argv", ["server_train", "--config", str(path)])
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(server_train.subprocess, "run", execute)
    with pytest.raises(subprocess.CalledProcessError):
        server_train.main()
    assert len(calls) == 2
    assert calls[0][3] == "train" and calls[1][3] == "export"
