import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from awbc_training.config import SimEvalConfig
from awbc_training.runtime import rng_state
from awbc_training.sim_validation import evaluate_snapshot
from test_training import trainer


def enable(run):
    run.config.sim_eval = SimEvalConfig(enabled=True, update_interval=2, episodes=2)
    return run


def test_schedules_after_updates_and_resumes_without_duplicates(tmp_path, monkeypatch):
    calls = []

    def evaluate(run):
        calls.append((run.global_step, run.ema.num_updates))
        assert all(parameter.grad is None for parameter in run.policy.parameters())
        return {"sim/complete": True, "sim/success_rate": 0.5}

    monkeypatch.setattr("awbc_training.trainer.evaluate_snapshot", evaluate)
    first = enable(trainer(tmp_path, "first", num_steps=4, max_updates=2))
    first.config.sim_eval.start_step = 2
    first.fit()
    checkpoint = first.directory / "checkpoints/latest_checkpoint.pt"
    state = torch.load(checkpoint, weights_only=False)
    first.logger.close()
    assert state["last_sim_eval_step"] == 2 and calls == [(2, 2)]
    resumed = enable(trainer(tmp_path, "resumed", num_steps=4, max_updates=4, state=state))
    resumed.config.sim_eval.start_step = 2
    resumed.fit()
    resumed.logger.close()
    assert calls == [(2, 2), (4, 4)]


def test_final_step_runs_once_without_changing_training(tmp_path, monkeypatch):
    monkeypatch.setattr("awbc_training.trainer.evaluate_snapshot",
                        lambda run: {"sim/complete": False, "sim/success_rate": None})
    torch.manual_seed(40)
    plain = trainer(tmp_path, "plain", num_steps=3, max_updates=3)
    plain.fit()
    torch.manual_seed(40)
    evaluated = enable(trainer(tmp_path, "evaluated", num_steps=3, max_updates=3))
    evaluated.fit()
    assert evaluated.last_sim_eval_step == 3
    for key, parameter in plain.policy.state_dict().items():
        torch.testing.assert_close(parameter, evaluated.policy.state_dict()[key], rtol=0, atol=0)
    for run in (plain, evaluated):
        run.logger.close()


@pytest.mark.parametrize(("start", "expected"), [(2, [2, 4, 5]), (6, [])])
def test_optional_start_step_and_final_evaluation(tmp_path, monkeypatch, start, expected):
    calls = []

    def evaluate(run):
        calls.append(run.global_step)
        return {"sim/complete": True, "sim/success_rate": 0.5}

    monkeypatch.setattr("awbc_training.trainer.evaluate_snapshot", evaluate)
    run = enable(trainer(tmp_path, "delayed", num_steps=5, max_updates=5))
    run.config.sim_eval.start_step = start
    run.fit()
    run.logger.close()
    assert calls == expected


@pytest.mark.parametrize("failure", [None, "timeout", "incomplete"])
def test_snapshot_process_arguments_and_errors(tmp_path, monkeypatch, failure):
    run = enable(trainer(tmp_path, "train", max_updates=1))
    run.global_step = 5
    before_rng = rng_state()["torch"]
    before = {key: value.clone() for key, value in run.policy.state_dict().items()}

    def launch(command, *, stdout, stderr, timeout, check):
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        snapshot = torch.load(checkpoint, weights_only=False)
        assert snapshot["global_step"] == 5 and snapshot["ema"] is not None
        assert command[command.index("--weights") + 1] == "ema"
        assert command[command.index("--seed") + 1] == "10000"
        assert timeout == 3600 and check is False
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        (output / "summary.json").write_text(json.dumps({
            "complete": failure != "incomplete", "successes": 1, "completed_episodes": 2,
            "success_rate": 0.5 if failure is None else None,
        }))
        return SimpleNamespace(returncode=0 if failure is None else 1)

    monkeypatch.setattr("awbc_training.sim_validation.subprocess.run", launch)
    try:
        result = evaluate_snapshot(run)
        assert result["sim/complete"] == (failure is None)
        assert result["sim/success_rate"] == (0.5 if failure is None else None)
        directory = Path(result["sim/result_directory"])
        assert (directory / "evaluation.json").is_file()
        assert (directory / "stdout.log").is_file() and (directory / "stderr.log").is_file()
        assert torch.equal(before_rng, rng_state()["torch"])
        assert run.policy.training
        for key, value in run.policy.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        repeated = evaluate_snapshot(run)
        assert repeated["sim/result_directory"] != result["sim/result_directory"]
    finally:
        run.logger.close()


def test_config_and_missing_ema(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        SimEvalConfig(enabled=True, update_interval=0)
    run = enable(trainer(tmp_path, "missing_ema"))
    run.ema = None
    result = evaluate_snapshot(run)
    assert not result["sim/complete"] and result["sim/success_rate"] is None
    report = json.loads((Path(result["sim/result_directory"]) / "evaluation.json").read_text())
    assert "ema.enabled" in report["error"]["message"]
    run.logger.close()


def test_python_symlink_is_preserved(tmp_path, monkeypatch):
    base_python = tmp_path / "base-python"
    base_python.touch()
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    monkeypatch.setenv("HOME", str(tmp_path))
    run = enable(trainer(tmp_path, "symlink"))
    run.config.sim_eval.python = "~/venv/bin/python"
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("awbc_training.sim_validation.subprocess.run", launch)
    try:
        evaluate_snapshot(run)
        assert commands[0][0] == str(venv_python)
        assert commands[0][0] != str(base_python)
    finally:
        run.logger.close()


@pytest.mark.parametrize("exit_code", [1, 0])
def test_missing_summary_reports_process_failure_first(tmp_path, monkeypatch, exit_code):
    run = enable(trainer(tmp_path, "missing_summary"))
    monkeypatch.setattr("awbc_training.sim_validation.subprocess.run",
                        lambda command, **kwargs: SimpleNamespace(returncode=exit_code))
    try:
        result = evaluate_snapshot(run)
        directory = Path(result["sim/result_directory"])
        report = json.loads((directory / "evaluation.json").read_text())
        assert not result["sim/complete"] and result["sim/success_rate"] is None
        assert not (directory / "results/summary.json").exists()
        message = report["error"]["message"]
        if exit_code:
            assert report["error"]["type"] == "RuntimeError"
            assert "code 1" in message
            assert str(directory / "stderr.log") in message
        else:
            assert report["error"]["type"] == "FileNotFoundError"
            assert "summary.json is missing" in message and "code 0" in message
    finally:
        run.logger.close()


def test_real_child_process_loads_snapshot_without_mujoco(tmp_path, monkeypatch):
    import sys
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1] / "src"))

    run = enable(trainer(tmp_path, "child"))
    script = tmp_path / "child_eval.py"
    script.write_text(
        "import json, pathlib, sys, torch\n"
        "from awbc_training.inference import load_policy\n"
        "args = sys.argv[1:]\n"
        "checkpoint = args[args.index('--checkpoint') + 1]\n"
        "policy = load_policy(checkpoint, use_ema=True, device='cpu')\n"
        "assert not policy.training\n"
        "output = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "output.mkdir()\n"
        "(output / 'summary.json').write_text(json.dumps(dict(complete=True, successes=1, "
        "completed_episodes=2, success_rate=0.5)))\n"
    )
    # A small Python launcher delegates the eval command to a test child process.
    launcher = tmp_path / "python"
    launcher.write_text(f"#!{sys.executable}\nimport os, sys\nos.execv(sys.executable, "
                        f"[sys.executable, {str(script)!r}, *sys.argv[3:]])\n")
    launcher.chmod(0o755)
    from test_advantage_weighting import tiny_policy
    from awbc_training.ema import EMA

    run.policy, run.stats = tiny_policy()
    run.ema = EMA(run.policy)
    run.config.sim_eval.python = str(launcher)
    result = evaluate_snapshot(run)
    assert result["sim/complete"] and result["sim/success_rate"] == 0.5
    run.logger.close()
