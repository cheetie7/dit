"""Run a frozen actor snapshot in the simulator's separate Python process."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from time import perf_counter

from .checkpoint import atomic_save


SIM_METRICS = ("sim/success_rate", "sim/successes", "sim/completed_episodes", "sim/complete",
               "sim/evaluation_seconds")


def evaluate_snapshot(trainer):
    config = trainer.config.sim_eval
    parent = trainer.directory / "sim_eval"
    parent.mkdir(exist_ok=True)
    directory = parent / f"step_{trainer.global_step:08d}"
    # A crash between evaluation and training-checkpoint save leaves a result on disk.
    # Keep it as a separate attempt when replaying the saved training cursor.
    attempt = 1
    while directory.exists():
        directory = parent / f"step_{trainer.global_step:08d}_attempt_{attempt:02d}"
        attempt += 1
    directory.mkdir()
    checkpoint = directory / "actor.pt"
    started = perf_counter()
    summary = {"complete": False, "success_rate": None, "successes": 0, "completed_episodes": 0}
    command = []
    try:
        if config.weights == "ema" and trainer.ema is None:
            raise ValueError("sim_eval.weights=ema requires ema.enabled=true")
        atomic_save({
            "policy": trainer.policy.state_dict(),
            "ema": trainer.ema.state_dict() if trainer.ema else None,
            "policy_config": asdict(trainer.policy.config),
            "normalization_stats": trainer.stats, "global_step": trainer.global_step,
        }, checkpoint)
        command = [
            str(Path(config.python).resolve()), "-m", "awbc_training.eval_sim",
            "--checkpoint", str(checkpoint.resolve()), "--weights", config.weights,
            "--mapping", str(Path(config.mapping).resolve()),
            "--simulation-root", str(Path(config.simulation_root).resolve()),
            "--dataset-root", str(Path(trainer.config.dataset.root).resolve()),
            "--device", config.device, "--seed", str(config.seed), "--noise-seed", str(config.noise_seed),
            "--episodes", str(config.episodes), "--max-control-ticks", str(config.max_control_ticks),
            "--output-dir", str((directory / "results").resolve()),
        ]
        with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=config.timeout_seconds, check=False)
        result = json.loads((directory / "results/summary.json").read_text())
        if process.returncode != 0 or not result["complete"]:
            raise RuntimeError(f"Simulation evaluation incomplete (exit={process.returncode}); see {directory}")
        if (result["completed_episodes"] != config.episodes
                or not 0 <= result["successes"] <= config.episodes
                or result["success_rate"] != result["successes"] / config.episodes):
            raise ValueError("Simulation summary counts and success rate are inconsistent")
        summary.update(result)
    except Exception as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
    summary.update(global_step=trainer.global_step, evaluation_seconds=perf_counter() - started,
                   command=command, weights=config.weights)
    (directory / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return {**{f"sim/{key}": summary[key] for key in ("complete", "success_rate", "successes", "completed_episodes",
                                                   "evaluation_seconds")},
            "sim/result_directory": str(directory.resolve())}
