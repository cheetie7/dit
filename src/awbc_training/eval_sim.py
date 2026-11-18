"""Evaluate a DiT actor synchronously in simulation-simple's block insertion task."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from .inference import load_policy
from .sim_mapping import SimulationMapping, action_repeat


def _error(stage, exception):
    return {"stage": stage, "type": type(exception).__name__, "message": str(exception)}


def rollout(env, policy, mapping, *, seed, noise_seed, action_fps, max_control_ticks):
    """Run one episode using the same reset targets and RNG throughout."""
    result = {"seed": seed, "noise_seed": noise_seed, "success": False, "control_ticks": 0, "inference_calls": 0}
    stage = "reset"
    try:
        observation, info = env.reset(seed=seed)
        target = mapping.reset_target(env)
        control_fps = float(env.control_frequency_hz)
        repeat = action_repeat(action_fps, control_fps)
        result.update(control_fps=control_fps, ticks_per_action=repeat)
        if info.get("success", False):
            return dict(result, success=True, termination_reason="success")

        rng = torch.Generator(device="cpu").manual_seed(noise_seed)
        while result["control_ticks"] < max_control_ticks:
            stage = "map_observation"
            raw = mapping.observation(env, observation)
            stage = "predict_action"
            result["inference_calls"] += 1
            with torch.inference_mode():
                prepared = policy.prepare_observation(raw)
                initial_noise = torch.randn((1, mapping.action_horizon, len(mapping.action_slots)), generator=rng)
                prediction = policy.predict_action(prepared, initial_noise=initial_noise)
            actions = prediction["action"].detach().to(device="cpu", dtype=torch.float32).numpy()
            shape = (1, mapping.n_action_steps, len(mapping.action_slots))
            if actions.shape != shape or not np.isfinite(actions).all():
                raise ValueError(f"Policy action chunk must be finite with shape {shape}, got {actions.shape}")
            for action in actions[0]:
                command = mapping.command(target, action)
                stage = "step"
                for _ in range(repeat):
                    observation, _, terminated, truncated, info = env.step(command)
                    result["control_ticks"] += 1
                    if info["success"]:
                        return dict(result, success=True, termination_reason="success")
                    if terminated or truncated:
                        return dict(result, termination_reason="terminated" if terminated else "truncated")
                    if result["control_ticks"] >= max_control_ticks:
                        return dict(result, termination_reason="max_control_ticks")
        return dict(result, termination_reason="max_control_ticks")
    except Exception as exception:
        return dict(result, success=None, termination_reason="error", error=_error(stage, exception))


def summarize_rollouts(records, errors, requested_episodes):
    completed = [row for row in records if row["termination_reason"] != "error"]
    successes = sum(row["success"] for row in completed)
    complete = len(completed) == requested_episodes and not errors
    return {"complete": complete, "requested_episodes": requested_episodes, "attempted_episodes": len(records),
            "completed_episodes": len(completed), "successes": successes,
            "success_rate": successes / len(completed) if complete and completed else None, "error_count": len(errors)}


def resolve_action_fps(explicit, checkpoint, dataset_info):
    dataset = dataset_info.get("fps") if dataset_info is not None else None
    sources = {name: rate for name, rate in (("--action-fps", explicit), ("checkpoint action_fps", checkpoint),
                                             ("dataset fps", dataset)) if rate is not None}
    if not sources:
        raise ValueError("Checkpoint has no action_fps. Provide --dataset-root with meta/info.json or --action-fps")
    fps = next(iter(sources.values()))
    if any(not math.isfinite(rate) or rate <= 0 for rate in sources.values()):
        raise ValueError(f"Action frequencies must be finite and positive, got {sources}")
    if any(not math.isclose(rate, fps, rel_tol=0, abs_tol=1e-6) for rate in sources.values()):
        raise ValueError(f"Action frequency sources disagree: {sources}")
    return float(fps)


def create_environment(simulation_root, mapping):
    root = Path(simulation_root).resolve()
    if not (root / "mujoco_sim/envs/block_insertion.py").is_file():
        raise FileNotFoundError(f"simulation-simple block_insertion checkout not found at {root}")
    sys.path.insert(0, str(root))
    try:
        from mujoco_sim.envs.block_insertion import BlockInsertionEnv

        # Render at inference boundaries. Physics advances solely in step().
        return BlockInsertionEnv(image_width=mapping.width, image_height=mapping.height,
                                 include_camera_obs=False, include_depth_obs=False)
    finally:
        sys.path.pop(0)


def _positive_int(value):
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def _write_jsonl(path, records):
    with path.open("w") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mapping", type=Path, required=True, help="JSON observation and action mapping")
    parser.add_argument("--dataset-root", type=Path, help="Training dataset with meta/info.json")
    parser.add_argument("--action-fps", type=float, help="Training action frequency when metadata is unavailable")
    parser.add_argument("--simulation-root", type=Path, default=Path("external/simulation-simple"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--episodes", type=_positive_int, default=10)
    parser.add_argument("--max-control-ticks", type=_positive_int, default=2500)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.seed, args.noise_seed) < 0:
        parser.error("seed and noise-seed must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records, errors = [], []
    frequency = None
    env = None
    try:
        stage = "load_policy"
        policy = load_policy(args.checkpoint, use_ema=args.weights == "ema", device=args.device)
        stage = "evaluation_mapping"
        mapping_json = json.loads(args.mapping.read_text())
        dataset_info = json.loads((args.dataset_root / "meta/info.json").read_text()) if args.dataset_root else None
        mapping = SimulationMapping(policy.config, mapping_json, dataset_info)
        frequency = resolve_action_fps(args.action_fps, policy.config.action_fps, dataset_info)
        stage = "environment_setup"
        env = create_environment(args.simulation_root, mapping)
        mapping.validate_environment(env)
        action_repeat(frequency, env.control_frequency_hz)
        for offset in range(args.episodes):
            record = rollout(env, policy, mapping, seed=args.seed + offset, noise_seed=args.noise_seed + offset,
                             action_fps=frequency, max_control_ticks=args.max_control_ticks)
            records.append(record)
            if "error" in record:
                errors.append({**record["error"], "seed": record["seed"], "noise_seed": record["noise_seed"]})
                break
    except Exception as exception:
        errors.append(_error(stage, exception))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exception:
                errors.append(_error("environment_close", exception))

    summary = summarize_rollouts(records, errors, args.episodes)
    summary.update(
        task="block_insertion", checkpoint=str(args.checkpoint.resolve()), weights=args.weights, device=args.device,
        mapping=str(args.mapping.resolve()), simulation_root=str(args.simulation_root.resolve()),
        dataset_root=str(args.dataset_root.resolve()) if args.dataset_root else None,
        action_fps=frequency, seed=args.seed, noise_seed=args.noise_seed, max_control_ticks=args.max_control_ticks,
    )
    _write_jsonl(args.output_dir / "rollouts.jsonl", records)
    _write_jsonl(args.output_dir / "errors.jsonl", errors)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for error in errors:
        print(f"{error['stage']}: {error['type']}: {error['message']}", file=sys.stderr)
    print(json.dumps(summary, allow_nan=False))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
