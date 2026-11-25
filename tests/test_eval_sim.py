import json
from pathlib import Path

import numpy as np
import pytest
import torch

from awbc_models import FlowMatchingPolicyConfig
from awbc_training import eval_sim
from awbc_training.sim_mapping import (
    ACTION_CONTRACT, ACTION_NAMES, ENV_ACTION_ORDER, STATE_FIELDS, STATE_NAMES, SimulationMapping, action_repeat,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/eval_sim_block_insertion.json"
CAMERAS = ("observation.images.eye", "observation.images.left_wrist", "observation.images.right_wrist")
RGB_KEYS = ("eye_rgb", "left_wrist_rgb", "right_wrist_rgb")


class FakeEnv:
    policy_image_keys = RGB_KEYS
    control_frequency_hz = 100

    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.seeds = []
        self.commands = []
        self.render_ticks = []
        self.closed = False
        self.ticks = 0

    def action_metadata(self):
        return {**ACTION_CONTRACT, "order": ENV_ACTION_ORDER, "shape": [16]}

    def reset(self, *, seed):
        self.seeds.append(seed)
        self.seed = seed
        self.ticks = 0
        self.last_applied_action = np.arange(16, dtype=np.float32) + seed
        return self.observe(), {"success": False}

    def observe(self):
        return {field: np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 100 * index + self.ticks
                for index, (field, (shape, _)) in enumerate(STATE_FIELDS.items())}

    def sampled_camera_observations(self, *, width, height, include_depth, force):
        assert (height, width) == (4, 6) and force and not include_depth
        self.render_ticks.append(self.ticks)
        return {key: np.full((height, width, 3), level, dtype=np.uint8)
                for key, level in zip(RGB_KEYS, (10, 20, 30), strict=True)}

    def step(self, action):
        self.commands.append(action.copy())
        self.ticks += 1
        self.last_applied_action = action.copy() + 1000  # Subsequent targets must still use the reset copy.
        outcome = self.outcomes.get((self.seed, self.ticks))
        if outcome == "error":
            raise RuntimeError("physics failed")
        return self.observe(), 0., outcome == "terminated", outcome == "truncated", {"success": outcome == "success"}

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.observations = []
        self.noises = []
        self.actions = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 10 + 0.2

    def prepare_observation(self, raw):
        self.observed_tick = self.env.ticks
        self.observations.append(raw)
        return raw

    def predict_action(self, prepared, *, initial_noise):
        assert self.env.ticks == self.observed_tick
        self.noises.append(initial_noise.clone())
        return {"action": self.actions, "action_normalized": torch.full_like(self.actions, -99.)}


@pytest.fixture
def evaluation():
    config = FlowMatchingPolicyConfig.from_dict({
        "camera_names": CAMERAS, "observation": {"state_dim": 8},
        "state_indices": [0, 1, 2, 3, 4, 5, 6, 28], "raw_state_dim": 44,
        "dit": {"action_horizon": 4, "action_dim": 8},
        "action_indices": list(range(8)), "raw_action_dim": 16, "n_action_steps": 2,
    })
    mapping = json.loads(CONFIG_PATH.read_text())
    mapping["image_size"] = [4, 6]
    metadata = {"fps": 25, "features": {
        **{camera: {"dtype": "video", "shape": [4, 6, 3]} for camera in CAMERAS},
        "observation.state": {"shape": [44], "names": STATE_NAMES},
        "action": {"shape": [16], "names": ACTION_NAMES},
    }}
    return config, mapping, metadata


def test_raw_mapping_uses_simulation_order_and_checkpoint_slots(evaluation):
    config, mapping, metadata = evaluation
    # Reorder both configuration entries and selected actions to exercise the explicit mapping.
    mapping["camera_map"] = dict(reversed(list(mapping["camera_map"].items())))
    config.action_indices = list(reversed(config.action_indices))
    env = FakeEnv()
    obs, _ = env.reset(seed=17)
    adapter = SimulationMapping(config, mapping, metadata)
    adapter.validate_environment(env)
    raw = adapter.observation(env, obs)
    assert list(raw["images"]) == list(CAMERAS)
    assert [frame[0, 0, 0] for frame in raw["images"].values()] == [10, 20, 30]
    expected = np.concatenate([obs[key].reshape(-1) for key in STATE_FIELDS])
    np.testing.assert_array_equal(raw["state"], expected)
    reset_target = adapter.reset_target(env)
    action = np.arange(8, dtype=np.float32)
    target = adapter.command(reset_target, action)
    np.testing.assert_array_equal(target[:8], action[::-1])
    np.testing.assert_array_equal(target[8:], np.arange(8, 16) + 17)


@pytest.mark.parametrize(("change", "message"), [
    ("camera", "camera_map keys"),
    ("state", "raw_state_dim=44"),
    ("actions", "raw_action_dim=16"),
    ("side", "controlled_arms"),
    ("units", "action_contract"),
    ("metadata", "metadata observation.state names/order"),
])
def test_rejects_unmatched_mapping(evaluation, change, message):
    config, mapping, metadata = evaluation
    if change == "camera":
        mapping["camera_map"].pop(CAMERAS[0])
    elif change == "state":
        mapping["state_names"] = STATE_NAMES[:-1]
    elif change == "actions":
        mapping["action_names"] = ACTION_NAMES[:8]
    elif change == "side":
        mapping["controlled_arms"] = ["right"]
    elif change == "units":
        mapping["action_contract"]["arm_unit"] = "degree"
    elif change == "metadata":
        metadata["features"]["observation.state"]["names"] = STATE_NAMES[::-1]
    with pytest.raises(ValueError, match=message):
        SimulationMapping(config, mapping, metadata)


def test_frequency_sources_and_integer_control_ratio():
    assert action_repeat(25, 100) == 4
    with pytest.raises(ValueError, match="100 / 30 Hz"):
        action_repeat(30, 100)
    with pytest.raises(ValueError, match="finite and positive"):
        action_repeat(0, 100)
    assert eval_sim.resolve_action_fps(None, None, {"fps": 25}) == 25
    assert eval_sim.resolve_action_fps(None, 25, None) == 25
    with pytest.raises(ValueError, match="sources disagree"):
        eval_sim.resolve_action_fps(30, 25, None)
    with pytest.raises(ValueError, match="Provide --dataset-root"):
        eval_sim.resolve_action_fps(None, None, None)


@pytest.mark.parametrize(("outcome", "tick", "budget", "reason", "calls"), [
    ("success", 3, 20, "success", 1),
    ("terminated", 2, 20, "terminated", 1),
    ("truncated", 6, 20, "truncated", 1),
    (None, 6, 6, "max_control_ticks", 1),
    (None, 10, 10, "max_control_ticks", 2),
])
def test_rollout_checks_every_tick(evaluation, outcome, tick, budget, reason, calls):
    config, mapping, metadata = evaluation
    env = FakeEnv({(19, tick): outcome})
    policy = FakePolicy(config, env)
    adapter = SimulationMapping(config, mapping, metadata)
    result = eval_sim.rollout(env, policy, adapter, seed=19, noise_seed=50, action_fps=25, max_control_ticks=budget)
    assert result["control_ticks"] == tick == len(env.commands)
    assert result["termination_reason"] == reason and result["success"] == (outcome == "success")
    assert result["ticks_per_action"] == 4
    assert result["inference_calls"] == calls == len(policy.noises)
    assert env.render_ticks == [0, 8][:calls]
    for index, command in enumerate(env.commands):
        np.testing.assert_array_equal(command[:8], policy.actions[0, index // 4 % 2].numpy())
        np.testing.assert_array_equal(command[8:], np.arange(8, 16) + 19)
    assert policy.noises[0].shape == (1, 4, 8)
    if calls == 2:
        with torch.random.fork_rng():
            torch.manual_seed(133)
            noise_before = torch.get_rng_state().clone()
            eval_sim.rollout(env, policy, adapter, seed=19, noise_seed=50, action_fps=25, max_control_ticks=budget)
            assert torch.equal(torch.get_rng_state(), noise_before)
        torch.testing.assert_close(policy.noises[2], policy.noises[0], rtol=0, atol=0)
        torch.testing.assert_close(policy.noises[3], policy.noises[1], rtol=0, atol=0)


@pytest.mark.parametrize("save_video", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_cli_writes_rollouts_and_handles_incomplete_evaluation(evaluation, tmp_path, monkeypatch, failure, save_video):
    config, mapping_json, dataset_info = evaluation
    env = FakeEnv({(20, 1): "success", (21, 2): "error" if failure else "truncated"})
    policy = FakePolicy(config, env)
    loads = []

    def load(checkpoint, *, use_ema, device):
        loads.append((checkpoint, use_ema, device))
        return policy

    monkeypatch.setattr(eval_sim, "load_policy", load)
    monkeypatch.setattr(eval_sim, "create_environment", lambda root, mapping: env)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping_json))
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps(dataset_info))
    output = tmp_path / "output"
    checkpoint = tmp_path / "actor.pt"
    status = eval_sim.main([
        "--checkpoint", str(checkpoint), "--mapping", str(mapping_path), "--dataset-root", str(tmp_path),
        "--weights", "ema", "--device", "cpu", "--seed", "20", "--noise-seed", "95",
        "--episodes", "2", "--output-dir", str(output),
    ] + (["--save-video"] if save_video else []))
    assert loads == [(checkpoint, True, "cpu")] and env.closed
    assert env.max_episode_steps == 2500
    assert env.seeds == [20, 21]
    records = [json.loads(row) for row in (output / "rollouts.jsonl").read_text().splitlines()]
    assert [row["noise_seed"] for row in records] == [95, 96]
    for row in records:
        if save_video and row['termination_reason'] != 'error':
            assert row['video_path'] == f"videos/seed_{row['seed']}.mp4"
            assert (output / row['video_path']).is_file()
        else:
            assert 'video_path' not in row
    if not save_video:
        assert not (output / 'videos').exists()
    summary = json.loads((output / "summary.json").read_text())
    errors = [json.loads(row) for row in (output / "errors.jsonl").read_text().splitlines()]
    assert summary["successes"] == 1 and summary["action_fps"] == 25
    if failure:
        assert status == 1 and not summary["complete"] and summary["completed_episodes"] == 1
        assert summary["success_rate"] is None and summary["error_count"] == 1
        assert errors == [{"stage": "step", "type": "RuntimeError", "message": "physics failed",
                           "seed": 21, "noise_seed": 96}]
    else:
        assert status == 0 and summary["complete"] and summary["completed_episodes"] == 2
        assert summary["success_rate"] == 0.5 and not errors


def test_inference_exception_is_recorded(evaluation, monkeypatch):
    config, mapping, metadata = evaluation
    env = FakeEnv()
    policy = FakePolicy(config, env)

    def fail(*args, **kwargs):
        raise FloatingPointError("invalid prediction")

    monkeypatch.setattr(policy, "predict_action", fail)
    result = eval_sim.rollout(env, policy, SimulationMapping(config, mapping, metadata),
                              seed=0, noise_seed=0, action_fps=25, max_control_ticks=2)
    assert result["inference_calls"] == 1 and result["control_ticks"] == 0
    assert result["error"]["stage"] == "predict_action"


def test_cli_help_exits_before_loading_simulation(monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("help must not load the model or simulator")

    monkeypatch.setattr(eval_sim, "create_environment", fail)
    monkeypatch.setattr(eval_sim, "load_policy", fail)
    with pytest.raises(SystemExit) as status:
        eval_sim.main(["--help"])
    assert status.value.code == 0 and "--max-control-ticks" in capsys.readouterr().out


@pytest.mark.parametrize(('outcome', 'tick', 'expected'), [
    ('success', 3, [0, 3]), ('terminated', 2, [0, 2]),
    ('truncated', 6, [0, 4, 6]), (None, 10, [0, 4, 8, 10]),
    ('success', 4, [0, 4]),
])
def test_video_decodes_action_frames_and_final_frame(evaluation, tmp_path, outcome, tick, expected):
    import av

    config, mapping, metadata = evaluation

    class VideoEnv(FakeEnv):
        def sampled_camera_observations(self, **kwargs):
            images = super().sampled_camera_observations(**kwargs)
            images['right_wrist_rgb'][:] = self.ticks * 20
            return images

    env = VideoEnv({(19, tick): outcome})
    path = tmp_path / 'videos/seed_19.mp4'
    result = eval_sim.rollout(env, FakePolicy(config, env), SimulationMapping(config, mapping, metadata),
                              seed=19, noise_seed=50, action_fps=25, max_control_ticks=tick,
                              video_path=path, video_camera='right_wrist_rgb')
    assert result['termination_reason'] != 'error'
    with av.open(str(path)) as reader:
        assert float(reader.streams.video[0].average_rate) == 25
        frames = [f.to_ndarray(format='rgb24') for f in reader.decode(video=0)]
    assert len(frames) == len(expected)
    for frame, frame_tick in zip(frames, expected, strict=True):
        assert abs(float(frame.mean()) - frame_tick * 20) <= 3
    assert not list(path.parent.glob('*.partial.mp4'))


@pytest.mark.parametrize('failure', ['inference', 'capture', 'finalize'])
def test_video_failure_removes_partial_file(evaluation, tmp_path, monkeypatch, failure):
    config, mapping, metadata = evaluation
    env = FakeEnv()
    policy = FakePolicy(config, env)

    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')

    if failure == 'inference':
        monkeypatch.setattr(policy, 'predict_action', fail)
    elif failure == 'capture':
        monkeypatch.setattr(eval_sim.EpisodeVideo, 'capture', fail)
    else:
        monkeypatch.setattr(eval_sim.EpisodeVideo, 'finish', fail)
    path = tmp_path / 'videos/seed_0.mp4'
    result = eval_sim.rollout(env, policy, SimulationMapping(config, mapping, metadata),
                              seed=0, noise_seed=0, action_fps=25, max_control_ticks=2, video_path=path)
    assert result['termination_reason'] == 'error'
    assert result['error']['stage'] == {'inference': 'predict_action', 'capture': 'video_capture',
                                        'finalize': 'video_finalize'}[failure]
    assert not list(path.parent.iterdir())


def test_invalid_video_camera_fails_before_reset(evaluation, tmp_path):
    config, mapping, metadata = evaluation
    env = FakeEnv()
    result = eval_sim.rollout(env, FakePolicy(config, env), SimulationMapping(config, mapping, metadata),
                              seed=0, noise_seed=0, action_fps=25, max_control_ticks=2,
                              video_path=tmp_path / 'video.mp4', video_camera='missing_rgb')
    assert not env.seeds
    assert result['error']['stage'] == 'video_setup'
    assert 'Available cameras: eye_rgb, left_wrist_rgb, right_wrist_rgb' in result['error']['message']


def test_video_reset_success_saves_single_frame(evaluation, tmp_path):
    import av

    config, mapping, metadata = evaluation

    class ResetSuccessEnv(FakeEnv):
        def reset(self, **kwargs):
            observation, _ = super().reset(**kwargs)
            return observation, {'success': True}

    env = ResetSuccessEnv()
    path = tmp_path / 'reset.mp4'
    result = eval_sim.rollout(env, FakePolicy(config, env), SimulationMapping(config, mapping, metadata),
                              seed=0, noise_seed=0, action_fps=25, max_control_ticks=2, video_path=path)
    assert result['success'] and not env.commands
    with av.open(str(path)) as reader:
        assert len(list(reader.decode(video=0))) == 1
