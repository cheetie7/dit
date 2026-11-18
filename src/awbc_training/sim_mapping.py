"""Explicit mappings from simulation-simple observations and absolute controls."""

import math

import numpy as np


ARMS = ("left", "right")
LAYOUT = "simulation_simple_lerobot_v3"
STATE_FIELDS = {
    "joint_pos": ((14,), [f"joint_pos_{arm}_{i}" for arm in ARMS for i in range(1, 8)]),
    "joint_vel": ((14,), [f"joint_vel_{arm}_{i}" for arm in ARMS for i in range(1, 8)]),
    "gripper_pos": ((2,), [f"gripper_pos_{arm}" for arm in ARMS]),
    "ee_pos": ((2, 3), [f"ee_pos_{arm}_{axis}" for arm in ARMS for axis in "xyz"]),
    "ee_quat": ((2, 4), [f"ee_quat_{arm}_{axis}" for arm in ARMS for axis in "wxyz"]),
}
STATE_NAMES = [name for _, names in STATE_FIELDS.values() for name in names]
ACTION_NAMES = [name for arm in ARMS for name in
                [*(f"{arm}_joint_{i}" for i in range(1, 8)), f"{arm}_gripper"]]
ACTION_CONTRACT = {"type": "absolute_joint_position", "arm_unit": "radian",
                   "gripper_unit": "actuator_coordinate"}
ENV_ACTION_ORDER = ["left_joint1..7", "left_gripper", "right_joint1..7", "right_gripper"]


def _ordered_names(mapping, feature, default):
    key = f"{feature}_names"
    names = mapping.get(key)
    if names is None and mapping.get(f"{feature}_layout") == LAYOUT:
        return default
    if not isinstance(names, list) or not names:
        raise ValueError(f"Specify {feature}_layout={LAYOUT!r} or an ordered {key} list")
    if any(not isinstance(name, str) or name not in default for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{key} must contain distinct simulation-simple {feature} names")
    return names


class SimulationMapping:
    def __init__(self, config, mapping, dataset_info=None):
        camera_map = mapping.get("camera_map", {})
        expected_cameras = tuple(config.camera_names)
        if set(camera_map) != set(expected_cameras) or len(expected_cameras) != config.vision.num_cameras:
            raise ValueError(f"camera_map keys must match checkpoint cameras {expected_cameras}, got {list(camera_map)}")
        self.cameras = {name: camera_map[name] for name in expected_cameras}
        if any(not isinstance(name, str) or not name for name in self.cameras.values()):
            raise ValueError("camera_map values must be simulation RGB field names")
        if len(set(self.cameras.values())) != len(self.cameras):
            raise ValueError("camera_map must map distinct cameras to distinct simulation RGB fields")
        size = mapping.get("image_size")
        if not isinstance(size, list) or len(size) != 2 or any(type(n) is not int or n <= 0 for n in size):
            raise ValueError("image_size must be the raw render [height, width]")
        self.height, self.width = size

        self.state_names = _ordered_names(mapping, "state", STATE_NAMES) if config.observation.use_state else []
        if config.observation.use_state and len(self.state_names) != config.raw_state_dim:
            raise ValueError(f"State mapping has {len(self.state_names)} entries, checkpoint expects raw_state_dim={config.raw_state_dim}")
        state_columns = {name: (key, index) for key, (_, names) in STATE_FIELDS.items()
                         for index, name in enumerate(names)}
        self.state_columns = [state_columns[name] for name in self.state_names]

        action_names = _ordered_names(mapping, "action", ACTION_NAMES)
        if len(action_names) != config.raw_action_dim:
            raise ValueError(f"Action mapping has {len(action_names)} entries, checkpoint expects raw_action_dim={config.raw_action_dim}")
        declared = mapping.get("action_contract")
        if declared != ACTION_CONTRACT:
            raise ValueError(f"action_contract must explicitly declare {ACTION_CONTRACT}, got {declared!r}")
        used_names = action_names if config.action_indices is None else [action_names[i] for i in config.action_indices]
        if len(used_names) != config.dit.action_dim:
            raise ValueError(f"Mapped action has {len(used_names)} slots, checkpoint expects {config.dit.action_dim}")
        arms = mapping.get("controlled_arms")
        actual_arms = sorted({name.split("_", 1)[0] for name in used_names})
        if not isinstance(arms, list) or arms != actual_arms:
            raise ValueError(f"controlled_arms={arms!r} differs from checkpoint action mapping {actual_arms}")
        self.action_slots = [ACTION_NAMES.index(name) for name in used_names]
        self.action_horizon = config.dit.action_horizon
        self.n_action_steps = config.n_action_steps

        if dataset_info is not None:
            features = dataset_info["features"]
            for name, columns in (("observation.state", self.state_names), ("action", action_names)):
                feature = features.get(name, {})
                if columns and (feature.get("names") != columns or feature.get("shape") != [len(columns)]):
                    raise ValueError(f"Training metadata {name} names/order or shape differs from the evaluation mapping")
            for name in expected_cameras:
                feature = features.get(name, {})
                if feature.get("dtype") not in ("video", "image") or feature.get("shape") != [self.height, self.width, 3]:
                    raise ValueError(f"Training metadata camera {name!r} does not match mapped RGB size {size}")

    def validate_environment(self, env):
        missing = set(self.cameras.values()) - set(env.policy_image_keys)
        if missing:
            raise ValueError(f"Simulation is missing mapped RGB fields: {sorted(missing)}")
        metadata = env.action_metadata()
        for key, expected in {**ACTION_CONTRACT, "shape": [len(ACTION_NAMES)], "order": ENV_ACTION_ORDER}.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Simulation action {key}={metadata.get(key)!r}, expected {expected!r}")

    def observation(self, env, observation):
        fields = {}
        for key in dict.fromkeys(key for key, _ in self.state_columns):
            if key not in observation:
                raise ValueError(f"Simulation observation is missing state field {key!r}")
            value = np.asarray(observation[key], dtype=np.float32)
            shape, _ = STATE_FIELDS[key]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Simulation state field {key!r} must be finite with shape {shape}, got {value.shape}")
            fields[key] = value.reshape(-1)

        images = {}
        if self.cameras:
            rendered = env.sampled_camera_observations(
                width=self.width, height=self.height, include_depth=False, force=True
            )
            for camera, key in self.cameras.items():
                if key not in rendered:
                    raise ValueError(f"Simulation did not render RGB field {key!r} for checkpoint camera {camera!r}")
                image = np.asarray(rendered[key])
                if image.dtype != np.uint8 or image.shape != (self.height, self.width, 3):
                    raise ValueError(f"Simulation RGB field {key!r} must be uint8 [{self.height},{self.width},3]")
                images[camera] = image
        raw = {"images": images}
        if self.state_columns:
            raw["state"] = np.asarray([fields[key][index] for key, index in self.state_columns], dtype=np.float32)
        return raw

    def reset_target(self, env):
        target = np.asarray(env.last_applied_action, dtype=np.float32)
        if target.shape != (len(ACTION_NAMES),) or not np.isfinite(target).all():
            raise ValueError("Simulation reset control target must be a finite 16-element action")
        return target.copy()

    def command(self, reset_target, action):
        command = reset_target.copy()
        command[self.action_slots] = action
        return command


def action_repeat(action_fps, control_fps):
    if any(not math.isfinite(rate) or rate <= 0 for rate in (action_fps, control_fps)):
        raise ValueError(f"Action and control frequencies must be finite and positive, got {action_fps}/{control_fps}")
    ticks = round(control_fps / action_fps)
    if ticks < 1 or not math.isclose(control_fps, ticks * action_fps, abs_tol=1e-6, rel_tol=0):
        raise ValueError(f"Control frequency must be an integer multiple of action frequency: {control_fps:g} / {action_fps:g} Hz")
    return ticks
