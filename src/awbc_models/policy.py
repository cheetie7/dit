"""Composition of the stage 2-5 modules with training forward and Euler inference."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from awbc_data import ImagePreprocessConfig, ImagePreprocessor
from .dit import ConditionalDiT, ConditionalDiTConfig
from .euler import sample_euler, synchronize, timed_call
from .flow_matching_objective import FlowMatchingObjective
from .normalization import MinMaxNormalizer
from .observation import ObservationEncoder, ObservationEncoderConfig
from .vision import ACTVisualEncoder, VisualEncoderConfig


@dataclass
class FlowMatchingPolicyConfig:
    vision: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    observation: ObservationEncoderConfig = field(default_factory=ObservationEncoderConfig)
    dit: ConditionalDiTConfig = field(default_factory=ConditionalDiTConfig)
    image_preprocess: ImagePreprocessConfig = field(default_factory=ImagePreprocessConfig)
    camera_names: tuple[str, ...] = ("observation.images.top_camera", "observation.images.left_camera",
                                     "observation.images.right_camera")
    normalization_stats_path: str | None = None
    num_inference_steps: int = 8
    n_action_steps: int = 8

    def __post_init__(self):
        self.camera_names = tuple(self.camera_names)
        if not 1 <= self.n_action_steps <= self.dit.action_horizon:
            raise ValueError("n_action_steps must be between 1 and action_horizon")

    @classmethod
    def from_json(cls, path, dataset_info=None):
        path = Path(path)
        values = json.loads(path.read_text())
        if values.get("normalization_stats_path") is not None:
            values["normalization_stats_path"] = str((path.parent / values["normalization_stats_path"]).resolve())
        return cls.from_dict(values, dataset_info)

    @classmethod
    def from_dict(cls, values, dataset_info=None):
        values = dict(values)
        vision = dict(values.pop("vision", {}))
        observation = dict(values.pop("observation", {}))
        dit = dict(values.pop("dit", {}))
        if dataset_info is not None:
            features = dataset_info["features"]
            names = tuple(k for k, v in features.items() if v["dtype"] == "video")
            values["camera_names"] = names
            vision["num_cameras"] = len(names)
            dit["action_dim"] = features["action"]["shape"][0]
            if "observation.state" in features:
                observation["state_dim"] = features["observation.state"]["shape"][0]
            else:
                observation["use_state"] = False
        preprocessing = ImagePreprocessConfig(**values.pop("image_preprocess", {}))
        return cls(vision=VisualEncoderConfig(**vision), observation=ObservationEncoderConfig(**observation),
                   dit=ConditionalDiTConfig(**dit), image_preprocess=preprocessing, **values)

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")


class FlowMatchingPolicy(nn.Module):
    """Dataset-ready forward; predict methods take already-preprocessed observations."""

    def __init__(self, config: FlowMatchingPolicyConfig, stats=None):
        super().__init__()
        self.config = config
        if stats is None:
            if config.normalization_stats_path is None:
                raise ValueError("Training normalization statistics are required: provide stats or normalization_stats_path")
            stats = json.loads(Path(config.normalization_stats_path).read_text())
        self.action_normalizer = MinMaxNormalizer(stats["action"])
        self.state_normalizer = MinMaxNormalizer(stats["observation.state"]) if config.observation.use_state else None
        self.image_preprocessor = ImagePreprocessor(config.image_preprocess)
        self.vision_encoder = ACTVisualEncoder(config.vision)
        self.observation_encoder = ObservationEncoder(config.observation)
        self.dit = ConditionalDiT(config.dit)
        self.flow_matching_objective = FlowMatchingObjective()

    def encode_observation(self, observation, timings=None):
        """Input is Dataset's images/state dict, already normalized; keep its graph."""
        device = observation["images"].device
        if timings is not None:
            synchronize(device)
            start = perf_counter()
        features = self.vision_encoder(observation["images"])
        if timings is not None:
            synchronize(device)
            timings["vision_ms"] = (perf_counter()-start)*1000
            start = perf_counter()
        encoded = self.observation_encoder(
            **features, state=observation.get("state"),
            visual_padding_mask=observation.get("visual_padding_mask"))
        if timings is not None:
            synchronize(device)
            timings["observation_ms"] = (perf_counter()-start)*1000
        return encoded

    def prepare_observation(self, raw_observation):
        """One raw observation: images is a camera-name -> uint8 RGB/PIL mapping."""
        reference = self.dit.action_proj.weight
        images = torch.stack([self.image_preprocessor(raw_observation["images"][camera])
                              for camera in self.config.camera_names]).unsqueeze(0).to(reference)
        output = {"images": images}
        if self.config.observation.use_state:
            state = torch.as_tensor(raw_observation["state"], device=reference.device, dtype=torch.float32)
            output["state"] = self.state_normalizer.normalize(state).unsqueeze(0).to(reference)
        return output

    def _prepare_action_chunk(self, action, mask):
        padding = self.config.dit.action_horizon-action.shape[1]
        if padding < 0:
            raise ValueError("Dataset action chunk exceeds policy action_horizon; truncation is not implicit")
        if padding:
            if mask is None:
                mask = torch.zeros(action.shape[:2], dtype=torch.bool, device=action.device)
            action = torch.cat((action, action[:, -1:].expand(-1, padding, -1)), dim=1)
            mask = torch.cat((mask, torch.ones(action.shape[0], padding, dtype=torch.bool, device=action.device)), dim=1)
        return action, mask

    def forward(self, batch, sample_weight=None, *, noise=None, timestep=None):
        encoded = self.encode_observation(batch["observation"])
        action, mask = self._prepare_action_chunk(batch["action"], batch.get("action_is_pad"))
        return self.flow_matching_objective(self.dit, action, **encoded,
                                             action_padding_mask=mask, sample_weight=sample_weight,
                                             noise=noise, timestep=timestep)

    @torch.inference_mode()
    def predict_action_chunk(self, observation, initial_noise=None, num_inference_steps=None, generator=None):
        device = self.dit.action_proj.weight.device
        synchronize(device)
        start = perf_counter()
        timings = {}
        self.eval()
        encoded = self.encode_observation(observation, timings)
        steps = self.config.num_inference_steps if num_inference_steps is None else num_inference_steps
        normalized = sample_euler(self.dit, **encoded, action_horizon=self.config.dit.action_horizon,
                                  action_dim=self.config.dit.action_dim, initial_noise=initial_noise,
                                  num_inference_steps=steps, generator=generator,
                                  dtype=self.dit.action_proj.weight.dtype, timings=timings)
        actions, elapsed = timed_call(lambda: self.action_normalizer.denormalize(normalized.float()), device)
        timings["denormalization_ms"] = elapsed
        if not torch.isfinite(actions).all():
            raise FloatingPointError("De-normalized action chunk contains NaN or Inf")
        synchronize(device)
        timings["total_ms"] = (perf_counter()-start)*1000
        return {"action_chunk_normalized": normalized, "action_chunk": actions, "timings": timings}

    def predict_action(self, observation, initial_noise=None, num_inference_steps=None, generator=None):
        output = self.predict_action_chunk(observation, initial_noise, num_inference_steps, generator)
        n = self.config.n_action_steps
        return {"action_normalized": output["action_chunk_normalized"][:, :n],
                "action": output["action_chunk"][:, :n], "timings": output["timings"]}

    def parameter_counts(self):
        return {name: {"total": sum(p.numel() for p in module.parameters()),
                       "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
                for name, module in self.named_children()}
