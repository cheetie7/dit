"""Run the Dataset -> pretrained vision -> observation memory acceptance check."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from importlib.resources import files

import torch
from torch.utils.data import DataLoader

from awbc_data import build_datasets
from awbc_data.inspect import tensor_report
from .observation import ObservationEncoder, ObservationEncoderConfig
from .vision import ACTVisualEncoder, VisualEncoderConfig


def gradient_norms(module):
    result = {}
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            result[name] = parameter.grad.norm().item()
            assert result[name] > 0, name
    return result


def describe_output(output):
    description = tensor_report({k: v for k, v in output.items() if v is not None})
    if output["observation_padding_mask"] is None:
        description["observation_padding_mask"] = None
    return description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/awbc_observation"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    protected = [Path(__file__).with_name("vision.py"), Path(str(files("awbc_data").joinpath("dataset.py")))]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    train, val = build_datasets(args.root)
    batch = next(iter(DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(args.seed))))
    state = batch["observation"]["state"]
    state_dim = train.source.info["features"]["observation.state"]["shape"][0]
    assert state.shape[-1] == state_dim
    vision_config = VisualEncoderConfig(num_cameras=len(train.source.cameras))
    vision = ACTVisualEncoder(vision_config)
    observation_config = ObservationEncoderConfig(state_dim=state_dim)
    encoder = ObservationEncoder(observation_config)
    features = vision(batch["observation"]["images"])
    features["visual_tokens"].retain_grad()
    features["visual_pos"].retain_grad()
    prepared = encoder.prepare_tokens(**features, state=state)
    output = encoder(**features, state=state)
    memory = output["observation_memory"]
    # A random linear readout avoids the near-constant mean of a LayerNorm output.
    loss = (memory * torch.randn_like(memory)).mean()
    loss.backward()
    encoder_gradients = gradient_norms(encoder)
    vision_gradients = gradient_norms(vision)
    assert all(p.grad is None for p in vision.backbone.parameters())
    assert features["visual_tokens"].grad.norm() > 0
    assert features["visual_pos"].grad.norm() > 0
    encoder.eval()
    with torch.no_grad():
        evaluation = encoder(**features, state=state)
        changed_state = encoder(**features, state=state + 0.1)
        change = (evaluation["observation_memory"][:, 1:] - changed_state["observation_memory"][:, 1:]).abs().max().item()
        assert change > 0
        visual_only = ObservationEncoder(ObservationEncoderConfig(use_state=False)).eval()
        no_state = visual_only(**features)
        mask = torch.zeros(features["visual_tokens"].shape[:2], dtype=torch.bool)
        mask[:, -10:] = True
        padded = encoder(**features, state=state, visual_padding_mask=mask)
    for p in protected:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)]
    report = {
        "state_field": "observation.state", "state_dim_from_metadata": state_dim,
        "state_normalization": "Dataset train-only min_max; reused without extra normalization",
        "camera_order": list(train.source.cameras),
        "episode_indices": batch["episode_index"].tolist(), "frame_indices": batch["frame_index"].tolist(),
        "vision_config": asdict(vision_config), "observation_config": asdict(observation_config),
        "input": tensor_report({"images": batch["observation"]["images"], "state": state, **features}),
        "state_token_shape": list(encoder.state_projection(state).unsqueeze(1).shape),
        "prepared": describe_output(prepared), "with_state": describe_output(output),
        "without_state": describe_output(no_state), "with_padding": describe_output(padded),
        "backward_check": {
            "objective": "mean(memory * random_readout)", "loss": loss.item(),
            "encoder_gradient_norms": encoder_gradients, "vision_gradient_norms": vision_gradients,
            "visual_tokens_grad_norm": features["visual_tokens"].grad.norm().item(),
            "visual_pos_grad_norm": features["visual_pos"].grad.norm().item(),
            "frozen_backbone_has_gradients": False},
        "state_change_max_visual_memory_difference": change,
        "all_reported_tensors_finite": True, "source_sha256": hashes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "input_config.json").write_text(json.dumps({
        "image_preprocess": asdict(train.image_preprocessor.config),
        "camera_order": list(train.source.cameras), "vision": asdict(vision_config),
        "observation": asdict(observation_config)}, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("state_dim_from_metadata", "state_token_shape",
                                            "with_state", "without_state", "with_padding",
                                            "all_reported_tensors_finite")}, indent=2))
    print(f"Gradient and configuration details: {args.output / 'report.json'}")
    train.close()
    val.close()


if __name__ == "__main__":
    main()
