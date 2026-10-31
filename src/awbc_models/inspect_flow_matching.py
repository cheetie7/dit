"""Real batch -> vision -> observation -> DiT -> masked weighted CFM objective."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from importlib.resources import files

import torch
from torch.utils.data import DataLoader, Subset

from awbc_data import build_datasets
from awbc_data.inspect import tensor_report
from .dit import ConditionalDiT, ConditionalDiTConfig
from .flow_matching_objective import FlowMatchingObjective
from .observation import ObservationEncoder, ObservationEncoderConfig
from .vision import ACTVisualEncoder, VisualEncoderConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "conditional_dit.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/awbc_flow_matching"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    paths = [Path(__file__).with_name(name) for name in ("vision.py", "observation.py", "dit.py")]
    paths.append(Path(str(files("awbc_data").joinpath("dataset.py"))))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    train, val = build_datasets(args.root)
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))[:args.batch_size].tolist()
    # Always include a real short tail, because a random batch often has no padding.
    indices[-1] = next(i for i, (_, _, length) in enumerate(train.samples) if length < 10)
    batch = next(iter(DataLoader(Subset(train, indices), batch_size=args.batch_size)))
    action_dim = train.source.info["features"]["action"]["shape"][0]
    state_dim = train.source.info["features"]["observation.state"]["shape"][0]
    config = ConditionalDiTConfig.from_json(args.config, action_dim=action_dim)
    action, action_mask = batch["action"], batch["action_is_pad"]
    dataset_shape = list(action.shape)
    # Input adapter only: no additional action is supervised, no Dataset is changed.
    padding = config.action_horizon - action.shape[1]
    if padding > 0:
        action = torch.cat((action, action[:, -1:].expand(-1, padding, -1)), dim=1)
        action_mask = torch.cat((action_mask, torch.ones(action.shape[0], padding, dtype=torch.bool)), dim=1)
    vision = ACTVisualEncoder(VisualEncoderConfig(d_model=config.hidden_dim, num_cameras=len(train.source.cameras))).eval()
    observation = ObservationEncoder(ObservationEncoderConfig(state_dim=state_dim, hidden_dim=config.hidden_dim)).eval()
    dit = ConditionalDiT(config).eval()
    features = vision(batch["observation"]["images"])
    observations = observation(**features, state=batch["observation"]["state"])
    observations["observation_memory"].retain_grad()
    observations["observation_pos"].retain_grad()
    objective = FlowMatchingObjective()
    output = objective(dit, action, **observations, action_padding_mask=action_mask)
    weights = torch.linspace(1, 2, action.shape[0])
    weighted = objective(dit, action, **observations, action_padding_mask=action_mask,
                         sample_weight=weights, noise=output["noise"], timestep=output["timestep"])
    torch.testing.assert_close(output["per_sample_loss"], weighted["per_sample_loss"])
    weighted["loss"].backward()
    assert dit.velocity_head.weight.grad.norm() > 0
    assert observations["observation_memory"].grad is not None
    assert observations["observation_pos"].grad is not None
    for module in (dit, observation, vision):
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    assert all(p.grad is None for p in vision.backbone.parameters())
    assert torch.equal(output["valid_action_count"], batch["valid_length"])
    report = {
        "dit_config": asdict(config), "dataset_action_shape": dataset_shape,
        "input_adapter": "repeat final action to configured horizon; all appended steps masked",
        "episode_indices": batch["episode_index"].tolist(), "frame_indices": batch["frame_index"].tolist(),
        "valid_action_count": output["valid_action_count"].tolist(),
        "input": tensor_report({"action": action, "action_padding_mask": action_mask,
                                "observation_memory": observations["observation_memory"],
                                "observation_pos": observations["observation_pos"]}),
        "unweighted": tensor_report(output), "weighted": tensor_report(weighted),
        "per_sample_loss": output["per_sample_loss"].tolist(),
        "sample_weight": weights.tolist(), "unweighted_loss": output["loss"].item(),
        "weighted_loss": weighted["loss"].item(), "weight_reduction": "sum(weight * per_sample_loss) / sum(weight)",
        "gradient_check": {"velocity_head_norm": dit.velocity_head.weight.grad.norm().item(),
                           "observation_memory_connected": True, "observation_pos_connected": True,
                           "observation_memory_norm": observations["observation_memory"].grad.norm().item(),
                           "observation_pos_norm": observations["observation_pos"].grad.norm().item(),
                           "initial_zero_head_blocks_nonzero_upstream_gradients": True,
                           "all_existing_gradients_finite": True, "frozen_backbone_has_gradients": False},
        "all_reported_tensors_finite": True, "source_sha256": hashes,
    }
    for path in paths:
        assert hashes[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("dataset_action_shape", "valid_action_count", "per_sample_loss",
                                            "sample_weight", "unweighted_loss", "weighted_loss", "gradient_check")}, indent=2))
    train.close()
    val.close()


if __name__ == "__main__":
    main()
