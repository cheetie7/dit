"""Real-observation forward check only; supplied action tensors are not sampled noise."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from importlib.resources import files

import torch
from torch.utils.data import DataLoader

from awbc_data import build_datasets
from awbc_data.inspect import tensor_report
from .dit import ConditionalDiT, ConditionalDiTConfig, sinusoidal_embedding
from .observation import ObservationEncoder, ObservationEncoderConfig
from .vision import ACTVisualEncoder, VisualEncoderConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "conditional_dit.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/awbc_dit"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    paths = [Path(__file__).with_name("vision.py"), Path(__file__).with_name("observation.py"),
             Path(str(files("awbc_data").joinpath("dataset.py")))]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    train, val = build_datasets(args.root)
    batch = next(iter(DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(args.seed))))
    action_dim = train.source.info["features"]["action"]["shape"][0]
    state_dim = train.source.info["features"]["observation.state"]["shape"][0]
    config = ConditionalDiTConfig.from_json(args.config, action_dim=action_dim)
    vision = ACTVisualEncoder(VisualEncoderConfig(d_model=config.hidden_dim, num_cameras=len(train.source.cameras))).eval()
    observation = ObservationEncoder(ObservationEncoderConfig(state_dim=state_dim, hidden_dim=config.hidden_dim)).eval()
    with torch.no_grad():
        features = vision(batch["observation"]["images"])
        observations = observation(**features, state=batch["observation"]["state"])
    dit = ConditionalDiT(config).eval()
    b, h, a = args.batch_size, config.action_horizon, config.action_dim
    supplied_action = torch.linspace(-1, 1, b*h*a).reshape(b, h, a)
    timestep = torch.linspace(0, 1, b)
    traces, handles = {}, []
    for name, module in [("action_projection", dit.action_proj), ("timestep_condition", dit.timestep_embedder),
                         *[(f"block_{i}", block) for i, block in enumerate(dit.blocks)],
                         ("final_hidden", dit.final_layer)]:
        handles.append(module.register_forward_hook(
            lambda module, inputs, output, name=name: traces.__setitem__(name, output.detach())))
    velocity = dit(supplied_action, timestep, **observations)
    for handle in handles:
        handle.remove()
    assert velocity.shape == supplied_action.shape and torch.count_nonzero(velocity) == 0
    # One diagnostic backward, no optimizer step or policy objective.
    readout = torch.cos(torch.arange(velocity.numel()).reshape_as(velocity)*0.23)
    (velocity*readout).mean().backward()
    assert dit.velocity_head.weight.grad.norm() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in dit.parameters())
    dataset_horizon = batch["action"].shape[1]
    with torch.no_grad():
        chunk_dit = ConditionalDiT(replace(config, action_horizon=dataset_horizon)).eval()
        chunk_velocity = chunk_dit(batch["action"], timestep, **observations)
        mask = torch.zeros(observations["observation_memory"].shape[:2], dtype=torch.bool)
        mask[:, -10:] = True
        masked_velocity = dit(supplied_action, timestep, observations["observation_memory"],
                              observations["observation_pos"], mask)
    block_counts = [{name: sum(p.numel() for p in module.parameters())
                     for name, module in block.named_children()} for block in dit.blocks]
    report = {
        "config": asdict(config), "action_dim_from_metadata": action_dim,
        "input_kind": "deterministic external action tensor; no noise sampling or corruption",
        "camera_order": list(train.source.cameras),
        "episode_indices": batch["episode_index"].tolist(), "frame_indices": batch["frame_index"].tolist(),
        "input": tensor_report({"supplied_action": supplied_action, "timestep": timestep,
                                "observation_memory": observations["observation_memory"],
                                "observation_pos": observations["observation_pos"]}),
        "intermediate": tensor_report({**traces, "fourier_timestep": sinusoidal_embedding(
            timestep, config.timestep_embedding_dim, config.max_period)}),
        "output": tensor_report({"velocity": velocity, "dataset_chunk_velocity": chunk_velocity,
                                 "masked_velocity": masked_velocity}),
        "observation_padding_mask": None, "all_outputs_finite": True,
        "initial_velocity_is_exactly_zero": True,
        "initial_velocity_head_gradient_norm": dit.velocity_head.weight.grad.norm().item(),
        "initial_block_self_attention_gradient_norm": dit.blocks[0].self_attn.in_proj_weight.grad.norm().item(),
        "parameter_counts": dit.parameter_counts(), "per_block_parameter_counts": block_counts,
        "dataset_horizon_unchanged": dataset_horizon, "source_sha256": hashes,
    }
    for path in paths:
        assert hashes[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("action_dim_from_metadata", "output", "parameter_counts",
                                            "all_outputs_finite", "initial_velocity_is_exactly_zero")}, indent=2))
    train.close()
    val.close()


if __name__ == "__main__":
    main()
