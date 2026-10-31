"""Stage 6 acceptance: real forward/backward, timed Euler inference and state_dict roundtrip."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from importlib.resources import files

import torch
from torch.utils.data import DataLoader, Subset

from awbc_data import build_datasets
from awbc_data.inspect import tensor_report
from .policy import FlowMatchingPolicy, FlowMatchingPolicyConfig


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value.to(device)


def gradient_summary(module):
    trainable = [p for p in module.parameters() if p.requires_grad]
    gradients = [p.grad for p in trainable if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients)
    return {"trainable_tensors": len(trainable), "gradient_tensors": len(gradients),
            "gradient_norm": sum(g.float().square().sum().item() for g in gradients)**0.5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "flow_matching_policy.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/awbc_policy"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    protected = [Path(__file__).with_name(name) for name in
                 ("vision.py", "observation.py", "dit.py", "flow_matching_objective.py")]
    protected.extend((Path(str(files("awbc_data").joinpath("dataset.py"))), Path(str(files("awbc_data").joinpath("images.py")))))
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    info = json.loads((args.root / "meta/info.json").read_text())
    config = FlowMatchingPolicyConfig.from_json(args.config, dataset_info=info)
    train, val = build_datasets(args.root, image_preprocess=config.image_preprocess)
    args.output.mkdir(parents=True, exist_ok=True)
    stats_path = args.output / "normalization.json"
    stats_path.write_text(json.dumps(train.stats, indent=2) + "\n")
    config.normalization_stats_path = str(stats_path.resolve())
    config.to_json(args.output / "config.json")
    model = FlowMatchingPolicy(config).to(args.device)
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed))[:args.batch_size].tolist()
    indices[-1] = next(i for i, (_, _, length) in enumerate(train.samples) if length < 10)
    batch = to_device(next(iter(DataLoader(Subset(train, indices), batch_size=args.batch_size))), args.device)
    counts = {"vision": 0, "observation": 0, "dit": 0}
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            counts[name] += 1
        return hook

    for name, module in (("vision", model.vision_encoder), ("observation", model.observation_encoder), ("dit", model.dit)):
        handles.append(module.register_forward_hook(capture(name)))
    output = model(batch, sample_weight=torch.linspace(1, 2, args.batch_size, device=args.device))
    output["loss"].backward()
    assert counts == {"vision": 1, "observation": 1, "dit": 1}
    training_report = tensor_report(output)
    training_loss = output["loss"].item()
    valid_action_count = output["valid_action_count"].tolist()
    gradients = {name: gradient_summary(module) for name, module in
                 (("dit", model.dit), ("observation", model.observation_encoder), ("vision", model.vision_encoder))}
    assert gradients["dit"]["gradient_norm"] > 0
    assert gradients["observation"]["trainable_tensors"] == gradients["observation"]["gradient_tensors"]
    assert all(p.grad is None for p in model.vision_encoder.backbone.parameters())
    del output
    model.zero_grad(set_to_none=True)
    model.eval()
    shape = (args.batch_size, config.dit.action_horizon, config.dit.action_dim)
    noise = torch.linspace(-1, 1, torch.tensor(shape).prod().item(), device=args.device).reshape(shape)
    model.predict_action_chunk(batch["observation"], initial_noise=noise)  # Warm-up.
    sampling = {}
    for steps in (1, 4, 8, 16):
        counts.update(vision=0, observation=0, dit=0)
        generated = model.predict_action_chunk(batch["observation"], initial_noise=noise, num_inference_steps=steps)
        assert counts == {"vision": 1, "observation": 1, "dit": steps}
        expected = train.denormalize("action", generated["action_chunk_normalized"].float().cpu())
        torch.testing.assert_close(generated["action_chunk"].cpu(), expected, rtol=0, atol=0)
        sampling[str(steps)] = {"calls": dict(counts), "timings": generated["timings"],
                                "tensors": tensor_report({k: v for k, v in generated.items() if k != "timings"})}
    reference = model.predict_action_chunk(batch["observation"], initial_noise=noise)
    repeated = model.predict_action_chunk(batch["observation"], initial_noise=noise)
    torch.testing.assert_close(reference["action_chunk"], repeated["action_chunk"], rtol=0, atol=0)
    execution = model.predict_action(batch["observation"], initial_noise=noise)
    torch.testing.assert_close(execution["action"], reference["action_chunk"][:, :config.n_action_steps])
    counts.update(vision=0, observation=0, dit=0)
    gaussian = model.predict_action_chunk(batch["observation"])
    assert counts == {"vision": 1, "observation": 1, "dit": config.num_inference_steps}
    e, t, _ = train.samples[indices[0]]
    raw_state = torch.tensor(train.source.rows[e]["observation.state"][t], device=args.device)
    if config.observation.use_state:
        torch.testing.assert_close(model.state_normalizer.normalize(raw_state), batch["observation"]["state"][0], rtol=0, atol=0)
    assert model.image_preprocessor.config == train.image_preprocessor.config
    # State-dict roundtrip in memory; no checkpoint manager or persistent weight dump.
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = FlowMatchingPolicy(FlowMatchingPolicyConfig.from_json(args.output / "config.json")).to(args.device)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=args.device))
    reloaded = restored.predict_action_chunk(batch["observation"], initial_noise=noise)
    torch.testing.assert_close(reference["action_chunk"], reloaded["action_chunk"], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    for handle in handles:
        handle.remove()
    parameters = model.parameter_counts()
    report = {
        "device": str(args.device), "batch_size": args.batch_size,
        "torch_version": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
        "dataset_action_shape": list(batch["action"].shape), "valid_action_count": valid_action_count,
        "training_loss": training_loss, "training": training_report, "gradients": gradients,
        "sampling": sampling, "execution_shape": list(execution["action"].shape),
        "gaussian_sampling": tensor_report({k: v for k, v in gaussian.items() if k != "timings"}),
        "parameters": parameters, "total_parameters": sum(v["total"] for v in parameters.values()),
        "trainable_parameters": sum(v["trainable"] for v in parameters.values()),
        "frozen_backbone_has_gradients": False, "fixed_noise_deterministic": True,
        "dataset_denormalization_exact_match": True, "dataset_preprocessing_exact_match": True,
        "state_dict_roundtrip_exact_match": True, "normalization_buffers_restored": True,
        "inference_no_gradients": all(p.grad is None for p in model.parameters()),
        "note": "Untrained zero velocity head: sampling checks the interface, not action quality",
        "source_sha256": hashes,
    }
    for path in protected:
        assert hashes[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("training_loss", "valid_action_count", "execution_shape",
                                            "total_parameters", "trainable_parameters", "state_dict_roundtrip_exact_match")}, indent=2))
    print(json.dumps({steps: {key: value for key, value in row["timings"].items() if key != "dit_step_ms"}
                      for steps, row in sampling.items()}, indent=2))
    train.close()
    val.close()


if __name__ == "__main__":
    main()
