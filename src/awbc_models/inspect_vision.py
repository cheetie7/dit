"""Exercise the pretrained frozen visual encoder on a real Dataset batch."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from awbc_data import build_datasets, ImagePreprocessConfig
from awbc_data.images import IMAGENET_MEAN, IMAGENET_STD
from awbc_data.inspect import tensor_report
from .vision import ACTVisualEncoder, VisualEncoderConfig


def visualize_features(images, feature_maps, cameras, path):
    """Show RGB and coarse layer4 activation magnitude, not attention weights."""
    b, n, _, h, w = images.shape
    canvas = Image.new("RGB", (n*w, b*(2*h+40)), "white")
    draw = ImageDraw.Draw(canvas)
    for i in range(b):
        for c in range(n):
            x, y = c*w, i*(2*h+40)
            rgb = (images[i, c].cpu()*IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
            picture = Image.fromarray((rgb.permute(1, 2, 0).numpy()*255).round().astype(np.uint8))
            canvas.paste(picture, (x, y+20))
            energy = feature_maps[i, c].detach().float().square().mean(0).sqrt().cpu()
            energy = (energy-energy.min()) / (energy.max()-energy.min()).clamp_min(1e-8)
            heat = Image.fromarray((energy.numpy()*255).astype(np.uint8)).convert("RGB")
            canvas.paste(heat.resize((w, h), Image.Resampling.NEAREST), (x, y+h+40))
            draw.text((x+4, y+4), f"sample {i} | {cameras[c].split('.')[-1]} | RGB", fill="black")
            draw.text((x+4, y+h+24), f"layer4 RMS: {energy.shape[0]}x{energy.shape[1]} (per-view scale)", fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/awbc_vision"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize-mode", choices=("resize", "letterbox"), default="resize")
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    preprocessing = ImagePreprocessConfig(size=(args.height, args.width), mode=args.resize_mode)
    train, val = build_datasets(args.root, image_preprocess=preprocessing)
    batch = next(iter(DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(args.seed))))
    config = VisualEncoderConfig(d_model=args.d_model, num_cameras=len(train.source.cameras))
    model = ACTVisualEncoder(config)
    images = batch["observation"]["images"]
    output = model.encode_cameras(images)
    (output["visual_tokens"].square().mean() + output["visual_pos"].square().mean()).backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert model.input_proj.weight.grad.abs().sum() > 0
    assert model.camera_embedding.weight.grad.abs().sum() > 0
    # Compare actual backbone weights and FrozenBN buffers with the downloaded checkpoint.
    checkpoint = torch.load(Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth", weights_only=True)
    for name, value in model.backbone.body.state_dict().items():
        torch.testing.assert_close(value, checkpoint[name], rtol=0, atol=0)
    with torch.no_grad():
        single = model(images[:, 0])
        maps = model.forward_features(images.flatten(0, 1))
        smoke = model(torch.randn(1, 3, 240, 320))
        other_size = model(torch.nn.functional.interpolate(images[:, 0], size=(255, 341), mode="bilinear", align_corners=False))
    report = {"weights": "ResNet18_Weights.IMAGENET1K_V1", "backbone_mode": "frozen",
              "pretrained_parameters_and_bn_buffers_match": True,
              "backbone_has_gradients": False, "projection_has_gradients": True,
              "camera_embedding_has_gradients": True,
              "preprocessing": asdict(preprocessing), "encoder_config": asdict(config),
              "camera_order": list(train.source.cameras),
              "input": tensor_report(images, "images"),
              "single_camera": tensor_report(single), "three_cameras": tensor_report(output),
              "maps": tensor_report(maps), "random_smoke": tensor_report(smoke),
              "input_255x341": tensor_report(other_size),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "total_parameters": sum(p.numel() for p in model.parameters())}
    args.output.mkdir(parents=True, exist_ok=True)
    visualize_features(images, maps["feature_map"].unflatten(0, images.shape[:2]),
                       train.source.cameras, args.output / "features.png")
    (args.output / "input_config.json").write_text(json.dumps({
        "image_preprocess": asdict(preprocessing), "encoder": asdict(config),
        "camera_order": list(train.source.cameras)}, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    train.close()
    val.close()


if __name__ == "__main__":
    main()
