"""Run: python -m awbc_data.inspect --root /path/to/dataset."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import IMAGENET_MEAN, IMAGENET_STD, build_datasets


def tensor_report(value, prefix=""):
    if isinstance(value, dict):
        return {k: v for name, item in value.items()
                for k, v in tensor_report(item, f"{prefix}.{name}".strip(".")).items()}
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite tensor: {prefix}")
    return {prefix: {"shape": list(value.shape), "dtype": str(value.dtype),
                     "min": value.min().item(), "max": value.max().item()}}


def visualize(batch, dataset, path):
    """RGB observations and raw action curves; dashed gray segments are padding."""
    count = min(4, len(batch["action"]))
    h, w = dataset.image_size
    width = max(w * len(dataset.source.cameras), 1000)
    canvas = Image.new("RGB", (width, count * (h + 260)), "white")
    draw = ImageDraw.Draw(canvas)
    colors = ["red", "blue", "green", "orange", "purple", "brown", "teal", "magenta"]
    for i in range(count):
        y = i * (h + 260)
        for c, name in enumerate(dataset.source.cameras):
            rgb = (batch["observation"]["images"][i, c] * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
            picture = Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
            canvas.paste(picture, (c * w, y))
            draw.text((c * w + 4, y + 4), name.split(".")[-1], fill="yellow", stroke_width=1, stroke_fill="black")
        length = int(batch["valid_length"][i])
        action = dataset.denormalize("action", batch["action"][i]).numpy()
        state = dataset.denormalize("observation.state", batch["observation"]["state"][i]).numpy()
        header = (f"ep={batch['episode_index'][i]} frame={batch['frame_index'][i]} "
                  f"valid={length}/10 A={batch['advantage'][i]:.5f} labeled={batch['advantage_valid'][i]}")
        draw.text((10, y + h + 5), header, fill="black")
        # Separate panel per action dimension preserves both gripper and joint scales.
        for d in range(action.shape[1]):
            col, row = d % 4, d // 4
            x0, y0 = 15 + col * (width // 4), y + h + 38 + row * 105
            lo, hi = float(action[:, d].min()), float(action[:, d].max())
            span = max(hi - lo, 1e-5)
            points = [(x0 + j * (width // 4 - 35) / 9, y0 + 70 - (float(v) - lo) / span * 48)
                      for j, v in enumerate(action[:, d])]
            draw.text((x0, y0), f"a{d} [{lo:.4f}, {hi:.4f}] s={state[d]:.4f}", fill=colors[d % 8])
            for j in range(9):
                if j + 1 < length:
                    draw.line([points[j], points[j + 1]], fill=colors[d % 8], width=2)
                else:
                    draw.line([points[j], ((points[j][0] + points[j + 1][0]) / 2, points[j][1])], fill="gray", width=2)
            for j, point in enumerate(points):
                draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill=colors[d % 8] if j < length else "gray")
            draw.text((x0, y0 + 76), "steps 0..9 (gray = padding)", fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/awbc_data"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-episodes", action="store_true")
    parser.add_argument("--require-advantage", action="store_true")
    args = parser.parse_args()
    train, val = build_datasets(args.root, shuffle_episodes=args.shuffle_episodes, seed=args.seed,
                                require_advantage=args.require_advantage)
    args.output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    batch = next(iter(DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                generator=generator, num_workers=args.num_workers)))
    assert not set(train.episode_ids) & set(val.episode_ids)
    assert torch.equal((~batch["action_is_pad"]).sum(1), batch["valid_length"])
    assert ((batch["video_timestamp"] - batch["video_target_timestamp"]).abs() <= 0.5/train.source.fps + 1e-3).all()
    report = {"train_episodes": len(train.episode_ids), "val_episodes": len(val.episode_ids),
              "train_chunks": len(train), "val_chunks": len(val),
              "camera_order": list(train.source.cameras), "has_advantage": train.source.has_advantage,
              "batch": tensor_report(batch)}
    print(json.dumps(report, indent=2))
    (args.output / "batch_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "normalization.json").write_text(json.dumps(train.stats, indent=2) + "\n")
    (args.output / "split.json").write_text(json.dumps({"train": train.episode_ids, "val": val.episode_ids}, indent=2) + "\n")
    visualize(batch, train, args.output / "batch.png")
    # Explicitly inspect a short tail too, since a random batch may contain none.
    tail_index = next((i for i, (_, _, length) in enumerate(train.samples) if length < 10), None)
    if tail_index is not None:
        tail = next(iter(DataLoader(Subset(train, [tail_index]), batch_size=1)))
        visualize(tail, train, args.output / "tail.png")
    train.close()
    val.close()


if __name__ == "__main__":
    main()
