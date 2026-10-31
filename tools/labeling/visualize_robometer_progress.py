#!/usr/bin/env python3
"""Create an annotated video and sampled-frame contact sheet for RoboMeter progress."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def annotate_frame(
    frame: Image.Image,
    frame_index: int,
    native_fps: float,
    sample_indices: np.ndarray,
    rewards: np.ndarray,
) -> Image.Image:
    image = frame.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    progress = float(np.interp(frame_index, sample_indices, rewards))
    exact = np.flatnonzero(sample_indices == frame_index)
    sample_text = (
        f"Model sample {int(exact[0]) + 1}/{len(rewards)}"
        if exact.size
        else "Interpolated between model samples"
    )

    panel_x0, panel_y0 = 18, 18
    panel_x1, panel_y1 = min(width - 18, 465), 150
    draw.rounded_rectangle(
        (panel_x0, panel_y0, panel_x1, panel_y1),
        radius=14,
        fill=(0, 0, 0, 180),
    )
    title_font = load_font(27, bold=True)
    text_font = load_font(19)
    small_font = load_font(15)
    draw.text((34, 31), f"Progress  {progress:.3f}", font=title_font, fill=(255, 255, 255, 255))
    draw.text(
        (34, 69),
        f"Time  {frame_index / native_fps:5.2f} s",
        font=text_font,
        fill=(235, 235, 235, 255),
    )
    draw.text((34, 99), sample_text, font=small_font, fill=(210, 210, 210, 255))

    bar_x0, bar_y0 = 34, 126
    bar_x1 = panel_x1 - 20
    bar_y1 = 140
    draw.rounded_rectangle((bar_x0, bar_y0, bar_x1, bar_y1), radius=7, fill=(80, 80, 80, 255))
    fill_x = bar_x0 + int((bar_x1 - bar_x0) * np.clip(progress, 0.0, 1.0))
    if fill_x > bar_x0:
        draw.rounded_rectangle((bar_x0, bar_y0, fill_x, bar_y1), radius=7, fill=(45, 205, 105, 255))

    return Image.alpha_composite(image, overlay).convert("RGB")


def make_contact_sheet(
    samples: list[Image.Image],
    sample_indices: np.ndarray,
    rewards: np.ndarray,
    native_fps: float,
    output_path: Path,
) -> None:
    columns = 5
    thumb_width = 256
    thumb_height = 192
    label_height = 52
    gap = 10
    margin = 16
    rows = (len(samples) + columns - 1) // columns
    sheet_width = margin * 2 + columns * thumb_width + (columns - 1) * gap
    sheet_height = margin * 2 + rows * (thumb_height + label_height) + (rows - 1) * gap
    sheet = Image.new("RGB", (sheet_width, sheet_height), "#171717")
    draw = ImageDraw.Draw(sheet)
    label_font = load_font(16, bold=True)
    small_font = load_font(14)

    for i, frame in enumerate(samples):
        row, column = divmod(i, columns)
        x = margin + column * (thumb_width + gap)
        y = margin + row * (thumb_height + label_height + gap)
        thumb = frame.copy()
        thumb.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (thumb_width, thumb_height), "black")
        cell.paste(thumb, ((thumb_width - thumb.width) // 2, (thumb_height - thumb.height) // 2))
        sheet.paste(cell, (x, y))
        timestamp = float(sample_indices[i] / native_fps)
        draw.text((x, y + thumb_height + 5), f"Sample {i + 1:02d}   t={timestamp:5.2f}s", font=label_font, fill="white")
        draw.text((x, y + thumb_height + 28), f"progress={float(rewards[i]):.3f}", font=small_font, fill="#54d889")

    sheet.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--rewards", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-sheet", required=True)
    args = parser.parse_args()

    video_path = Path(args.video)
    rewards = np.asarray(np.load(args.rewards), dtype=np.float32).reshape(-1)
    if rewards.size < 2:
        raise ValueError("At least two progress values are required")

    total_frames, _ = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    reader = imageio_ffmpeg.read_frames(str(video_path), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    native_fps = float(metadata["fps"])
    sample_indices = np.linspace(0, total_frames - 1, rewards.size, dtype=int)
    sample_lookup = {int(frame_index): i for i, frame_index in enumerate(sample_indices)}

    output_video = Path(args.output_video)
    output_sheet = Path(args.output_sheet)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_sheet.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio_ffmpeg.write_frames(
        str(output_video),
        (width, height),
        fps=native_fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart"],
    )
    writer.send(None)
    sampled_frames: list[Image.Image | None] = [None] * rewards.size
    processed = 0
    try:
        for frame_index, raw_frame in enumerate(reader):
            frame = Image.frombytes("RGB", (width, height), raw_frame)
            if frame_index in sample_lookup:
                sampled_frames[sample_lookup[frame_index]] = frame.copy()
            annotated = annotate_frame(frame, frame_index, native_fps, sample_indices, rewards)
            writer.send(np.asarray(annotated, dtype=np.uint8).tobytes())
            processed += 1
    finally:
        reader.close()
        writer.close()

    if processed != total_frames:
        raise RuntimeError(f"Decoded {processed} frames, expected {total_frames}")
    if any(frame is None for frame in sampled_frames):
        raise RuntimeError("Failed to collect every sampled frame")
    make_contact_sheet(
        [frame for frame in sampled_frames if frame is not None],
        sample_indices,
        rewards,
        native_fps,
        output_sheet,
    )

    print(f"video={output_video}")
    print(f"contact_sheet={output_sheet}")
    print(f"source_frames={total_frames}")
    print(f"source_fps={native_fps}")
    print(f"model_samples={rewards.size}")


if __name__ == "__main__":
    main()
