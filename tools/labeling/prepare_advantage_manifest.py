#!/usr/bin/env python3
"""Prepare deterministic RoboMeter boundary and action-chunk manifests.

This script is read-only with respect to the source LeRobot v3 dataset. It builds:

* boundaries.jsonl: one record per progress evaluation boundary;
* chunks.jsonl: one record per non-overlapping action chunk;
* pilot_*.jsonl: records for the first N episodes;
* summary.json: counts and labeling configuration.

Progress inference and dataset rewriting happen in later stages.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-key", default="observation.images.top_camera")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--chunks-per-group", type=int, default=2)
    parser.add_argument("--max-model-frames", type=int, default=8)
    parser.add_argument("--pilot-episodes", type=int, default=5)
    parser.add_argument("--task", default=None, help="Override the task text from episode metadata")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def fixed_history_indices(boundary: int, max_frames: int) -> list[int]:
    """Match RoboMeter's prefix handling: linspace downsample or right-pad last frame."""
    prefix_length = boundary + 1
    if prefix_length > max_frames:
        return np.linspace(0, boundary, max_frames, dtype=int).tolist()
    indices = list(range(prefix_length))
    indices.extend([boundary] * (max_frames - prefix_length))
    return indices


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    if args.chunk_size <= 0 or args.chunks_per_group <= 0 or args.max_model_frames <= 0:
        raise ValueError("chunk-size, chunks-per-group, and max-model-frames must be positive")

    info_path = root / "meta/info.json"
    episode_path = root / "meta/episodes/chunk-000/file-000.parquet"
    if not info_path.is_file() or not episode_path.is_file():
        raise FileNotFoundError(f"Not a supported LeRobot v3 dataset: {root}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}")
    if args.camera_key not in info.get("features", {}):
        raise KeyError(f"Camera feature not found: {args.camera_key}")

    fps = float(info["fps"])
    prefix = f"videos/{args.camera_key}"
    episode_columns = [
        "episode_index",
        "dataset_from_index",
        "dataset_to_index",
        "length",
        "tasks",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
        f"{prefix}/to_timestamp",
    ]
    table = pq.read_table(episode_path, columns=episode_columns)
    episodes = table.to_pylist()
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"Episode count mismatch: metadata={info['total_episodes']}, parquet={len(episodes)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    boundaries: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    group_span = args.chunk_size * args.chunks_per_group

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        transition_count = length - 1
        dataset_from = int(episode["dataset_from_index"])
        dataset_to = int(episode["dataset_to_index"])
        if dataset_to - dataset_from != length:
            raise ValueError(
                f"Episode {episode_index}: dataset range {dataset_from}:{dataset_to} does not match length {length}"
            )
        if transition_count <= 0:
            raise ValueError(f"Episode {episode_index} has no action transitions")

        video_chunk = int(episode[f"{prefix}/chunk_index"])
        video_file = int(episode[f"{prefix}/file_index"])
        video_from = float(episode[f"{prefix}/from_timestamp"])
        video_to = float(episode[f"{prefix}/to_timestamp"])
        video_relpath = (
            f"videos/{args.camera_key}/chunk-{video_chunk:03d}/file-{video_file:03d}.mp4"
        )
        if not (root / video_relpath).is_file():
            raise FileNotFoundError(root / video_relpath)

        task_items = episode.get("tasks") or []
        task = args.task if args.task is not None else (str(task_items[0]) if task_items else "")

        boundary_frames = list(range(0, transition_count + 1, group_span))
        if boundary_frames[-1] != transition_count:
            boundary_frames.append(transition_count)

        boundary_ids: dict[int, str] = {}
        for ordinal, boundary in enumerate(boundary_frames):
            boundary_id = f"ep{episode_index:06d}_b{boundary:06d}"
            boundary_ids[boundary] = boundary_id
            sampled_relative = fixed_history_indices(boundary, args.max_model_frames)
            sample_timestamps = [video_from + frame / fps for frame in sampled_relative]
            video_frame_indices = [int(round(timestamp * fps)) for timestamp in sample_timestamps]
            is_terminal = boundary == transition_count
            boundaries.append(
                {
                    "boundary_id": boundary_id,
                    "episode_index": episode_index,
                    "boundary_ordinal": ordinal,
                    "relative_frame_index": boundary,
                    "dataset_index": dataset_from + boundary,
                    "is_terminal": is_terminal,
                    "force_progress_one": is_terminal,
                    "task": task,
                    "camera_key": args.camera_key,
                    "video_path": video_relpath,
                    "video_chunk_index": video_chunk,
                    "video_file_index": video_file,
                    "episode_video_from_timestamp": video_from,
                    "episode_video_to_timestamp": video_to,
                    "boundary_timestamp": video_from + boundary / fps,
                    "sample_relative_frame_indices": sampled_relative,
                    "sample_timestamps": sample_timestamps,
                    "sample_video_frame_indices": video_frame_indices,
                    "model_frame_count": args.max_model_frames,
                }
            )

        group_index = 0
        for group_start in range(0, transition_count, group_span):
            group_end = min(group_start + group_span, transition_count)
            group_length = group_end - group_start
            chunk_count = math.ceil(group_length / args.chunk_size)
            if group_start not in boundary_ids or group_end not in boundary_ids:
                raise AssertionError(f"Missing boundary for episode {episode_index}, group {group_start}:{group_end}")

            for chunk_in_group in range(chunk_count):
                chunk_start = group_start + chunk_in_group * args.chunk_size
                chunk_end = min(chunk_start + args.chunk_size, group_end)
                valid_length = chunk_end - chunk_start
                chunk_id = f"ep{episode_index:06d}_c{chunk_start:06d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "episode_index": episode_index,
                        "group_index": group_index,
                        "chunk_in_group": chunk_in_group,
                        "chunks_in_group": chunk_count,
                        "relative_start_frame": chunk_start,
                        "relative_end_frame": chunk_end,
                        "dataset_start_index": dataset_from + chunk_start,
                        "dataset_end_index_exclusive": dataset_from + chunk_end,
                        "valid_length": valid_length,
                        "target_chunk_length": args.chunk_size,
                        "is_partial": valid_length < args.chunk_size,
                        "is_terminal_chunk": chunk_end == transition_count,
                        "start_boundary_id": boundary_ids[group_start],
                        "end_boundary_id": boundary_ids[group_end],
                        "advantage_divisor": chunk_count,
                        "task": task,
                    }
                )
            group_index += 1

    pilot_episode_count = min(args.pilot_episodes, len(episodes))
    pilot_episode_ids = set(range(pilot_episode_count))
    pilot_boundaries = [row for row in boundaries if row["episode_index"] in pilot_episode_ids]
    pilot_chunks = [row for row in chunks if row["episode_index"] in pilot_episode_ids]

    partial_chunks = sum(bool(row["is_partial"]) for row in chunks)
    padded_action_slots = sum(args.chunk_size - int(row["valid_length"]) for row in chunks)
    summary = {
        "source_dataset": str(root),
        "codebase_version": info["codebase_version"],
        "total_source_episodes": int(info["total_episodes"]),
        "total_source_frames": int(info["total_frames"]),
        "fps": fps,
        "camera_key": args.camera_key,
        "task_override": args.task,
        "chunk_size_transitions": args.chunk_size,
        "chunks_per_progress_group": args.chunks_per_group,
        "group_span_transitions": group_span,
        "max_model_frames": args.max_model_frames,
        "history_policy": "episode_prefix_linspace_then_right_pad_last",
        "terminal_progress_policy": "force_to_1.0",
        "total_boundaries": len(boundaries),
        "total_action_chunks": len(chunks),
        "partial_action_chunks": partial_chunks,
        "padded_action_slots": padded_action_slots,
        "pilot_episode_count": pilot_episode_count,
        "pilot_boundary_count": len(pilot_boundaries),
        "pilot_action_chunk_count": len(pilot_chunks),
    }

    write_jsonl(output_dir / "boundaries.jsonl", boundaries)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_jsonl(output_dir / "pilot_boundaries.jsonl", pilot_boundaries)
    write_jsonl(output_dir / "pilot_chunks.jsonl", pilot_chunks)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
