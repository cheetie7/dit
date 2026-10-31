#!/usr/bin/env python3
"""Run RoboMeter once per boundary manifest entry and derive chunk advantages.

The model is loaded once. Boundary results are appended to JSONL and can be
resumed safely. Terminal progress is forced to 1.0 while the raw model output is
retained for auditing.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import decord
import numpy as np
import torch

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--boundary-manifest", type=Path, required=True)
    parser.add_argument("--chunk-manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    completed: dict[str, dict[str, Any]] = {}
    for row in rows:
        boundary_id = str(row["boundary_id"])
        if boundary_id in completed:
            raise ValueError(f"Duplicate boundary result in {path}: {boundary_id}")
        completed[boundary_id] = row
    return completed


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [move_to_device(item, device) for item in value]
        return type(value)(converted)
    if hasattr(value, "to"):
        return value.to(device)
    return value


class VideoReaderCache:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root
        self.path: Path | None = None
        self.reader: decord.VideoReader | None = None

    def get_frames(self, relative_path: str, indices: list[int]) -> np.ndarray:
        path = (self.dataset_root / relative_path).resolve()
        if path != self.path:
            if not path.is_file():
                raise FileNotFoundError(path)
            self.reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
            self.path = path
        assert self.reader is not None
        if not indices:
            raise ValueError("Empty frame index list")
        if min(indices) < 0 or max(indices) >= len(self.reader):
            raise IndexError(
                f"Video frame index out of range for {path}: "
                f"min={min(indices)}, max={max(indices)}, video_frames={len(self.reader)}"
            )
        frames = self.reader.get_batch(indices).asnumpy()
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        return frames


def make_sample(row: dict[str, Any], frames: np.ndarray) -> ProgressSample:
    trajectory = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=str(row["task"]),
        id=str(row["boundary_id"]),
        metadata={
            "subsequence_length": int(frames.shape[0]),
            "boundary_id": str(row["boundary_id"]),
            "episode_index": int(row["episode_index"]),
            "relative_frame_index": int(row["relative_frame_index"]),
        },
        video_embeddings=None,
    )
    return ProgressSample(trajectory=trajectory, sample_type="progress")


def scalar_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def write_chunk_labels(
    chunk_manifest: Path,
    boundary_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    chunks = read_jsonl(chunk_manifest)
    labeled: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            start_id = str(chunk["start_boundary_id"])
            end_id = str(chunk["end_boundary_id"])
            if start_id not in boundary_results or end_id not in boundary_results:
                continue
            start = boundary_results[start_id]
            end = boundary_results[end_id]
            progress_start = float(start["progress_used"])
            progress_end = float(end["progress_used"])
            delta = progress_end - progress_start
            divisor = int(chunk["advantage_divisor"])
            if divisor <= 0:
                raise ValueError(f"Invalid advantage divisor for {chunk['chunk_id']}: {divisor}")
            row = {
                **chunk,
                "progress_start_raw": float(start["progress_raw"]),
                "progress_end_raw": float(end["progress_raw"]),
                "progress_start": progress_start,
                "progress_end": progress_end,
                "terminal_progress_forced": bool(end["terminal_progress_forced"]),
                "progress_delta": delta,
                "advantage": delta / divisor,
                "advantage_valid": True,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            labeled.append(row)
    return labeled


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    dataset_root = args.dataset_root.resolve()
    boundary_manifest = args.boundary_manifest.resolve()
    chunk_manifest = args.chunk_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary_rows = read_jsonl(boundary_manifest)
    if args.limit is not None:
        boundary_rows = boundary_rows[: args.limit]
    expected_ids = {str(row["boundary_id"]) for row in boundary_rows}
    if len(expected_ids) != len(boundary_rows):
        raise ValueError("Boundary manifest contains duplicate boundary_id values")

    results_path = output_dir / "boundary_progress.jsonl"
    completed = load_completed(results_path)
    unexpected = set(completed) - expected_ids
    if unexpected:
        raise ValueError(f"Output contains boundary IDs not present in this manifest: {sorted(unexpected)[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for RoboMeter boundary inference")

    print(f"Loading RoboMeter from {args.model_path} on {device}", flush=True)
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=args.model_path,
        device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config is not None
        else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    pending = [row for row in boundary_rows if str(row["boundary_id"]) not in completed]
    print(
        f"Boundaries: total={len(boundary_rows)}, completed={len(completed)}, pending={len(pending)}",
        flush=True,
    )
    video_cache = VideoReaderCache(dataset_root)

    with results_path.open("a", encoding="utf-8") as output_handle:
        for batch_start in range(0, len(pending), args.batch_size):
            rows = pending[batch_start : batch_start + args.batch_size]
            samples: list[ProgressSample] = []
            for row in rows:
                frames = video_cache.get_frames(
                    str(row["video_path"]),
                    [int(index) for index in row["sample_video_frame_indices"]],
                )
                if frames.shape[0] != int(row["model_frame_count"]):
                    raise ValueError(
                        f"{row['boundary_id']}: extracted {frames.shape[0]} frames, "
                        f"expected {row['model_frame_count']}"
                    )
                samples.append(make_sample(row, frames))

            batch = batch_collator(samples)
            progress_inputs = move_to_device(batch["progress_inputs"], device)
            with torch.inference_mode():
                outputs = compute_batch_outputs(
                    reward_model,
                    tokenizer,
                    progress_inputs,
                    sample_type="progress",
                    is_discrete_mode=is_discrete,
                    num_bins=num_bins,
                )
            predictions = outputs.get("progress_pred", [])
            if len(predictions) != len(rows):
                raise RuntimeError(
                    f"Model returned {len(predictions)} predictions for batch of {len(rows)} boundaries"
                )

            for row, sequence in zip(rows, predictions, strict=True):
                values = np.asarray(sequence, dtype=np.float32).reshape(-1)
                if values.size == 0:
                    raise RuntimeError(f"Empty progress prediction for {row['boundary_id']}")
                raw_progress = float(values[-1])
                terminal_forced = bool(row["force_progress_one"])
                progress_used = 1.0 if terminal_forced else raw_progress
                result = {
                    "boundary_id": str(row["boundary_id"]),
                    "episode_index": int(row["episode_index"]),
                    "relative_frame_index": int(row["relative_frame_index"]),
                    "dataset_index": int(row["dataset_index"]),
                    "is_terminal": bool(row["is_terminal"]),
                    "sample_relative_frame_indices": row["sample_relative_frame_indices"],
                    "progress_sequence_raw": values.tolist(),
                    "progress_raw": raw_progress,
                    "progress_used": progress_used,
                    "terminal_progress_forced": terminal_forced,
                }
                output_handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                output_handle.flush()
                completed[result["boundary_id"]] = result

            done = len(completed)
            print(f"Completed {done}/{len(boundary_rows)} boundaries", flush=True)

    chunk_labels = write_chunk_labels(
        chunk_manifest=chunk_manifest,
        boundary_results=completed,
        output_path=output_dir / "chunk_advantages.jsonl",
    )
    raw_progress = [float(row["progress_raw"]) for row in completed.values()]
    advantages = [float(row["advantage"]) for row in chunk_labels]
    deltas = [float(row["progress_delta"]) for row in chunk_labels]
    summary = {
        "model_path": args.model_path,
        "device": str(device),
        "boundary_manifest": str(boundary_manifest),
        "chunk_manifest": str(chunk_manifest),
        "boundaries_scored": len(completed),
        "chunks_labeled": len(chunk_labels),
        "terminal_boundaries_forced": sum(
            bool(row["terminal_progress_forced"]) for row in completed.values()
        ),
        "raw_progress": scalar_stats(raw_progress),
        "progress_delta": scalar_stats(deltas),
        "advantage": scalar_stats(advantages),
        "negative_advantage_chunks": sum(value < 0 for value in advantages),
        "positive_advantage_chunks": sum(value > 0 for value in advantages),
        "zero_advantage_chunks": sum(value == 0 for value in advantages),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
