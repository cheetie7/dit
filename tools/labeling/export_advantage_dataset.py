#!/usr/bin/env python3
"""Audit RoboMeter labels and export a separate, annotated LeRobot v3 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {"min": [float(x.min())], "max": [float(x.max())],
            "mean": [float(x.mean())], "std": [float(x.std())], "count": [len(x)]}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    root, output = args.dataset_root.resolve(), args.output_root.resolve()
    labels_dir = args.labels_dir.resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Output must be separate from, and not contain, the source dataset")
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected LeRobot v3.0")
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    episodes = [ep for p in episode_files for ep in pq.read_table(p).to_pylist()]
    episodes.sort(key=lambda ep: ep["episode_index"])
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata count mismatch")
    n = info["total_frames"]
    cursor = 0
    for ep in episodes:
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        if start != cursor or end - start != ep["length"] or ep["length"] < 2:
            raise ValueError(f"Noncontiguous or invalid episode range: {ep['episode_index']}")
        cursor = end
    if cursor != n:
        raise ValueError("Episode ranges do not cover all source rows")

    boundary_rows = read_rows(labels_dir / "boundary_progress.jsonl")
    boundaries = {r["boundary_id"]: r for r in boundary_rows}
    if len(boundaries) != len(boundary_rows):
        raise ValueError("Duplicate boundary IDs")
    chunk_rows = read_rows(labels_dir / "chunk_advantages.jsonl")
    chunks = {r["chunk_id"]: r for r in chunk_rows}
    if len(chunks) != len(chunk_rows):
        raise ValueError("Duplicate chunk IDs")
    # Original rows remain intact. Only known transitions receive labels.
    floats = ["advantage", "robometer_progress_start", "robometer_progress_end",
              "robometer_progress_delta", "robometer_progress_start_raw", "robometer_progress_end_raw"]
    bools = ["advantage_valid", "action_chunk_start", "action_chunk_is_partial",
             "is_terminal_chunk", "terminal_progress_forced", "is_episode_terminal"]
    ints = ["action_chunk_index", "chunk_step", "action_chunk_valid_length", "pair_index"]
    arrays = {"annotation." + k: np.zeros(n, dtype=np.float32) for k in floats}
    arrays.update({"annotation." + k: np.zeros(n, dtype=np.bool_) for k in bools})
    arrays.update({"annotation." + k: np.full(n, -1, dtype=np.int64) for k in ints})
    arrays["source_episode_index"] = np.empty(n, dtype=np.int64)
    arrays["source_frame_index"] = np.empty(n, dtype=np.int64)
    used_boundaries, used_chunks = set(), set()
    audit_episodes = []
    label_mapping = {"advantage": "advantage", "robometer_progress_start": "progress_start",
                     "robometer_progress_end": "progress_end", "robometer_progress_delta": "progress_delta",
                     "robometer_progress_start_raw": "progress_start_raw", "robometer_progress_end_raw": "progress_end_raw"}
    for ep in episodes:
        eid, length = ep["episode_index"], ep["length"]
        offset, transitions = ep["dataset_from_index"], length - 1
        arrays["source_episode_index"][offset:offset + length] = eid
        arrays["source_frame_index"][offset:offset + length] = np.arange(length)
        arrays["annotation.is_episode_terminal"][offset + transitions] = True
        positions = list(range(0, transitions + 1, 20))
        if positions[-1] != transitions:
            positions.append(transitions)
        for pos in positions:
            bid = f"ep{eid:06d}_b{pos:06d}"
            b = boundaries[bid]
            used_boundaries.add(bid)
            if (b["episode_index"], b["relative_frame_index"], b["dataset_index"]) != (eid, pos, offset + pos):
                raise ValueError(f"Boundary location mismatch: {bid}")
            raw, value = float(b["progress_raw"]), float(b["progress_used"])
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in (raw, value)):
                raise ValueError(f"Nonfinite or out-of-range progress: {bid}")
            terminal = pos == transitions
            if bool(b["terminal_progress_forced"]) != terminal or bool(b["is_terminal"]) != terminal:
                raise ValueError(f"Terminal flag mismatch: {bid}")
            if not math.isclose(value, 1.0 if terminal else raw, abs_tol=1e-7):
                raise ValueError(f"Progress override mismatch: {bid}")
        episode_advantages = []
        for group_start, group_end in zip(positions, positions[1:]):
            start_id, end_id = f"ep{eid:06d}_b{group_start:06d}", f"ep{eid:06d}_b{group_end:06d}"
            b0, b1 = boundaries[start_id], boundaries[end_id]
            delta = b1["progress_used"] - b0["progress_used"]
            divisor = math.ceil((group_end - group_start) / 10)
            for start in range(group_start, group_end, 10):
                end = min(start + 10, group_end)
                cid = f"ep{eid:06d}_c{start:06d}"
                c = chunks[cid]
                used_chunks.add(cid)
                expected = {"episode_index": eid, "relative_start_frame": start, "relative_end_frame": end,
                            "dataset_start_index": offset + start, "dataset_end_index_exclusive": offset + end,
                            "valid_length": end - start, "target_chunk_length": 10,
                            "advantage_divisor": divisor, "chunks_in_group": divisor,
                            "group_index": group_start // 20, "chunk_in_group": (start - group_start) // 10,
                            "start_boundary_id": start_id, "end_boundary_id": end_id,
                            "is_partial": end - start < 10, "is_terminal_chunk": end == transitions,
                            "terminal_progress_forced": end == transitions or group_end == transitions,
                            "advantage_valid": True}
                for key, value in expected.items():
                    if c[key] != value:
                        raise ValueError(f"Chunk mismatch {cid}: {key}={c[key]!r}, expected {value!r}")
                targets = {"progress_start": b0["progress_used"], "progress_end": b1["progress_used"],
                           "progress_start_raw": b0["progress_raw"], "progress_end_raw": b1["progress_raw"],
                           "progress_delta": delta, "advantage": delta / divisor}
                for key, value in targets.items():
                    if not math.isfinite(c[key]) or not math.isclose(c[key], value, abs_tol=1e-7):
                        raise ValueError(f"Invalid {key} for {cid}")
                span = slice(offset + start, offset + end)
                for column, key in label_mapping.items():
                    arrays["annotation." + column][span] = c[key]
                arrays["annotation.advantage_valid"][span] = True
                arrays["annotation.action_chunk_start"][offset + start] = True
                arrays["annotation.action_chunk_index"][span] = start // 10
                arrays["annotation.chunk_step"][span] = np.arange(end - start)
                arrays["annotation.action_chunk_valid_length"][span] = end - start
                arrays["annotation.pair_index"][span] = group_start // 20
                for key in ["action_chunk_is_partial", "is_terminal_chunk", "terminal_progress_forced"]:
                    source_key = "is_partial" if key == "action_chunk_is_partial" else key
                    arrays["annotation." + key][span] = c[source_key]
                episode_advantages.append(c["advantage"])
        expected_sum = 1.0 - boundaries[f"ep{eid:06d}_b000000"]["progress_used"]
        if not math.isclose(sum(episode_advantages), expected_sum, abs_tol=1e-6):
            raise ValueError(f"Advantage telescoping identity failed for episode {eid}")
        audit_episodes.append({"episode_index": eid, "chunks": len(episode_advantages),
                               "sum_chunk_advantage": sum(episode_advantages), "expected_sum": expected_sum})
    if used_boundaries != set(boundaries) or used_chunks != set(chunks):
        raise ValueError("Unexpected label rows outside the source dataset")
    valid_count = int(arrays["annotation.advantage_valid"].sum())
    if valid_count != n - len(episodes):
        raise ValueError("Transition coverage mismatch")
    expected_indices = []
    for p in sorted((root / "data").rglob("*.parquet")):
        expected_indices.extend(pq.read_table(p, columns=["index"]).column("index").to_pylist())
    if sorted(expected_indices) != list(range(n)):
        raise ValueError("Source parquet global indices are not a unique 0..N-1 sequence")
    report = {"source_dataset": str(root), "source_frames": n, "episodes": len(episodes),
              "boundaries": len(boundaries), "chunks": len(chunks), "valid_transitions": valid_count,
              "unlabeled_terminal_rows": n - valid_count,
              "partial_chunks": sum(c["is_partial"] for c in chunk_rows),
              "negative_chunk_fraction": sum(c["advantage"] < 0 for c in chunk_rows) / len(chunk_rows),
              "all_consistency_checks_passed": True,
              "largest_advantages": sorted(chunk_rows, key=lambda c: c["advantage"], reverse=True)[:10],
              "smallest_advantages": sorted(chunk_rows, key=lambda c: c["advantage"])[:10],
              "episode_checks": audit_episodes}
    if args.check_only:
        save_json(labels_dir / "export_audit.json", report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("episode_checks", "largest_advantages", "smallest_advantages")}, indent=2))
        print(f"Audit: {labels_dir / 'export_audit.json'}")
        return
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    # A failed export deliberately lacks EXPORT_COMPLETE.json and is not usable.
    shutil.copytree(root / "meta", output / "meta")
    shutil.copytree(root / "videos", output / "videos")
    for p in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(p)
        indices = np.asarray(table["index"].to_pylist(), dtype=np.int64)
        for key, values in arrays.items():
            if key in table.column_names:
                raise ValueError(f"Source already contains target column: {key}")
            table = table.append_column(key, pa.array(values[indices]))
        # Source pandas metadata describes only the original columns.
        table = table.replace_schema_metadata(None)
        dest = output / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest, compression="zstd")
        check = pq.read_table(dest)
        for name in pq.read_schema(p).names:
            if not check[name].equals(pq.read_table(p, columns=[name])[name]):
                raise ValueError(f"Original field changed: {p} / {name}")
    for key, values in arrays.items():
        info["features"][key] = {"dtype": str(values.dtype), "shape": [1], "names": None}
    save_json(output / "meta/info.json", info)
    original_stats = json.loads((root / "meta/stats.json").read_text())
    original_stats.update({key: stats(values) for key, values in arrays.items()})
    save_json(output / "meta/stats.json", original_stats)
    for p in episode_files:
        table = pq.read_table(p)
        eps = table.to_pylist()
        for key, values in arrays.items():
            per_episode = [stats(values[ep["dataset_from_index"]:ep["dataset_to_index"]]) for ep in eps]
            for metric in ["min", "max", "mean", "std", "count"]:
                table = table.append_column(f"stats/{key}/{metric}", pa.array([s[metric] for s in per_episode]))
        pq.write_table(table.replace_schema_metadata(None), output / p.relative_to(root), compression="zstd")
    # Verify copied videos by SHA256; preserve original timestamp mappings.
    video_hashes = {}
    for p in sorted((root / "videos").rglob("*.mp4")):
        rel = str(p.relative_to(root))
        video_hashes[rel] = digest(p)
        if digest(output / rel) != video_hashes[rel]:
            raise ValueError(f"Video copy mismatch: {rel}")
    audit_dir = output / "labeling"
    audit_dir.mkdir()
    for name in ["boundary_progress.jsonl", "chunk_advantages.jsonl", "summary.json"]:
        if (labels_dir / name).exists():
            shutil.copy2(labels_dir / name, audit_dir / name)
    save_json(audit_dir / "export_audit.json", report)
    save_json(audit_dir / "video_sha256.json", video_hashes)
    (output / "ADVANTAGE_README.md").write_text(
        "# RoboMeter action-chunk annotations\n\n"
        "Original episodes, observations, actions, camera videos, and timing are preserved.\n"
        "Use only annotation.action_chunk_start == true as training chunk starts.\n"
        "Read valid_length real actions starting there; pad to 10 in the dataloader.\n"
        "Mark padded positions with action_is_pad and exclude them from action loss.\n"
        "Normalize each chunk loss by its valid steps, not by 10.\n"
        "annotation.advantage is a chunk-level progress increment repeated on its real action rows; do not sum those repeated rows.\n"
        "Two chunks share the boundary progress delta equally; a single tail chunk receives the full delta.\n"
        "Episode-final progress is forced to 1.0. Negative labels are retained.\n"
        "The final observation has no observed next state: advantage_valid=false, float annotation placeholders=0, chunk indices=-1.\n"
        "These placeholders are excluded from training by the validity/start masks, but included in stored feature statistics.\n"
        "No terminal success is inferred at earlier frames; is_episode_terminal marks the recorded endpoint only.\n"
        "For value-based offline RL, use valid_length for duration/discount handling; terminal chunks end at the recorded endpoint.\n"
        "action_is_pad is generated at load time; no synthetic action/video rows are stored.\n",
        encoding="utf-8")
    save_json(output / "EXPORT_COMPLETE.json", {"complete": True, "episodes": len(episodes),
                                               "frames": n, "chunks": len(chunks), "valid_transitions": valid_count})
    print(f"Export complete: {output}")
    print(json.dumps({"episodes": len(episodes), "frames": n, "chunks": len(chunks), "valid_transitions": valid_count}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args())
