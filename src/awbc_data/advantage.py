"""Precomputed window labels, joined by trajectory/start/horizon rather than row order."""
from abc import ABC, abstractmethod
import hashlib
import json
import math
from pathlib import Path

import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Subset


def window_metadata(episode, start, length, horizon=10):
    return {"sample_id": f"ep{episode:06d}_t{start:06d}_h{horizon}", "trajectory_id": int(episode),
            "start_timestep": int(start), "end_timestep": int(start+length), "action_horizon": horizon}


def window_key(record):
    return (record["trajectory_id"], record["start_timestep"], record["action_horizon"])


def sample_metadata(dataset, index, horizon=10):
    while isinstance(dataset, Subset):
        index, dataset = dataset.indices[index], dataset.dataset
    episode, start, length = dataset.samples[index]
    return window_metadata(episode, start, length, horizon)


class AdvantageProvider(ABC):
    @abstractmethod
    def get_advantage(self, sample_metadata):
        """Return the scalar window metric declared by provider.metadata."""


class RewardToAdvantageAdapter(ABC):
    @abstractmethod
    def convert(self, raw_scores, sample_metadata):
        """Implement only after the score-to-advantage definition is explicitly established."""


class PrecomputedAdvantageProvider(AdvantageProvider):
    def __init__(self, records, metadata):
        self.metadata = dict(metadata)
        if metadata["semantics"] not in ("advantage", "progress_increment_proxy"):
            raise ValueError("Raw reward/progress/quality scores need an explicit score-to-advantage adapter")
        if not metadata["advantage_definition"]:
            raise ValueError("advantage_definition must declare how the metric was computed")
        self.records, ids = {}, set()
        for original in records:
            row = dict(original)
            key = window_key(row)
            if row["sample_id"] in ids or key in self.records:
                raise ValueError(f"Duplicate sample_id or window key: {row['sample_id']} / {key}")
            if not math.isfinite(float(row["advantage"])):
                raise ValueError(f"Nonfinite advantage at {key}")
            if row["action_horizon"] != metadata["window_length"]:
                raise ValueError(f"Label horizon differs from metadata at {key}")
            row.setdefault("advantage_source", metadata["source"])
            row.setdefault("advantage_model_version", metadata.get("model_version"))
            if row["advantage_source"] != metadata["source"] or row["advantage_model_version"] != metadata.get("model_version"):
                raise ValueError("Row source/version differs from provider provenance")
            ids.add(row["sample_id"])
            self.records[key] = row
        declared = metadata.get("evaluation_trajectory_ids")
        if declared is not None and not {key[0] for key in self.records}.issubset(set(declared)):
            raise ValueError("Unexpected labels outside the declared evaluation trajectory range")
        canonical = {"metadata": self.metadata, "records": [self.records[k] for k in sorted(self.records)]}
        self.fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_file(cls, path, metadata):
        path = Path(path)
        if path.suffix == ".parquet":
            rows = pq.read_table(path).to_pylist()
        elif path.suffix == ".csv":
            rows = pcsv.read_csv(path).to_pylist()
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            rows = json.loads(path.read_text())
        # Existing RoboMeter exporter already computed the proxy. Never re-derive raw scores here.
        if rows and "relative_start_frame" in rows[0]:
            if any(not row.get("advantage_valid", True) for row in rows):
                raise ValueError("RoboMeter chunk table contains invalid advantage labels")
            rows = [{"sample_id": row["chunk_id"], "trajectory_id": row["episode_index"],
                     "start_timestep": row["relative_start_frame"], "end_timestep": row["relative_end_frame"],
                     "action_horizon": row["target_chunk_length"], "advantage": row["advantage"]} for row in rows]
        return cls(rows, metadata)

    @classmethod
    def from_robometer_dataset(cls, source, metadata):
        if not source.has_advantage:
            raise ValueError("Dataset has no annotation.advantage columns; supply the scored export or a label table")
        rows = []
        for episode, data in source.rows.items():
            for start in range(0, len(data["action"])-1, 10):
                if not data["annotation.action_chunk_start"][start] or not data["annotation.advantage_valid"][start]:
                    raise ValueError(f"Invalid annotated chunk at {episode}/{start}")
                length = int(data["annotation.action_chunk_valid_length"][start])
                rows.append({**window_metadata(episode, start, length),
                             "advantage": float(data["annotation.advantage"][start])})
        return cls(rows, metadata)

    def validate_windows(self, windows, sampling_frequency, missing_policy="error"):
        if missing_policy not in ("error", "drop"):
            raise ValueError("missing_label_policy must be error or explicitly drop")
        if self.metadata["sampling_frequency"] != sampling_frequency:
            raise ValueError("Label sampling_frequency does not match Dataset fps")
        expected = {window_key(row): row for row in windows}
        extra, missing = set(self.records)-set(expected), set(expected)-set(self.records)
        if extra:
            raise ValueError(f"Unexpected labels or mismatched horizons: {len(extra)} windows, e.g. {next(iter(extra))}")
        if missing and missing_policy == "error":
            raise ValueError(f"Missing advantage for {len(missing)} windows, e.g. {next(iter(missing))}")
        for key in set(expected) & set(self.records):
            if self.records[key]["end_timestep"] != expected[key]["end_timestep"]:
                raise ValueError(f"Label end_timestep does not match real valid actions at {key}")
        return {"expected_windows": len(expected), "label_count": len(self.records),
                "missing_windows": len(missing), "missing_label_policy": missing_policy}

    def get_record(self, metadata):
        return self.records[window_key(metadata)]

    def get_advantage(self, metadata):
        return float(self.get_record(metadata)["advantage"])

    def provenance_report(self, train_ids, val_ids):
        trained = self.metadata.get("training_trajectory_ids")
        return {"metadata": self.metadata, "fingerprint": self.fingerprint,
                "policy_train_overlap_with_estimator_train": sorted(set(trained)&set(train_ids)) if trained is not None else None,
                "policy_val_overlap_with_estimator_train": sorted(set(trained)&set(val_ids)) if trained is not None else None,
                "estimator_training_range_known": trained is not None,
                "labeled_trajectory_ids": sorted({key[0] for key in self.records})}


class AdvantageDataset(Dataset):
    def __init__(self, dataset, provider, horizon=10, missing_policy="error"):
        self.dataset = dataset
        self.records, self.indices = [], []
        for index in range(len(dataset)):
            metadata = sample_metadata(dataset, index, horizon)
            if window_key(metadata) not in provider.records:
                if missing_policy == "drop":
                    continue
                raise ValueError(f"Missing advantage at {window_key(metadata)}")
            row = provider.get_record(metadata)
            if row["end_timestep"] != metadata["end_timestep"]:
                raise ValueError(f"Label end_timestep does not match valid actions at {window_key(metadata)}")
            self.records.append(row)
            self.indices.append(index)
        self.advantages = torch.tensor([row["advantage"] for row in self.records], dtype=torch.float32)
        if not self.records:
            raise ValueError("No labeled windows remain after the missing-label policy")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.records[index]
        numeric = {key: torch.tensor(row[key]) for key in ("trajectory_id", "start_timestep", "end_timestep", "action_horizon")}
        return {**self.dataset[self.indices[index]], **numeric, "sample_id": row["sample_id"],
                "advantage": self.advantages[index], "advantage_valid": torch.tensor(True),
                "advantage_source": row["advantage_source"],
                "advantage_model_version": row["advantage_model_version"] or "unknown"}
