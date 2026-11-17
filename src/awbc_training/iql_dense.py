"""Select progress-shaped rewards for the shared IQL train and export commands."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import ClassVar

from awbc_data.iql_dense import DenseRLChunkDataset, load_progress
from . import iql


@dataclass(kw_only=True)
class IQLDenseConfig(iql.IQLConfig):
    reward_mode: ClassVar[str] = "dense"
    progress_path: str
    shaping_scale: float = 1.0

    def __post_init__(self):
        self.progress_path = str(Path(self.progress_path).resolve())
        if not math.isfinite(self.shaping_scale):
            raise ValueError("shaping_scale must be finite")

    def prepare_rewards(self, train, val):
        potentials, fingerprint = load_progress(self.progress_path, train.chunks.source)
        contract = {
            "reward_definition": "discounted_sparse_terminal_success_plus_single_gamma_potential_difference_v1",
            "progress_labels_sha256": fingerprint,
            "progress_interpolation": "linear_between_raw_labeled_frames",
            "shaping_scale": self.shaping_scale,
        }
        return (DenseRLChunkDataset(train, potentials, shaping_scale=self.shaping_scale),
                DenseRLChunkDataset(val, potentials, shaping_scale=self.shaping_scale), contract)


def main(argv=None):
    return iql.main(argv, config_type=IQLDenseConfig, default_config="configs/iql_dense_critic.json")


if __name__ == "__main__":
    main()
