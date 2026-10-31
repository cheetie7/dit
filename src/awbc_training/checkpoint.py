import os
from pathlib import Path

import torch


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def save_checkpoint(state, directory, epoch_number=None, keep_last_n=3, *, step_number=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_save(state, directory / "latest_checkpoint.pt")
    if step_number is not None or epoch_number is not None:
        prefix = "step" if step_number is not None else "epoch"
        name = f"step_{step_number:08d}.pt" if step_number is not None else f"epoch_{epoch_number:04d}.pt"
        atomic_save(state, directory / name)
        periodic = sorted(directory.glob(f"{prefix}_*.pt"))
        for old in periodic[:-keep_last_n] if keep_last_n else periodic:
            old.unlink()
