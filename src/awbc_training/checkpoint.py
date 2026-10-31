import os
from pathlib import Path

import torch


def atomic_save(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def save_checkpoint(state, directory, epoch_number=None, keep_last_n=3):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_save(state, directory / "latest_checkpoint.pt")
    if epoch_number is not None:
        atomic_save(state, directory / f"epoch_{epoch_number:04d}.pt")
        periodic = sorted(directory.glob("epoch_*.pt"))
        for old in periodic[:-keep_last_n] if keep_last_n else periodic:
            old.unlink()
