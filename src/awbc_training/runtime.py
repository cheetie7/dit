import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(value, device):
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    return value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but no MPS device is available")
    return device


class Precision:
    def __init__(self, config, device):
        self.device_type = device.type
        self.dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[config.dtype]
        self.enabled = config.enabled and self.dtype != torch.float32
        if self.enabled:
            if not torch.amp.autocast_mode.is_autocast_available(device.type):
                raise ValueError(f"Autocast is unavailable on {device.type}")
            if self.dtype == torch.bfloat16 and device.type == "cuda":
                with torch.cuda.device(device):
                    supported = torch.cuda.is_bf16_supported()
                if not supported:
                    raise ValueError("This CUDA device does not support bfloat16; select float16 or float32")
            if self.dtype == torch.bfloat16 and device.type == "mps" and not torch.backends.mps.is_macos_or_newer(14, 0):
                raise ValueError("MPS bfloat16 requires macOS 14 or newer; select float32")
            if self.dtype == torch.float16 and device.type not in ("cpu", "cuda"):
                raise ValueError("FP16 GradScaler training is supported here on CPU/CUDA only")
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.enabled and self.dtype == torch.float16)

    def autocast(self):
        return torch.autocast(self.device_type, dtype=self.dtype, enabled=self.enabled)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())
