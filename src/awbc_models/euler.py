"""Left-endpoint explicit Euler integration on [0,1], with cached observations."""
from time import perf_counter

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed_call(function, device):
    synchronize(device)
    start = perf_counter()
    output = function()
    synchronize(device)
    return output, (perf_counter()-start)*1000


@torch.inference_mode()
def sample_euler(dit, observation_memory, action_horizon, action_dim, observation_pos=None,
                 observation_padding_mask=None, initial_noise=None, num_inference_steps=8,
                 generator=None, dtype=None, timings=None):
    """Return normalized [B,H,A]; evaluate at 0/N,...,(N-1)/N exactly once each."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be positive")
    device = observation_memory.device
    dtype = dtype or observation_memory.dtype
    shape = (observation_memory.shape[0], action_horizon, action_dim)
    synchronize(device)
    start = perf_counter()
    if initial_noise is None:
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    else:
        if tuple(initial_noise.shape) != shape:
            raise ValueError(f"initial_noise must have shape {shape}, got {tuple(initial_noise.shape)}")
        x = initial_noise.to(device=device, dtype=dtype)
    step_times = []
    dt = 1.0 / num_inference_steps
    dit.eval()
    for i in range(num_inference_steps):
        timestep = torch.full((shape[0],), i/num_inference_steps, device=device, dtype=torch.float32)
        velocity, elapsed = timed_call(lambda: dit(
            noisy_action=x, timestep=timestep, observation_memory=observation_memory,
            observation_pos=observation_pos, observation_padding_mask=observation_padding_mask), device)
        x = x + dt*velocity.to(dtype=x.dtype)
        step_times.append(elapsed)
    if not torch.isfinite(x).all():
        raise FloatingPointError("Euler sampling produced NaN or Inf")
    synchronize(device)
    if timings is not None:
        timings.update(dit_step_ms=step_times, euler_ms=(perf_counter()-start)*1000)
    return x
