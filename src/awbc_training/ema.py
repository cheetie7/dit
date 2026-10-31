from copy import deepcopy

import torch


class EMA:
    def __init__(self, policy, decay=0.999):
        self.policy = deepcopy(policy).eval().requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, policy):
        source = dict(policy.named_parameters())
        for name, parameter in self.policy.named_parameters():
            if source[name].requires_grad:
                parameter.lerp_(source[name], 1-self.decay)
            else:
                parameter.copy_(source[name])
        buffers = dict(policy.named_buffers())
        for name, buffer in self.policy.named_buffers():
            buffer.copy_(buffers[name])
        self.num_updates += 1

    def state_dict(self):
        return {"policy": self.policy.state_dict(), "decay": self.decay, "num_updates": self.num_updates}

    def load_state_dict(self, state):
        self.policy.load_state_dict(state["policy"])
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
