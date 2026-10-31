import math

import torch


def build_optimizer(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("vision_encoder.backbone."):
            section, lr = "backbone", config.backbone_learning_rate
        elif name.startswith("vision_encoder."):
            section, lr = "visual_projection", config.learning_rate
        elif name.startswith("observation_encoder."):
            section, lr = "observation_encoder", config.learning_rate
        elif name.startswith("dit."):
            section, lr = "dit", config.learning_rate
        else:
            section, lr = "policy", config.learning_rate
        decay = config.weight_decay
        if config.no_decay_bias_and_norm and (parameter.ndim <= 1 or name.endswith(".bias")):
            decay = 0.
        key = (section, decay)
        if key not in groups:
            groups[key] = {"name": section + ("/no_decay" if decay == 0 else "/decay"),
                           "params": [], "lr": lr, "weight_decay": decay}
        groups[key]["params"].append(parameter)
    return {"adamw": torch.optim.AdamW}[config.name](list(groups.values()))


def build_scheduler(optimizer, config, total_steps):
    if config.name != "cosine":
        raise ValueError("This baseline implements the cosine scheduler")
    def factor(step):
        if step < config.warmup_steps:
            return (step+1)/config.warmup_steps
        progress = min(1., (step-config.warmup_steps)/max(1, total_steps-config.warmup_steps-1))
        return config.min_lr_ratio + (1-config.min_lr_ratio)*(1+math.cos(math.pi*progress))/2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def optimizer_summary(model, optimizer):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "groups": [{"name": group["name"], "parameters": sum(p.numel() for p in group["params"]),
                        "initial_learning_rate": group["lr"], "weight_decay": group["weight_decay"]}
                       for group in optimizer.param_groups]}
