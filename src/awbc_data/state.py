"""Describe ordered robot feature columns across data, models and checkpoints."""


def state_selection(raw_state_dim, state_indices=None):
    return _selection("state", raw_state_dim, state_indices)


def action_selection(raw_action_dim, action_indices=None):
    return _selection("action", raw_action_dim, action_indices)


def _selection(feature, raw_dim, indices):
    name = f"{feature}_indices"
    if indices is not None:
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"{name} must be a nonempty list")
        if any(type(index) is not int for index in indices):
            raise ValueError(f"{name} must contain integers")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} must not contain duplicates")
        if raw_dim is None or any(not 0 <= index < raw_dim for index in indices):
            raise ValueError(f"{name} must be within the raw {feature} dimension {raw_dim}")
        indices = list(indices)
    return {
        name: indices,
        f"raw_{feature}_dim": raw_dim,
        f"{feature}_dim": raw_dim if indices is None else len(indices),
    }
