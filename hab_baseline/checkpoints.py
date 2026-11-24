"""Strict, backward-compatible checkpoint and best-weight handling."""

from pathlib import Path

import torch


def torch_load(path, map_location="cpu"):
    """Use safe, memory-mapped tensor loading where PyTorch supports it."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True, mmap=True)
    except TypeError:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_best_model(model, path):
    """Keep the legacy raw state_dict format used by best_model_*.pth."""
    _atomic_torch_save(model.state_dict(), path)


def save_checkpoint(model, optimizer, epoch, best_composite_score, is_best, path):
    """Keep all keys used by the original training scripts."""
    _atomic_torch_save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_composite_score": best_composite_score,
        "is_best": is_best,
    }, path)


def load_best_model(model, path, map_location="cpu", strict=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Best-model file does not exist: {path}")
    payload = torch_load(path, map_location=map_location)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"Best-model payload is not a state_dict: {path}")
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return {
        "path": str(path),
        "parameters": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def load_checkpoint(model, optimizer, path, map_location="cpu", strict=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = torch_load(path, map_location=map_location)
    required = {"epoch", "model_state_dict", "optimizer_state_dict"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint {path} lacks keys: {sorted(missing)}")
    parameters = len(payload["model_state_dict"])
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=strict)
    # Release the mapped model tensors before expanding large optimizer states.
    del payload["model_state_dict"]
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {
        "path": str(path),
        "epoch": int(payload["epoch"]),
        "next_epoch": int(payload["epoch"]) + 1,
        "best_composite_score": float(payload.get("best_composite_score", 0.0)),
        "parameters": parameters,
        "optimizer_state_entries": len(optimizer.state_dict()["state"]),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
