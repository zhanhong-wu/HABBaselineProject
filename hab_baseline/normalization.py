"""Training-only normalization-statistic loading and computation."""

import json
from pathlib import Path

import torch
from torchvision import transforms

from .data import get_dataset_stats, make_loader


def load_or_compute_stats(dataset, stats_path: Path, batch_size: int,
                          patch_size: int, scale_factor: float):
    if stats_path.is_file():
        with stats_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("patch_size") != patch_size or payload.get("scale_factor") != scale_factor:
            raise ValueError(f"Statistics configuration does not match: {stats_path}")
        mean = torch.tensor(payload["mean"], dtype=torch.float32)
        std = torch.tensor(payload["std"], dtype=torch.float32)
        print(f"Loaded training-set normalization statistics: {stats_path}")
    else:
        loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        mean, std = get_dataset_stats(loader)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump({"mean": mean.tolist(), "std": std.tolist(),
                       "patch_size": patch_size, "scale_factor": scale_factor}, handle, indent=2)
        print(f"Saved training-set normalization statistics: {stats_path}")
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError("Invalid channel statistics shape.")
    if torch.any(std <= 0):
        raise ValueError("All channel standard deviations must be positive.")
    return mean, std, transforms.Normalize(mean=mean, std=std)
