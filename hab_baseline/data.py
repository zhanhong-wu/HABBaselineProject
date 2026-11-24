"""GeoTIFF block reading and non-overlapping patch extraction."""

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TiledRemoteSensingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, patch_size=256,
                 stride=256, scale_factor=10000.0, in_channels=12):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.patches = []
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

        image_files = sorted(p for p in self.image_dir.glob("*.tif") if not p.name.startswith("."))
        for image_path in tqdm(image_files, desc=f"Indexing {self.image_dir.name} patches"):
            mask_path = self.mask_dir / f"{image_path.stem}_label{image_path.suffix}"
            if not mask_path.is_file():
                print(f"Warning: missing paired label, skipped: {mask_path}")
                continue
            with rasterio.open(image_path) as src:
                height, width = src.height, src.width
            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    self.patches.append({
                        "img_path": str(image_path),
                        "mask_path": str(mask_path),
                        "window": Window(x, y, patch_size, patch_size),
                    })
        print(f"Indexed {len(self.patches)} patches from {len(image_files)} images in {self.image_dir}")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        info = self.patches[idx]
        try:
            with rasterio.open(info["img_path"]) as src:
                image = src.read(range(1, self.in_channels + 1), window=info["window"])
            with rasterio.open(info["mask_path"]) as src:
                mask = src.read(1, window=info["window"])
        except rasterio.errors.RasterioIOError as exc:
            print(f"Warning: failed to read {info['img_path']}: {exc}")
            return None, None
        image = torch.from_numpy(image.astype(np.float32) / self.scale_factor).float()
        mask = torch.from_numpy(mask).unsqueeze(0).long()
        if self.transform is not None:
            image = self.transform(image)
        return image, mask


def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)


def make_loader(dataset, batch_size, shuffle, num_workers=8):
    options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    if num_workers > 0:
        options.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, **options)


def get_dataset_stats(dataloader):
    """Reproduce the original batch-mean normalization-statistic calculation."""
    channels_sum = 0
    channels_squared_sum = 0
    num_batches = 0
    for images, _ in tqdm(dataloader, desc="Computing training statistics"):
        if images is None or images.nelement() == 0:
            continue
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1
    if num_batches == 0:
        raise RuntimeError("No valid training batches were available for statistics.")
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    std[std == 0] = 1e-6
    return mean, std
