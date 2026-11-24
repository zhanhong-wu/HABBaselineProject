"""Prediction visualization and stitched raster/vector export."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from tqdm import tqdm


@torch.no_grad()
def visualize_predictions(model, data_loader, num_samples, device, output_path):
    model.to(device)
    model.eval()
    try:
        images, masks = next(iter(data_loader))
    except (StopIteration, TypeError):
        print("No valid samples were available for visualization.")
        return
    if images is None:
        return
    images = images[:num_samples].to(device)
    masks_np = masks[:num_samples].squeeze(1).cpu().numpy()
    predictions = torch.argmax(model(images), dim=1).cpu().numpy()
    images_np = images.cpu().numpy()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = np.array([axes])
    for i in range(num_samples):
        rgb = np.transpose(images_np[i][[3, 2, 1]], (1, 2, 0))
        low, high = np.percentile(rgb, (2, 98))
        if high > low:
            rgb = (rgb - low) / (high - low)
        axes[i, 0].imshow(np.clip(rgb, 0, 1))
        axes[i, 0].set_title("Input Patch (RGB)")
        axes[i, 1].imshow(masks_np[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(predictions[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction")
        for axis in axes[i]:
            axis.axis("off")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


@torch.no_grad()
def generate_inference_results(model, dataset, output_root, model_name, device, target_filenames):
    if not target_filenames:
        return
    result_dir = Path(output_root) / model_name
    png_dir = result_dir / "PNG_Full"
    geojson_dir = result_dir / "GeoJSON_Full"
    png_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    model.eval()

    image_to_patches = defaultdict(list)
    for index, info in enumerate(dataset.patches):
        filename = Path(info["img_path"]).name
        if filename in target_filenames:
            image_to_patches[filename].append(index)

    for filename in target_filenames:
        indices = image_to_patches.get(filename, [])
        if not indices:
            print(f"Warning: no indexed patches for {filename}")
            continue
        image_path = dataset.patches[indices[0]]["img_path"]
        with rasterio.open(image_path) as src:
            height, width, transform, crs = src.height, src.width, src.transform, src.crs
        prediction = np.zeros((height, width), dtype=np.uint8)
        for index in tqdm(indices, desc=f"Stitching {filename}", leave=False):
            image, _ = dataset[index]
            if image is None:
                continue
            window = dataset.patches[index]["window"]
            output = model(image.unsqueeze(0).to(device))
            patch = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            row, col = int(window.row_off), int(window.col_off)
            prediction[row:row + int(window.height), col:col + int(window.width)] = patch

        stem = f"{Path(filename).stem}-{model_name}-prediction"
        plt.imsave(png_dir / f"{stem}.png", prediction, cmap="gray", vmin=0, vmax=1)
        features = []
        for geometry, value in rasterio.features.shapes(prediction, mask=prediction, transform=transform):
            if int(value) == 1:
                features.append({"type": "Feature", "geometry": geometry,
                                 "properties": {"class": 1, "source_image": filename,
                                                "model": model_name}})
        crs_json = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
        if crs is not None and crs.to_epsg() is not None:
            crs_json = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs.to_epsg()}"}}
        with (geojson_dir / f"{stem}.geojson").open("w", encoding="utf-8") as handle:
            json.dump({"type": "FeatureCollection", "crs": crs_json, "features": features}, handle)
