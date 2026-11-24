#!/usr/bin/env python3
"""Unified training and benchmark-test evaluation entry point."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from hab_baseline.checkpoints import load_best_model
from hab_baseline.data import TiledRemoteSensingDataset, make_loader
from hab_baseline.engine import composite_score, evaluate_model, train_model
from hab_baseline.inference import generate_inference_results, visualize_predictions
from hab_baseline.models import BASELINE_MODEL_NAMES, available_models, build_model
from hab_baseline.normalization import load_or_compute_stats
from hab_baseline.utils import project_path, resolve_device, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--models", nargs="+", help="Model names, or 'all' for the 15 baselines.")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--stats-path")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from output checkpoints.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing output best_model.pth files.")
    parser.add_argument("--skip-inference", action="store_true", help="Skip PNG visualizations and full-block exports.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = project_path(PROJECT_ROOT, args.config)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    for argument, key in [("data_root", "data_root"), ("output_root", "output_root"),
                          ("stats_path", "stats_path"), ("epochs", "num_epochs"),
                          ("batch_size", "batch_size"), ("num_workers", "num_workers")]:
        value = getattr(args, argument)
        if value is not None:
            config[key] = value

    model_names = args.models or config["models"]
    if model_names == ["all"]:
        model_names = BASELINE_MODEL_NAMES
    unknown = set(model_names) - set(available_models())
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    data_root = project_path(PROJECT_ROOT, config["data_root"])
    output_root = project_path(PROJECT_ROOT, config["output_root"])
    stats_path = project_path(PROJECT_ROOT, config["stats_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    set_seed(config["seed"])
    torch.set_num_threads(config["torch_threads"])
    device = resolve_device(args.device)
    print(f"Device: {device}; data: {data_root}; outputs: {output_root}")

    patch_size = config["patch_size"]
    scale_factor = config["scale_factor"]
    raw_train = TiledRemoteSensingDataset(
        data_root / "Train" / "Image", data_root / "Train" / "Label",
        transform=None, patch_size=patch_size, stride=patch_size,
        scale_factor=scale_factor, in_channels=config["in_channels"])
    _, _, transform = load_or_compute_stats(
        raw_train, stats_path, config["batch_size"], patch_size, scale_factor)

    datasets = {}
    loaders = {}
    for split in ("Train", "Val", "Test"):
        datasets[split] = TiledRemoteSensingDataset(
            data_root / split / "Image", data_root / split / "Label",
            transform=transform, patch_size=patch_size, stride=patch_size,
            scale_factor=scale_factor, in_channels=config["in_channels"])
        loaders[split] = make_loader(
            datasets[split], config["batch_size"], shuffle=(split == "Train"),
            num_workers=config["num_workers"])
    if len(datasets["Train"]) == 0:
        raise RuntimeError("The training dataset is empty.")

    results = {}
    for model_name in model_names:
        print(f"\n{'=' * 24} {model_name} {'=' * 24}")
        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / "checkpoint.pth"
        best_model_path = model_dir / "best_model.pth"
        writer = SummaryWriter(log_dir=str(model_dir / "runs"))
        model = build_model(model_name, config["in_channels"], config["num_classes"])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
        if not args.eval_only:
            train_model(
                model, loaders["Train"], loaders["Val"], criterion, optimizer,
                config["num_epochs"], device, checkpoint_path, best_model_path,
                config["metric_weights"], writer, resume=not args.no_resume)
        writer.close()

        load_best_model(model, best_model_path, map_location=device)
        metrics, test_loss = evaluate_model(model, loaders["Test"], criterion, device)
        results[model_name] = {**metrics, "Test-Loss": test_loss,
                               "Composite-Score": composite_score(metrics, config["metric_weights"])}
        print(f"Benchmark test: {json.dumps(results[model_name], indent=2)}")
        if not args.skip_inference:
            visualize_predictions(model, loaders["Test"],
                                  min(4, config["batch_size"]), device,
                                  model_dir / "predictions.png")
            generate_inference_results(model, datasets["Test"], model_dir / "full_blocks",
                                       model_name, device, config["target_test_images"])
        del model, optimizer

    with (output_root / "benchmark_test_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved benchmark results: {output_root / 'benchmark_test_results.json'}")


if __name__ == "__main__":
    main()
