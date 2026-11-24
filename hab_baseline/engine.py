"""Training and evaluation loops using aggregate bloom-class confusion counts."""

import torch
from tqdm import tqdm

from .checkpoints import load_checkpoint, save_best_model, save_checkpoint


def composite_score(metrics, weights):
    return (
        metrics["F1-Score"] * weights["f1"]
        + metrics["IoU"] * weights["iou"]
        + metrics["Precision"] * weights["precision"]
        + metrics["Recall"] * weights["recall"]
    )


@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_tp = total_fp = total_fn = 0
    for images, masks in data_loader:
        if images is None or masks is None:
            continue
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1))
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        targets = masks.squeeze(1)
        total_tp += torch.logical_and(predictions == 1, targets == 1).sum().float()
        total_fp += torch.logical_and(predictions == 1, targets == 0).sum().float()
        total_fn += torch.logical_and(predictions == 0, targets == 1).sum().float()
    average_loss = total_loss / max(len(data_loader), 1)
    epsilon = 1e-6
    metrics = {
        "IoU": (total_tp / (total_tp + total_fp + total_fn + epsilon)).cpu().item(),
        "Precision": (total_tp / (total_tp + total_fp + epsilon)).cpu().item(),
        "Recall": (total_tp / (total_tp + total_fn + epsilon)).cpu().item(),
        "F1-Score": (2 * total_tp / (2 * total_tp + total_fp + total_fn + epsilon)).cpu().item(),
    }
    return metrics, average_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs,
                device, checkpoint_path, best_model_path, metric_weights, writer,
                resume=True):
    model.to(device)
    best_score = 0.0
    start_epoch = 0
    if resume and checkpoint_path.is_file():
        state = load_checkpoint(model, optimizer, checkpoint_path, map_location=device)
        start_epoch = state["next_epoch"]
        best_score = state["best_composite_score"]
        print(f"Resumed {checkpoint_path}: next epoch={start_epoch}, best={best_score:.4f}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for images, masks in progress:
            if images is None or masks is None:
                continue
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        average_train_loss = train_loss / max(len(train_loader), 1)
        metrics, average_val_loss = evaluate_model(model, val_loader, criterion, device)
        score = composite_score(metrics, metric_weights)
        print(f"Epoch {epoch + 1}: train={average_train_loss:.4f}, val={average_val_loss:.4f}, "
              f"IoU={metrics['IoU']:.4f}, F1={metrics['F1-Score']:.4f}, composite={score:.4f}")
        writer.add_scalars("Loss", {"train": average_train_loss, "validation": average_val_loss}, epoch)
        for key, value in metrics.items():
            writer.add_scalar(f"Metrics/{key}", value, epoch)
        writer.add_scalar("Metrics/Composite_Score", score, epoch)

        is_best = len(val_loader.dataset) > 0 and score > best_score
        if is_best:
            best_score = score
            save_best_model(model, best_model_path)
            print(f"Saved new best model: score={best_score:.4f}")
        save_checkpoint(model, optimizer, epoch, best_score, is_best, checkpoint_path)
    return best_score
