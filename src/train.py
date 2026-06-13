"""
Train Pokemon type classifier on MPS (Apple Silicon).
Logs metrics to metrics.csv for visualization.

Usage:
    python src/train.py --data-dir data --epochs 30 --batch-size 32
    python src/plot.py  # visualize after training
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import IDX_TO_TYPE, NUM_CLASSES, create_dataloaders
from model import PokemonTypeCNN


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(label_counts):
    total = sum(label_counts)
    weights = []
    for c in label_counts:
        if c > 0:
            weights.append(total / (NUM_CLASSES * c))
        else:
            weights.append(0.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        for label, pred in zip(labels, preds):
            class_total[label.item()] += 1
            if label == pred:
                class_correct[label.item()] += 1

    avg_loss = running_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc, class_correct, class_total


def main():
    parser = argparse.ArgumentParser(description="Train Pokemon type classifier")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = str(project_root / args.data_dir)

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, label_counts = create_dataloaders(
        data_dir, batch_size=args.batch_size, val_split=0.2,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Class counts:  {dict(zip(IDX_TO_TYPE.values(), label_counts))}")

    class_weights = compute_class_weights(label_counts).to(device)

    model = PokemonTypeCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 6

    metrics_path = project_root / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_epochs", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr", "elapsed_s"])

    per_class_path = project_root / "per_class.csv"
    with open(per_class_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + [IDX_TO_TYPE[i] for i in range(NUM_CLASSES)])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, class_correct, class_total = validate(
            model, val_loader, criterion, device,
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train loss: {train_loss:.3f} acc: {train_acc:.3f} | "
              f"val loss: {val_loss:.3f} acc: {val_acc:.3f} | "
              f"lr: {current_lr:.1e} | {elapsed:.1f}s")

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
                             current_lr, elapsed])

        per_class_accs = []
        for i in range(NUM_CLASSES):
            if class_total[i] > 0:
                per_class_accs.append(class_correct[i] / class_total[i])
            else:
                per_class_accs.append(0.0)
        with open(per_class_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + per_class_accs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), project_root / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("\nPer-class accuracy on best model:")
    model.load_state_dict(torch.load(project_root / "best_model.pt"))
    _, _, class_correct, class_total = validate(model, val_loader, criterion, device)
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {IDX_TO_TYPE[i]:12s}: {acc:.3f} ({class_correct[i]}/{class_total[i]})")


if __name__ == "__main__":
    main()
