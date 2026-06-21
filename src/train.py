"""
Train Pokemon type classifier on MPS (Apple Silicon).
Logs metrics to metrics.csv for visualization.

Usage:
    python src/train.py --data-dir data --epochs 30 --batch-size 32
    python src/plot.py  # visualize after training
"""

import argparse
import csv
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from dataset import IDX_TO_TYPE, NUM_CLASSES, TYPE_TO_IDX, create_dataloaders, PokemonDataset
from dataset import get_train_transform, get_val_transform
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


def create_filtered_loaders(data_dir, classes, batch_size):
    from torch.utils.data import DataLoader

    full_dataset = PokemonDataset(data_dir, transform=get_train_transform())
    target_idxs = [TYPE_TO_IDX[c] for c in classes]

    filtered_samples = [(p, target_idxs.index(l)) for p, l in full_dataset.samples
                        if l in target_idxs]
    filtered_dataset = PokemonDataset.__new__(PokemonDataset)
    filtered_dataset.data_dir = full_dataset.data_dir
    filtered_dataset.samples = filtered_samples
    filtered_dataset.transform = get_train_transform()

    val_size = int(len(filtered_dataset) * 0.2)
    train_size = len(filtered_dataset) - val_size

    train_ds, val_ds = random_split(
        filtered_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    val_ds.transform = get_val_transform()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    label_counts = [0] * len(classes)
    for _, label in filtered_samples:
        label_counts[label] += 1

    return train_loader, val_loader, label_counts


def _get_param_layers(model):
    """Yield (name, param) for layers with learnable weights (skip bias-only)."""
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:
            yield name, param


def compute_weight_stats(model):
    """Return dict of {layer_name: {weight_norm, weight_mean, weight_std}}."""
    stats = {}
    for name, param in _get_param_layers(model):
        w = param.data.float()
        stats[name] = {
            "weight_norm": w.norm(2).item(),
            "weight_mean": w.mean().item(),
            "weight_std": w.std().item(),
        }
    return stats


def _setup_grad_hooks(model):
    """Register backward hooks on parameters to accumulate grad norms across batches.
    Returns a dict {name: accumulated_norm} that gets updated during backward().
    Call reset_grad_hooks() before each epoch."""
    accum = {}  # {name: [sum_sq_norm, count]}
    handles = []

    def _make_hook(name):
        def hook(grad):
            if grad is not None:
                if name not in accum:
                    accum[name] = [0.0, 0]
                accum[name][0] += grad.float().norm(2).item() ** 2
                accum[name][1] += 1
        return hook

    for name, param in _get_param_layers(model):
        handle = param.register_hook(_make_hook(name))
        handles.append(handle)

    return accum, handles


def _remove_grad_hooks(handles):
    for h in handles:
        h.remove()


def compute_grad_stats(accum):
    """Convert accumulated sums to avg per-batch norms."""
    stats = {}
    for name, (sum_sq, count) in accum.items():
        if count > 0:
            stats[name] = (sum_sq / count) ** 0.5
        else:
            stats[name] = 0.0
    return stats


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    accum, handles = _setup_grad_hooks(model)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type=str(device)):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    grad_stats = compute_grad_stats(accum)
    _remove_grad_hooks(handles)

    return running_loss / total, correct / total, grad_stats


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

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
    parser.add_argument("--run", default="v0.2", help="Run name for saving artifacts")
    parser.add_argument("--pretrain", type=str, nargs="*",
                        help="Classes to pretrain on first (e.g. Colorless Metal)")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a run directory (e.g. runs/v0.5-rcrop)")
    parser.add_argument("--resume-epochs", type=int, default=15,
                        help="Additional epochs when resuming")
    parser.add_argument("--preload", action="store_true", help="Preload images into RAM")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    project_root = Path(__file__).resolve().parent.parent
    run_dir = project_root / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = str(project_root / args.data_dir)

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, label_counts = create_dataloaders(
        data_dir, batch_size=args.batch_size, val_split=0.2,
        preload=args.preload, num_workers=args.workers,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Class counts:  {dict(zip(IDX_TO_TYPE.values(), label_counts))}")

    class_weights = compute_class_weights(label_counts).to(device)

    # --- Resume handling ---
    resume_from_epoch = 0
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.is_absolute():
            resume_dir = project_root / args.resume
        ckpt_path = resume_dir / "model.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            return
        print(f"Resuming from {ckpt_path}")
        # Find last completed epoch from metrics
        resume_metrics = resume_dir / "metrics.csv"
        if resume_metrics.exists():
            resume_df = pd.read_csv(resume_metrics)
            if len(resume_df) > 0:
                resume_from_epoch = int(resume_df["epoch"].iloc[-1])
                last_lr = float(resume_df["lr"].iloc[-1])
                print(f"  Last epoch: {resume_from_epoch}, last LR: {last_lr:.2e}")
                args.lr = last_lr  # use last LR as starting point

    model = PokemonTypeCNN(num_classes=NUM_CLASSES).to(device)
    if resume_from_epoch > 0:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"  Loaded weights from checkpoint")

    pretrain_classes = [] if resume_from_epoch > 0 else (args.pretrain or [])
    if pretrain_classes:
        print(f"\nPhase 1: Pretraining on {pretrain_classes} only")
        print("-" * 40)

        pt_train_loader, pt_val_loader, pt_counts = create_filtered_loaders(
            data_dir, pretrain_classes, args.batch_size,
        )
        print(f"Pretrain samples: {sum(pt_counts)} ({dict(zip(pretrain_classes, pt_counts))})")

        pt_model = PokemonTypeCNN(num_classes=len(pretrain_classes)).to(device)
        pt_model.features.load_state_dict(model.features.state_dict())
        pt_criterion = nn.CrossEntropyLoss()
        pt_optimizer = AdamW(pt_model.parameters(), lr=args.lr, weight_decay=1e-4)
        pt_scaler = torch.amp.GradScaler(device.type) if device.type == "mps" else None

        pt_best = float("inf")
        for epoch in range(1, args.pretrain_epochs + 1):
            train_loss, train_acc, _ = train_epoch(pt_model, pt_train_loader, pt_criterion,
                                                     pt_optimizer, device, pt_scaler)
            val_loss, val_acc, _, _ = validate(pt_model, pt_val_loader, pt_criterion,
                                                device, num_classes=len(pretrain_classes))
            print(f"  Pretrain epoch {epoch:2d}/{args.pretrain_epochs} | "
                  f"train loss: {train_loss:.3f} acc: {train_acc:.3f} | "
                  f"val loss: {val_loss:.3f} acc: {val_acc:.3f}")
            if val_loss < pt_best:
                pt_best = val_loss
                torch.save(pt_model.state_dict(), run_dir / "pretrain_model.pt")

        pt_model.load_state_dict(torch.load(run_dir / "pretrain_model.pt"))
        model.features.load_state_dict(pt_model.features.state_dict())
        print(f"  Pretrain done. Best val loss: {pt_best:.4f}\n")

    if resume_from_epoch > 0:
        print(f"Phase 2 (resumed): Full training from epoch {resume_from_epoch + 1}")
    else:
        print("Phase 2: Full training")
    print("-" * 40)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type) if device.type == "mps" else None
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 6

    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_epochs", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr", "elapsed_s"])

    per_class_path = run_dir / "per_class.csv"
    if resume_from_epoch > 0:
        # When resuming, copy existing CSVs into new run dir so logs are contiguous
        import shutil
        resume_dir = Path(args.resume)
        if not resume_dir.is_absolute():
            resume_dir = project_root / args.resume
        shutil.copy(resume_dir / "metrics.csv", metrics_path)
        shutil.copy(resume_dir / "per_class.csv", per_class_path)
        weights_src = resume_dir / "weights.csv"
        if weights_src.exists():
            shutil.copy(weights_src, run_dir / "weights.csv")
        print(f"  Copied existing logs to {run_dir}")
    else:
        with open(per_class_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + [IDX_TO_TYPE[i] for i in range(NUM_CLASSES)])

    weights_path = run_dir / "weights.csv"
    weight_header_written = False

    start_epoch = resume_from_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, grad_stats = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
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

        # Log weight stats
        weight_stats = compute_weight_stats(model)
        fieldnames = ["epoch"] + list(weight_stats.keys())
        if not weight_header_written:
            with open(weights_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fieldnames)
            weight_header_written = True
        row = [epoch]
        for name in weight_stats:
            row.append(weight_stats[name]["weight_norm"])
        with open(weights_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        if val_loss < best_val_loss or (resume_from_epoch > 0 and epoch == start_epoch):
            if epoch == start_epoch and resume_from_epoch > 0:
                # Seed best_val_loss from the first resumed epoch
                best_val_loss = val_loss
            else:
                best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("\nPer-class accuracy on best model:")
    model.load_state_dict(torch.load(run_dir / "model.pt"))
    _, _, class_correct, class_total = validate(model, val_loader, criterion, device)
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  {IDX_TO_TYPE[i]:12s}: {acc:.3f} ({class_correct[i]}/{class_total[i]})")


if __name__ == "__main__":
    main()
