"""
Visualize training metrics from metrics.csv.

Usage:
    python src/plot.py [--metrics metrics.csv] [--output plot.png] [--no-show]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd


def plot(metrics_path: str, output_path: str = None, show: bool = True):
    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pokemon Type Classifier — Training Metrics", fontsize=14)

    # Loss
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="train", marker="o", markersize=3)
    ax.plot(df["epoch"], df["val_loss"], label="val", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_acc"], label="train", marker="o", markersize=3)
    ax.plot(df["epoch"], df["val_acc"], label="val", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Learning rate
    ax = axes[1, 0]
    ax.plot(df["epoch"], df["lr"], marker="o", markersize=3, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Time per epoch
    ax = axes[1, 1]
    ax.bar(df["epoch"], df["elapsed_s"], color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title("Time per Epoch")
    ax.axhline(y=df["elapsed_s"].mean(), color="red", linestyle="--",
               label=f'mean = {df["elapsed_s"].mean():.1f}s')
    ax.legend()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--metrics", default="metrics.csv", help="Path to metrics CSV")
    parser.add_argument("--output", default="plot.png", help="Output image path")
    parser.add_argument("--no-show", action="store_true", help="Don't show plot")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    metrics_path = project_root / args.metrics
    if not metrics_path.exists():
        print(f"No metrics file at {metrics_path}. Train first: python src/train.py")
        return

    output_path = str(project_root / args.output) if args.output else None
    plot(str(metrics_path), output_path, show=not args.no_show)


if __name__ == "__main__":
    main()
