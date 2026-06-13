"""
Live terminal visualizer for training metrics using Rich.

Usage (standalone):
    python src/viz.py metrics.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="loss_acc", ratio=1),
        Layout(name="per_class", ratio=1),
    )
    return layout


def loss_plot(values, width: int = 40, color: str = "cyan") -> str:
    if not values:
        return "—" * width
    chars = " ▁▂▃▄▅▆▇█"
    val_min, val_max = min(values), max(values)
    span = val_max - val_min if val_max != val_min else 1
    scaled = [int((v - val_min) / span * (len(chars) - 1)) for v in values]
    max_bar = width
    step = max(1, len(values) // max_bar)
    return "".join(chars[min(s, len(chars) - 1)] for s in scaled[::step])[-max_bar:]


def render(console: Console, epoch: int, total_epochs: int,
           train_losses: list, val_losses: list,
           train_accs: list, val_accs: list,
           elapsed: float, eta: float,
           per_class: dict = None):
    layout = make_layout()

    # Header
    header_text = Text(
        f"Pokemon NN — Training Epoch {epoch}/{total_epochs}    "
        f"{elapsed:.1f}s elapsed    ETA {eta:.0f}s    "
        f"MPS (Apple Silicon)",
        style="bold white on blue",
    )
    layout["header"].update(Panel(header_text))

    # Loss & Accuracy panel
    la_table = Table(expand=True, show_header=False, padding=(0, 1))
    la_table.add_column("metric")
    la_table.add_column("value")
    la_table.add_column("spark")

    if train_losses:
        la_table.add_row(
            "train loss", f"[cyan]{train_losses[-1]:.4f}[/]",
            f"[dim cyan]{loss_plot(train_losses, 35, 'cyan')}[/]"
        )
        la_table.add_row(
            "val loss  ", f"[green]{val_losses[-1]:.4f}[/]",
            f"[dim green]{loss_plot(val_losses, 35, 'green')}[/]"
        )
        la_table.add_row("", "", "")
        la_table.add_row(
            "train acc ", f"[cyan]{train_accs[-1]:.3f}[/]",
            f"[dim cyan]{loss_plot([a for a in train_accs], 35, 'cyan')}[/]"
        )
        la_table.add_row(
            "val acc   ", f"[green]{val_accs[-1]:.3f}[/]",
            f"[dim green]{loss_plot([a for a in val_accs], 35, 'green')}[/]"
        )
        val_acc_pct = val_accs[-1] * 100
        color = "green" if val_acc_pct > 80 else "yellow" if val_acc_pct > 60 else "red"
        la_table.add_row("val acc bar", "", "")
        la_table.add_row(
            ProgressBar(total=100, completed=int(val_acc_pct), width=30),
            f"[{color}]{val_acc_pct:.1f}%[/]",
            "",
        )

    layout["loss_acc"].update(Panel(la_table, title="Loss & Accuracy"))

    # Per-class panel
    if per_class:
        pc_table = Table(expand=True, padding=(0, 1))
        pc_table.add_column("Class", style="bold")
        pc_table.add_column("Acc")
        pc_table.add_column("Bar")
        for cls_name, acc in sorted(per_class.items(), key=lambda x: -x[1]):
            pct = int(acc * 100)
            color = "green" if pct > 80 else "yellow" if pct > 50 else "red"
            pc_table.add_row(
                cls_name,
                f"[{color}]{acc:.3f}[/]",
                ProgressBar(total=100, completed=pct, width=20),
            )
        layout["per_class"].update(Panel(pc_table, title="Per-Class Accuracy"))

    return layout


def watch(metrics_path: str, refresh: float = 2.0):
    """Live-watch a training run by polling metrics.csv."""
    console = Console()
    path = Path(metrics_path)
    per_class_path = path.parent / "per_class.csv"

    if not path.exists():
        console.print(f"[red]Waiting for {metrics_path}... (start training)[/]")

    total_epochs = "?"
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    per_class = {}
    last_lines = 0
    epoch_start = None

    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            if not path.exists():
                time.sleep(refresh)
                continue

            try:
                df = pd.read_csv(path)
            except Exception:
                time.sleep(refresh)
                continue

            if len(df) == 0:
                time.sleep(refresh)
                continue

            if len(df) != last_lines:
                last_lines = len(df)
                last = df.iloc[-1]
                epoch = int(last["epoch"])
                if "total_epochs" in df.columns:
                    total_epochs = str(int(df["total_epochs"].iloc[-1]))

                train_losses.append(last["train_loss"])
                val_losses.append(last["val_loss"])
                train_accs.append(last["train_acc"])
                val_accs.append(last["val_acc"])

                if epoch_start is None:
                    epoch_start = time.time()
                elapsed = time.time() - epoch_start
                eta = (elapsed / epoch) * (int(total_epochs if total_epochs != "?" else 30) - epoch)

            # Always try to load per-class (may be written after metrics)
            try:
                if per_class_path.exists():
                    pc_df = pd.read_csv(per_class_path)
                    if len(pc_df) > 0:
                        latest = pc_df.iloc[-1]
                        per_class = {col: float(latest[col])
                                     for col in pc_df.columns if col != "epoch"}
            except Exception:
                pass

            live_layout = render(
                console, epoch if last_lines else 0, total_epochs,
                train_losses, val_losses, train_accs, val_accs,
                elapsed if last_lines else 0, eta if last_lines else 0,
                per_class,
            )
            live.update(live_layout)

            time.sleep(refresh)


def main():
    parser = argparse.ArgumentParser(description="Live training visualizer")
    parser.add_argument("metrics", nargs="?", default="metrics.csv", help="Path to metrics CSV")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval (s)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    metrics_path = project_root / args.metrics
    watch(str(metrics_path), args.refresh)


if __name__ == "__main__":
    main()
