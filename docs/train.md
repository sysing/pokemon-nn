# train.py

Training script for the Pokemon type classifier CNN. Runs on Apple Silicon (MPS), CUDA, or CPU.

## Quick start

```bash
python src/train.py --epochs 30 --batch-size 32
```

## CLI reference

```
python src/train.py [--data-dir DIR] [--epochs N] [--batch-size N] [--lr LR]
                    [--run NAME] [--pretrain TYPE...] [--pretrain-epochs N]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data-dir` | `str` | `data` | Directory with `images/` and `labels.csv` |
| `--epochs` | `int` | `30` | Max training epochs (early stopping may cut it short) |
| `--batch-size` | `int` | `32` | Images per batch |
| `--lr` | `float` | `1e-3` | Starting learning rate |
| `--run` | `str` | `v0.2` | Run name for artifact directory: `runs/<name>/` |
| `--pretrain` | `str` (nargs) | _none_ | Types to pretrain on, e.g. `Dragon Fire` |
| `--pretrain-epochs` | `int` | `10` | Number of pretraining epochs |

## Device selection

```
MPS (Apple Silicon) > CUDA > CPU
```

Auto-detected. On Mac, it uses the GPU via Metal Performance Shaders.

## Training pipeline

### Step-by-step per epoch

```
┌─────────────────────────────────────────────────────────┐
│  EPOCH N                                                │
│                                                         │
│  1. TRAINING LOOP                                       │
│     for batch in train_loader:                          │
│       images, labels → GPU                              │
│       optimizer.zero_grad()                             │
│       outputs = model(images)       # forward           │
│       loss = criterion(outputs, labels)                 │
│       loss.backward()              # compute gradients  │
│       optimizer.step()             # update weights     │
│                                                         │
│  2. VALIDATION                                          │
│     with torch.no_grad():                               │
│       for batch in val_loader:                          │
│         outputs = model(images)                         │
│         track loss, accuracy, per-class accuracy        │
│                                                         │
│  3. LOG                                                 │
│     append to runs/<run>/metrics.csv                    │
│     append to runs/<run>/per_class.csv                  │
│                                                         │
│  4. SAVE                                                │
│     if val_loss < best_val_loss:                        │
│       save model.pt                                     │
│                                                         │
│  5. SCHEDULE                                            │
│     scheduler.step(val_loss)                            │
│     ReduceLROnPlateau: halve LR if stall > 3 epochs     │
│                                                         │
│  6. EARLY STOPPING                                      │
│     if val_loss not improved for 6 epochs:              │
│       break                                             │
└─────────────────────────────────────────────────────────┘
```

### Loss function

**`CrossEntropyLoss`** with:

- **Class weights** — computed as `total_samples / (num_classes × count_per_class)`. Rebalances the loss so the model doesn't tilt toward over-represented types.
- **Label smoothing (α = 0.1)** — target distribution becomes `(1 - α)` on the true class and `α / (num_classes - 1)` on the rest. Prevents overconfidence; improves generalisation on small datasets.

```
Without smoothing:  target = [0, 0, 1, 0, ...]  (100% Fire)
With smoothing:     target = [0.011, 0.011, 0.9, 0.011, ...]  (90% Fire)
```

### Optimizer

**AdamW** — Adam with decoupled weight decay. Hyperparameters:

| Param | Value |
|-------|-------|
| `lr` | `1e-3` (configurable) |
| `weight_decay` | `1e-4` |

### Scheduler

**`ReduceLROnPlateau`** monitors validation loss:

- `mode = "min"` — triggers when val loss stops decreasing
- `factor = 0.5` — halves the learning rate
- `patience = 3` — waits 3 epochs of no improvement before triggering

### Early stopping

Tracks the best validation loss. If the model doesn't beat it for **6 consecutive epochs**, training halts. The best checkpoint (`model.pt`) is always preserved.

## Two-phase training (pretrain)

When `--pretrain` is specified, training runs in two phases:

### Phase 1: pretraining

A new `PokemonTypeCNN` is created with `num_classes = len(pretrain_classes)`. The feature extractor weights are copied from the full model (so they start from the same random init).

```
pretrain types: ["Dragon", "Fire", "Grass"]
pretrain model:  features (shared init) + classifier (3 outputs)
pretrain data:   only Dragon, Fire, Grass images (remapped to labels 0,1,2)
pretrain epochs: 10
save:            runs/<run>/pretrain_model.pt
```

### Phase 2: full training

The pretrained feature extractor weights (`features.*`) are copied into the full 10-class model. Training then proceeds normally on all types.

```
full model:  pretrained features + fresh classifier (10 outputs)
full data:   all 10 types
full epochs: 30
save:        runs/<run>/model.pt
```

This is a form of self-supervised pretraining — no external data needed. It helps when certain types are hard to distinguish and benefit from the features learned on easier types.

## Output files

All saved under `runs/<run_name>/`:

| File | Schema | Description |
|------|--------|-------------|
| `model.pt` | PyTorch state_dict | Best model (lowest val loss) |
| `pretrain_model.pt` | PyTorch state_dict | Best pretrain model (if `--pretrain`) |
| `metrics.csv` | `epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, lr, elapsed_s` | Epoch-level metrics |
| `per_class.csv` | `epoch, Colorless, Darkness, Dragon, Fighting, Fire, Grass, Lightning, Metal, Psychic, Water` | Per-type val accuracy per epoch |

## Expected results

On ~934 images (M4 MacBook Air, MPS):

| Epoch | Val Accuracy | Time |
|-------|-------------|------|
| 1 | ~70% | ~15s |
| 6 | ~86% | ~90s |
| 15 | ~90% | ~3.5m |
| 30 | ~92% | ~7m |

Hardest types: Colorless (~80%), Fighting (~87%), Psychic (~87%)  
Easiest types: Grass (~100%), Lightning (~100%), Darkness (~94%)

## Visualise while training

In a second terminal:

```bash
python src/viz.py runs/<run>/metrics.csv
```

Or after training:

```bash
python src/plot.py --metrics runs/<run>/metrics.csv --output runs/<run>/plot.png
```

## Complete example

```bash
# 1. Collect data
python src/collect.py

# 2. Basic training
python src/train.py --epochs 30 --batch-size 32 --run v0.4

# 3. Or with pretraining
python src/train.py --pretrain Dragon Fire Grass --pretrain-epochs 10 --epochs 30 --run v0.4-pt

# 4. Predict
python src/predict.py path/to/card.jpg --model runs/v0.4/model.pt
```
