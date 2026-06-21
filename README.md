# pokemon-nn

Toy neural network for classifying Pokemon card types from card images. Trains on Apple Silicon (MPS).

**Goal**: given a Pokemon card image, predict the Pokemon type (Fire, Water, Grass, etc.).

**[Project State & Experiment Log](https://sysing.github.io/pokemon-nn/state.html)** — full release history, per-type accuracy comparison, architecture details.

## Setup

```bash
pip install -r requirements.txt
python src/collect.py        # download ~900 card images + labels
```

## Train

### Quick start

```bash
python src/train.py --epochs 30 --batch-size 32
```

Runs on MPS (Apple Silicon GPU) by default, falls back to CUDA then CPU.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data` | Directory containing `images/` and `labels.csv` |
| `--epochs` | `30` | Maximum training epochs (early stopping may end sooner) |
| `--batch-size` | `32` | Images per batch. Larger = faster but more memory |
| `--lr` | `1e-3` | Starting learning rate for AdamW |
| `--run` | `v0.2` | Run name — artifacts saved to `runs/<name>/` |
| `--pretrain` | *(none)* | Space-separated type names to pretrain on first |
| `--pretrain-epochs` | `10` | Number of pretraining epochs |

### Two-phase training (pretrain)

You can optionally pretrain the feature extractor on a subset of types before training on all 10:

```bash
python src/train.py --pretrain Dragon Fire Grass --pretrain-epochs 10 --epochs 30
```

1. **Phase 1:** A temporary model trains *only* on the specified types. Teaches the convolutional blocks to focus on distinctive visual patterns (e.g., flames vs. leaves vs. dragon silhouettes).
2. **Phase 2:** The learned features are copied into the full model, which then trains on all 10 types.

This is most useful when the overall dataset is small or some types are hard to distinguish.

### What happens during training

Each epoch:

1. **Training loop** — for each batch of 32 images:
   - Forward pass through the CNN → 10 logits per image
   - Compute **CrossEntropyLoss** (with class weights + label smoothing 0.1)
   - Backward pass → AdamW optimizer step
2. **Validation** — run the held-out 20% through the model (no gradients) and compute loss + per-class accuracy
3. **Log** — append epoch metrics to `runs/<name>/metrics.csv` and per-class accuracy to `per_class.csv`
4. **Save** — checkpoint `model.pt` if validation loss improved
5. **Schedule** — `ReduceLROnPlateau` halves the learning rate if val loss stalls for 3 epochs
6. **Stop** — early stopping triggers if val loss hasn't improved in 6 epochs

### Loss function

`CrossEntropyLoss` with two enhancements:

- **Class weights** — `total / (num_classes × count)` per class. Prevents the model from ignoring underrepresented types.
- **Label smoothing (0.1)** — trains against 90% confidence instead of 100%. Reduces overfitting and improves calibration.

### Output files (in `runs/<name>/`)

| File | Contents |
|------|----------|
| `model.pt` | Best checkpoint (lowest validation loss) |
| `metrics.csv` | Epoch-level: `epoch, train_loss, train_acc, val_loss, val_acc, lr, elapsed_s` |
| `per_class.csv` | Per-type validation accuracy per epoch |
| `pretrain_model.pt` | Best pretrain checkpoint (only if `--pretrain` used) |

## Visualize

**Live terminal dashboard** (run in a second terminal while training):

```bash
python src/viz.py
```

**Static plot** after training:

```bash
python src/plot.py
```

## Predict

```bash
python src/predict.py path/to/card.jpg
```

## Architecture

| Layer | Details |
|-------|---------|
| Conv Block 1 | Conv2d(3→32) → BN → ReLU → MaxPool |
| Conv Block 2 | Conv2d(32→64) → BN → ReLU → MaxPool |
| Conv Block 3 | Conv2d(64→128) → BN → ReLU → MaxPool |
| Conv Block 4 | Conv2d(128→256) → BN → ReLU → AdaptiveAvgPool |
| Classifier | Dropout(0.5) → Linear(256→10) |

~600K parameters. 10 output classes (Colorless, Darkness, Dragon, Fighting, Fire, Grass, Lightning, Metal, Psychic, Water).

## Data

~900 images from [pokemontcg.io](https://pokemontcg.io) API. Includes multi-type cards (labeled by primary type). Class weights compensate for type imbalance.

## Results

~82% validation accuracy in 5 epochs, ~90%+ in 30 epochs on M4 MacBook Air.
