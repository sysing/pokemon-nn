# pokemon-nn

Toy neural network for classifying Pokemon card types from card images. Trains on Apple Silicon (MPS).

**Goal**: given a Pokemon card image, predict the Pokemon type (Fire, Water, Grass, etc.).

## Setup

```bash
pip install -r requirements.txt
python src/collect.py        # download ~900 card images + labels
```

## Train

```bash
python src/train.py --epochs 30 --batch-size 32
```

Runs on MPS (Apple Silicon GPU) by default. Logs metrics to `metrics.csv` and `per_class.csv`.

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
