# Specification

## Project Scope

This project recognizes chess board positions from schematic board images. The target is the piece or empty state at each of the 64 board squares.

The pipeline is a single root script, `cpr.py`. It reads the local Kaggle Chess Positions images from `data/` on every run, parses labels from filenames in memory, trains one scratch PyTorch CNN, and writes evaluation artifacts.

## Recognition Target

| Field        | Definition                                                                    |
| ------------ | ----------------------------------------------------------------------------- |
| Target       | 64 square labels: `empty`, white pieces, or black pieces                      |
| Input        | 400 by 400 RGB chessboard image                                               |
| Label rule   | Filename stem is FEN ranks with dashes instead of slashes                     |
| Split rule   | Random validation split from `data/train/`; final test uses `data/test/`      |
| Main metric  | Square-level accuracy                                                         |
| Extra metrics | Occupied-square accuracy, empty-square accuracy, and full-board accuracy     |

## Inputs

The source dataset is [Chess Positions](https://www.kaggle.com/datasets/koryakinp/chess-positions). The dataset must be placed locally under `data/` before running the pipeline.

| Path           | Format                  | Purpose                         |
| -------------- | ----------------------- | ------------------------------- |
| `data/train/`  | JPEG board images       | Training and validation source  |
| `data/test/`   | JPEG board images       | Final held-out evaluation       |

No Kaggle download step is implemented. No processed dataset cache is written; images and filename labels are read on each run.

## Data Contract

Each image represents one chessboard.

| Field Group | Required Content                                                   |
| ----------- | ------------------------------------------------------------------ |
| Image       | RGB chessboard image                                               |
| Label       | Filename stem containing 8 dash-separated FEN ranks                |
| Squares     | 64 labels mapped to 13 class ids                                   |
| Split       | `train`, `validation`, or `test`                                   |
| Classes     | `empty`, `P`, `N`, `B`, `R`, `Q`, `K`, `p`, `n`, `b`, `r`, `q`, `k` |

## Outputs

| Output              | Format             | Directory             |
| ------------------- | ------------------ | --------------------- |
| Model checkpoint    | PyTorch checkpoint | `output/models/`      |
| Predictions         | CSV and Parquet    | `output/predictions/` |
| Metrics and reports | JSON, CSV, Markdown | `output/reports/`    |
| Figures             | PNG                | `output/figures/`     |

Training reports include epoch-level history and per-batch training/validation history. Training figures show batch-level loss and square accuracy, with validation epoch points overlaid on the main training curves.

## Usage

Local run:

```bash
just sync
just run
```

Short CPU smoke run:

```bash
just smoke
```

Direct script run:

```bash
uv run python cpr.py
```

## Configuration

Pipeline and model settings are Python constants near the top of `cpr.py`. Edit those constants before running the script.

| Constant                    | Default                 | Description                                      |
| --------------------------- | ----------------------- | ------------------------------------------------ |
| `SEED`                      | `67`                    | Random seed.                                     |
| `EPOCHS`                    | `40`                    | Maximum training epochs.                         |
| `PATIENCE`                  | `6`                     | Early-stopping patience.                         |
| `BATCH_SIZE`                | `128`                   | Mini-batch image count.                          |
| `LEARNING_RATE`             | `0.0008`                | AdamW learning rate.                             |
| `WEIGHT_DECAY`              | `0.0001`                | AdamW weight decay.                              |
| `DEVICE`                    | `cuda`                  | Training device: `cuda`, `cpu`, or `auto`.       |
| `VALIDATION_SIZE`           | `0.1`                   | Validation fraction from training images.        |
| `IMAGE_SIZE`                | `256`                   | Resized image side length.                       |
| `DROPOUT`                   | `0.15`                  | CNN dropout.                                     |
| `LABEL_SMOOTHING`           | `0.01`                  | Cross-entropy label smoothing.                   |
| `EMPTY_CLASS_WEIGHT`        | `0.45`                  | Loss weight for empty squares.                   |
| `DATALOADER_WORKERS`        | `4`                     | DataLoader worker count.                         |
| `MIXED_PRECISION`           | `True`                  | Use CUDA automatic mixed precision.              |
| `COMPILE_MODEL`             | `False`                 | Compile the PyTorch model before training.       |
| `ROW_LIMIT`                 | `0`                     | Optional smoke/debug image limit; `0` means all. |
| `CHECKPOINT_INTERVAL_EPOCHS` | `1`                    | Save periodic checkpoints every N epochs.        |

## Success Criteria

The project is complete when `just run` reads local chessboard images, trains the scratch PyTorch CNN, evaluates validation and test splits, and generates report-ready outputs. The only expected remaining work after migration is running extended local RTX 4080 training and filling final article metrics.
