# Chess Position Recognition with a Scratch PyTorch CNN

## Abstract

This project predicts the piece or empty state at every square of a chessboard image with a reproducible deep learning pipeline. The model reads local board images, parses the square labels from dash-separated FEN filenames, and trains a scratch PyTorch convolutional neural network without fastai, pretrained models, pretrained weights, or transfer learning.

Final result placeholders in this article should be filled after the extended RTX 4080 training run.

## Problem Definition

Each image represents one chessboard. The model receives the board image and predicts one of 13 labels for each of the 64 board squares.

| Class Group | Labels                                      |
| ----------- | ------------------------------------------- |
| Empty       | `empty`                                     |
| White       | `P`, `N`, `B`, `R`, `Q`, `K`                |
| Black       | `p`, `n`, `b`, `r`, `q`, `k`                |

## Data

Raw data comes from the [Chess Positions](https://www.kaggle.com/datasets/koryakinp/chess-positions) dataset. The files are placed locally before running the pipeline:

- `data/train/` for training and validation images.
- `data/test/` for final held-out evaluation images.

Each image is a 400 by 400 schematic chessboard. The filename stem is a FEN board description with dashes instead of slashes. The pipeline reads these image files on every run. No processed dataset cache is written.

## Image Processing

The pipeline builds labels and tensors in memory:

- dash-separated FEN ranks are expanded into an 8 by 8 label grid;
- piece letters map to piece classes and digits expand to `empty` squares;
- images are opened as RGB, resized to `IMAGE_SIZE`, and normalized to `[-1, 1]`;
- training-time horizontal and vertical flips also flip the label grid;
- brightness, contrast, and color jitter preserve the square labels.

## Model

Only one active architecture is trained: a scratch residual CNN implemented directly with PyTorch.

For each board:

- residual convolution stages learn board, square, and piece visual features;
- downsampling produces an 8 by 8 feature map aligned to board squares;
- a convolutional classification head emits 13-class logits for every square.

This architecture is appropriate for the course scope because it exercises convolutional neural networks, data preprocessing, augmentation, optimization, and end-to-end PyTorch implementation without relying on fastai or pretrained models.

## Evaluation

The training image directory is split into train and validation partitions. The test image directory remains isolated until final evaluation.

Metrics:

- Square accuracy measures the percentage of correctly classified board squares and is the primary metric.
- Occupied-square accuracy measures recognition quality on piece-containing squares.
- Empty-square accuracy measures empty-square recognition quality.
- Board accuracy measures the percentage of images where all 64 squares are correct.
- Per-class precision, recall, and F1 show class-specific behavior.
- Confusion matrix shows square-label error structure.

## Results

Replace this section after the extended training run.

| Split      | Square Accuracy | Occupied Accuracy | Board Accuracy |
| ---------- | --------------- | ----------------- | -------------- |
| Validation | TODO            | TODO              | TODO           |
| Test       | TODO            | TODO              | TODO           |

Generated figures after `just run`:

- `output/figures/confusion_matrix.png`
- `output/figures/class_distribution.png`
- `output/figures/prediction_confidence.png`
- `output/figures/training_loss.png`
- `output/figures/training_accuracy.png`
- `output/figures/batch_loss.png`
- `output/figures/batch_accuracy.png`

## Reproducibility

Run the full pipeline:

```bash
just sync
just run
```

Short CPU smoke run:

```bash
just smoke
```

## Limitations

The model assumes aligned full-board schematic images matching the dataset contract. Images are resized to the configured `IMAGE_SIZE`, so final quality may improve with careful tuning of image size, batch size, learning rate, dropout, and empty-square loss weighting. Final report quality depends on the extended local GPU training run.
