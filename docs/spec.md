# Specification

## Project Scope

This project predicts Yelp review polarity from raw review text. The target is binary sentiment: `negative` or `positive`.

The pipeline is a single root script, `yrs.py`. It reads the local Kaggle Yelp CSV files from `data/` on every run, builds an in-memory vocabulary from the training split, trains one scratch PyTorch TextCNN classifier, and writes evaluation artifacts.

## Forecast Target

| Field        | Definition                                                       |
| ------------ | ---------------------------------------------------------------- |
| Target       | Yelp review sentiment: `negative` or `positive`                  |
| Input        | Raw English review text                                          |
| Label rule   | Kaggle label `1` maps to `negative`; label `2` maps to `positive` |
| Split rule   | Stratified validation split from `data/train.csv`; final test uses `data/test.csv` |
| Main metrics | Accuracy, macro F1, and log loss                                 |

## Inputs

The source dataset is [Yelp Review Dataset](https://www.kaggle.com/datasets/ilhamfp31/yelp-review-dataset). The dataset must be placed locally under `data/` before running the pipeline.

| File             | Format                  | Purpose                         |
| ---------------- | ----------------------- | ------------------------------- |
| `data/train.csv` | Headerless CSV: label, text | Training and validation source |
| `data/test.csv`  | Headerless CSV: label, text | Final held-out evaluation      |

No Kaggle download step is implemented. No processed dataset cache is written; text preprocessing is rebuilt in memory on each run.

## Data Contract

Each row represents one Yelp review.

| Field Group | Required Content                                      |
| ----------- | ----------------------------------------------------- |
| Label       | Raw label `1` or `2`, mapped to numeric class id      |
| Text        | Review text tokenized by the pipeline                 |
| Split       | `train`, `validation`, or `test`                      |
| Vocabulary  | Built only from the active training split each run    |

## Outputs

| Output              | Format             | Directory             |
| ------------------- | ------------------ | --------------------- |
| Model checkpoint    | PyTorch checkpoint | `output/models/`      |
| Predictions         | CSV and Parquet    | `output/predictions/` |
| Metrics and reports | JSON, CSV, Markdown | `output/reports/`    |
| Figures             | PNG                | `output/figures/`     |

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
uv run python yrs.py
```

## Configuration

Pipeline and model settings are Python constants near the top of `yrs.py`. Edit those constants before running the script.

| Constant                    | Default                 | Description                                      |
| --------------------------- | ----------------------- | ------------------------------------------------ |
| `SEED`                      | `67`                    | Random seed.                                     |
| `EPOCHS`                    | `20`                    | Maximum training epochs.                         |
| `PATIENCE`                  | `4`                     | Early-stopping patience.                         |
| `BATCH_SIZE`                | `256`                   | Mini-batch size.                                 |
| `LEARNING_RATE`             | `0.001`                 | AdamW learning rate.                             |
| `WEIGHT_DECAY`              | `0.0001`                | AdamW weight decay.                              |
| `DEVICE`                    | `cuda`                  | Training device: `cuda`, `cpu`, or `auto`.       |
| `VALIDATION_SIZE`           | `0.1`                   | Stratified validation fraction from train CSV.   |
| `MAX_VOCAB_SIZE`            | `80000`                 | Maximum vocabulary size including special tokens. |
| `MIN_TOKEN_FREQUENCY`       | `2`                     | Minimum frequency for vocabulary inclusion.      |
| `MAX_SEQUENCE_LENGTH`       | `256`                   | Review token length after truncation/padding.    |
| `EMBEDDING_DIM`             | `256`                   | Learned word embedding width.                    |
| `FILTER_COUNT`              | `256`                   | Filters per convolution width.                   |
| `KERNEL_SIZES`              | `(2, 3, 4, 5)`          | TextCNN convolution widths.                      |
| `HIDDEN_SIZE`               | `256`                   | Classifier hidden size.                          |
| `DROPOUT`                   | `0.45`                  | Classifier dropout.                              |
| `WORD_DROPOUT`              | `0.04`                  | Training-time token replacement augmentation.    |
| `LABEL_SMOOTHING`           | `0.02`                  | Cross-entropy label smoothing.                   |
| `DATALOADER_WORKERS`        | `4`                     | DataLoader worker count.                         |
| `MIXED_PRECISION`           | `True`                  | Use CUDA automatic mixed precision.              |
| `COMPILE_MODEL`             | `False`                 | Compile the PyTorch model before training.       |
| `ROW_LIMIT`                 | `0`                     | Optional smoke/debug row limit; `0` means all.   |
| `CHECKPOINT_INTERVAL_EPOCHS` | `1`                    | Save periodic checkpoints every N epochs.        |

## Success Criteria

The project is complete when `just run` reads the local Yelp CSV files, trains the scratch PyTorch TextCNN, evaluates validation and test splits, and generates report-ready outputs. The only expected remaining work after migration is running extended local GPU training and filling final article metrics.
