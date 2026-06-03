# Specification

## Project Scope

This project predicts the final result of a football match from in-play information available through minute 60. The target is a three-class outcome: home win, draw, or away win.

The pipeline is a single root script, `fig.py`. It downloads the Kaggle ESPN Soccer dataset when local raw files are missing, builds leakage-safe first-60-minute features, trains one raw-PyTorch classifier, and writes evaluation artifacts.

## Forecast Target

| Field          | Definition                                                                                |
| -------------- | ----------------------------------------------------------------------------------------- |
| Target         | Final match outcome: `home`, `draw`, or `away`                                            |
| Forecast time  | Minute 60                                                                                 |
| Forecast rule  | Use only plays, key events, commentary, and safe lineup data available through minute 60. |
| Split strategy | Chronological train, validation, and test periods within each league-season key           |
| Main metric    | Accuracy and macro F1                                                                     |

## Inputs

The source dataset is [ESPN Soccer Data](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data). `fig.py` uses Kaggle credentials to download it into `data/raw/` if the expected directories are absent.

| Directory                 | Purpose                                                                             | Used                                     |
| ------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| `base_data/fixtures.csv`  | Match dates, teams, status, final scores, and labels                                | Yes                                      |
| `base_data/leagues.csv`   | League and season metadata                                                          | Yes                                      |
| `plays_data/*.csv`        | Play-by-play event text, clocks, teams, scoring flags, event types, and coordinates | Yes                                      |
| `keyEvents_data/*.csv`    | Important events with clocks, teams, text, event types, and coordinates             | Yes                                      |
| `commentary_data/*.csv`   | Minute-by-minute commentary text                                                    | Yes                                      |
| `lineup_data/*.csv`       | Formations, starters, positions, and substitution metadata                          | Safe pre-match fields only               |
| `playerStats_data/*.csv`  | Season player aggregates                                                            | No, excluded until lagged features exist |
| `base_data/teamStats.csv` | Full-match team statistics                                                          | No, excluded to prevent leakage          |
| `base_data/standings.csv` | Scrape-time standings snapshots                                                     | No, excluded until lagged features exist |

## Data Contract

Each observation in `data/processed/model_dataset.parquet` represents one completed match sampled at minute 60.

| Field Group  | Required Content                                                                      |
| ------------ | ------------------------------------------------------------------------------------- |
| Time keys    | Match date, split, league, league-season key, event id                                |
| Target       | Final result label and numeric class id                                               |
| Text         | One tokenized first-60-minute text sequence per match                                 |
| Numeric      | One first-60-minute numeric feature vector per match                                  |
| Leakage rule | No play, key event, commentary, or unsafe lineup data after minute 60 may enter input |

## Outputs

| Output              | Format             | Directory             |
| ------------------- | ------------------ | --------------------- |
| Model-ready dataset | Parquet            | `data/processed/`     |
| Split summaries     | CSV                | `data/processed/`     |
| Model checkpoint    | PyTorch checkpoint | `output/models/`      |
| Predictions         | CSV and Parquet    | `output/predictions/` |
| Metrics and reports | JSON and Markdown  | `output/reports/`     |
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
uv run python fig.py
```

## Kaggle Credentials

Local options:

- Place `kaggle.json` at `~/.kaggle/kaggle.json` and run `chmod 600 ~/.kaggle/kaggle.json`.
- Or export `KAGGLE_USERNAME` and `KAGGLE_KEY` in the shell.

Colab options:

- Upload `kaggle.json` to `/root/.kaggle/kaggle.json` and run `chmod 600 /root/.kaggle/kaggle.json`.
- Or store `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab secrets and assign them to environment variables before running `fig.py`.

Never commit `kaggle.json` or credential values.

## Configuration

Pipeline and model settings are Python constants near the top of `fig.py`. Edit those constants before running the script.

| Constant                   | Default     | Description                                      |
| -------------------------- | ----------- | ------------------------------------------------ |
| `SEED`                     | `67`        | Random seed.                                     |
| `EPOCHS`                   | `1200`      | Maximum training epochs.                         |
| `PATIENCE`                 | `50`        | Early-stopping patience.                         |
| `BATCH_SIZE`               | `512`       | Mini-batch size.                                 |
| `LEARNING_RATE`            | `0.0001`    | AdamW learning rate.                             |
| `WEIGHT_DECAY`             | `0.0001`    | AdamW weight decay.                              |
| `EARLY_STOPPING_MIN_DELTA` | `0.001`     | Minimum validation-loss improvement.             |
| `DEVICE`                   | `cuda`      | Training device: `cuda`, `cpu`, or `auto`.       |
| `CUTOFF_MINUTE`            | `60`        | Last match minute allowed in model inputs.       |
| `MAX_TOKENS`               | `256`       | Maximum first-60-minute text tokens per match.   |
| `MAX_VOCAB_SIZE`           | `6000`      | Maximum train-split vocabulary size.             |
| `TEXT_EMBEDDING_DIM`       | `64`        | Text embedding dimension.                        |
| `TEXT_CHANNEL_COUNT`       | `48`        | TextCNN channels per kernel size.                |
| `TEXT_KERNEL_SIZES`        | `(3, 4, 5)` | TextCNN kernel sizes.                            |
| `NUMERIC_HIDDEN_SIZE`      | `128`       | Numeric MLP hidden width.                        |
| `FUSION_HIDDEN_SIZE`       | `128`       | Final fusion classifier width.                   |
| `DROPOUT`                  | `0.25`      | Numeric and fusion dropout.                      |
| `TEXT_DROPOUT`             | `0.25`      | Text branch dropout.                             |
| `DATALOADER_WORKERS`       | `4`         | DataLoader worker count.                         |
| `MIXED_PRECISION`          | `True`      | Use CUDA automatic mixed precision.              |
| `COMPILE_MODEL`            | `False`     | Compile the PyTorch model before training.       |
| `MATCH_LIMIT`              | `0`         | Optional smoke/debug match limit; `0` means all. |

## Success Criteria

The project is complete when `just run` can download or reuse local ESPN data, build leakage-safe first-60-minute match features, train the single TextCNN plus numeric MLP classifier, evaluate chronological league-aware splits, and generate report-ready outputs.
