---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Map

The repository is command-first and centered on one root script.

```
/
├── data/
│   ├── train.csv              # Local Kaggle Yelp training CSV, ignored
│   └── test.csv               # Local Kaggle Yelp test CSV, ignored
├── docs/                      # Documentation source
├── output/
│   ├── figures/               # Training and evaluation plots
│   ├── models/                # PyTorch checkpoints
│   ├── predictions/           # Sentiment prediction outputs
│   └── reports/               # Metrics and generated summaries
├── yrs.py                     # Read, train, evaluate, and generate artifacts
├── article.md                 # Article draft using generated outputs
├── justfile                   # Project command wrapper
├── pyproject.toml             # Python dependency config
├── uv.lock                    # Python dependency lock
└── zensical.toml              # Website configuration
```

Generated model artifacts live outside source-controlled code. The raw Yelp CSV files are local inputs and are also ignored.

## Script Responsibilities

| Part of `yrs.py` | Responsibility                                                               |
| ---------------- | ---------------------------------------------------------------------------- |
| Data loading     | Read `data/train.csv` and `data/test.csv` every run                          |
| Splitting        | Create a stratified validation split from `data/train.csv`                   |
| Vocabulary       | Build an in-memory token vocabulary from the training split only             |
| Encoding         | Convert review text into padded fixed-length token id sequences              |
| Training         | Train the scratch PyTorch TextCNN classifier                                 |
| Evaluation       | Write predictions, metrics, per-class reports, checkpoints, and figures      |

## Data Files

| Path             | Purpose                         | Git Policy |
| ---------------- | ------------------------------- | ---------- |
| `data/train.csv` | Training and validation source  | ignored    |
| `data/test.csv`  | Final held-out evaluation split | ignored    |

No `data/processed/` cache is used by the active pipeline. Tokenization, vocabulary construction, and sequence encoding run in memory each time.

## Output Directories

| Path                  | Purpose                                                        | Git Policy         |
| --------------------- | -------------------------------------------------------------- | ------------------ |
| `output/models/`      | Trained PyTorch checkpoints                                    | ignored            |
| `output/predictions/` | Prediction CSV and Parquet outputs                             | ignored            |
| `output/reports/`     | Metrics, class reports, training summaries, and history tables | ignored            |
| `output/figures/`     | Training and evaluation plots                                  | ignored by default |

## Command Interface

| Recipe       | Command                    | Responsibility                           |
| ------------ | -------------------------- | ---------------------------------------- |
| `just sync`  | `uv sync`                  | Install or update the Python environment |
| `just run`   | `uv run python yrs.py`     | Run the complete pipeline                |
| `just smoke` | constrained `yrs.py` run   | Run a short CPU smoke pipeline           |
| `just check` | formatting and lint checks | Verify source and documentation style    |
| `just fix`   | format and lint fixes      | Apply automated formatting fixes         |
| `just ci`    | `just check`               | Run the current verification gate        |

### Deployment

Documentation is deployed through `.github/workflows/docs.yml`. On pushes to `main`, GitHub Pages builds the site with Zensical and publishes the generated site directory.
