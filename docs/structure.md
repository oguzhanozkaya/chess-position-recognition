---
description: Project structure, file organization, and tooling reference.
---

# Structure

## Repository Map

The repository is command-first and centered on one root script.

```
/
├── data/
│   ├── train/                 # Local Kaggle chess training images, ignored
│   └── test/                  # Local Kaggle chess test images, ignored
├── docs/                      # Documentation source
├── output/
│   ├── figures/               # Training and evaluation plots
│   ├── models/                # PyTorch checkpoints
│   ├── predictions/           # Square prediction outputs
│   └── reports/               # Metrics and generated summaries
├── presentation/
│   └── chess-position-recognition.tex # Single-file Beamer presentation source
├── cpr.py                     # Read, train, evaluate, and generate artifacts
├── article.md                 # Article draft using generated outputs
├── justfile                   # Project command wrapper
├── pyproject.toml             # Python dependency config
├── uv.lock                    # Python dependency lock
└── zensical.toml              # Website configuration
```

Generated model artifacts live outside source-controlled code. The raw chess image files are local inputs and are also ignored.

## Script Responsibilities

| Part of `cpr.py` | Responsibility                                                               |
| ---------------- | ---------------------------------------------------------------------------- |
| Data loading     | Read `data/train/` and `data/test/` images every run                         |
| Label parsing    | Parse 64 square labels from dash-separated FEN filename stems                |
| Splitting        | Create a random validation split from `data/train/`                          |
| Training         | Train the scratch PyTorch board CNN classifier                               |
| Evaluation       | Write predictions, metrics, per-class reports, checkpoints, and figures      |

## Data Files

| Path             | Purpose                         | Git Policy |
| ---------------- | ------------------------------- | ---------- |
| `data/train/` | Training and validation source  | ignored    |
| `data/test/`  | Final held-out evaluation split | ignored    |

No `data/processed/` cache is used by the active pipeline. Images and filename labels are read in memory each time.

## Output Directories

| Path                  | Purpose                                                        | Git Policy         |
| --------------------- | -------------------------------------------------------------- | ------------------ |
| `output/models/`      | Trained PyTorch checkpoints                                    | ignored            |
| `output/predictions/` | Prediction CSV and Parquet outputs                             | ignored            |
| `output/reports/`     | Metrics, class reports, training summaries, and history tables | ignored            |
| `output/figures/`     | Training and evaluation plots                                  | ignored by default |
| `presentation/build/`  | Intermediate TeX build files                                   | ignored            |
| `presentation/*.pdf`   | Generated presentation PDFs                                    | ignored            |

## Command Interface

| Recipe       | Command                    | Responsibility                           |
| ------------ | -------------------------- | ---------------------------------------- |
| `just sync`  | `uv sync`                  | Install or update the Python environment |
| `just run`   | `uv run python cpr.py`     | Run the complete pipeline                |
| `just smoke` | constrained `cpr.py` run   | Run a short CPU smoke pipeline           |
| `just check` | formatting and lint checks | Verify source and documentation style    |
| `just fix`   | format and lint fixes      | Apply automated formatting fixes         |
| `just ci`    | `just check`               | Run the current verification gate        |
| `just presentation-build` | `lualatex` build       | Build the Beamer presentation PDF        |
| `just presentation-clean` | generated file cleanup  | Remove presentation build artifacts      |

### Deployment

Documentation is deployed through `.github/workflows/docs.yml`. On pushes to `main`, GitHub Pages builds the site with Zensical and publishes the generated site directory.
