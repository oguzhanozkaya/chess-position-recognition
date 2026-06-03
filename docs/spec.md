# Specification

## Project Scope

This project forecasts next-month consumer price inflation using a reproducible deep learning pipeline. The target variable is monthly CPI inflation, not yearly CPI inflation.

The project combines structured macro-financial indicators with economic text sources. Numeric data captures observable economic conditions such as currency rates, commodity prices, policy rates, and CPI history. Text data captures forward-looking inflation pressure from central bank publications and economic news.

## Forecast Target

| Field          | Definition                                                   |
| -------------- | ------------------------------------------------------------ |
| Target         | CPI month-over-month inflation                               |
| Forecast step  | One month ahead                                              |
| Forecast rule  | At the end of month `t`, forecast CPI MoM for month `t + 1`. |
| Split strategy | Chronological train, validation, and test periods            |
| Main metric    | Mean absolute error                                          |

The target is CPI MoM because it directly measures the monthly inflation movement to be forecast. CPI YoY is not used as a target.

## Inputs

### Numeric Sources

The numeric pipeline should use reproducible source downloads instead of committed full datasets.

| Category        | Examples                                                   | Purpose                                 |
| --------------- | ---------------------------------------------------------- | --------------------------------------- |
| Inflation       | CPI index, CPI MoM                                         | Target construction and lagged features |
| Monetary policy | CBRT policy rate, funding rates                            | Domestic monetary stance                |
| FX              | USD/TRY, EUR/TRY, currency basket                          | Imported inflation pressure             |
| Commodities     | Brent oil, natural gas, food or commodity indexes          | External cost pressure                  |
| Markets         | BIST index, bond yields, CDS if reproducibly available     | Financial conditions                    |
| Expectations    | Inflation expectations if available from reproducible data | Forward-looking macro-financial context |

The implemented numeric sources include the CBRT Consumer Prices page, the CBRT public indicative exchange-rate XML archive, and public FRED CSV series for Brent crude oil, Turkish industrial production growth, and Turkish unemployment. The preprocessing stage builds the CPI MoM target table, normalized long numeric observations, and a monthly numeric feature base from these public sources.

### Text Sources

Text sources must be downloaded or scraped reproducibly. The initial text scope should prioritize stable official publications before broader news sources.

| Source Type               | Examples                                   | Purpose                                          |
| ------------------------- | ------------------------------------------ | ------------------------------------------------ |
| Central bank publications | Inflation reports, MPC summaries, speeches | Official inflation narrative and policy guidance |
| Economic news             | Reproducible public archives or feeds      | Market and public inflation-pressure signal      |
| Source metadata           | Publication date, title, source, URL       | Time alignment and auditability                  |

Text models must be trained from scratch. External pretrained language models, pretrained embeddings, and large language model APIs are not part of the project.

The initial implemented text sources are CBRT MPC meeting decision and meeting summary pages. The download stage fetches the listing pages and each discovered official document page. The preprocessing stage extracts document metadata, publication dates, and clean body text from those raw HTML snapshots.

## Data Contract

Each observation in the final modeling dataset represents one forecast month.

| Field Group      | Required Content                                                            |
| ---------------- | --------------------------------------------------------------------------- |
| Time keys        | Forecast origin month, target month, source publication dates               |
| Numeric features | Lagged, rolling, and transformed macro-financial variables                  |
| Text features    | Tokenized text windows or text-derived inflation-pressure scores            |
| Target           | CPI MoM for the target month                                                |
| Availability     | Feature availability flags or cutoff rules that prevent future data leakage |

The pipeline must not use data published after the forecast origin. This rule applies to CPI values, macro-financial indicators, and text documents.

## Outputs

| Output                 | Format                | Directory             |
| ---------------------- | --------------------- | --------------------- |
| Raw downloaded data    | Source-native formats | `data/raw/`           |
| Model-ready dataset    | Parquet               | `data/processed/`     |
| Model checkpoints      | PyTorch checkpoint    | `output/models/`      |
| Forecasts              | CSV and Parquet       | `output/predictions/` |
| Metrics and summaries  | JSON and Markdown     | `output/reports/`     |
| Article/report figures | PNG or SVG            | `output/figures/`     |

Generated data and model outputs are reproducible artifacts. They should be generated by commands instead of committed as source files, except for selected small figures or tables intentionally referenced by documentation.

Current data foundation artifacts:

| Artifact                                 | Command           | Purpose                                                                       |
| ---------------------------------------- | ----------------- | ----------------------------------------------------------------------------- |
| `data/raw/source_registry.json`          | `just download`   | Snapshot of configured sources                                                |
| `data/raw/source_manifest.json`          | `just download`   | Download status, local paths, hashes, and byte counts                         |
| `data/raw/numeric/cbrt_fx_month_end/`    | `just download`   | Official CBRT month-end FX XML snapshots                                      |
| `data/raw/numeric/fred/*.csv`            | `just download`   | Public FRED macro-financial CSV snapshots                                     |
| `data/raw/text/documents/*.html`         | `just download`   | Official CBRT MPC document page snapshots                                     |
| `data/processed/cpi_mom.parquet`         | `just preprocess` | Forecast-origin and target-month CPI MoM table                                |
| `data/processed/numeric_series.parquet`  | `just preprocess` | Normalized long numeric source observations                                   |
| `data/processed/monthly_numeric.parquet` | `just preprocess` | Monthly numeric feature base                                                  |
| `data/processed/text_documents.parquet`  | `just preprocess` | Official CBRT text metadata, dates, and body text                             |
| `data/processed/model_dataset.parquet`   | `just preprocess` | Leakage-safe monthly modeling table                                           |
| `data/processed/feature_metadata.json`   | `just preprocess` | Feature columns, split metadata, and tokenizer config                         |
| `data/processed/text_vocabulary.json`    | `just preprocess` | Train-split vocabulary built from project text                                |
| `output/models/*.pt`                     | `just train`      | Raw PyTorch model checkpoints                                                 |
| `output/predictions/predictions.*`       | `just train`      | Forecasts for all models and chronological splits                             |
| `output/reports/training_history.*`      | `just train`      | Per-epoch train and validation losses for neural models                       |
| `output/figures/training_*.png`          | `just train`      | Neural model training-loss curves                                             |
| `output/reports/metrics.*`               | `just evaluate`   | MAE, RMSE, direction accuracy, volatility-normalized MAE, and baseline deltas |
| `output/figures/*.png`                   | `just evaluate`   | CPI, prediction, residual, and comparison figures                             |
| `output/reports/*_summary.md`            | `just evaluate`   | Split, data, best-model, feature, and text-corpus summaries                   |
| `output/reports/error_examples.md`       | `just evaluate`   | Largest forecast errors for the selected model                                |

## Usage

### Prerequisites

| Tool   | Version / Requirement  | Required                               |
| ------ | ---------------------- | -------------------------------------- |
| uv     | 0.11+                  | For Python environment management      |
| Python | Project-pinned version | For pipeline execution                 |
| just   | Any recent version     | For standardized project commands      |
| Nix    | Flakes enabled         | For the reproducible development shell |

### Clone the Project

```bash
git clone https://github.com/oguzhanozkaya/turkish-inflation-forecasting.git
cd turkish-inflation-forecasting
```

### Sync

```bash
just sync
```

Manual equivalent:

```bash
uv sync
```

### Run

The complete pipeline is runnable from a single command:

```bash
just run
```

The project also exposes stage-level commands:

```bash
just download
just preprocess
just train
just evaluate
```

Training runtime can be controlled through environment variables without changing the command interface:

```bash
TIF_EPOCHS=200 TIF_PATIENCE=20 just train
```

Training, preprocessing, and model architecture variables:

| Variable                             | Default        | Description                                                                                |
| ------------------------------------ | -------------- | ------------------------------------------------------------------------------------------ |
| `TIF_SEED`                           | `447`          | Random seed for NumPy, Python, PyTorch, Random Forest, and deterministic experiment setup. |
| `TIF_EPOCHS`                         | `36000`        | Maximum number of epochs for each neural model.                                            |
| `TIF_PATIENCE`                       | `100000`       | Early-stopping patience measured in epochs without validation improvement.                 |
| `TIF_BATCH_SIZE`                     | `32`           | Mini-batch size for PyTorch training and neural-model prediction.                          |
| `TIF_LEARNING_RATE`                  | `0.00001`      | Adam optimizer learning rate for all neural models.                                        |
| `TIF_WEIGHT_DECAY`                   | `0.0`          | Adam optimizer L2 weight decay for neural models.                                          |
| `TIF_EARLY_STOPPING_MIN_DELTA`       | `0.00000001`   | Minimum validation-loss improvement required to reset early stopping.                      |
| `TIF_RIDGE_ALPHA`                    | `1.0`          | Ridge regression regularization strength.                                                  |
| `TIF_RANDOM_FOREST_TREES`            | `200`          | Number of trees in the Random Forest baseline.                                             |
| `TIF_RANDOM_FOREST_MIN_SAMPLES_LEAF` | `3`            | Minimum samples per Random Forest leaf.                                                    |
| `TIF_DEVICE`                         | `cuda`         | Training device: `cuda`, `cpu`, or `auto`.                                                 |
| `TIF_SEQUENCE_STEPS`                 | `12,6,3,2,1,0` | Lag steps used to build numeric sequence tensors for the GRU.                              |
| `TIF_MAX_TOKENS`                     | `256`          | Maximum token IDs kept per text window.                                                    |
| `TIF_MAX_VOCAB_SIZE`                 | `5000`         | Maximum train-split vocabulary size for project text.                                      |
| `TIF_TEXT_LOOKBACK_MONTHS`           | `12`           | Number of months of documents included in each forecast-origin text window.                |
| `TIF_NUMERIC_MLP_HIDDEN_SIZE`        | `128`          | Hidden width of the numeric MLP.                                                           |
| `TIF_NUMERIC_MLP_DROPOUT`            | `0.15`         | Dropout probability in the numeric MLP.                                                    |
| `TIF_NUMERIC_GRU_HIDDEN_SIZE`        | `64`           | Hidden size of the numeric GRU.                                                            |
| `TIF_NUMERIC_GRU_DROPOUT`            | `0.10`         | Dropout probability in the numeric GRU regression head.                                    |
| `TIF_TEXT_EMBEDDING_DIM`             | `64`           | Dimension of the randomly initialized text embedding layer.                                |
| `TIF_TEXT_CHANNEL_COUNT`             | `48`           | Number of convolution channels per TextCNN kernel size.                                    |
| `TIF_TEXT_KERNEL_SIZES`              | `3,4,5`        | Comma-separated convolution kernel sizes for TextCNN encoders.                             |
| `TIF_TEXT_DROPOUT`                   | `0.20`         | Dropout probability after TextCNN pooled representations.                                  |
| `TIF_FUSION_HIDDEN_SIZE`             | `128`          | Hidden width of the numeric projection and fusion head.                                    |
| `TIF_FUSION_DROPOUT`                 | `0.15`         | Dropout probability in the fusion regression head.                                         |

## Success Criteria

The project is complete when the repository can reproducibly:

1. Download or scrape all required source data.
2. Build a leakage-safe monthly modeling dataset.
3. Train baseline, numeric deep learning, text, and fusion models.
4. Evaluate all models on chronological validation and test periods.
5. Generate predictions, metrics, plots, and article-ready outputs.
6. Explain the architecture, validation strategy, and results in the documentation.
