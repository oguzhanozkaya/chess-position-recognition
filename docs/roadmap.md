---
description: Tasks, priorities, known bugs, and the project roadmap.
---

# Roadmap

## Status Overview

The repository is in the foundation stage. The project structure, reproducibility contract, command workflow, and implementation milestones are documented. The next goal is to replace the stage entrypoint placeholders with real data and modeling logic.

| Area                     | Status                 |
| ------------------------ | ---------------------- |
| Documentation structure  | In progress            |
| Python package structure | Foundation implemented |
| Command workflow         | Foundation implemented |
| Data download pipeline   | Planned                |
| Text collection pipeline | Planned                |
| Feature generation       | Planned                |
| Baselines                | Planned                |
| Deep learning models     | Planned                |
| Evaluation reports       | Planned                |
| Article draft            | Planned                |

## Active Tasks

| Priority | Task                                       | Exit Criteria                                                        |
| -------- | ------------------------------------------ | -------------------------------------------------------------------- |
| High     | Define source registry                     | Numeric and text sources have documented download metadata           |
| High     | Build CPI target pipeline                  | CPI MoM target is reproducibly generated without leakage             |
| High     | Implement official text collection         | Text documents and metadata are downloaded reproducibly              |
| High     | Replace stage placeholders with real logic | `just download`, `preprocess`, and `features` produce real artifacts |

## Milestones

### Phase 1: Project Foundation

| Task                                           | Output                                                               |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| Document architecture and repository structure | Updated `docs/` pages                                                |
| Create package layout                          | `src/turkish_inflation_forecasting/`                                 |
| Configure stage entrypoints                    | Stage-level commands callable through `uv run` and `just`            |
| Add tests directory and fixtures policy        | Deterministic tests can be added without committing full datasets    |
| Update ignore rules                            | Generated `data/` and `output/` artifacts stay out of source control |

### Phase 2: Data Pipeline

| Task                                     | Output                                        |
| ---------------------------------------- | --------------------------------------------- |
| Implement source definitions             | Reproducible numeric and text source registry |
| Download CPI and macro-financial data    | Raw files under `data/raw/`                   |
| Download or scrape official text sources | Raw text and metadata under `data/raw/`       |
| Preprocess raw sources                   | Cleaned tables under `data/interim/`          |
| Align monthly observations               | One row per forecast origin and target month  |

### Phase 3: Feature Engineering

| Task                              | Output                                    |
| --------------------------------- | ----------------------------------------- |
| Build CPI MoM target              | Leakage-safe target column                |
| Generate lag and rolling features | Numeric feature table                     |
| Train project tokenizer           | Vocabulary built only from project corpus |
| Convert text into model inputs    | Token IDs or monthly text windows         |
| Build model-ready dataset         | Parquet files under `data/processed/`     |

### Phase 4: Modeling

| Task                                  | Output                                                    |
| ------------------------------------- | --------------------------------------------------------- |
| Implement naive and rolling baselines | Baseline forecasts and metrics                            |
| Implement classical baselines         | Ridge and random forest comparisons                       |
| Implement numeric deep models         | MLP, 1D CNN, GRU or LSTM experiments                      |
| Implement text encoder                | TextCNN, BiGRU, or small Transformer trained from scratch |
| Implement fusion model                | Combined numeric and text forecast model                  |

### Phase 5: Evaluation and Reporting

| Task                           | Output                                                        |
| ------------------------------ | ------------------------------------------------------------- |
| Add chronological validation   | Train, validation, and test splits                            |
| Add rolling-origin backtesting | Forecast history across multiple origins                      |
| Generate metrics               | JSON and Markdown reports                                     |
| Generate figures               | CPI, features, predictions, residuals, and model comparisons  |
| Add interpretability outputs   | Feature importance, ablations, and text contribution analysis |

### Phase 6: Final Article and Polish

| Task                                   | Output                                            |
| -------------------------------------- | ------------------------------------------------- |
| Write article draft in `docs/index.md` | Reviewable project article                        |
| Add reproducibility instructions       | Clear clone, sync, run, and evaluate workflow     |
| Summarize limitations                  | Honest explanation of data and model constraints  |
| Prepare final GitHub presentation      | Clean README, docs, figures, and command workflow |

## Delivered Capabilities

- Initial documentation scaffold exists.
- `uv`, `just`, and Zensical are already present in the repository template.
- Python package foundation exists under `src/turkish_inflation_forecasting/`.
- Stage-specific console entrypoints are registered through `pyproject.toml`.
- `just` exposes pipeline, quality, test, and documentation commands.
- Generated `data/` and `output/` directory placeholders are tracked while generated contents are ignored.
- Initial smoke tests cover path setup and stage entrypoint behavior.

## Limitations

- The current repository has runnable stage placeholders, not a complete forecasting pipeline.
- Stage-level data and modeling commands are registered but their real implementation is pending.
- Data source availability and publication-delay handling still need implementation validation.
- Text collection should start with stable official sources before broader news scraping.

## Future Work

- Add self-supervised text pretraining on the project corpus if it improves the text branch.
- Compare numeric-only, text-only, and fusion models.
- Add ablation studies for text source categories and numeric feature groups.
- Add a lightweight forecast command that loads the latest generated model and writes a next-month forecast.
- Add deployment-oriented documentation after the pipeline is stable.
