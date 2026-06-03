---
description: Tasks, priorities, known bugs, and the project roadmap.
---

# Roadmap

## Status Overview

The repository is implemented as a single-script Yelp review sentiment pipeline. The current pipeline reads local Kaggle CSV files, builds an in-memory vocabulary, trains one scratch PyTorch TextCNN classifier, and evaluates negative/positive classification outputs.

| Area                         | Status      |
| ---------------------------- | ----------- |
| Documentation structure      | Implemented |
| Single root script           | Implemented |
| Command workflow             | Implemented |
| Local CSV loading            | Implemented |
| In-memory vocabulary         | Implemented |
| Scratch TextCNN classifier   | Implemented |
| Evaluation reports           | Implemented |
| Article draft                | Implemented |

## Active Tasks

| Priority | Task                             | Exit Criteria                                                       |
| -------- | -------------------------------- | ------------------------------------------------------------------- |
| High     | Run extended GPU training        | `just run` completes on the RTX 4080 target machine                 |
| High     | Regenerate reports and figures   | `output/reports/` and `output/figures/` contain final run artifacts |
| High     | Fill article result placeholders | Final accuracy, macro F1, figures, and interpretation are added to `article.md` |

## Delivered Capabilities

- `yrs.py` runs the complete pipeline from local CSV loading through evaluation.
- `just run` executes `uv run python yrs.py`.
- `just smoke` runs a short CPU end-to-end pipeline through a direct `Config` override.
- The script reads `data/train.csv` and `data/test.csv` every run.
- No processed dataset cache is written.
- Vocabulary is built from the active training split only.
- The model has no fastai, pretrained embeddings, pretrained language model, transformer library, or language model API dependency.
- Training writes PyTorch checkpoints, predictions, training history, and a training-loss figure.
- Evaluation writes JSON/Markdown/CSV classification metrics, a per-class report, and evaluation figures.

## Limitations

- The active model is intentionally limited to one scratch TextCNN architecture.
- Very long reviews are truncated to `MAX_SEQUENCE_LENGTH` tokens.
- Vocabulary is word-level rather than subword-level, so rare spelling variants map to `<unk>`.
- Final reported results depend on the selected long GPU training run and must be regenerated after tuning.

## Future Work

- Tune `MAX_SEQUENCE_LENGTH`, vocabulary size, convolution width, and dropout for the RTX 4080 run.
- Add calibration analysis and probability reliability reports.
- Add high-confidence error examples for article interpretation.
- Compare against a scratch recurrent baseline if the course report needs a second architecture.
