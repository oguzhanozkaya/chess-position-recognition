---
description: Tasks, priorities, known bugs, and the project roadmap.
---

# Roadmap

## Status Overview

The repository is implemented as a single-script chess position recognition pipeline. The current pipeline reads local Kaggle board images, parses FEN filename labels in memory, trains one scratch PyTorch CNN, and evaluates 64 square predictions per board.

| Area                         | Status      |
| ---------------------------- | ----------- |
| Documentation structure      | Implemented |
| Single root script           | Implemented |
| Command workflow             | Implemented |
| Local image loading          | Implemented |
| Filename FEN parsing         | Implemented |
| Scratch board CNN classifier | Implemented |
| Evaluation reports           | Implemented |
| Article draft                | Implemented |

## Delivered Capabilities

- `cpr.py` runs the complete pipeline from local image loading through evaluation.
- `just run` executes `uv run python cpr.py`.
- `just smoke` runs a short CPU end-to-end pipeline through a direct `Config` override.
- The script reads `data/train/` and `data/test/` every run.
- No processed dataset cache is written.
- Labels are parsed from filename stems on each run.
- The model has no fastai, pretrained model, pretrained weight, or transfer-learning dependency.
- Training writes PyTorch checkpoints, predictions, training history, and training figures.
- Evaluation writes JSON/Markdown/CSV square metrics, a per-class report, and evaluation figures.

## Limitations

- The active model is intentionally limited to one scratch CNN architecture.
- Input images are resized to `IMAGE_SIZE`, so very small piece details depend on that setting.
- Board images are assumed to be aligned full-board schematics matching the dataset contract.
- Final reported results depend on the selected long GPU training run and must be regenerated after tuning.

## Future Work

- Tune `IMAGE_SIZE`, batch size, empty-square loss weight, learning rate, and dropout for the RTX 4080 run.
- Add calibration analysis and probability reliability reports.
- Add high-confidence error examples for article interpretation.
- Compare against a second scratch CNN variant if the course report needs another architecture.
