# Project recipes
# Usage: just <recipe>
# Run `just --list` to see all available recipes.

# Default environemnt variables
export TIF_SEED := env("TIF_SEED", "447")
export TIF_EPOCHS := env("TIF_EPOCHS", "36000")
export TIF_PATIENCE := env("TIF_PATIENCE", "100000")
export TIF_BATCH_SIZE := env("TIF_BATCH_SIZE", "32")
export TIF_LEARNING_RATE := env("TIF_LEARNING_RATE", "0.00001")
export TIF_WEIGHT_DECAY := env("TIF_WEIGHT_DECAY", "0.0")
export TIF_EARLY_STOPPING_MIN_DELTA := env("TIF_EARLY_STOPPING_MIN_DELTA", "0.00000001")
export TIF_RIDGE_ALPHA := env("TIF_RIDGE_ALPHA", "1.0")
export TIF_RANDOM_FOREST_TREES := env("TIF_RANDOM_FOREST_TREES", "200")
export TIF_RANDOM_FOREST_MIN_SAMPLES_LEAF := env("TIF_RANDOM_FOREST_MIN_SAMPLES_LEAF", "3")
export TIF_DEVICE := env("TIF_DEVICE", "cuda")
export TIF_SEQUENCE_STEPS := env("TIF_SEQUENCE_STEPS", "12,6,3,2,1,0")
export TIF_MAX_TOKENS := env("TIF_MAX_TOKENS", "256")
export TIF_MAX_VOCAB_SIZE := env("TIF_MAX_VOCAB_SIZE", "5000")
export TIF_TEXT_LOOKBACK_MONTHS := env("TIF_TEXT_LOOKBACK_MONTHS", "12")
export TIF_NUMERIC_MLP_HIDDEN_SIZE := env("TIF_NUMERIC_MLP_HIDDEN_SIZE", "128")
export TIF_NUMERIC_MLP_DROPOUT := env("TIF_NUMERIC_MLP_DROPOUT", "0.15")
export TIF_NUMERIC_GRU_HIDDEN_SIZE := env("TIF_NUMERIC_GRU_HIDDEN_SIZE", "64")
export TIF_NUMERIC_GRU_DROPOUT := env("TIF_NUMERIC_GRU_DROPOUT", "0.10")
export TIF_TEXT_EMBEDDING_DIM := env("TIF_TEXT_EMBEDDING_DIM", "64")
export TIF_TEXT_CHANNEL_COUNT := env("TIF_TEXT_CHANNEL_COUNT", "48")
export TIF_TEXT_KERNEL_SIZES := env("TIF_TEXT_KERNEL_SIZES", "3,4,5")
export TIF_TEXT_DROPOUT := env("TIF_TEXT_DROPOUT", "0.20")
export TIF_FUSION_HIDDEN_SIZE := env("TIF_FUSION_HIDDEN_SIZE", "128")
export TIF_FUSION_DROPOUT := env("TIF_FUSION_DROPOUT", "0.15")

# Sync python environment
[group('dev')]
sync:
  uv sync

# Run the full pipeline
[group('run')]
run: download preprocess train evaluate

# Download numeric data and text sources
[group('run')]
download:
  uv run tif-download

# Clean raw source files and build processed model data
[group('run')]
preprocess:
  uv run tif-preprocess

# Train baselines and deep learning models
[group('run')]
[default]
train:
  uv run tif-train

# Evaluate models on chronological splits and generate figures
[group('run')]
evaluate:
  uv run tif-evaluate


# Fix: format and lint
[group('qual')]
fix:
  bunx prettier --log-level=warn --write .
  uv run ruff format .
  uv run ruff check . --fix

# Check code: format and lint
[group('qual')]
check:
  bunx prettier --log-level warn --check .
  uv run ruff format . --check
  uv run ruff check .

# Run tests
[group('qual')]
test:
  uv run pytest

# Full check + test gate (github ci runs this command)
[group('qual')]
ci: check test


# Remove build artifacts
[group('clean')]
clean:
  uv run ruff clean

# Remove all output and generated artifacts
[group('clean')]
clean-outputs:
  rm -rf output/figures/*
  rm -rf output/models/*
  rm -rf output/predictions/*
  rm -rf output/reports/*


# Clean and start docs website at localhost
[group('docs')]
docs:
  rm -rf .site/
  uv run --only-group docs zensical serve

# Build web page with clean cache
[group('docs')]
docs-build:
  uv run --only-group docs zensical build --clean
