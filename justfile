# Inflation Forecasting - Project Commands
# Usage: just <recipe>
# Run `just --list` to see all available recipes.


# Sync python environment
[group('dev')]
sync:
  uv sync

# Run the full pipeline
[default]
[group('run')]
run:
  uv run turkish-inflation-run

# Download numeric data and text sources
[group('run')]
download:
  uv run turkish-inflation-download

# Clean raw source files and build interim tables
[group('run')]
preprocess:
  uv run turkish-inflation-preprocess

# Build model-ready numeric and text features
[group('run')]
features:
  uv run turkish-inflation-features

# Train baselines and deep learning models
[group('run')]
train:
  uv run turkish-inflation-train

# Evaluate models on chronological splits
[group('run')]
evaluate:
  uv run turkish-inflation-evaluate

# Generate figures for reports and article drafts
[group('run')]
plots:
  uv run turkish-inflation-plots


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
  rm -rf output/figures/* output/models/* output/predictions/* output/reports/*


# Clean and start docs website at localhost
[group('docs')]
docs:
  rm -rf .site/
  uv run --only-group docs zensical serve

# Build web page with clean cache
[group('docs')]
docs-build:
  uv run --only-group docs zensical build --clean
