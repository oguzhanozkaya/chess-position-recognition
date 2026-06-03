# Project recipes
# Usage: just <recipe>

[group('dev')]
sync:
  uv sync

[group('run')]
[default]
run:
  uv run python yrs.py

[group('run')]
smoke:
  uv run python -c 'import yrs; yrs.run_pipeline(yrs.Config(device="cpu", epochs=1, patience=1, dataloader_workers=0, mixed_precision=False, row_limit=1000, max_vocab_size=5000, max_sequence_length=96, embedding_dim=64, filter_count=32, hidden_size=64, batch_size=64))'

[group('qual')]
fix:
  uv run ruff format .
  uv run ruff check . --fix

[group('qual')]
check:
  uv run ruff format . --check
  uv run ruff check .

[group('qual')]
ci: check

[group('clean')]
clean:
  uv run ruff clean

[group('clean')]
clean-outputs:
  rm -rf output/figures/*
  rm -rf output/models/*
  rm -rf output/predictions/*
  rm -rf output/reports/*

[group('docs')]
docs:
  rm -rf .site/
  uv run --only-group docs zensical serve

[group('docs')]
docs-build:
  uv run --only-group docs zensical build --clean
