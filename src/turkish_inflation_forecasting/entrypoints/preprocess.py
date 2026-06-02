"""Preprocess stage entrypoint."""

from turkish_inflation_forecasting.pipeline import run_preprocess


def main() -> int:
    return run_preprocess()
