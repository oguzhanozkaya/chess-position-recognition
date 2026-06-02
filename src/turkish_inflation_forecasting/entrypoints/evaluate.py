"""Evaluation stage entrypoint."""

from turkish_inflation_forecasting.pipeline import run_evaluate


def main() -> int:
    return run_evaluate()
