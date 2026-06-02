"""Training stage entrypoint."""

from turkish_inflation_forecasting.pipeline import run_train


def main() -> int:
    return run_train()
