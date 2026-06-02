"""Feature generation stage entrypoint."""

from turkish_inflation_forecasting.pipeline import run_features


def main() -> int:
    return run_features()
