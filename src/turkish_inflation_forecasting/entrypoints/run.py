"""Full pipeline entrypoint."""

from turkish_inflation_forecasting.pipeline import run_pipeline


def main() -> int:
    return run_pipeline()
