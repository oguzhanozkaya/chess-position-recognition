"""Plot generation stage entrypoint."""

from turkish_inflation_forecasting.pipeline import run_plots


def main() -> int:
    return run_plots()
