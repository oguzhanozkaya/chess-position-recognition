"""Download stage entrypoint."""

from turkish_inflation_forecasting.pipeline import run_download


def main() -> int:
    return run_download()
