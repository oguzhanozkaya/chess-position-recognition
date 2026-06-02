"""Pipeline stage entry functions.

The project exposes separate console entrypoints for each stage. These functions
hold the stage behavior so entrypoint modules stay small and testable.
"""

from __future__ import annotations

from turkish_inflation_forecasting.config import DEFAULT_PATHS, ProjectPaths, ensure_generated_directories


def _run_pending_stage(stage_name: str, next_step: str, paths: ProjectPaths = DEFAULT_PATHS) -> int:
    ensure_generated_directories(paths)
    print(f"{stage_name}: project directories are ready.")
    print(f"{stage_name}: {next_step}")
    return 0


def run_download(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("download", "source registry and download logic are not implemented yet.", paths)


def run_preprocess(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("preprocess", "raw source cleaning is not implemented yet.", paths)


def run_features(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("features", "numeric and text feature generation is not implemented yet.", paths)


def run_train(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("train", "baseline and deep model training is not implemented yet.", paths)


def run_evaluate(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("evaluate", "chronological evaluation is not implemented yet.", paths)


def run_plots(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    return _run_pending_stage("plots", "report figure generation is not implemented yet.", paths)


def run_pipeline(paths: ProjectPaths = DEFAULT_PATHS) -> int:
    ensure_generated_directories(paths)
    print("run: project directories are ready.")
    print("run: full forecasting pipeline is registered but not implemented yet.")
    return 0
