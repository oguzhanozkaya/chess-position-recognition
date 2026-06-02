from pathlib import Path

from turkish_inflation_forecasting.config import build_paths
from turkish_inflation_forecasting.pipeline import run_download, run_pipeline


def test_download_stage_initializes_directories(tmp_path: Path, capsys) -> None:
    paths = build_paths(tmp_path)

    exit_code = run_download(paths)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert paths.raw_data.is_dir()
    assert "download" in captured.out
    assert "not implemented yet" in captured.out


def test_run_entrypoint_reports_registered_pipeline(tmp_path: Path, capsys) -> None:
    paths = build_paths(tmp_path)

    exit_code = run_pipeline(paths)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert paths.reports.is_dir()
    assert "full forecasting pipeline" in captured.out
