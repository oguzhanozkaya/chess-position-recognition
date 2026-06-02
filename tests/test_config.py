from pathlib import Path

from turkish_inflation_forecasting.config import PROJECT_ROOT, build_paths, ensure_generated_directories


def test_project_root_points_to_repository() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_build_paths_uses_given_root(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)

    assert paths.root == tmp_path.resolve()
    assert paths.raw_data == tmp_path.resolve() / "data" / "raw"
    assert paths.reports == tmp_path.resolve() / "output" / "reports"


def test_ensure_generated_directories_creates_pipeline_layout(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)

    directories = ensure_generated_directories(paths)

    assert directories == paths.generated_directories()
    assert all(directory.is_dir() for directory in directories)
