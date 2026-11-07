"""Non-destructive guards for fresh outputs and explicitly resumable runs."""
from pathlib import Path


def require_empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Output is not empty: {path}. Choose a new output directory.')


def require_resumable_output(path: Path, metadata_name: str, resume: bool = True) -> None:
    if not resume:
        require_empty_output(path)
    elif path.exists() and not (path / metadata_name).is_file():
        require_empty_output(path)
