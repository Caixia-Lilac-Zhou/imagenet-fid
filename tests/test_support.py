"""Keep test fixtures for inspection; tests never delete files or directories."""
from contextlib import contextmanager
from pathlib import Path
import tempfile


@contextmanager
def preserved_directory():
    path = Path(tempfile.mkdtemp(prefix='imagenet-fid-test-'))
    try:
        yield str(path)
    finally:
        print(f'Test fixtures retained: {path}')
