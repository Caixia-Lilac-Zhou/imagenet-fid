"""Check that fresh-run flags preserve previous files and manifests stay portable."""
from pathlib import Path
import numpy as np
from bootstrap_fid_uncertainty import run_bootstrap
from evaluate import evaluate
from make_manifest import create_manifest
from output_safety import require_empty_output, require_resumable_output
from subsample_fid_stability import run_subsampling
from test_support import preserved_directory
from verify_release import verify


def must_raise(error, fn):
    try:
        fn()
    except error:
        return
    raise AssertionError(f'Expected {error.__name__}')


def main():
    verify()
    with preserved_directory() as temporary:
        root = Path(temporary)
        existing = root / 'existing'
        existing.mkdir()
        sentinel = existing / 'keep.txt'
        sentinel.write_text('preserve me')
        must_raise(FileExistsError, lambda: require_empty_output(existing))
        must_raise(FileExistsError, lambda: require_resumable_output(existing, 'metadata.json'))
        assert sentinel.read_text() == 'preserve me'
        features = np.random.default_rng(7).normal(size=(12, 3))
        labels = np.repeat([0, 1], 6)
        generated, real = root / 'generated.npz', root / 'real.npz'
        np.savez(generated, features=features + 0.25, labels=labels)
        np.savez(real, features=features, labels=labels)
        second = root / 'second.npz'
        np.savez(second, features=features + 0.5, labels=labels)
        manifest = create_manifest({'M': generated, 'N': second}, {'val': real}, {}, root / 'manifest.csv')
        assert str(root) not in manifest.read_text()
        points = evaluate(manifest, root / 'points.csv')
        assert abs(float(points[0]['point_fd']) - 3 * 0.25 ** 2) < 1e-8
        must_raise(FileExistsError, lambda: evaluate(manifest, root / 'points.csv'))
        result = run_bootstrap(manifest, root / 'bootstrap', replicates=4, quiet=True)
        before = {k: p.read_bytes() for k, p in result.items()}
        must_raise(FileExistsError, lambda: run_bootstrap(manifest, root / 'bootstrap', replicates=4, resume=False, quiet=True))
        must_raise(ValueError, lambda: run_bootstrap(manifest, root / 'bootstrap', replicates=4, seed=17, quiet=True))
        assert all(p.read_bytes() == before[k] for k, p in result.items())
        result = run_subsampling(manifest, root / 'size', [3], [], trials=3, quiet=True)
        before = {k: p.read_bytes() for k, p in result.items()}
        must_raise(FileExistsError, lambda: run_subsampling(manifest, root / 'size', [3], [], trials=3, resume=False, quiet=True))
        assert all(p.read_bytes() == before[k] for k, p in result.items())
    print('release_safety: all tests passed')
