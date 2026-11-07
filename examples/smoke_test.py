#!/usr/bin/env python3
"""CPU-only synthetic demonstration; these are not paper or image-quality results."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
sys.path.insert(0, str(ROOT / 'scripts'))
from bootstrap_fid_uncertainty import run_bootstrap
from evaluate import evaluate
from fid_core import compute_stats
from make_manifest import create_manifest
from subsample_fid_stability import run_subsampling


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(2026)
    labels = np.repeat(np.arange(3), 16)
    real = rng.normal(size=(48, 6)) + labels[:, None] * 0.2
    generated = {}
    for name, shift in [('A', 0.1), ('B', 0.5), ('C', 1.0)]:
        path = output / f'{name}.npz'
        np.savez(path, features=real + shift, labels=labels)
        generated[name] = path
    reference = output / 'real.npz'
    np.savez(reference, features=real, labels=labels)
    fixed = output / 'real_stats.npz'
    stats = compute_stats(real)
    np.savez(fixed, mu=stats.mu, sigma=stats.cov)
    manifest = create_manifest(generated, {'validation': reference}, {'fixed_train': fixed}, output / 'manifest.csv')
    evaluate(manifest, output / 'point_scores.csv')
    run_bootstrap(manifest, output / 'bootstrap', replicates=24, seed=2026, quiet=True)
    run_subsampling(manifest, output / 'subsampling', generated_sizes=[4, 8],
                    reference_sizes=[4, 8], trials=8, seed=2026, quiet=True)
    print(f'Synthetic smoke test complete: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/smoke'))
    run(parser.parse_args().output_dir.resolve())


if __name__ == '__main__':
    main()
