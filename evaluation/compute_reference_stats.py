#!/usr/bin/env python3
"""Compute mu/sigma reference statistics from a validated per-image feature bundle."""
import argparse
from pathlib import Path
import numpy as np
from fid_core import get_backend, load_feature_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace {args.output}')
    bundle = load_feature_bundle(args.bundle)
    stats = get_backend('numpy').stats(bundle.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        np.savez(f, mu=stats.mu, sigma=stats.cov)
    print(args.output)


if __name__ == '__main__':
    main()
