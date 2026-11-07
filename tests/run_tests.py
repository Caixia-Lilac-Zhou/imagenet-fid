#!/usr/bin/env python3
"""Run lightweight release tests, with optional PyTorch extraction checks."""
import importlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for name in ['evaluation', 'scripts', 'analysis', 'examples']:
    sys.path.insert(0, str(ROOT / name))


def main():
    if not __debug__:
        raise RuntimeError('Run tests without Python optimization; assertions must remain enabled')
    modules = ['test_fid_core', 'test_bootstrap_fid_uncertainty',
               'test_subsample_fid_stability', 'test_release_safety', 'test_paper_results',
               'test_model_sources']
    if importlib.util.find_spec('torch') is not None:
        modules.append('test_feature_pipeline')
    else:
        print('SKIP: PyTorch-dependent extraction/sharding test (PyTorch not installed)')
    for name in modules:
        print(f'RUN: {name}', flush=True)
        importlib.import_module(name).main()
    print(f'PASS: {len(modules)} test modules. All temporary fixtures were retained.')


if __name__ == '__main__':
    main()
