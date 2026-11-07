#!/usr/bin/env python3
"""Numerical and bundle-format regression tests for fid_core.py."""

from __future__ import annotations

import json
import importlib.util
from test_support import preserved_directory
from pathlib import Path

import numpy as np

from fid_core import (
    NumpyBackend,
    TorchBackend,
    compute_stats,
    frechet_distance,
    load_feature_bundle,
)


def main() -> None:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(73, 7)).astype(np.float32)
    indices = np.sort(rng.choice(features.shape[0], size=51, replace=True))
    backend = NumpyBackend(chunk_size=13)
    stats = backend.stats(features, indices=indices, cache_sqrt=True)
    expected = features[indices].astype(np.float64)
    assert np.allclose(stats.mu, expected.mean(axis=0), atol=1e-12)
    assert np.allclose(stats.cov, np.cov(expected, rowvar=False), atol=1e-12)
    assert backend.distance(stats, stats) < 1e-8

    if importlib.util.find_spec('torch') is not None:
        torch_backend = TorchBackend(device="cpu", chunk_size=11)
        torch_stats = torch_backend.stats(features, indices=indices, cache_sqrt=True)
        assert np.allclose(torch_stats.mu.cpu().numpy(), stats.mu, atol=1e-12)
        assert np.allclose(torch_stats.cov.cpu().numpy(), stats.cov, atol=1e-12)
        assert torch_backend.distance(torch_stats, torch_stats) < 1e-8
    else:
        print('SKIP: optional TorchBackend comparison (PyTorch not installed)')

    shifted = compute_stats(features + 0.25, cache_sqrt=True)
    original = compute_stats(features, cache_sqrt=True)
    assert frechet_distance(original, shifted) > 0.0

    with preserved_directory() as temporary_dir:
        root = Path(temporary_dir) / "bundle"
        root.mkdir()
        np.save(root / "features.npy", features)
        np.save(root / "labels.npy", np.repeat(np.arange(73), 1).astype(np.int32))
        (root / "metadata.json").write_text(
            json.dumps({"complete": True}), encoding="utf-8"
        )
        bundle = load_feature_bundle(root)
        assert bundle.features.shape == features.shape
        assert isinstance(bundle.features, np.memmap)

    print("fid_core: all tests passed")


if __name__ == "__main__":
    main()
