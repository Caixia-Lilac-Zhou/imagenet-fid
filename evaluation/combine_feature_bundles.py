#!/usr/bin/env python3
"""Concatenate compatible feature bundles without loading them fully into memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from output_safety import require_empty_output

from fid_core import atomic_write_json, class_count_summary, load_feature_bundle


def combine(inputs: list[Path], output: Path, chunk_size: int = 8192) -> Path:
    if len(inputs) < 2 or chunk_size < 1:
        raise ValueError("Provide at least two inputs and a positive chunk size")
    bundles = [load_feature_bundle(path) for path in inputs]
    dimensions = {int(bundle.features.shape[1]) for bundle in bundles}
    if len(dimensions) != 1:
        raise ValueError(f"Feature dimensions disagree: {sorted(dimensions)}")
    dimension = dimensions.pop()
    total = sum(int(bundle.features.shape[0]) for bundle in bundles)
    output = output.resolve()
    require_empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    features_tmp = output / "features.npy.tmp"
    labels_tmp = output / "labels.npy.tmp"
    features = np.lib.format.open_memmap(
        features_tmp, mode="w+", dtype=np.float32, shape=(total, dimension)
    )
    labels = np.lib.format.open_memmap(labels_tmp, mode="w+", dtype=np.int32, shape=(total,))
    offset = 0
    for bundle in bundles:
        count = int(bundle.features.shape[0])
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            values = np.asarray(bundle.features[start:end], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite features in {bundle.source_path}")
            features[offset + start : offset + end] = values
            labels[offset + start : offset + end] = np.asarray(
                bundle.labels[start:end], dtype=np.int32
            )
        offset += count
    features.flush()
    labels.flush()
    counts = class_count_summary(labels)
    features_tmp.replace(output / "features.npy")
    labels_tmp.replace(output / "labels.npy")
    atomic_write_json(
        output / "metadata.json",
        {
            "schema_version": 1,
            "name": output.name,
            "source_bundles": [str(path.resolve()) for path in inputs],
            "samples": total,
            "feature_dimension": dimension,
            "class_counts": counts,
            "completed_samples": total,
            "complete": True,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    print(combine(args.input, args.output, args.chunk_size))


if __name__ == "__main__":
    main()
