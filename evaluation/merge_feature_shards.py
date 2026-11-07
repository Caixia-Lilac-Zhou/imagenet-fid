#!/usr/bin/env python3
"""Merge interleaved feature-extraction shards into one validated bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from output_safety import require_empty_output

from fid_core import atomic_write_json, class_count_summary, load_feature_bundle


def merge_shards(
    input_root: Path,
    output_dir: Path,
    num_shards: int,
    expected_count: int | None = None,
    expected_classes: int | None = None,
    chunk_size: int = 4096,
) -> Path:
    if num_shards < 2 or chunk_size < 1:
        raise ValueError("num_shards must be at least two and chunk_size must be positive")
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    require_empty_output(output_dir)
    shard_paths = [
        input_root / f"shard-{index:03d}-of-{num_shards:03d}"
        for index in range(num_shards)
    ]
    bundles = [load_feature_bundle(path) for path in shard_paths]
    metadata = [
        json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        for path in shard_paths
    ]
    mappings = [
        json.loads((path / "class_mapping.json").read_text(encoding="utf-8"))
        for path in shard_paths
    ]
    if any(mapping != mappings[0] for mapping in mappings[1:]):
        raise ValueError("Shard class mappings disagree")

    total = sum(int(bundle.features.shape[0]) for bundle in bundles)
    dimensions = {int(bundle.features.shape[1]) for bundle in bundles}
    if len(dimensions) != 1:
        raise ValueError(f"Shard feature dimensions disagree: {sorted(dimensions)}")
    dimension = dimensions.pop()
    if expected_count is not None and total != expected_count:
        raise ValueError(f"Expected {expected_count} merged samples, found {total}")
    for index, (bundle, item) in enumerate(zip(bundles, metadata)):
        descriptor = item.get("source", {}).get("shard", {})
        expected_rows = len(range(index, total, num_shards))
        if descriptor != {
            "full_samples": total,
            "index": index,
            "num_shards": num_shards,
            "scheme": "global_index_modulo",
        }:
            raise ValueError(f"Invalid shard descriptor for {shard_paths[index]}: {descriptor}")
        if bundle.features.shape[0] != expected_rows:
            raise ValueError(
                f"Shard {index} has {bundle.features.shape[0]} rows; expected {expected_rows}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    features_temp = output_dir / "features.npy.tmp"
    labels_temp = output_dir / "labels.npy.tmp"
    merged_features = np.lib.format.open_memmap(
        features_temp, mode="w+", dtype=np.float32, shape=(total, dimension)
    )
    merged_labels = np.lib.format.open_memmap(
        labels_temp, mode="w+", dtype=np.int32, shape=(total,)
    )
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        global_indices = np.arange(start, end, dtype=np.int64)
        feature_chunk = np.empty((end - start, dimension), dtype=np.float32)
        label_chunk = np.empty(end - start, dtype=np.int32)
        for index, bundle in enumerate(bundles):
            positions = np.flatnonzero(global_indices % num_shards == index)
            if positions.size == 0:
                continue
            local_indices = global_indices[positions] // num_shards
            values = np.asarray(bundle.features[local_indices], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Shard {index} contains non-finite features")
            feature_chunk[positions] = values
            label_chunk[positions] = np.asarray(bundle.labels[local_indices], dtype=np.int32)
        merged_features[start:end] = feature_chunk
        merged_labels[start:end] = label_chunk
    merged_features.flush()
    merged_labels.flush()

    counts = class_count_summary(merged_labels)
    if expected_classes is not None and counts["classes"] != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {counts['classes']}")
    features_path = output_dir / "features.npy"
    labels_path = output_dir / "labels.npy"
    features_temp.replace(features_path)
    labels_temp.replace(labels_path)
    atomic_write_json(output_dir / "class_mapping.json", mappings[0])
    first = metadata[0]
    source = dict(first.get("source", {}))
    source.pop("shard", None)
    source["merged_shards"] = {
        "num_shards": num_shards,
        "scheme": "global_index_modulo",
        "input_root": str(input_root),
    }
    atomic_write_json(
        output_dir / "metadata.json",
        {
            **first,
            "name": str(first.get("name", "features")),
            "source": source,
            "samples": total,
            "class_counts": counts,
            "completed_samples": total,
            "complete": True,
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-classes", type=int)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        merge_shards(
            args.input_root,
            args.output_dir,
            args.num_shards,
            expected_count=args.expected_count,
            expected_classes=args.expected_classes,
            chunk_size=args.chunk_size,
        )
    )


if __name__ == "__main__":
    main()
