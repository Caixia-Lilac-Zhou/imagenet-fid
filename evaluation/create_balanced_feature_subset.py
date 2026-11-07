#!/usr/bin/env python3
"""Create a deterministic, resumable class-balanced subset of a feature bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from output_safety import require_resumable_output

from fid_core import (
    atomic_write_json,
    class_count_summary,
    load_feature_bundle,
    sha256_file,
    sha256_text,
    stable_seed,
    stratified_subsample_indices,
)


def _source_identity(path: Path) -> str:
    if path.is_dir() and (path / "metadata.json").exists():
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        portable = metadata.get("source", {}).get("portable_identity")
        if portable:
            return str(portable)
        return sha256_file(path / "metadata.json")
    if path.is_file():
        stat = path.stat()
        return sha256_text(f"{path.resolve()}\n{stat.st_size}\n{stat.st_mtime_ns}")
    return sha256_text(str(path.resolve()))


def create_balanced_subset(
    source_path: Path,
    output_dir: Path,
    per_class: int,
    seed: int,
    copy_chunk_size: int = 8192,
    overwrite: bool = False,
    quiet: bool = False,
) -> Path:
    if per_class < 2:
        raise ValueError("per_class must be at least two")
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    if source_path == output_dir:
        raise ValueError('Source and output must be different directories')
    require_resumable_output(output_dir, 'metadata.json', resume=not overwrite)
    source = load_feature_bundle(source_path)
    source_identity = _source_identity(source_path)
    rng = np.random.default_rng(
        stable_seed(seed, "balanced_subset", source_identity, per_class)
    )
    indices = stratified_subsample_indices(source.labels, per_class, rng)
    if indices is None:
        _, counts = np.unique(source.labels, return_counts=True)
        raise ValueError(
            f"Requested {per_class}/class, but source minimum is {int(counts.min())}/class"
        )
    output_labels = np.asarray(source.labels[indices], dtype=np.int32)
    identity = {
        "schema_version": 1,
        "source": str(source_path),
        "source_identity": source_identity,
        "per_class": per_class,
        "seed": seed,
        "samples": int(indices.size),
        "feature_dimension": int(source.features.shape[1]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    features_path = output_dir / "features.npy"
    labels_path = output_dir / "labels.npy"
    indices_path = output_dir / "source_indices.npy"
    completed = 0
    if metadata_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in identity.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Existing subset at {output_dir} has different settings: {mismatches}"
            )
        if metadata.get("complete") is True:
            return output_dir
        completed = int(metadata.get("completed_samples", 0))

    if completed == 0:
        for path, array in [(labels_path, output_labels), (indices_path, indices.astype(np.int64))]:
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            temporary.replace(path)
        output_features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=source.features.dtype,
            shape=(indices.size, source.features.shape[1]),
        )
    else:
        output_features = np.lib.format.open_memmap(features_path, mode="r+")
        saved_indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
        saved_labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if not np.array_equal(saved_indices, indices) or not np.array_equal(saved_labels, output_labels):
            raise ValueError("Partial subset indices/labels do not match deterministic reconstruction")

    atomic_write_json(
        metadata_path,
        {**identity, "class_counts": class_count_summary(output_labels), "completed_samples": completed, "complete": False},
    )
    for start in range(completed, indices.size, copy_chunk_size):
        end = min(start + copy_chunk_size, indices.size)
        output_features[start:end] = source.features[indices[start:end]]
        output_features.flush()
        atomic_write_json(
            metadata_path,
            {
                **identity,
                "class_counts": class_count_summary(output_labels),
                "completed_samples": end,
                "complete": False,
            },
        )
        if not quiet:
            print(f"Balanced-{per_class}: {end}/{indices.size}", flush=True)
    output_features.flush()
    atomic_write_json(
        metadata_path,
        {
            **identity,
            "class_counts": class_count_summary(output_labels),
            "completed_samples": int(indices.size),
            "complete": True,
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--copy-chunk-size", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = create_balanced_subset(
        args.source,
        args.output,
        per_class=args.per_class,
        seed=args.seed,
        copy_chunk_size=args.copy_chunk_size,
        overwrite=args.overwrite,
    )
    print(path)


if __name__ == "__main__":
    main()
