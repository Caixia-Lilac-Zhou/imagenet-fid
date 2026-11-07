#!/usr/bin/env python3
"""Synthetic-image smoke test for extraction and balanced subset creation."""

from __future__ import annotations

import json
from test_support import preserved_directory
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import extract_inception_features as extraction
from create_balanced_feature_subset import create_balanced_subset
from combine_feature_bundles import combine
from fid_core import load_feature_bundle, validate_bundle_counts
from merge_feature_shards import merge_shards


class ToyExtractor:
    def eval(self):
        return self

    def requires_grad_(self, _value):
        return self

    def __call__(self, batch: torch.Tensor):
        batch = batch.to(torch.float32)
        channel_means = batch.mean(dim=(2, 3))
        features = torch.zeros((batch.shape[0], 2048), dtype=torch.float32, device=batch.device)
        features[:, :3] = channel_means
        features[:, 3] = batch.std(dim=(1, 2, 3))
        return (features,)


def main() -> None:
    rng = np.random.default_rng(23)
    with preserved_directory() as temporary_dir:
        root = Path(temporary_dir)
        image_root = root / "images"
        for class_index in range(3):
            class_dir = image_root / f"class_{class_index}"
            class_dir.mkdir(parents=True)
            for sample_index in range(5):
                height = 20 + sample_index
                width = 24 + class_index
                pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(class_dir / f"{sample_index:03d}.png")

        spec = extraction.DatasetSpec(
            name="synthetic",
            path=image_root,
            format="images",
            preprocess="jit_center_crop",
            image_size=16,
            label_mode="parent",
            expected_count=15,
            expected_classes=3,
            expected_per_class=5,
        )
        dataset, labels, _mapping, counts, _source, _signature = extraction.build_dataset(spec)
        assert tuple(dataset[0].shape) == (3, 16, 16)
        assert counts["balanced"] is True
        assert np.array_equal(np.bincount(labels), np.asarray([5, 5, 5]))

        original_loader = extraction._load_feature_extractor
        extraction._load_feature_extractor = lambda _weights, _device: (
            ToyExtractor(),
            torch.device("cpu"),
        )
        try:
            bundle_dir = extraction.extract_features(
                spec,
                root / "features",
                batch_size=4,
                num_workers=0,
                device="cpu",
                checkpoint_every=1,
                quiet=True,
            )
            bundle = load_feature_bundle(bundle_dir)
            validate_bundle_counts(bundle, expected_count=15, expected_classes=3, expected_per_class=5)
            assert bundle.features.shape == (15, 2048)

            shard_root = root / "feature_shards"
            for shard_index in range(2):
                extraction.extract_features(
                    spec,
                    shard_root / f"shard-{shard_index:03d}-of-002",
                    batch_size=4,
                    num_workers=0,
                    device="cpu",
                    checkpoint_every=1,
                    quiet=True,
                    shard_index=shard_index,
                    num_shards=2,
                )
            merged_dir = merge_shards(
                shard_root,
                root / "merged_features",
                2,
                expected_count=15,
                expected_classes=3,
                chunk_size=4,
            )
            merged = load_feature_bundle(merged_dir)
            assert np.array_equal(merged.labels, bundle.labels)
            assert np.array_equal(merged.features, bundle.features)

            combined_dir = combine([bundle_dir, merged_dir], root / "combined_features", 4)
            combined = load_feature_bundle(combined_dir)
            validate_bundle_counts(
                combined, expected_count=30, expected_classes=3, expected_per_class=10
            )
            assert np.array_equal(combined.features[:15], bundle.features)
            assert np.array_equal(combined.features[15:], merged.features)

            metadata_path = bundle_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["complete"] = False
            metadata["completed_samples"] = 7
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            mutable = np.lib.format.open_memmap(bundle_dir / "features.npy", mode="r+")
            mutable[7:] = np.nan
            mutable.flush()
            extraction.extract_features(
                spec,
                bundle_dir,
                batch_size=4,
                num_workers=0,
                device="cpu",
                checkpoint_every=1,
                quiet=True,
            )
            resumed = load_feature_bundle(bundle_dir)
            assert np.isfinite(np.asarray(resumed.features)).all()

            subset_dir = create_balanced_subset(
                bundle_dir,
                root / "balanced3",
                per_class=3,
                seed=99,
                copy_chunk_size=4,
                quiet=True,
            )
            subset = load_feature_bundle(subset_dir)
            validate_bundle_counts(subset, expected_count=9, expected_classes=3, expected_per_class=3)
            second = create_balanced_subset(
                bundle_dir,
                root / "balanced3",
                per_class=3,
                seed=99,
                quiet=True,
            )
            assert second == subset_dir
        finally:
            extraction._load_feature_extractor = original_loader

    print("feature_pipeline: all tests passed")


if __name__ == "__main__":
    main()
