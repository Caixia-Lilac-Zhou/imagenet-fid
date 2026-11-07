#!/usr/bin/env python3
"""Extract resumable per-image features for the CIFAR-10 backbone check."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from output_safety import require_resumable_output
from extract_inception_features import (
    _collate_same_shape,
    build_dataset,
    dataset_spec_from_dict,
)
from fid_core import atomic_write_json


BACKBONE_DIMENSIONS = {
    "dinov2-vit-b-14": 768,
    "clip-vit-b-16": 512,
}


def load_extractor(backbone: str, device: str):
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if backbone == "dinov2-vit-b-14":
        from torch_fidelity.feature_extractor_dinov2 import FeatureExtractorDinoV2

        extractor = FeatureExtractorDinoV2(
            name=backbone,
            features_list=["dinov2"],
            feature_extractor_weights_path=None,
            feature_extractor_internal_dtype="float32",
        )
    elif backbone == "clip-vit-b-16":
        from torch_fidelity.feature_extractor_clip import FeatureExtractorCLIP

        extractor = FeatureExtractorCLIP(
            name=backbone,
            features_list=["clip"],
            feature_extractor_weights_path=None,
            feature_extractor_internal_dtype="float32",
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    extractor = extractor.to(device)
    return extractor.eval().requires_grad_(False), torch.device(device)


def write_metadata(path: Path, value: dict[str, object]) -> None:
    atomic_write_json(path, value)
    path.chmod(0o644)


def extract(
    spec_json: Path,
    output_dir: Path,
    backbone: str,
    batch_size: int,
    num_workers: int,
    device: str,
    overwrite: bool,
) -> Path:
    import torch
    from torch.utils.data import DataLoader, Subset

    raw = json.loads(spec_json.read_text(encoding="utf-8"))
    spec = dataset_spec_from_dict(raw, spec_json.resolve().parent)
    dataset, labels, mapping, counts, source, signature = build_dataset(spec)
    dimension = BACKBONE_DIMENSIONS[backbone]
    require_resumable_output(output_dir, 'metadata.json', resume=not overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.npy"
    labels_path = output_dir / "labels.npy"
    metadata_path = output_dir / "metadata.json"
    mapping_path = output_dir / "class_mapping.json"

    completed = 0
    if metadata_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("input_signature") != signature or metadata.get("backbone") != backbone:
            raise ValueError(f"Existing bundle at {output_dir} has different inputs or backbone")
        if metadata.get("complete") is True:
            print(f"Feature bundle already complete: {output_dir}", file=sys.stderr)
            return output_dir
        completed = int(metadata.get("completed_samples", 0))

    if completed == 0:
        labels_temp = labels_path.with_suffix(".npy.tmp")
        with labels_temp.open("wb") as handle:
            np.save(handle, labels.astype(np.int32, copy=False), allow_pickle=False)
        labels_temp.replace(labels_path)
        labels_path.chmod(0o644)
        atomic_write_json(mapping_path, {"class_to_index": mapping})
        mapping_path.chmod(0o644)
        features = np.lib.format.open_memmap(
            features_path, mode="w+", dtype=np.float32, shape=(len(dataset), dimension)
        )
        features_path.chmod(0o644)
    else:
        features = np.lib.format.open_memmap(features_path, mode="r+")
        saved_labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if features.shape != (len(dataset), dimension) or not np.array_equal(saved_labels, labels):
            raise ValueError(f"Partial feature bundle does not match {spec.name}")

    extractor, torch_device = load_extractor(backbone, device)
    versions = {}
    for package in ("torch", "torchvision", "torch-fidelity", "numpy", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    base_metadata = {
        "schema_version": 1,
        "name": spec.name,
        "input_signature": signature,
        "source": source,
        "samples": len(dataset),
        "feature_dimension": dimension,
        "feature_extractor": f"torch-fidelity/{backbone}",
        "backbone": backbone,
        "preprocess": spec.preprocess,
        "class_counts": counts,
        "versions": versions,
        "device": str(torch_device),
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    write_metadata(
        metadata_path, {**base_metadata, "completed_samples": completed, "complete": False}
    )

    if completed < len(dataset):
        loader = DataLoader(
            Subset(dataset, range(completed, len(dataset))),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch_device.type == "cuda",
            persistent_workers=num_workers > 0,
            collate_fn=_collate_same_shape,
        )
        total_batches = len(loader)
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                extracted = (
                    extractor(batch.to(torch_device, non_blocking=True))[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                if extracted.ndim != 2 or extracted.shape[1] != dimension:
                    raise RuntimeError(f"Unexpected {backbone} output shape {extracted.shape}")
                if not np.isfinite(extracted).all():
                    raise ValueError(f"{backbone} produced non-finite features")
                end = completed + extracted.shape[0]
                features[completed:end] = extracted
                completed = end
                if (batch_index + 1) % 10 == 0 or batch_index + 1 == total_batches:
                    features.flush()
                    write_metadata(
                        metadata_path,
                        {**base_metadata, "completed_samples": completed, "complete": False},
                    )
                if (batch_index + 1) % max(1, total_batches // 20) == 0:
                    print(f"{spec.name}: {completed}/{len(dataset)} {backbone} features", file=sys.stderr)

    features.flush()
    write_metadata(
        metadata_path, {**base_metadata, "completed_samples": completed, "complete": True}
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=tuple(BACKBONE_DIMENSIONS), default="dinov2-vit-b-14")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        extract(
            args.spec_json,
            args.output_dir,
            args.backbone,
            args.batch_size,
            args.num_workers,
            args.device,
            args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
