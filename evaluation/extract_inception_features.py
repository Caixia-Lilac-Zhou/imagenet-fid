#!/usr/bin/env python3
"""Resumable per-image Inception feature extraction for rebuttal experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from output_safety import require_resumable_output

from fid_core import atomic_write_json, class_count_summary, sha256_text


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
FEATURE_DIMENSION = 2048


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    format: str = "auto"
    preprocess: str = "none"
    image_size: int = 256
    label_mode: str = "auto"
    labels_path: Path | None = None
    filename_regex: str | None = None
    image_key: str | None = None
    label_key: str | None = None
    expected_count: int | None = None
    expected_classes: int | None = None
    expected_per_class: int | None = None


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def dataset_spec_from_dict(raw: dict[str, Any], base_dir: Path) -> DatasetSpec:
    if not raw.get("name") or not raw.get("path"):
        raise ValueError("Every dataset requires non-empty name and path")
    source_path = Path(str(raw["path"])).expanduser()
    source_path = source_path.resolve() if source_path.is_absolute() else (base_dir / source_path).resolve()
    labels_path = None
    if raw.get("labels_path"):
        labels_path = Path(str(raw["labels_path"])).expanduser()
        labels_path = (
            labels_path.resolve()
            if labels_path.is_absolute()
            else (base_dir / labels_path).resolve()
        )
    return DatasetSpec(
        name=str(raw["name"]),
        path=source_path,
        format=str(raw.get("format", "auto")),
        preprocess=str(raw.get("preprocess", "none")),
        image_size=int(raw.get("image_size", 256)),
        label_mode=str(raw.get("label_mode", "auto")),
        labels_path=labels_path,
        filename_regex=raw.get("filename_regex"),
        image_key=raw.get("image_key"),
        label_key=raw.get("label_key"),
        expected_count=_optional_int(raw.get("expected_count")),
        expected_classes=_optional_int(raw.get("expected_classes")),
        expected_per_class=_optional_int(raw.get("expected_per_class")),
    )


def center_crop_arr(pil_image, image_size: int):
    """ADM/JiT center crop; upstream notices are in licenses/ADM_JiT_MIT.txt."""
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=resampling.BOX,
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=resampling.BICUBIC,
    )
    array = np.asarray(pil_image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(array[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _to_uint8_chw(image: Any, preprocess: str, image_size: int):
    import torch
    from PIL import Image

    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        elif array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Expected HxWxC or CxHxW image, got {array.shape}")
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            values = np.asarray(array, dtype=np.float32)
            minimum = float(np.nanmin(values))
            maximum = float(np.nanmax(values))
            if not np.isfinite([minimum, maximum]).all():
                raise ValueError("Image contains non-finite pixels")
            if minimum >= -1.01 and maximum <= 1.01 and minimum < -0.01:
                values = (values + 1.0) * 127.5
            elif minimum >= -0.01 and maximum <= 1.01:
                values = values * 255.0
            array = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(array, mode="RGB")

    if preprocess == "jit_center_crop":
        pil_image = center_crop_arr(pil_image, image_size)
    elif preprocess == "resize_center_crop":
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image)
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(value * scale) for value in pil_image.size),
            resample=resampling.BICUBIC,
        )
        left = (pil_image.width - image_size) // 2
        top = (pil_image.height - image_size) // 2
        pil_image = pil_image.crop((left, top, left + image_size, top + image_size))
    elif preprocess != "none":
        raise ValueError(
            f"Unknown preprocessing {preprocess!r}; choose none, jit_center_crop, or resize_center_crop"
        )
    array = np.asarray(pil_image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_sidecar_labels(
    path: Path,
    count: int,
    relative_paths: Sequence[str] | None = None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        labels = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = next((key for key in ["labels", "y", "classes", "arr_0"] if key in archive), None)
            if key is None:
                raise ValueError(f"No label array found in {path}; keys={list(archive.files)}")
            labels = np.asarray(archive[key])
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "label" not in rows[0]:
            raise ValueError(f"CSV sidecar {path} must contain a label column")
        if "path" in rows[0] and relative_paths is not None:
            mapping = {str(row["path"]): row["label"] for row in rows}
            missing = [relative for relative in relative_paths if relative not in mapping]
            if missing:
                raise ValueError(f"CSV sidecar lacks {len(missing)} image paths; first={missing[0]}")
            labels = np.asarray([mapping[relative] for relative in relative_paths])
        else:
            labels = np.asarray([row["label"] for row in rows])
    else:
        labels = np.asarray(
            [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        )
    labels = np.asarray(labels).reshape(-1)
    if labels.size != count:
        raise ValueError(f"Label sidecar has {labels.size} entries; expected {count}")
    return labels


def _encode_labels(raw_labels: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.asarray(raw_labels).reshape(-1)
    if labels.dtype.kind in "iu":
        integer_labels = labels.astype(np.int64, copy=False)
        classes = sorted(int(value) for value in np.unique(integer_labels))
        mapping = {str(value): index for index, value in enumerate(classes)}
        if classes == list(range(len(classes))):
            return integer_labels, mapping
        encoded = np.asarray([mapping[str(int(value))] for value in integer_labels], dtype=np.int64)
        return encoded, mapping
    string_labels = labels.astype(str)
    classes = sorted(np.unique(string_labels).tolist())
    mapping = {value: index for index, value in enumerate(classes)}
    encoded = np.asarray([mapping[value] for value in string_labels], dtype=np.int64)
    return encoded, mapping


class ImagePathsDataset:
    def __init__(self, paths: Sequence[Path], preprocess: str, image_size: int):
        self.paths = list(paths)
        self.preprocess = preprocess
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        from PIL import Image

        path = self.paths[index]
        try:
            with Image.open(path) as image:
                return _to_uint8_chw(image, self.preprocess, self.image_size)
        except Exception as error:
            raise RuntimeError(f"Failed to decode {path}") from error


class ArrayImagesDataset:
    def __init__(self, images: np.ndarray, preprocess: str, image_size: int):
        self.images = images
        self.preprocess = preprocess
        self.image_size = image_size

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        return _to_uint8_chw(self.images[index], self.preprocess, self.image_size)


def _collate_same_shape(images):
    import torch

    shapes = {tuple(image.shape) for image in images}
    if len(shapes) != 1:
        raise ValueError(
            f"Images in one batch have different shapes {sorted(shapes)}. "
            "Use preprocess=jit_center_crop for raw ImageNet images."
        )
    return torch.stack(images, dim=0)


def _discover_image_paths(root: Path) -> list[Path]:
    paths = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found under {root}")
    return paths


def _resolve_format(spec: DatasetSpec) -> str:
    if spec.format != "auto":
        return spec.format
    if spec.path.is_dir():
        return "images"
    if spec.path.suffix.lower() == ".npz":
        return "npz"
    if spec.path.suffix.lower() == ".npy":
        return "npy"
    raise ValueError(f"Cannot infer input format for {spec.path}")


def _labels_from_mode(
    spec: DatasetSpec,
    count: int,
    relative_paths: Sequence[str] | None,
    embedded_labels: np.ndarray | None,
) -> np.ndarray:
    mode = spec.label_mode
    if mode == "auto":
        if embedded_labels is not None:
            mode = "embedded"
        elif spec.labels_path is not None:
            mode = "sidecar"
        elif relative_paths is not None and any("/" in path for path in relative_paths):
            mode = "parent"
        else:
            raise ValueError(
                f"Cannot infer labels for {spec.name}; set label_mode and/or labels_path"
            )
    if mode == "embedded":
        if embedded_labels is None:
            raise ValueError(f"{spec.name} has no embedded label array")
        labels = embedded_labels
    elif mode == "sidecar":
        if spec.labels_path is None:
            raise ValueError(f"{spec.name} label_mode=sidecar requires labels_path")
        labels = _load_sidecar_labels(spec.labels_path, count, relative_paths)
    elif mode == "parent":
        if relative_paths is None:
            raise ValueError("label_mode=parent is only valid for image directories")
        labels = np.asarray([Path(path).parts[0] for path in relative_paths])
    elif mode == "filename_regex":
        if relative_paths is None or not spec.filename_regex:
            raise ValueError("filename_regex mode requires image paths and filename_regex")
        pattern = re.compile(spec.filename_regex)
        parsed = []
        for relative in relative_paths:
            match = pattern.search(Path(relative).name)
            if match is None:
                raise ValueError(f"filename_regex did not match {relative}")
            parsed.append(match.groupdict().get("label") or match.group(1))
        labels = np.asarray(parsed)
    elif mode in {"balanced_block", "balanced_cycle"}:
        if spec.expected_classes is None:
            raise ValueError(f"{mode} requires expected_classes")
        if count % spec.expected_classes != 0:
            raise ValueError(f"{count} samples are not divisible by {spec.expected_classes} classes")
        per_class = count // spec.expected_classes
        labels = (
            np.repeat(np.arange(spec.expected_classes), per_class)
            if mode == "balanced_block"
            else np.tile(np.arange(spec.expected_classes), per_class)
        )
    else:
        raise ValueError(f"Unsupported label_mode {mode!r}")
    labels = np.asarray(labels).reshape(-1)
    if labels.size != count:
        raise ValueError(f"Resolved {labels.size} labels for {count} samples")
    return labels


def build_dataset(spec: DatasetSpec):
    resolved_format = _resolve_format(spec)
    relative_paths: list[str] | None = None
    embedded_labels = None
    source_descriptor: dict[str, object]
    if resolved_format == "images":
        if not spec.path.is_dir():
            raise ValueError(f"Image input is not a directory: {spec.path}")
        resolved_root = spec.path.resolve()
        paths = _discover_image_paths(resolved_root)
        relative_paths = [path.relative_to(resolved_root).as_posix() for path in paths]
        labels = _labels_from_mode(spec, len(paths), relative_paths, None)
        dataset = ImagePathsDataset(paths, spec.preprocess, spec.image_size)
        source_descriptor = {
            "format": resolved_format,
            "root": str(resolved_root),
            "relative_paths_sha256": sha256_text("\n".join(relative_paths)),
        }
    elif resolved_format in {"npz", "npy"}:
        if not spec.path.is_file():
            raise ValueError(f"Array input is not a file: {spec.path}")
        if resolved_format == "npy":
            images = np.load(spec.path, mmap_mode="r", allow_pickle=False)
        else:
            with np.load(spec.path, allow_pickle=False) as archive:
                image_key = spec.image_key or next(
                    (key for key in ["images", "arr_0", "samples", "x"] if key in archive),
                    None,
                )
                if image_key is None:
                    raise ValueError(f"No image array found in {spec.path}; keys={list(archive.files)}")
                images = np.asarray(archive[image_key])
                label_key = spec.label_key or next(
                    (key for key in ["labels", "y", "classes", "arr_1"] if key in archive),
                    None,
                )
                if label_key is not None:
                    embedded_labels = np.asarray(archive[label_key])
        if images.ndim != 4:
            raise ValueError(f"Expected a four-dimensional image array, got {images.shape}")
        labels = _labels_from_mode(spec, images.shape[0], None, embedded_labels)
        dataset = ArrayImagesDataset(images, spec.preprocess, spec.image_size)
        stat = spec.path.stat()
        source_descriptor = {
            "format": resolved_format,
            "path": str(spec.path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "shape": list(images.shape),
            "dtype": str(images.dtype),
        }
    else:
        raise ValueError(f"Unsupported format {resolved_format!r}")

    encoded_labels, mapping = _encode_labels(labels)
    summary = class_count_summary(encoded_labels)
    if spec.expected_count is not None and summary["samples"] != spec.expected_count:
        raise ValueError(
            f"{spec.name}: expected {spec.expected_count} samples, found {summary['samples']}"
        )
    if spec.expected_classes is not None and summary["classes"] != spec.expected_classes:
        raise ValueError(
            f"{spec.name}: expected {spec.expected_classes} classes, found {summary['classes']}"
        )
    if spec.expected_per_class is not None and (
        summary["min_per_class"] != spec.expected_per_class
        or summary["max_per_class"] != spec.expected_per_class
    ):
        raise ValueError(
            f"{spec.name}: expected {spec.expected_per_class}/class, found "
            f"{summary['min_per_class']}--{summary['max_per_class']}"
        )
    labels_sha256 = hashlib.sha256(
        np.ascontiguousarray(encoded_labels).view(np.uint8)
    ).hexdigest()
    portable_source = {
        key: value
        for key, value in source_descriptor.items()
        if key not in {"root", "path", "mtime_ns"}
    }
    source_descriptor["portable_identity"] = sha256_text(
        json.dumps(
            {
                "name": spec.name,
                "source": portable_source,
                "labels_sha256": labels_sha256,
                "preprocess": spec.preprocess,
                "image_size": spec.image_size,
            },
            sort_keys=True,
        )
    )
    signature_payload = {
        "name": spec.name,
        "source": source_descriptor,
        "preprocess": spec.preprocess,
        "image_size": spec.image_size,
        "label_mode": spec.label_mode,
        "labels_path": str(spec.labels_path) if spec.labels_path else None,
        "filename_regex": spec.filename_regex,
        "count_summary": summary,
        "labels_sha256": labels_sha256,
    }
    signature = sha256_text(json.dumps(signature_payload, sort_keys=True))
    return dataset, encoded_labels, mapping, summary, source_descriptor, signature


def _load_feature_extractor(weights_path: Path | None, device: str):
    try:
        import torch
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    except ImportError as error:
        raise RuntimeError(
            "Feature extraction requires torch and torch-fidelity==0.4.0"
        ) from error
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    extractor = FeatureExtractorInceptionV3(
        name="inception-v3-compat",
        features_list=["2048"],
        feature_extractor_weights_path=str(weights_path) if weights_path else None,
        feature_extractor_internal_dtype="float32",
        verbose=True,
    ).to(device)
    extractor.eval().requires_grad_(False)
    return extractor, torch.device(device)


def extract_features(
    spec: DatasetSpec,
    output_dir: Path,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "auto",
    weights_path: Path | None = None,
    checkpoint_every: int = 10,
    overwrite: bool = False,
    quiet: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Path:
    import torch
    from torch.utils.data import DataLoader, Subset

    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"Require num_shards >= 1 and 0 <= shard_index < num_shards; "
            f"got shard_index={shard_index}, num_shards={num_shards}"
        )
    dataset, labels, mapping, counts, source_descriptor, signature = build_dataset(spec)
    if num_shards > 1:
        full_count = len(dataset)
        shard_indices = np.arange(shard_index, full_count, num_shards, dtype=np.int64)
        dataset = Subset(dataset, shard_indices.tolist())
        labels = labels[shard_indices]
        counts = class_count_summary(labels)
        shard_descriptor = {
            "index": shard_index,
            "num_shards": num_shards,
            "scheme": "global_index_modulo",
            "full_samples": full_count,
        }
        source_descriptor = {**source_descriptor, "shard": shard_descriptor}
        signature = sha256_text(
            json.dumps(
                {"parent_input_signature": signature, "shard": shard_descriptor},
                sort_keys=True,
            )
        )
    output_dir = output_dir.resolve()
    require_resumable_output(output_dir, 'metadata.json', resume=not overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.npy"
    labels_path = output_dir / "labels.npy"
    metadata_path = output_dir / "metadata.json"
    mapping_path = output_dir / "class_mapping.json"

    completed = 0
    if metadata_path.exists() and not overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("input_signature") != signature:
            raise ValueError(
                f"Existing feature bundle at {output_dir} was built from different inputs/settings; "
                "choose a new output directory"
            )
        if metadata.get("complete") is True:
            if not quiet:
                print(f"Feature bundle already complete: {output_dir}", file=sys.stderr)
            return output_dir
        completed = int(metadata.get("completed_samples", 0))
    elif overwrite:
        completed = 0

    if completed == 0:
        labels_temp = labels_path.with_suffix(".npy.tmp")
        with labels_temp.open("wb") as handle:
            np.save(handle, labels.astype(np.int32, copy=False), allow_pickle=False)
        labels_temp.replace(labels_path)
        atomic_write_json(mapping_path, {"class_to_index": mapping})
        features = np.lib.format.open_memmap(
            features_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(dataset), FEATURE_DIMENSION),
        )
    else:
        features = np.lib.format.open_memmap(features_path, mode="r+")
        saved_labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if features.shape != (len(dataset), FEATURE_DIMENSION) or not np.array_equal(saved_labels, labels):
            raise ValueError(f"Partial feature bundle shape/labels do not match input: {output_dir}")
    if not (0 <= completed <= len(dataset)):
        raise ValueError(f"Invalid completed_samples={completed} in {metadata_path}")

    extractor, torch_device = _load_feature_extractor(weights_path, device)
    package_versions = {}
    for package in ["torch", "torchvision", "torch-fidelity", "numpy", "Pillow"]:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "unknown"
    base_metadata: dict[str, object] = {
        "schema_version": 1,
        "name": spec.name,
        "input_signature": signature,
        "source": source_descriptor,
        "samples": len(dataset),
        "feature_dimension": FEATURE_DIMENSION,
        "feature_extractor": "torch-fidelity/inception-v3-compat:2048",
        "preprocess": spec.preprocess,
        "preprocess_size": spec.image_size,
        "decode": "PIL RGB for files; deterministic uint8 conversion for arrays",
        "inception_internal_preprocess": "TensorFlow-1.x-compatible bilinear resize to 299; (x-128)/128",
        "class_counts": counts,
        "versions": package_versions,
        "device": str(torch_device),
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    atomic_write_json(
        metadata_path,
        {**base_metadata, "completed_samples": completed, "complete": False},
    )

    if completed < len(dataset):
        subset = Subset(dataset, range(completed, len(dataset)))
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch_device.type == "cuda",
            persistent_workers=num_workers > 0,
            collate_fn=_collate_same_shape,
        )
        total_batches = (len(subset) + batch_size - 1) // batch_size
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                batch = batch.to(torch_device, non_blocking=True)
                extracted = extractor(batch)[0].detach().cpu().numpy().astype(np.float32, copy=False)
                if extracted.ndim != 2 or extracted.shape[1] != FEATURE_DIMENSION:
                    raise RuntimeError(f"Unexpected Inception output shape {extracted.shape}")
                if not np.isfinite(extracted).all():
                    raise ValueError("Inception produced non-finite features")
                start = completed
                end = start + extracted.shape[0]
                features[start:end] = extracted
                completed = end
                if (
                    (batch_index + 1) % checkpoint_every == 0
                    or batch_index + 1 == total_batches
                ):
                    features.flush()
                    atomic_write_json(
                        metadata_path,
                        {
                            **base_metadata,
                            "completed_samples": completed,
                            "complete": False,
                        },
                    )
                if not quiet and (
                    (batch_index + 1) % max(1, total_batches // 20) == 0
                    or batch_index + 1 == total_batches
                ):
                    print(
                        f"{spec.name}: {completed}/{len(dataset)} features",
                        file=sys.stderr,
                    )

    features.flush()
    if completed != len(dataset):
        raise RuntimeError(f"Extraction ended at {completed}/{len(dataset)} samples")
    atomic_write_json(
        metadata_path,
        {**base_metadata, "completed_samples": completed, "complete": True},
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(args.spec_json.read_text(encoding="utf-8"))
    spec = dataset_spec_from_dict(raw, args.spec_json.resolve().parent)
    path = extract_features(
        spec,
        args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        weights_path=args.weights,
        overwrite=args.overwrite,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    print(path)


if __name__ == "__main__":
    main()
