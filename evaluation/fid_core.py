#!/usr/bin/env python3
"""Shared data, manifest, and numerical utilities for rebuttal FD experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np


REQUIRED_MANIFEST_COLUMNS = {
    "model",
    "protocol",
    "generated_npz",
    "reference_npz",
    "reference_stats_npz",
}


@dataclass(frozen=True)
class ManifestRow:
    model: str
    protocol: str
    generated_path: Path
    reference_path: Path | None
    reference_stats_path: Path | None
    expected_fd: float | None


@dataclass(frozen=True)
class FeatureBundle:
    features: np.ndarray
    labels: np.ndarray
    source_path: Path | None = None


@dataclass(frozen=True)
class FeatureStats:
    mu: Any
    cov: Any
    cov_sqrt: Any | None = None


class FDBackend(Protocol):
    name: str
    device: str

    def stats(
        self,
        features: np.ndarray,
        indices: np.ndarray | None = None,
        cache_sqrt: bool = False,
    ) -> FeatureStats: ...

    def from_numpy_stats(self, mu: np.ndarray, cov: np.ndarray, cache_sqrt: bool = True) -> FeatureStats: ...

    def distance(self, a: FeatureStats, b: FeatureStats) -> float: ...


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stratified_bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    chunks = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        chunks.append(rng.choice(members, size=members.size, replace=True))
    return np.sort(np.concatenate(chunks))


def stratified_subsample_indices(
    labels: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    chunks = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if members.size < per_class:
            return None
        chunks.append(rng.choice(members, size=per_class, replace=False))
    return np.sort(np.concatenate(chunks))


def class_count_summary(labels: np.ndarray) -> dict[str, object]:
    classes, counts = np.unique(labels, return_counts=True)
    return {
        "samples": int(labels.size),
        "classes": int(classes.size),
        "min_per_class": int(counts.min()),
        "max_per_class": int(counts.max()),
        "mean_per_class": float(counts.mean()),
        "balanced": bool(np.all(counts == counts[0])),
    }


def validate_bundle_counts(
    bundle: FeatureBundle,
    expected_count: int | None = None,
    expected_classes: int | None = None,
    expected_per_class: int | None = None,
) -> dict[str, object]:
    summary = class_count_summary(bundle.labels)
    if expected_count is not None and summary["samples"] != expected_count:
        raise ValueError(f"Expected {expected_count} samples, found {summary['samples']}")
    if expected_classes is not None and summary["classes"] != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, found {summary['classes']}")
    if expected_per_class is not None and (
        summary["min_per_class"] != expected_per_class
        or summary["max_per_class"] != expected_per_class
    ):
        raise ValueError(
            f"Expected {expected_per_class} samples per class, found "
            f"{summary['min_per_class']}--{summary['max_per_class']}"
        )
    return summary


def _resolve_path(raw: str, base_dir: Path) -> Path | None:
    raw = raw.strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def read_manifest(path: Path) -> list[ManifestRow]:
    path = path.resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        records = list(reader)

    rows = []
    seen = set()
    for line_number, record in enumerate(records, start=2):
        model = record["model"].strip()
        protocol = record["protocol"].strip()
        if not model or not protocol:
            raise ValueError(f"Line {line_number}: model and protocol are required")
        key = (model, protocol)
        if key in seen:
            raise ValueError(f"Line {line_number}: duplicate model/protocol pair {key}")
        seen.add(key)

        generated_path = _resolve_path(record["generated_npz"], path.parent)
        reference_path = _resolve_path(record["reference_npz"], path.parent)
        reference_stats_path = _resolve_path(record["reference_stats_npz"], path.parent)
        if generated_path is None:
            raise ValueError(f"Line {line_number}: generated_npz is required")
        if (reference_path is None) == (reference_stats_path is None):
            raise ValueError(
                f"Line {line_number}: provide exactly one of reference_npz and "
                "reference_stats_npz"
            )
        expected_raw = record.get("expected_fd", "").strip()
        rows.append(
            ManifestRow(
                model=model,
                protocol=protocol,
                generated_path=generated_path,
                reference_path=reference_path,
                reference_stats_path=reference_stats_path,
                expected_fd=float(expected_raw) if expected_raw else None,
            )
        )
    if not rows:
        raise ValueError("Manifest contains no experiment rows")
    return rows


def _validate_feature_arrays(features: np.ndarray, labels: np.ndarray, path: Path) -> None:
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Invalid feature bundle {path}: features={features.shape}, labels={labels.shape}"
        )
    if features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError(f"{path} contains too few features")
    check_rows = min(1024, features.shape[0])
    probe_indices = np.linspace(0, features.shape[0] - 1, check_rows, dtype=np.int64)
    if not np.isfinite(np.asarray(features[probe_indices])).all():
        raise ValueError(f"{path} contains non-finite feature values")


def load_feature_bundle(path: Path) -> FeatureBundle:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("complete") is not True:
                raise ValueError(f"Feature extraction is incomplete: {path}")
        features_path = path / "features.npy"
        labels_path = path / "labels.npy"
        if not features_path.exists() or not labels_path.exists():
            raise ValueError(f"Feature directory must contain features.npy and labels.npy: {path}")
        features = np.load(features_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "features" not in archive or "labels" not in archive:
                raise ValueError(f"{path} must contain features and labels arrays")
            features = np.asarray(archive["features"])
            labels = np.asarray(archive["labels"])
    else:
        raise ValueError(f"Unsupported feature bundle {path}; use a bundle directory or .npz")
    _validate_feature_arrays(features, labels, path)
    return FeatureBundle(features=features, labels=labels, source_path=path)


def _iter_feature_chunks(
    features: np.ndarray,
    indices: np.ndarray | None,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    total = features.shape[0] if indices is None else indices.size
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        selection: slice | np.ndarray = slice(start, end) if indices is None else indices[start:end]
        yield np.asarray(features[selection])


def _numpy_matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    if float(eigvals.min()) < -1e-7 * scale:
        raise ValueError("Matrix is materially non-positive-semidefinite")
    return (eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))) @ eigvecs.T


class NumpyBackend:
    name = "numpy"
    device = "cpu"

    def __init__(self, chunk_size: int = 8192):
        if chunk_size < 2:
            raise ValueError("chunk_size must be at least two")
        self.chunk_size = chunk_size

    def stats(
        self,
        features: np.ndarray,
        indices: np.ndarray | None = None,
        cache_sqrt: bool = False,
    ) -> FeatureStats:
        if features.ndim != 2:
            raise ValueError(f"Expected N x D features, got {features.shape}")
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            if indices.ndim != 1 or indices.size < 2:
                raise ValueError("indices must contain at least two rows")
        count = features.shape[0] if indices is None else indices.size
        if count < 2:
            raise ValueError("At least two features are required")

        total = np.zeros(features.shape[1], dtype=np.float64)
        seen = 0
        for chunk in _iter_feature_chunks(features, indices, self.chunk_size):
            values = np.asarray(chunk, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Features contain non-finite values")
            total += values.sum(axis=0)
            seen += values.shape[0]
        if seen != count:
            raise RuntimeError(f"Feature count changed while reading: {seen} != {count}")
        mu = total / count

        cov_accumulator = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
        for chunk in _iter_feature_chunks(features, indices, self.chunk_size):
            centered = np.asarray(chunk, dtype=np.float64) - mu
            cov_accumulator += centered.T @ centered
        cov = cov_accumulator / (count - 1)
        cov = (cov + cov.T) * 0.5
        return FeatureStats(
            mu=mu,
            cov=cov,
            cov_sqrt=_numpy_matrix_sqrt_psd(cov) if cache_sqrt else None,
        )

    def from_numpy_stats(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        cache_sqrt: bool = True,
    ) -> FeatureStats:
        mu = np.asarray(mu, dtype=np.float64)
        cov = np.asarray(cov, dtype=np.float64)
        cov = (cov + cov.T) * 0.5
        return FeatureStats(
            mu=mu,
            cov=cov,
            cov_sqrt=_numpy_matrix_sqrt_psd(cov) if cache_sqrt else None,
        )

    def distance(self, a: FeatureStats, b: FeatureStats) -> float:
        if a.mu.shape != b.mu.shape or a.cov.shape != b.cov.shape:
            raise ValueError(
                f"Feature-statistic dimensions differ: {a.mu.shape}/{a.cov.shape} "
                f"versus {b.mu.shape}/{b.cov.shape}"
            )
        mean_term = float(np.square(a.mu - b.mu).sum())
        if b.cov_sqrt is not None:
            middle = b.cov_sqrt @ a.cov @ b.cov_sqrt
        else:
            sqrt_a = a.cov_sqrt if a.cov_sqrt is not None else _numpy_matrix_sqrt_psd(a.cov)
            middle = sqrt_a @ b.cov @ sqrt_a
        eigvals = np.linalg.eigvalsh((middle + middle.T) * 0.5)
        scale = max(float(np.max(np.abs(eigvals))), 1.0)
        if float(eigvals.min()) < -1e-7 * scale:
            raise ValueError("Covariance product is materially non-positive-semidefinite")
        trace_sqrt = float(np.sqrt(np.clip(eigvals, 0.0, None)).sum())
        fd = mean_term + float(np.trace(a.cov) + np.trace(b.cov)) - 2.0 * trace_sqrt
        if fd < -1e-6:
            raise ValueError(f"Computed a materially negative FD ({fd})")
        return max(fd, 0.0)


class TorchBackend:
    name = "torch"

    def __init__(self, device: str = "cuda", chunk_size: int = 8192):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Torch backend requested, but PyTorch is unavailable") from error
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested, but torch.cuda.is_available() is false")
        self.torch = torch
        self.torch_device = torch.device(device)
        self.device = str(self.torch_device)
        self.chunk_size = chunk_size

    def _matrix_sqrt_psd(self, matrix: Any) -> Any:
        torch = self.torch
        matrix = (matrix + matrix.T) * 0.5
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        scale = max(float(eigvals.abs().max().item()), 1.0)
        if float(eigvals.min().item()) < -1e-7 * scale:
            raise ValueError("Matrix is materially non-positive-semidefinite")
        eigvals = eigvals.clamp_min(0.0)
        return (eigvecs * eigvals.sqrt().unsqueeze(0)) @ eigvecs.T

    def stats(
        self,
        features: np.ndarray,
        indices: np.ndarray | None = None,
        cache_sqrt: bool = False,
    ) -> FeatureStats:
        torch = self.torch
        if features.ndim != 2:
            raise ValueError(f"Expected N x D features, got {features.shape}")
        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            if indices.ndim != 1 or indices.size < 2:
                raise ValueError("indices must contain at least two rows")
        count = features.shape[0] if indices is None else indices.size
        if count < 2:
            raise ValueError("At least two features are required")

        total = torch.zeros(features.shape[1], dtype=torch.float64, device=self.torch_device)
        seen = 0
        for chunk in _iter_feature_chunks(features, indices, self.chunk_size):
            values = torch.as_tensor(
                np.asarray(chunk), dtype=torch.float64, device=self.torch_device
            )
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError("Features contain non-finite values")
            total += values.sum(dim=0)
            seen += values.shape[0]
        if seen != count:
            raise RuntimeError(f"Feature count changed while reading: {seen} != {count}")
        mu = total / count

        cov_accumulator = torch.zeros(
            (features.shape[1], features.shape[1]),
            dtype=torch.float64,
            device=self.torch_device,
        )
        for chunk in _iter_feature_chunks(features, indices, self.chunk_size):
            values = torch.as_tensor(
                np.asarray(chunk), dtype=torch.float64, device=self.torch_device
            )
            centered = values - mu
            cov_accumulator.addmm_(centered.T, centered)
        cov = cov_accumulator / (count - 1)
        cov = (cov + cov.T) * 0.5
        return FeatureStats(
            mu=mu,
            cov=cov,
            cov_sqrt=self._matrix_sqrt_psd(cov) if cache_sqrt else None,
        )

    def from_numpy_stats(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        cache_sqrt: bool = True,
    ) -> FeatureStats:
        torch = self.torch
        mu_tensor = torch.as_tensor(
            np.asarray(mu), dtype=torch.float64, device=self.torch_device
        )
        cov_tensor = torch.as_tensor(
            np.asarray(cov), dtype=torch.float64, device=self.torch_device
        )
        cov_tensor = (cov_tensor + cov_tensor.T) * 0.5
        return FeatureStats(
            mu=mu_tensor,
            cov=cov_tensor,
            cov_sqrt=self._matrix_sqrt_psd(cov_tensor) if cache_sqrt else None,
        )

    def distance(self, a: FeatureStats, b: FeatureStats) -> float:
        torch = self.torch
        if a.mu.shape != b.mu.shape or a.cov.shape != b.cov.shape:
            raise ValueError(
                f"Feature-statistic dimensions differ: {a.mu.shape}/{a.cov.shape} "
                f"versus {b.mu.shape}/{b.cov.shape}"
            )
        mean_term = torch.square(a.mu - b.mu).sum()
        if b.cov_sqrt is not None:
            middle = b.cov_sqrt @ a.cov @ b.cov_sqrt
        else:
            sqrt_a = a.cov_sqrt if a.cov_sqrt is not None else self._matrix_sqrt_psd(a.cov)
            middle = sqrt_a @ b.cov @ sqrt_a
        eigvals = torch.linalg.eigvalsh((middle + middle.T) * 0.5)
        scale = max(float(eigvals.abs().max().item()), 1.0)
        if float(eigvals.min().item()) < -1e-7 * scale:
            raise ValueError("Covariance product is materially non-positive-semidefinite")
        trace_sqrt = eigvals.clamp_min(0.0).sqrt().sum()
        fd = mean_term + torch.trace(a.cov) + torch.trace(b.cov) - 2.0 * trace_sqrt
        value = float(fd.item())
        if value < -1e-6:
            raise ValueError(f"Computed a materially negative FD ({value})")
        return max(value, 0.0)


def get_backend(name: str = "numpy", device: str = "auto", chunk_size: int = 8192) -> FDBackend:
    normalized = name.lower()
    if normalized == "auto":
        try:
            import torch

            normalized = "torch" if torch.cuda.is_available() else "numpy"
        except ImportError:
            normalized = "numpy"
    if normalized == "numpy":
        return NumpyBackend(chunk_size=chunk_size)
    if normalized == "torch":
        return TorchBackend(device=device, chunk_size=chunk_size)
    raise ValueError(f"Unknown backend {name!r}; choose auto, numpy, or torch")


def compute_stats(features: np.ndarray, cache_sqrt: bool = False) -> FeatureStats:
    return NumpyBackend().stats(features, cache_sqrt=cache_sqrt)


def frechet_distance(a: FeatureStats, b: FeatureStats) -> float:
    return NumpyBackend().distance(a, b)


def load_reference_stats(path: Path, backend: FDBackend | None = None) -> FeatureStats:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        mu_key = "mu" if "mu" in archive else "m" if "m" in archive else None
        cov_key = (
            "sigma"
            if "sigma" in archive
            else "cov"
            if "cov" in archive
            else "s"
            if "s" in archive
            else None
        )
        if mu_key is None or cov_key is None:
            raise ValueError(
                f"{path} must contain mu/sigma (also accepts m/s or mu/cov); "
                f"found {list(archive.files)}"
            )
        mu = np.asarray(archive[mu_key], dtype=np.float64)
        cov = np.asarray(archive[cov_key], dtype=np.float64)
    if mu.ndim != 1 or cov.shape != (mu.size, mu.size):
        raise ValueError(f"Invalid statistics in {path}: mu={mu.shape}, cov={cov.shape}")
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError(f"{path} contains non-finite statistics")
    return (backend or NumpyBackend()).from_numpy_stats(mu, cov, cache_sqrt=True)

