#!/usr/bin/env python3
"""Paired class-stratified bootstrap uncertainty analysis for feature-space FD."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from output_safety import require_resumable_output

from fid_core import (
    FeatureStats,
    append_csv_rows,
    atomic_write_json,
    compute_stats,
    frechet_distance,
    get_backend,
    load_feature_bundle,
    load_reference_stats,
    read_csv_rows,
    read_manifest,
    sha256_file,
    stable_seed,
    stratified_bootstrap_indices,
    write_csv,
)


RAW_FIELDS = ["model", "protocol", "replicate", "fd"]


def _restore_complete_replicates(
    path: Path,
    expected_pairs: set[tuple[str, str]],
    target_replicates: int,
) -> tuple[list[dict[str, str]], set[int]]:
    records = read_csv_rows(path)
    by_replicate: dict[int, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        try:
            replicate = int(record["replicate"])
            float(record["fd"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= replicate < target_replicates:
            by_replicate[replicate].append(record)

    complete: set[int] = set()
    kept: list[dict[str, str]] = []
    for replicate, replicate_rows in sorted(by_replicate.items()):
        pairs = [(row.get("model", ""), row.get("protocol", "")) for row in replicate_rows]
        if len(pairs) == len(expected_pairs) and set(pairs) == expected_pairs:
            complete.add(replicate)
            kept.extend(sorted(replicate_rows, key=lambda row: (row["model"], row["protocol"])))
    if records != kept:
        write_csv(path, RAW_FIELDS, kept)
    return kept, complete


def _validate_or_create_metadata(
    metadata_path: Path,
    manifest_path: Path,
    seed: int,
    backend_name: str,
    device: str,
    chunk_size: int,
    replicates: int,
    resume: bool,
) -> None:
    identity = {
        "schema_version": 2,
        "manifest_sha256": sha256_file(manifest_path),
        "seed": seed,
        "backend": backend_name,
        "device": device,
        "chunk_size": chunk_size,
    }
    if metadata_path.exists() and resume:
        import json

        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing.get(key), value)
            for key, value in identity.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Cannot resume bootstrap with changed inputs/settings: "
                + ", ".join(
                    f"{key}={old!r}->{new!r}" for key, (old, new) in mismatches.items()
                )
            )
    atomic_write_json(
        metadata_path,
        {**identity, "target_replicates": replicates, "complete": False},
    )


def run_bootstrap(
    manifest_path: Path,
    output_dir: Path,
    replicates: int = 200,
    seed: int = 2026,
    expected_tolerance: float = 0.05,
    strict_expected: bool = False,
    quiet: bool = False,
    backend_name: str = "numpy",
    device: str = "auto",
    chunk_size: int = 8192,
    resume: bool = True,
) -> dict[str, Path]:
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    manifest_path = manifest_path.resolve()
    rows = read_manifest(manifest_path)
    require_resumable_output(output_dir, 'bootstrap_metadata.json', resume=resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend(backend_name, device=device, chunk_size=chunk_size)

    raw_path = output_dir / "bootstrap_replicates.csv"
    metadata_path = output_dir / "bootstrap_metadata.json"
    _validate_or_create_metadata(
        metadata_path,
        manifest_path,
        seed,
        backend.name,
        backend.device,
        chunk_size,
        replicates,
        resume,
    )

    feature_paths = {row.generated_path for row in rows}
    feature_paths.update(row.reference_path for row in rows if row.reference_path is not None)
    if not quiet:
        print(
            f"Loading {len(feature_paths)} feature bundles; backend={backend.name}/{backend.device}",
            file=sys.stderr,
        )
    bundles = {path: load_feature_bundle(path) for path in sorted(feature_paths)}
    stats_paths = {row.reference_stats_path for row in rows if row.reference_stats_path is not None}
    fixed_stats = {path: load_reference_stats(path, backend) for path in sorted(stats_paths)}

    generated_paths = sorted({row.generated_path for row in rows})
    reference_paths = sorted(
        {row.reference_path for row in rows if row.reference_path is not None}
    )
    generated_point_stats = {
        path: backend.stats(bundles[path].features) for path in generated_paths
    }
    reference_point_stats = {
        path: backend.stats(bundles[path].features, cache_sqrt=True)
        for path in reference_paths
    }
    generated_seed_ids = {
        path: "|".join(sorted({row.model for row in rows if row.generated_path == path}))
        for path in generated_paths
    }
    reference_seed_ids = {
        path: "|".join(sorted({row.protocol for row in rows if row.reference_path == path}))
        for path in reference_paths
    }

    point_fds: dict[tuple[str, str], float] = {}
    for row in rows:
        reference_stats = (
            reference_point_stats[row.reference_path]
            if row.reference_path is not None
            else fixed_stats[row.reference_stats_path]
        )
        point_fd = backend.distance(generated_point_stats[row.generated_path], reference_stats)
        point_fds[(row.model, row.protocol)] = point_fd
        if row.expected_fd is not None and abs(point_fd - row.expected_fd) > expected_tolerance:
            message = (
                f"{row.model}/{row.protocol}: recomputed FD {point_fd:.6f} differs from "
                f"expected {row.expected_fd:.6f} by more than {expected_tolerance:g}"
            )
            if strict_expected:
                raise ValueError(message)
            print(f"WARNING: {message}", file=sys.stderr)

    expected_pairs = {(row.model, row.protocol) for row in rows}
    existing_records, completed = _restore_complete_replicates(
        raw_path, expected_pairs, replicates
    )
    if not quiet and completed:
        print(f"Resuming with {len(completed)}/{replicates} replicates complete", file=sys.stderr)

    progress_every = max(1, replicates // 10)
    for replicate in range(replicates):
        if replicate in completed:
            continue
        boot_reference_stats: dict[Path, FeatureStats] = {}
        for reference_path in reference_paths:
            bundle = bundles[reference_path]
            rng = np.random.default_rng(
                stable_seed(seed, "reference", reference_seed_ids[reference_path], replicate)
            )
            indices = stratified_bootstrap_indices(bundle.labels, rng)
            boot_reference_stats[reference_path] = backend.stats(
                bundle.features,
                indices=indices,
                cache_sqrt=True,
            )

        # Pair protocols by using the same generated resample for a given archive.
        boot_generated_stats: dict[Path, FeatureStats] = {}
        for generated_path in generated_paths:
            bundle = bundles[generated_path]
            rng = np.random.default_rng(
                stable_seed(seed, "generated", generated_seed_ids[generated_path], replicate)
            )
            indices = stratified_bootstrap_indices(bundle.labels, rng)
            boot_generated_stats[generated_path] = backend.stats(
                bundle.features,
                indices=indices,
            )

        replicate_rows: list[dict[str, object]] = []
        for row in rows:
            reference_stats = (
                boot_reference_stats[row.reference_path]
                if row.reference_path is not None
                else fixed_stats[row.reference_stats_path]
            )
            score = backend.distance(boot_generated_stats[row.generated_path], reference_stats)
            replicate_rows.append(
                {
                    "model": row.model,
                    "protocol": row.protocol,
                    "replicate": replicate,
                    "fd": f"{score:.9f}",
                }
            )
        append_csv_rows(raw_path, RAW_FIELDS, replicate_rows)
        existing_records.extend({key: str(value) for key, value in record.items()} for record in replicate_rows)
        completed.add(replicate)
        if not quiet and (
            len(completed) % progress_every == 0 or len(completed) == replicates
        ):
            print(f"Completed {len(completed)}/{replicates} replicates", file=sys.stderr)

    records, completed = _restore_complete_replicates(raw_path, expected_pairs, replicates)
    if len(completed) != replicates:
        raise RuntimeError(f"Only {len(completed)}/{replicates} complete replicates were recovered")
    replicate_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in sorted(records, key=lambda row: (int(row["replicate"]), row["model"], row["protocol"])):
        replicate_scores[(record["model"], record["protocol"])].append(float(record["fd"]))

    model_ranges: dict[str, tuple[float, float, float]] = {}
    for model in sorted({row.model for row in rows}):
        values = [point_fds[(row.model, row.protocol)] for row in rows if row.model == model]
        model_ranges[model] = (min(values), max(values), max(values) - min(values))

    summary_records = []
    for row in rows:
        values = np.asarray(replicate_scores[(row.model, row.protocol)], dtype=np.float64)
        low, high = np.percentile(values, [2.5, 97.5])
        ci_width = float(high - low)
        range_min, range_max, protocol_range = model_ranges[row.model]
        summary_records.append(
            {
                "model": row.model,
                "protocol": row.protocol,
                "point_fd": f"{point_fds[(row.model, row.protocol)]:.9f}",
                "bootstrap_mean": f"{values.mean():.9f}",
                "bootstrap_std": f"{values.std(ddof=1):.9f}",
                "ci_2.5": f"{low:.9f}",
                "ci_97.5": f"{high:.9f}",
                "ci_width": f"{ci_width:.9f}",
                "protocol_min_point_fd": f"{range_min:.9f}",
                "protocol_max_point_fd": f"{range_max:.9f}",
                "protocol_range": f"{protocol_range:.9f}",
                "protocol_range_over_ci_width": (
                    f"{protocol_range / ci_width:.9f}" if ci_width > 0 else "inf"
                ),
                "real_side_bootstrapped": str(row.reference_path is not None).lower(),
                "replicates": replicates,
                "seed": seed,
                "backend": backend.name,
                "device": backend.device,
            }
        )

    pairwise_records = []
    by_protocol: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_protocol[row.protocol].append(row.model)
    for protocol, models in sorted(by_protocol.items()):
        for model_a, model_b in combinations(sorted(set(models)), 2):
            values_a = np.asarray(replicate_scores[(model_a, protocol)])
            values_b = np.asarray(replicate_scores[(model_b, protocol)])
            delta = values_a - values_b
            low, high = np.percentile(delta, [2.5, 97.5])
            pairwise_records.append(
                {
                    "protocol": protocol,
                    "model_a": model_a,
                    "model_b": model_b,
                    "point_delta_a_minus_b": f"{point_fds[(model_a, protocol)] - point_fds[(model_b, protocol)]:.9f}",
                    "bootstrap_delta_mean": f"{delta.mean():.9f}",
                    "delta_ci_2.5": f"{low:.9f}",
                    "delta_ci_97.5": f"{high:.9f}",
                    "probability_a_lower": f"{np.mean(values_a < values_b):.9f}",
                    "probability_tie": f"{np.mean(values_a == values_b):.9f}",
                    "replicates": replicates,
                }
            )

    protocol_difference_records = []
    by_model: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_model[row.model].append(row.protocol)
    for model, protocols in sorted(by_model.items()):
        for protocol_a, protocol_b in combinations(sorted(set(protocols)), 2):
            values_a = np.asarray(replicate_scores[(model, protocol_a)])
            values_b = np.asarray(replicate_scores[(model, protocol_b)])
            delta = values_a - values_b
            low, high = np.percentile(delta, [2.5, 97.5])
            protocol_difference_records.append(
                {
                    "model": model,
                    "protocol_a": protocol_a,
                    "protocol_b": protocol_b,
                    "point_delta_a_minus_b": f"{point_fds[(model, protocol_a)] - point_fds[(model, protocol_b)]:.9f}",
                    "bootstrap_delta_mean": f"{delta.mean():.9f}",
                    "delta_ci_2.5": f"{low:.9f}",
                    "delta_ci_97.5": f"{high:.9f}",
                    "probability_a_lower": f"{np.mean(values_a < values_b):.9f}",
                    "replicates": replicates,
                }
            )

    convergence_records = []
    checkpoints = sorted({value for value in [25, 50, 100, 200, replicates] if value <= replicates})
    for row in rows:
        values = np.asarray(replicate_scores[(row.model, row.protocol)])
        final_low, final_high = np.percentile(values, [2.5, 97.5])
        for checkpoint in checkpoints:
            low, high = np.percentile(values[:checkpoint], [2.5, 97.5])
            convergence_records.append(
                {
                    "model": row.model,
                    "protocol": row.protocol,
                    "replicates": checkpoint,
                    "ci_2.5": f"{low:.9f}",
                    "ci_97.5": f"{high:.9f}",
                    "lower_endpoint_abs_change_from_final": f"{abs(low - final_low):.9f}",
                    "upper_endpoint_abs_change_from_final": f"{abs(high - final_high):.9f}",
                }
            )

    paths = {
        "replicates": raw_path,
        "summary": output_dir / "bootstrap_summary.csv",
        "pairwise": output_dir / "bootstrap_pairwise.csv",
        "protocol_differences": output_dir / "bootstrap_protocol_differences.csv",
        "convergence": output_dir / "bootstrap_convergence.csv",
        "metadata": metadata_path,
    }
    write_csv(paths["summary"], list(summary_records[0]), summary_records)
    write_csv(
        paths["pairwise"],
        [
            "protocol", "model_a", "model_b", "point_delta_a_minus_b",
            "bootstrap_delta_mean", "delta_ci_2.5", "delta_ci_97.5",
            "probability_a_lower", "probability_tie", "replicates",
        ],
        pairwise_records,
    )
    write_csv(
        paths["protocol_differences"],
        [
            "model", "protocol_a", "protocol_b", "point_delta_a_minus_b",
            "bootstrap_delta_mean", "delta_ci_2.5", "delta_ci_97.5",
            "probability_a_lower", "replicates",
        ],
        protocol_difference_records,
    )
    write_csv(paths["convergence"], list(convergence_records[0]), convergence_records)
    atomic_write_json(
        metadata_path,
        {
            "schema_version": 2,
            "manifest_sha256": sha256_file(manifest_path),
            "seed": seed,
            "backend": backend.name,
            "device": backend.device,
            "chunk_size": chunk_size,
            "target_replicates": replicates,
            "completed_replicates": len(completed),
            "generated_resampling_ids": {
                generated_seed_ids[path]: str(path) for path in generated_paths
            },
            "reference_resampling_ids": {
                reference_seed_ids[path]: str(path) for path in reference_paths
            },
            "complete": True,
        },
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-tolerance", type=float, default=0.05)
    parser.add_argument("--strict-expected", action="store_true")
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_bootstrap(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        replicates=args.replicates,
        seed=args.seed,
        expected_tolerance=args.expected_tolerance,
        strict_expected=args.strict_expected,
        backend_name=args.backend,
        device=args.device,
        chunk_size=args.chunk_size,
        resume=not args.no_resume,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
