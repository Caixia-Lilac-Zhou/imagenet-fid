#!/usr/bin/env python3
"""Repeated class-balanced subsampling for FD score and ranking stability."""

from __future__ import annotations

import argparse
import json
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
    get_backend,
    load_feature_bundle,
    load_reference_stats,
    read_csv_rows,
    read_manifest,
    sha256_file,
    stable_seed,
    stratified_subsample_indices,
    write_csv,
)


RAW_FIELDS = ["axis", "size_per_class", "trial", "model", "protocol", "fd"]


def parse_sizes(raw: str) -> list[int]:
    if raw.strip().lower() in {"", "none"}:
        return []
    sizes = sorted({int(value.strip()) for value in raw.split(",")})
    if not sizes or sizes[0] < 2:
        raise ValueError("Subsample sizes must be integers of at least two per class")
    return sizes


def sample_per_class_indices(
    labels: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    return stratified_subsample_indices(labels, per_class, rng)


def rank_values(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda model: (values[model], model))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for model in ordered[start:end]:
            ranks[model] = average_rank
        start = end
    return ranks


def ranking_metrics(
    point_scores: dict[str, float],
    trial_scores: dict[str, float],
) -> dict[str, float]:
    models = sorted(set(point_scores) & set(trial_scores))
    if not models:
        raise ValueError("No common models for ranking metrics")
    point_ranks = rank_values({model: point_scores[model] for model in models})
    trial_ranks = rank_values({model: trial_scores[model] for model in models})
    point_vector = np.asarray([point_ranks[model] for model in models])
    trial_vector = np.asarray([trial_ranks[model] for model in models])
    spearman = (
        float(np.corrcoef(point_vector, trial_vector)[0, 1])
        if len(models) > 1 and point_vector.std() > 0 and trial_vector.std() > 0
        else float("nan")
    )
    top_k = min(3, len(models))
    point_order = sorted(models, key=lambda model: (point_scores[model], model))
    trial_order = sorted(models, key=lambda model: (trial_scores[model], model))
    changes = np.abs(point_vector - trial_vector)
    return {
        "n_models": float(len(models)),
        "spearman": spearman,
        "top1_retained": float(point_order[0] == trial_order[0]),
        "top3_overlap": len(set(point_order[:top_k]) & set(trial_order[:top_k])) / top_k,
        "mean_abs_rank_change": float(changes.mean()),
        "max_abs_rank_change": float(changes.max()),
    }


def _minimum_class_count(labels: np.ndarray) -> int:
    _, counts = np.unique(labels, return_counts=True)
    return int(counts.min())


def _unit_expected_pairs(rows, bundles, axis: str, size: int) -> set[tuple[str, str]]:
    expected = set()
    for row in rows:
        if axis == "generated":
            labels = bundles[row.generated_path].labels
        else:
            if row.reference_path is None:
                continue
            labels = bundles[row.reference_path].labels
        if _minimum_class_count(labels) >= size:
            expected.add((row.model, row.protocol))
    return expected


def _restore_complete_units(
    path: Path,
    expected_by_unit: dict[tuple[str, int, int], set[tuple[str, str]]],
) -> tuple[list[dict[str, str]], set[tuple[str, int, int]]]:
    records = read_csv_rows(path)
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        try:
            unit = (record["axis"], int(record["size_per_class"]), int(record["trial"]))
            float(record["fd"])
        except (KeyError, TypeError, ValueError):
            continue
        if unit in expected_by_unit:
            grouped[unit].append(record)

    complete = set()
    kept = []
    for unit, expected in sorted(expected_by_unit.items()):
        unit_rows = grouped.get(unit, [])
        pairs = [(row.get("model", ""), row.get("protocol", "")) for row in unit_rows]
        if expected and len(pairs) == len(expected) and set(pairs) == expected:
            complete.add(unit)
            kept.extend(sorted(unit_rows, key=lambda row: (row["model"], row["protocol"])))
    if records != kept:
        write_csv(path, RAW_FIELDS, kept)
    return kept, complete


def _prepare_metadata(
    path: Path,
    manifest_path: Path,
    seed: int,
    backend_name: str,
    device: str,
    chunk_size: int,
    trials: int,
    generated_sizes: list[int],
    reference_sizes: list[int],
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
    if path.exists() and resume:
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing.get(key), value)
            for key, value in identity.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Cannot resume subsampling with changed inputs/settings: "
                + ", ".join(
                    f"{key}={old!r}->{new!r}" for key, (old, new) in mismatches.items()
                )
            )
    atomic_write_json(
        path,
        {
            **identity,
            "trials": trials,
            "generated_sizes": generated_sizes,
            "reference_sizes": reference_sizes,
            "complete": False,
        },
    )


def run_subsampling(
    manifest_path: Path,
    output_dir: Path,
    generated_sizes: list[int],
    reference_sizes: list[int],
    trials: int = 20,
    seed: int = 2026,
    quiet: bool = False,
    backend_name: str = "numpy",
    device: str = "auto",
    chunk_size: int = 8192,
    resume: bool = True,
) -> dict[str, Path]:
    if trials < 2:
        raise ValueError("At least two trials are required")
    if not generated_sizes and not reference_sizes:
        raise ValueError("Enable at least one generated or reference size")
    if any(size < 2 for size in generated_sizes + reference_sizes):
        raise ValueError("Every per-class size must be at least two")

    manifest_path = manifest_path.resolve()
    rows = read_manifest(manifest_path)
    require_resumable_output(output_dir, 'subsample_metadata.json', resume=resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend(backend_name, device=device, chunk_size=chunk_size)
    raw_path = output_dir / "subsample_scores.csv"
    metadata_path = output_dir / "subsample_metadata.json"
    _prepare_metadata(
        metadata_path,
        manifest_path,
        seed,
        backend.name,
        backend.device,
        chunk_size,
        trials,
        generated_sizes,
        reference_sizes,
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
    generated_full_stats = {
        path: backend.stats(bundles[path].features) for path in generated_paths
    }
    reference_full_stats = {
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
    point_scores: dict[tuple[str, str], float] = {}
    for row in rows:
        reference_stats = (
            reference_full_stats[row.reference_path]
            if row.reference_path is not None
            else fixed_stats[row.reference_stats_path]
        )
        point_scores[(row.model, row.protocol)] = backend.distance(
            generated_full_stats[row.generated_path], reference_stats
        )

    expected_by_unit: dict[tuple[str, int, int], set[tuple[str, str]]] = {}
    for axis, sizes in [("generated", generated_sizes), ("reference", reference_sizes)]:
        for size in sizes:
            expected = _unit_expected_pairs(rows, bundles, axis, size)
            if not expected and not quiet:
                print(f"Skipping unsupported {axis} size {size}/class", file=sys.stderr)
            for trial in range(trials):
                if expected:
                    expected_by_unit[(axis, size, trial)] = expected

    records, completed = _restore_complete_units(raw_path, expected_by_unit)
    if not expected_by_unit:
        raise ValueError("No requested subset size is supported by the supplied feature bundles")
    if not quiet and completed:
        print(f"Resuming with {len(completed)}/{len(expected_by_unit)} units complete", file=sys.stderr)

    progress_every = max(1, len(expected_by_unit) // 10)
    for unit in sorted(expected_by_unit):
        if unit in completed:
            continue
        axis, per_class, trial = unit
        expected_pairs = expected_by_unit[unit]
        unit_rows: list[dict[str, object]] = []
        if axis == "generated":
            eligible_rows = [
                row for row in rows if (row.model, row.protocol) in expected_pairs
            ]
            stats_by_path: dict[Path, FeatureStats] = {}
            for generated_path in sorted({row.generated_path for row in eligible_rows}):
                bundle = bundles[generated_path]
                rng = np.random.default_rng(
                    stable_seed(
                        seed,
                        "generated_size",
                        generated_seed_ids[generated_path],
                        per_class,
                        trial,
                    )
                )
                indices = stratified_subsample_indices(bundle.labels, per_class, rng)
                if indices is None:
                    raise RuntimeError("Eligibility check and generated subsampling disagree")
                stats_by_path[generated_path] = backend.stats(
                    bundle.features,
                    indices=indices,
                )
            for row in eligible_rows:
                reference_stats = (
                    reference_full_stats[row.reference_path]
                    if row.reference_path is not None
                    else fixed_stats[row.reference_stats_path]
                )
                score = backend.distance(stats_by_path[row.generated_path], reference_stats)
                unit_rows.append(
                    {
                        "axis": axis,
                        "size_per_class": per_class,
                        "trial": trial,
                        "model": row.model,
                        "protocol": row.protocol,
                        "fd": f"{score:.9f}",
                    }
                )
        else:
            eligible_rows = [
                row for row in rows if (row.model, row.protocol) in expected_pairs
            ]
            stats_by_path = {}
            for reference_path in sorted({row.reference_path for row in eligible_rows}):
                if reference_path is None:
                    continue
                bundle = bundles[reference_path]
                rng = np.random.default_rng(
                    stable_seed(
                        seed,
                        "reference_size",
                        reference_seed_ids[reference_path],
                        per_class,
                        trial,
                    )
                )
                indices = stratified_subsample_indices(bundle.labels, per_class, rng)
                if indices is None:
                    raise RuntimeError("Eligibility check and reference subsampling disagree")
                stats_by_path[reference_path] = backend.stats(
                    bundle.features,
                    indices=indices,
                    cache_sqrt=True,
                )
            for row in eligible_rows:
                if row.reference_path is None:
                    continue
                score = backend.distance(
                    generated_full_stats[row.generated_path],
                    stats_by_path[row.reference_path],
                )
                unit_rows.append(
                    {
                        "axis": axis,
                        "size_per_class": per_class,
                        "trial": trial,
                        "model": row.model,
                        "protocol": row.protocol,
                        "fd": f"{score:.9f}",
                    }
                )
        append_csv_rows(raw_path, RAW_FIELDS, unit_rows)
        records.extend({key: str(value) for key, value in record.items()} for record in unit_rows)
        completed.add(unit)
        if not quiet and (
            len(completed) % progress_every == 0 or len(completed) == len(expected_by_unit)
        ):
            print(
                f"Completed {len(completed)}/{len(expected_by_unit)} size/seed units",
                file=sys.stderr,
            )

    records, completed = _restore_complete_units(raw_path, expected_by_unit)
    if len(completed) != len(expected_by_unit):
        raise RuntimeError(
            f"Only {len(completed)}/{len(expected_by_unit)} complete units were recovered"
        )

    score_groups: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    trial_groups: dict[tuple[str, int, int, str], dict[str, float]] = defaultdict(dict)
    for record in records:
        axis = record["axis"]
        size = int(record["size_per_class"])
        model = record["model"]
        protocol = record["protocol"]
        trial = int(record["trial"])
        score = float(record["fd"])
        score_groups[(axis, size, model, protocol)].append(score)
        trial_groups[(axis, size, trial, protocol)][model] = score

    summary_records = []
    for (axis, per_class, model, protocol), raw_values in sorted(score_groups.items()):
        values = np.asarray(raw_values)
        low, high = np.percentile(values, [2.5, 97.5])
        summary_records.append(
            {
                "axis": axis,
                "size_per_class": per_class,
                "total_samples": per_class * len(np.unique(
                    bundles[next(
                        row.generated_path if axis == "generated" else row.reference_path
                        for row in rows
                        if row.model == model and row.protocol == protocol
                    )].labels
                )),
                "model": model,
                "protocol": protocol,
                "full_sample_point_fd": f"{point_scores[(model, protocol)]:.9f}",
                "mean_fd": f"{values.mean():.9f}",
                "std_fd": f"{values.std(ddof=1):.9f}",
                "ci_2.5": f"{low:.9f}",
                "ci_97.5": f"{high:.9f}",
                "trials": len(values),
                "seed": seed,
            }
        )

    ranking_trial_records = []
    per_model_rank_values: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for (axis, per_class, trial, protocol), trial_scores in sorted(trial_groups.items()):
        baseline = {model: point_scores[(model, protocol)] for model in trial_scores}
        metrics = ranking_metrics(baseline, trial_scores)
        trial_ranks = rank_values(trial_scores)
        for model, rank in trial_ranks.items():
            per_model_rank_values[(axis, per_class, protocol, model)].append(rank)
        ranking_trial_records.append(
            {
                "axis": axis,
                "size_per_class": per_class,
                "trial": trial,
                "protocol": protocol,
                **{
                    key: (f"{value:.9f}" if key != "n_models" else int(value))
                    for key, value in metrics.items()
                },
            }
        )

    ranking_groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for record in ranking_trial_records:
        ranking_groups[
            (str(record["axis"]), int(record["size_per_class"]), str(record["protocol"]))
        ].append(record)
    ranking_summary_records = []
    metric_names = [
        "spearman",
        "top1_retained",
        "top3_overlap",
        "mean_abs_rank_change",
        "max_abs_rank_change",
    ]
    for (axis, per_class, protocol), group in sorted(ranking_groups.items()):
        output: dict[str, object] = {
            "axis": axis,
            "size_per_class": per_class,
            "protocol": protocol,
            "n_models": group[0]["n_models"],
            "trials": len(group),
        }
        for metric in metric_names:
            values = np.asarray([float(record[metric]) for record in group])
            output[f"mean_{metric}"] = f"{np.nanmean(values):.9f}"
            output[f"std_{metric}"] = f"{np.nanstd(values, ddof=1):.9f}"
        ranking_summary_records.append(output)

    model_rank_records = []
    for (axis, per_class, protocol, model), raw_values in sorted(per_model_rank_values.items()):
        values = np.asarray(raw_values)
        n_models = len(
            next(
                scores
                for (trial_axis, trial_size, _trial, trial_protocol), scores in trial_groups.items()
                if trial_axis == axis and trial_size == per_class and trial_protocol == protocol
            )
        )
        model_rank_records.append(
            {
                "axis": axis,
                "size_per_class": per_class,
                "protocol": protocol,
                "model": model,
                "mean_rank": f"{values.mean():.9f}",
                "std_rank": f"{values.std(ddof=1):.9f}",
                "min_rank": f"{values.min():.9f}",
                "max_rank": f"{values.max():.9f}",
                "rank_span": f"{values.max() - values.min():.9f}",
                "top1_frequency": f"{np.mean(values <= 1.0):.9f}",
                "top3_frequency": f"{np.mean(values <= min(3, n_models)):.9f}",
                "trials": len(values),
            }
        )

    pairwise_records = []
    by_axis_size_protocol: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (axis, per_class, model, protocol), values in score_groups.items():
        by_axis_size_protocol[(axis, per_class, protocol)][model] = values
    for (axis, per_class, protocol), model_scores in sorted(by_axis_size_protocol.items()):
        for model_a, model_b in combinations(sorted(model_scores), 2):
            values_a = np.asarray(model_scores[model_a])
            values_b = np.asarray(model_scores[model_b])
            pairwise_records.append(
                {
                    "axis": axis,
                    "size_per_class": per_class,
                    "protocol": protocol,
                    "model_a": model_a,
                    "model_b": model_b,
                    "probability_a_lower": f"{np.mean(values_a < values_b):.9f}",
                    "full_sample_a_lower": str(
                        point_scores[(model_a, protocol)] < point_scores[(model_b, protocol)]
                    ).lower(),
                    "trials": len(values_a),
                }
            )

    paths = {
        "scores": raw_path,
        "summary": output_dir / "subsample_summary.csv",
        "ranking_trials": output_dir / "subsample_ranking_trials.csv",
        "ranking_summary": output_dir / "subsample_ranking_summary.csv",
        "model_ranks": output_dir / "subsample_model_ranks.csv",
        "pairwise": output_dir / "subsample_pairwise.csv",
        "metadata": metadata_path,
    }
    write_csv(paths["summary"], list(summary_records[0]), summary_records)
    write_csv(paths["ranking_trials"], list(ranking_trial_records[0]), ranking_trial_records)
    write_csv(paths["ranking_summary"], list(ranking_summary_records[0]), ranking_summary_records)
    write_csv(paths["model_ranks"], list(model_rank_records[0]), model_rank_records)
    write_csv(
        paths["pairwise"],
        [
            "axis", "size_per_class", "protocol", "model_a", "model_b",
            "probability_a_lower", "full_sample_a_lower", "trials",
        ],
        pairwise_records,
    )
    atomic_write_json(
        metadata_path,
        {
            "schema_version": 2,
            "manifest_sha256": sha256_file(manifest_path),
            "seed": seed,
            "backend": backend.name,
            "device": backend.device,
            "chunk_size": chunk_size,
            "trials": trials,
            "generated_sizes": generated_sizes,
            "reference_sizes": reference_sizes,
            "completed_units": len(completed),
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
    parser.add_argument("--generated-per-class", default="5,10,25")
    parser.add_argument("--reference-per-class", default="50,100,250,500,732")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--backend", choices=["auto", "numpy", "torch"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_subsampling(
        args.manifest,
        args.output_dir,
        generated_sizes=parse_sizes(args.generated_per_class),
        reference_sizes=parse_sizes(args.reference_per_class),
        trials=args.trials,
        seed=args.seed,
        backend_name=args.backend,
        device=args.device,
        chunk_size=args.chunk_size,
        resume=not args.no_resume,
    )
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
