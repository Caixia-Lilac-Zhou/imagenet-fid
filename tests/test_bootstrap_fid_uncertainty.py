#!/usr/bin/env python3
"""Small deterministic test for bootstrap_fid_uncertainty.py."""

from __future__ import annotations

import csv
import json
from test_support import preserved_directory
from pathlib import Path

import numpy as np

from bootstrap_fid_uncertainty import (
    compute_stats,
    run_bootstrap,
    stratified_bootstrap_indices,
)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "model",
        "protocol",
        "generated_npz",
        "reference_npz",
        "reference_stats_npz",
        "expected_fd",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), 16)
    centers = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, -0.5, 0.3, 0.8], [-0.6, 0.7, 1.1, -0.2]])
    real = centers[labels] + rng.normal(scale=0.25, size=(labels.size, 4))
    model_a = real.copy()
    model_b = real + np.asarray([0.7, 0.0, 0.0, 0.0])

    sampled = stratified_bootstrap_indices(labels, np.random.default_rng(9))
    assert np.array_equal(np.bincount(labels[sampled]), np.bincount(labels))

    with preserved_directory() as temporary_dir:
        root = Path(temporary_dir)
        np.savez(root / "real.npz", features=real, labels=labels)
        np.savez(root / "model_a.npz", features=model_a, labels=labels)
        np.savez(root / "model_b.npz", features=model_b, labels=labels)
        real_stats = compute_stats(real)
        np.savez(root / "real_stats.npz", mu=real_stats.mu, sigma=real_stats.cov)

        manifest = root / "manifest.csv"
        write_manifest(
            manifest,
            [
                {
                    "model": "A",
                    "protocol": "validation",
                    "generated_npz": "model_a.npz",
                    "reference_npz": "real.npz",
                    "reference_stats_npz": "",
                    "expected_fd": "0",
                },
                {
                    "model": "B",
                    "protocol": "validation",
                    "generated_npz": "model_b.npz",
                    "reference_npz": "real.npz",
                    "reference_stats_npz": "",
                    "expected_fd": "0.49",
                },
                {
                    "model": "A",
                    "protocol": "fixed_train",
                    "generated_npz": "model_a.npz",
                    "reference_npz": "",
                    "reference_stats_npz": "real_stats.npz",
                    "expected_fd": "0",
                },
            ],
        )

        first = run_bootstrap(manifest, root / "out1", replicates=24, seed=123, quiet=True)
        second = run_bootstrap(manifest, root / "out2", replicates=24, seed=123, quiet=True)
        for key in first:
            assert first[key].read_bytes() == second[key].read_bytes(), key

        expected_bytes = {key: path.read_bytes() for key, path in first.items()}
        interrupted_rows = read_rows(first["replicates"])[:-1]
        with first["replicates"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model", "protocol", "replicate", "fd"])
            writer.writeheader()
            writer.writerows(interrupted_rows)
        metadata = json.loads(first["metadata"].read_text(encoding="utf-8"))
        metadata["complete"] = False
        first["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
        resumed = run_bootstrap(manifest, root / "out1", replicates=24, seed=123, quiet=True)
        for key, path in resumed.items():
            assert path.read_bytes() == expected_bytes[key], f"resume mismatch: {key}"

        replicate_rows = read_rows(first["replicates"])
        summary_rows = read_rows(first["summary"])
        pairwise_rows = read_rows(first["pairwise"])
        assert len(replicate_rows) == 3 * 24
        assert len(summary_rows) == 3
        assert len(pairwise_rows) == 1
        assert len(read_rows(first["protocol_differences"])) == 1
        assert float(pairwise_rows[0]["probability_a_lower"]) > 0.9
        fixed = next(row for row in summary_rows if row["protocol"] == "fixed_train")
        validation = next(
            row for row in summary_rows if row["model"] == "A" and row["protocol"] == "validation"
        )
        assert fixed["real_side_bootstrapped"] == "false"
        assert validation["real_side_bootstrapped"] == "true"
        assert abs(float(fixed["point_fd"])) < 1e-10

    print("bootstrap_fid_uncertainty: all tests passed")


if __name__ == "__main__":
    main()
