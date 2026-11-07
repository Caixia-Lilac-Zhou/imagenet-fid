#!/usr/bin/env python3
"""Small deterministic test for subsample_fid_stability.py."""

from __future__ import annotations

import csv
import json
from test_support import preserved_directory
from pathlib import Path

import numpy as np

from subsample_fid_stability import run_subsampling


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rng = np.random.default_rng(11)
    labels = np.repeat(np.arange(3), 16)
    centers = np.asarray([[0.0, 0.0, 0.0, 0.0], [0.8, -0.4, 0.2, 0.6], [-0.5, 0.8, 1.0, -0.1]])
    real = centers[labels] + rng.normal(scale=0.2, size=(labels.size, 4))

    with preserved_directory() as temporary_dir:
        root = Path(temporary_dir)
        np.savez(root / "real.npz", features=real, labels=labels)
        rows = []
        for index, shift in enumerate([0.0, 0.4, 0.8]):
            model = f"M{index}"
            np.savez(
                root / f"{model}.npz",
                features=real + np.asarray([shift, 0.0, 0.0, 0.0]),
                labels=labels,
            )
            rows.append(
                {
                    "model": model,
                    "protocol": "validation",
                    "generated_npz": f"{model}.npz",
                    "reference_npz": "real.npz",
                    "reference_stats_npz": "",
                    "expected_fd": "",
                }
            )

        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        first = run_subsampling(
            manifest,
            root / "out1",
            generated_sizes=[4, 8],
            reference_sizes=[4, 8],
            trials=8,
            seed=321,
            quiet=True,
        )
        second = run_subsampling(
            manifest,
            root / "out2",
            generated_sizes=[4, 8],
            reference_sizes=[4, 8],
            trials=8,
            seed=321,
            quiet=True,
        )
        for key in first:
            assert first[key].read_bytes() == second[key].read_bytes(), key

        expected_bytes = {key: path.read_bytes() for key, path in first.items()}
        interrupted_rows = read_rows(first["scores"])[:-1]
        with first["scores"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["axis", "size_per_class", "trial", "model", "protocol", "fd"],
            )
            writer.writeheader()
            writer.writerows(interrupted_rows)
        metadata = json.loads(first["metadata"].read_text(encoding="utf-8"))
        metadata["complete"] = False
        first["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
        resumed = run_subsampling(
            manifest,
            root / "out1",
            generated_sizes=[4, 8],
            reference_sizes=[4, 8],
            trials=8,
            seed=321,
            quiet=True,
        )
        for key, path in resumed.items():
            assert path.read_bytes() == expected_bytes[key], f"resume mismatch: {key}"

        assert len(read_rows(first["scores"])) == 2 * 2 * 8 * 3
        assert len(read_rows(first["summary"])) == 2 * 2 * 3
        assert len(read_rows(first["ranking_trials"])) == 2 * 2 * 8
        assert len(read_rows(first["ranking_summary"])) == 2 * 2
        assert len(read_rows(first["pairwise"])) == 2 * 2 * 3

    print("subsample_fid_stability: all tests passed")


if __name__ == "__main__":
    main()
