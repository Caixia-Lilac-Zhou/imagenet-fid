#!/usr/bin/env python3
"""Evaluate full-pool FD for a manifest, without resampling or image generation."""
import argparse
import csv
from pathlib import Path

from fid_core import get_backend, load_feature_bundle, load_reference_stats, read_manifest


def evaluate(manifest, output, backend_name='numpy', device='auto', chunk_size=8192):
    rows = read_manifest(manifest)
    if output.exists():
        raise FileExistsError(f'Refusing to replace {output}')
    backend = get_backend(backend_name, device=device, chunk_size=chunk_size)
    stats_cache = {}
    counts = {}

    def stats(path):
        if path not in stats_cache:
            bundle = load_feature_bundle(path)
            stats_cache[path] = backend.stats(bundle.features, cache_sqrt=True)
            counts[path] = bundle.features.shape[0]
        return stats_cache[path]

    values = []
    for row in rows:
        generated = stats(row.generated_path)
        reference = (stats(row.reference_path) if row.reference_path is not None else
                     load_reference_stats(row.reference_stats_path, backend))
        values.append({'model': row.model, 'protocol': row.protocol,
                       'point_fd': f'{backend.distance(generated, reference):.9f}',
                       'generated_samples': counts[row.generated_path],
                       'reference_samples': counts.get(row.reference_path, 'fixed_statistics')})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['numpy', 'torch', 'auto'], default='numpy')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--chunk-size', type=int, default=8192)
    args = parser.parse_args()
    for row in evaluate(args.manifest, args.output, args.backend, args.device, args.chunk_size):
        print(row)


if __name__ == '__main__':
    main()
