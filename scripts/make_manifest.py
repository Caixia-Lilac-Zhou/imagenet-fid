#!/usr/bin/env python3
"""Create a portable model x protocol manifest from existing feature bundles."""
import argparse
import csv
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from fid_core import load_feature_bundle, load_reference_stats


def pairs(items):
    result = {}
    for item in items:
        name, separator, value = item.partition('=')
        if not separator or not name.strip() or not value.strip() or name in result:
            raise ValueError(f'Expected a unique NAME=PATH argument: {item}')
        result[name] = Path(value).expanduser().resolve()
    return result


def create_manifest(generated, references, fixed, output):
    if not generated or not (references or fixed):
        raise ValueError('Provide generated models and at least one real reference')
    if set(references) & set(fixed):
        raise ValueError('Feature and fixed-statistics protocol names must be distinct')
    if output.exists():
        raise FileExistsError(f'Refusing to replace {output}')
    dimensions = set()
    for path in list(generated.values()) + list(references.values()):
        dimensions.add(load_feature_bundle(path).features.shape[1])
    for path in fixed.values():
        dimensions.add(load_reference_stats(path).mu.shape[0])
    if len(dimensions) != 1:
        raise ValueError(f'Feature dimensions disagree: {dimensions}')
    relative = lambda p: os.path.relpath(p.resolve(), output.resolve().parent)
    rows = []
    for model, source in generated.items():
        for protocol, target in {**references, **fixed}.items():
            rows.append({'model': model, 'protocol': protocol, 'generated_npz': relative(source),
                         'reference_npz': relative(target) if protocol in references else '',
                         'reference_stats_npz': relative(target) if protocol in fixed else '',
                         'expected_fd': ''})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated', action='append', default=[], metavar='MODEL=PATH')
    parser.add_argument('--reference', action='append', default=[], metavar='PROTOCOL=PATH')
    parser.add_argument('--fixed-reference', action='append', default=[], metavar='PROTOCOL=PATH')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(create_manifest(pairs(args.generated), pairs(args.reference), pairs(args.fixed_reference), args.output))


if __name__ == '__main__':
    main()
