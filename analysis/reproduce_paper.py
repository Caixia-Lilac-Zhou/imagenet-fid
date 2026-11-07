#!/usr/bin/env python3
"""Rebuild camera-ready figures and numerical tables from the released result CSVs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['BAR-L', 'REPA-E', 'RAE-DDT', 'LightningDiT', 'MAR-H', 'JiT-H-16',
          'AlphaFlow-XL-2+-2NFE', 'AFM-XL2-112L-1NFE']
NAMES = {'JiT-H-16': 'JiT-H/16', 'AlphaFlow-XL-2+-2NFE': 'AlphaFlow-XL/2+',
         'AFM-XL2-112L-1NFE': 'AFM-XL/2-112L'}


def read_rows(relative):
    with (ROOT / 'data' / relative).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def verify_data():
    manifest = json.loads((ROOT / 'data/SHA256SUMS.json').read_text())
    for relative, expected in manifest.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Released input checksum mismatch: {relative}')
    return len(manifest)


def winner_counts(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row['protocol'], row['replicate']][row['model']] = float(row['fd'])
    winners = defaultdict(Counter)
    for (protocol, _), values in groups.items():
        best = min(values.values())
        for model, value in values.items():
            if value == best:
                winners[protocol][model] += 1
    return {p: dict(counts) for p, counts in winners.items()}


def tex_escape(value):
    return str(value).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def write_table(out, name, caption, headers, values):
    with (out / f'{name}.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(values)
    markdown = f'### {caption}\n\n'
    markdown += '| ' + ' | '.join(headers) + ' |\n'
    markdown += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
    markdown += ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in values)
    with (out / f'{name}.md').open('x') as f:
        f.write(markdown)
    tex = '\\begin{table}[t]\n\\centering\\small\n\\caption{' + tex_escape(caption) + '}\n'
    tex += '\\begin{tabular}{l' + 'r' * (len(headers) - 1) + '}\n\\toprule\n'
    tex += ' & '.join(map(tex_escape, headers)) + ' \\\\\n\\midrule\n'
    tex += ''.join(' & '.join(map(tex_escape, row)) + ' \\\\\n' for row in values)
    tex += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    with (out / f'{name}.tex').open('x') as f:
        f.write(tex)
    return markdown


def reproduce(output, figures=True):
    input_count = verify_data()
    output.mkdir(parents=True, exist_ok=False)
    tables_dir = output / 'tables'
    tables_dir.mkdir()
    sections = []

    def table(*args):
        sections.append(write_table(tables_dir, *args))

    boot = read_rows('imagenet/bootstrap/bootstrap_summary.csv')
    b = {(r['model'], r['protocol']): r for r in boot}
    values = []
    for model in MODELS:
        train, val = b[model, 'conventional_train'], b[model, 'validation']
        values.append([NAMES.get(model, model)] + [f'{float(r[k]):.3f}' for r, k in
                      [(train, 'point_fd'), (train, 'ci_width'), (val, 'point_fd'), (val, 'ci_width')]]
                      + [f"{float(val['point_fd']) - float(train['point_fd']):.3f}"])
    table('table1_uncertainty', 'ImageNet: full-pool FD and central 95% bootstrap width',
          ['Model', 'Train FD', 'Train w95', 'Val FD', 'Val w95', 'Val - train'], values)

    generated = read_rows('imagenet/generated_size/subsample_ranking_summary.csv')
    gr = {(int(r['size_per_class']), r['protocol']): r for r in generated}
    values = []
    for size in [5, 10, 25]:
        row = [f'{size}K']
        for protocol in ['conventional_train', 'validation']:
            r = gr[size, protocol]
            count = round(float(r['mean_top1_retained']) * int(r['trials']))
            row += [f"{float(r['mean_spearman']):.3f}", f"{count}/{r['trials']}"]
        values.append(row)
    table('table2a_generated_stability', 'Generated-size stability vs each protocol\'s full-50K baseline',
          ['Images', 'Train rho', 'Train winner retained', 'Val rho', 'Val winner retained'], values)
    rr = read_rows('imagenet/reference_size/subsample_ranking_summary.csv')
    rr.sort(key=lambda r: int(r['size_per_class']))
    table('table2b_reference_stability', 'Real-reference stability vs recomputed full-training statistics',
          ['Images per class', 'Mean rho', 'Winner retained'],
          [[r['size_per_class'], f"{float(r['mean_spearman']):.3f}",
            f"{round(float(r['mean_top1_retained']) * int(r['trials']))}/{r['trials']}"] for r in rr])

    cf = {name: {(r['model'], r['protocol']): r for r in read_rows(f'cifar10/{name}/bootstrap_summary.csv')}
          for name in ['inception', 'dinov2', 'clip']}
    cifar_models = ['Neon-EDM', 'EDM', 'StyleGAN2-ADA', 'C2OT']
    values = []
    for model in cifar_models:
        row = [model]
        for backbone, protocol in [('inception', 'train_full'), ('inception', 'test_10k'),
                                   ('dinov2', 'test_10k'), ('clip', 'test_10k')]:
            r = cf[backbone][model, protocol]
            digits = 3 if backbone == 'clip' else 2
            row.append(f"{float(r['point_fd']):.{digits}f} / {float(r['ci_width']):.{digits}f}")
        values.append(row)
    table('table4_cifar10', 'CIFAR-10: full-pool FD / central 95% bootstrap width',
          ['Model', 'Inception train', 'Inception test', 'DINOv2 test', 'CLIP test'], values)

    gs = {(r['model'], r['protocol'], int(r['size_per_class'])): r
          for r in read_rows('imagenet/generated_size/subsample_summary.csv')}
    for protocol in ['conventional_train', 'validation']:
        values = []
        for model in MODELS:
            row = [NAMES.get(model, model)]
            for size in [5, 10, 25]:
                r = gs[model, protocol, size]
                row.append(f"{float(r['mean_fd']):.3f} +/- {float(r['std_fd']):.3f}")
            row.append(f"{float(b[model, protocol]['point_fd']):.3f}")
            values.append(row)
        table(f'appendix_generated_{protocol}', f'{protocol}: generated-size FD (mean +/- sample SD)',
              ['Model', '5K', '10K', '25K', '50K (once)'], values)
    rs = {(r['model'], int(r['size_per_class'])): r
          for r in read_rows('imagenet/reference_size/subsample_summary.csv')}
    table('appendix_reference_size', 'Balanced real-reference FD (mean +/- sample SD)',
          ['Model', '50', '100', '250', '500', '732'],
          [[NAMES.get(m, m)] + [f"{float(rs[m, n]['mean_fd']):.3f} +/- {float(rs[m, n]['std_fd']):.3f}"
                                for n in [50, 100, 250, 500, 732]] for m in MODELS])
    for backbone in cf:
        digits = 3 if backbone == 'clip' else 2
        values = []
        for model in cifar_models:
            row = [model]
            for protocol in ['train_full', 'train_10k_seed2026', 'train_10k_seed2027', 'test_10k']:
                r = cf[backbone][model, protocol]
                row.append(f"{float(r['point_fd']):.{digits}f} / {float(r['ci_width']):.{digits}f}")
            values.append(row)
        table(f'appendix_cifar10_{backbone}', f'CIFAR-10 {backbone}: FD / w95',
              ['Model', 'Train 50K', 'Balanced A', 'Balanced B', 'Test 10K'], values)

    shifts = [float(b[m, 'validation']['point_fd']) - float(b[m, 'conventional_train']['point_fd']) for m in MODELS]
    widths = [float(b[m, 'conventional_train']['ci_width']) for m in MODELS]
    larger = [max(float(b[m, p]['ci_width']) for p in ['conventional_train', 'validation']) for m in MODELS]
    facts = {'verified_input_files': input_count, 'imagenet_models': len(MODELS),
             'median_train_validation_shift': statistics.median(shifts),
             'median_conventional_bootstrap_width': statistics.median(widths),
             'median_per_model_shift_over_larger_width': statistics.median(s / w for s, w in zip(shifts, larger)),
             'imagenet_bootstrap_winners': winner_counts(read_rows('imagenet/bootstrap/bootstrap_replicates.csv')),
             'cifar10_bootstrap_winners': {name: winner_counts(read_rows(f'cifar10/{name}/bootstrap_replicates.csv')) for name in cf}}
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import paper_figures as f
        f.FIG_DIR = output / 'figures'
        f.FIG_DIR.mkdir()
        sources = {name: f.load_table(filename) for name, filename in f.FILES.items()}
        inception = sources['Inception']
        metrics = ['Train (openai stats)', 'Train (JiT stats)', 'Train (balanced)', 'Train (balanced732)', 'Val']
        common = [m for m in inception if all(f.finite(f.val(inception, m, 'CFG', p)) for p in metrics)]
        facts['coverage'] = {'historical_model_labels': len(f.load_table('fid_table_results_inception-v3-compat.csv')),
                             'plotting_subset_labels': len(inception), 'common_CFG_pools': len(common)}
        f.set_style()
        f.make_fid_leaderboard_progress(inception)
        f.make_protocol_scatter_panel(inception)
        f.make_backbone_scatter_panel(sources)
        f.make_protocol_ranges(inception)
    with (output / 'facts.json').open('x') as file:
        json.dump(facts, file, indent=2)
        file.write('\n')
    with (output / 'results.md').open('x') as file:
        file.write('# Reproduced Numerical Results\n\n')
        file.write('These tables re-analyze released CSVs, not freshly generated image pools. '
                   'w95 is a percentile-distribution width, not a calibrated population confidence interval. '
                   'Do not compare raw FD across backbones. Paper Table 3 is a conceptual recommendation, not an experiment.\n\n')
        file.write('\n'.join(sections))
    print(json.dumps(facts, indent=2))
    print(f'Results written to {output}')
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/paper'))
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    reproduce(args.output_dir.resolve(), figures=not args.no_figures)


if __name__ == '__main__':
    main()
