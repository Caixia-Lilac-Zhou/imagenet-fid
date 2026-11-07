"""Regression checks for camera-ready values rebuilt from released CSVs."""
from pathlib import Path
import csv
from reproduce_paper import reproduce
from test_support import preserved_directory


def main():
    with preserved_directory() as temporary:
        output = Path(temporary) / 'paper'
        facts = reproduce(output, figures=False)
        assert abs(facts['median_train_validation_shift'] - 1.3629706215) < 1e-8
        assert abs(facts['median_conventional_bootstrap_width'] - 0.066406497) < 1e-8
        assert facts['imagenet_bootstrap_winners']['conventional_train'] == {'BAR-L': 200}
        assert facts['imagenet_bootstrap_winners']['validation'] == {'RAE-DDT': 197, 'BAR-L': 3}
        assert all(w == {'Neon-EDM': 200} for backbone in facts['cifar10_bootstrap_winners'].values() for w in backbone.values())
        with (output / 'tables/table2a_generated_stability.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert [r['Val winner retained'] for r in rows] == ['0/20', '0/20', '20/20']
        assert [r['Train rho'] for r in rows] == ['0.724', '0.957', '0.999']
    print('paper_results: all tests passed')
