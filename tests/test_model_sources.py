"""Check complete original-model attribution without inventing run metadata."""
import csv
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    registry = json.loads((ROOT / 'data/provenance/imagenet_original_models.json').read_text())
    families = registry['families']
    assert registry['schema_version'] == 1
    assert registry['scope'] == 'original_47_label_imagenet_audit'
    assert len(families) == len({f['family'] for f in families}) == 15
    with (ROOT / 'data/original/fid_table_results_inception-v3-compat.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    expected = {r[0] for r in rows[2:] if r and r[0] and r[0] != 'Val'}
    models = [m for family in families for m in family['models']]
    labels = [m['audit_model_id'] for m in models]
    assert len(labels) == len(set(labels)) == len(expected) == 47
    assert set(labels) == expected

    document = (ROOT / 'docs/MODEL_SOURCES.md').read_text()
    for family in families:
        for key in ['repository_url', 'configuration_url']:
            parsed = urlparse(family[key])
            assert parsed.scheme == 'https' and parsed.netloc == 'github.com'
            assert family[key] in document
        assert family['entry_point'] and family['upstream_recipe_summary']
        for model in family['models']:
            assert model['upstream_identifier']
            assert f"`{model['audit_model_id']}`" in document
            assert 'generation_seed' not in model and 'checkpoint_sha256' not in model

    by_family = {f['family']: f for f in families}
    assert by_family['MeanFlow']['repository_url'] == 'https://github.com/snap-research/alphaflow'
    assert len(by_family['MeanFlow']['models']) == 6
    assert all(m['upstream_identifier'].startswith('meanflow-') for m in by_family['MeanFlow']['models'])
    assert {m['audit_model_id'] for m in by_family['LightningDiT']['models']} == {'VAVAE-64ep', 'VAVAE-800ep'}
    pmf = by_family['pMF']
    assert pmf['branch'] == 'torch' and pmf['implementation'] == 'PyTorch'
    assert pmf['branch_evidence'] == 'author_confirmation'
    assert pmf['branch_url'] == pmf['repository_url'] + '/tree/torch'
    assert pmf['configuration_url'] == pmf['repository_url'] + '/blob/torch/README.md'
    assert pmf['entry_point'] == 'evaluate.py evaluate'
    assert {m['audit_model_id'] for m in pmf['models']} == {'pMF-B-16', 'pMF-L-16', 'pMF-H-16'}
    assert 'historical branch remains unconfirmed' not in document.lower()
    assert 'generation_seed' in registry['evidence']['historical_fields_not_recovered']
    print('model_sources: all 47 labels / 15 families covered; provenance boundaries preserved')


if __name__ == '__main__':
    main()
