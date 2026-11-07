# ImageNet FID is a Pass Check, Not a Finish Line

Evaluation code and recorded results for the **NeurIPS 2026 Position Paper Track** paper by **Qinyu Zhao, Guangting Zheng, and Caixia Zhou**. [NeurIPS paper page](https://neurips.cc/virtual/2026/poster/140042)

The paper argues for retaining conventional ImageNet Fréchet Inception Distance (FID) as a comparability check. Once methods reach competitive and stable performance, the evaluation should focus on evidence for their technical contributions, such as efficiency, scalability, or representation quality.

## What the Audit Found

- The historical ImageNet audit covers 47 model/run labels across 15 families. Holding the generated samples fixed, changes to the real-image reference or feature extractor can change which model ranks first.
- In the eight-family ImageNet study, BAR-L led all 200 bootstrap replicates against the conventional training reference. RAE-DDT led 197 of 200 against the validation reference. This comparison separates sensitivity to the reference choice from sampling variability within each protocol.
- In the four-model CIFAR-10 comparison, Neon-EDM stayed first across the tested references, feature spaces, and bootstrap replicates.

## What This Repository Reproduces

| Entry point | Inputs | Outputs | GPU needed? |
| --- | --- | --- | --- |
| `analysis/reproduce_paper.py` | Included result CSVs | Four main PDF figures, numerical main/appendix tables, audited headline facts | No |
| `examples/smoke_test.py` | Synthetic low-dimensional features | Small end-to-end FD, bootstrap, and subsampling run | No |
| `evaluation/evaluate.py` | Your per-image features and/or reference statistics | Full-pool FD scores | Optional |
| `evaluation/bootstrap_fid_uncertainty.py` | Your feature manifest | Conditional bootstrap distributions, widths, paired differences | Recommended for 2048-D features |
| `evaluation/subsample_fid_stability.py` | Your feature manifest | Generated/reference-size sensitivity and ranking stability | Recommended for 2048-D features |
| `evaluation/extract_*features.py` | Your images and labels | Cached Inception, DINOv2, or CLIP features | Recommended |

The included CSVs support result-level reproduction. Image-level reruns require separately obtained datasets, model checkpoints or generated images, and per-image features. The historical audit has partial run-level provenance. See [provenance and release scope](docs/PROVENANCE.md).

## Generator Sources

The [model-source table](docs/MODEL_SOURCES.md) lists the official repositories and model-specific standard evaluation recipes for all **47 original audit labels across 15 families**. The authors confirm that MeanFlow uses the MeanFlow implementation/checkpoints in **AlphaFlow**, and pMF uses the official **PyTorch `torch` branch**. A [machine-readable registry](data/provenance/imagenet_original_models.json) preserves every original model label, including the legacy LightningDiT labels `VAVAE-64ep` and `VAVAE-800ep`.

The table distinguishes author-confirmed source choices from currently documented upstream parameters. It does not infer historical seeds or artifact hashes from today's repository defaults.

## Quick Start

Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
python tests/run_tests.py
python examples/smoke_test.py --output-dir outputs/smoke
python analysis/reproduce_paper.py --output-dir outputs/paper
```

Open `outputs/paper/results.md` for the numerical tables and `outputs/paper/figures/` for the four vector PDFs. The same tables are exported as CSV and LaTeX. `facts.json` records winner counts and headline statistics computed from the included results. Table 3 in the paper is a conceptual recommendation and is not an experimental output.

Output directories must be new. Use another name for a fresh run; no cleanup command is needed. Matching bootstrap, subsampling, and extraction jobs can resume in their existing directories.

The preparation environment and verified/skipped checks are recorded in [VALIDATION.md](docs/VALIDATION.md).

## Evaluate New Image Pools

Install PyTorch and torchvision appropriate to your hardware, then:

```bash
python -m pip install -r requirements-features.txt
python evaluation/extract_inception_features.py \
  --spec-json configs/imagenet_val.json \
  --output-dir features/imagenet_val --device cuda
python evaluation/evaluate.py configs/manifest.example.csv \
  --output outputs/new_scores.csv --backend torch --device cuda
```

Edit the example input paths and provide all bundles referenced by the manifest first. No dataset download or model generation is triggered by these examples. Extractor weights may be fetched by their upstream libraries on first use. See [the complete reproduction guide](docs/REPRODUCTION.md) for bootstrap, sample-size experiments, balanced references, and CIFAR-10.

## Protocol in Brief

- ImageNet resampling study: eight families, one fixed balanced 50K generated pool per family, 50 images per class.
- Bootstrap: 200 class-stratified replicates, master seed 2026. Conventional training statistics are fixed; validation FD resamples both sides and shares real-reference draws across models.
- Generated size: 5K / 10K / 25K, 20 subsets without replacement per size; the full 50K pool is evaluated once.
- Real-reference size: 50 / 100 / 250 / 500 / 732 images per class, 20 balanced subsets; rankings are compared with recomputed full-training statistics.
- CIFAR-10 boundary check: four models, four real references, three feature spaces, 200 replicates.

Bootstrap width is **not** a bias-corrected population confidence interval. Resampling a fixed pool is **not** independent generation or training. Raw FD values are **not** comparable across different feature spaces. The detailed definitions and historical coverage are in [PROTOCOLS.md](docs/PROTOCOLS.md).

## Layout

```text
analysis/       paper figures and numerical table reproduction
evaluation/     feature extraction, FD, bootstrap, subsampling
configs/        portable input specifications and manifest examples
data/           recorded result CSVs, provenance, and checksums
docs/           protocols, reproduction steps, scope and limitations
examples/       small CPU-only end-to-end example
scripts/        manifest creation, release checks, packaging
tests/          numerical, resume, input-safety and release tests
```

The original server launchers, credentials, private paths, internal reviews, and rebuttal drafts are not part of this release. No script deletes files or directories. Tests retain their small temporary fixtures for inspection.

## Citation and License

```bibtex
@inproceedings{zhao2026imagenetfid,
  title={ImageNet FID is a Pass Check, Not a Finish Line},
  author={Zhao, Qinyu and Zheng, Guangting and Zhou, Caixia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://neurips.cc/virtual/2026/poster/140042}
}
```

Project code is released under the [MIT License](LICENSE). Third-party datasets, pretrained weights, libraries, and upstream generator implementations retain their own licenses; they are not relicensed by this repository. See [THIRD_PARTY.md](THIRD_PARTY.md).
