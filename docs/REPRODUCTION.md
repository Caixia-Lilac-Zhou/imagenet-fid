# Reproduction Guide

Run commands from the repository root. Relative paths inside JSON specifications and CSV manifests resolve relative to those files, not the current shell directory. Provide data and checkpoints under their upstream terms before starting GPU work.

## 1. Rebuild the Paper from Recorded Results

```bash
python scripts/verify_release.py
python analysis/reproduce_paper.py --output-dir outputs/paper
```

This checks data hashes, creates the four main PDF figures with PNG previews, and exports numerical main/appendix tables as Markdown, CSV, and LaTeX. It does not rerun image generation or feature extraction. Use `--no-figures` for a NumPy-only table/fact check.

Expected checks include BAR-L winning conventional ImageNet FID in 200/200 replicates and RAE-DDT winning validation FD in 197/200. The median train-validation shift is approximately 1.363, and the median conventional bootstrap width is approximately 0.0664. These summaries refer to the eight-family study, not the historical figure pools.

## 2. Extract and Validate Features

Install torch/torchvision for the intended CPU or CUDA environment, then `python -m pip install -r requirements-features.txt`. No GPU job is launched automatically.

Edit `configs/imagenet_train.json`, `configs/imagenet_val.json`, and `configs/generated_imagenet.json` to point to your files. Keep generated labels from the official generation process; do not infer class ordering from an archive name.

```bash
python evaluation/extract_inception_features.py \
  --spec-json configs/imagenet_train.json \
  --output-dir features/imagenet_train --device cuda
python evaluation/extract_inception_features.py \
  --spec-json configs/imagenet_val.json \
  --output-dir features/imagenet_val --device cuda
python evaluation/extract_inception_features.py \
  --spec-json configs/generated_imagenet.json \
  --output-dir features/BAR-L --device cuda
```

The generated spec is a template; repeat for each actual model. Feature output contains `features.npy`, `labels.npy`, `metadata.json`, and a class mapping. The loader checks sample/class counts and resumes matching partial extraction. For multiple workers, extract interleaved shards with `--num-shards N --shard-index I` into distinct directories, then use `evaluation/merge_feature_shards.py`; it refuses nonempty destinations.

ImageNet validation folders must have correct class labels, or use an explicit label sidecar. The conventional reference must be the supplied Inception-compatible statistics archive (keys `mu` and `sigma`); it cannot be recovered from the published FD scalar values. The reference and generated features must use identical extractors and preprocessing contracts.

## 3. Full-Pool Scores and Bootstrap

Create a manifest from your own feature directories. Repeat `--generated` to include all eight families: BAR-L, REPA-E, RAE-DDT, LightningDiT, MAR-H, JiT-H-16, AlphaFlow-XL-2+-2NFE, and AFM-XL2-112L-1NFE.

```bash
python scripts/make_manifest.py --output outputs/manifests/imagenet.csv \
  --generated BAR-L=features/BAR-L --generated RAE-DDT=features/RAE-DDT \
  --reference validation=features/imagenet_val \
  --fixed-reference conventional_train=datasets/stats/openai_train.npz
python evaluation/evaluate.py outputs/manifests/imagenet.csv \
  --output outputs/imagenet_points.csv --backend torch --device cuda
python evaluation/bootstrap_fid_uncertainty.py outputs/manifests/imagenet.csv \
  --output-dir outputs/imagenet_bootstrap --replicates 200 --seed 2026 \
  --backend torch --device cuda
```

The manifest builder validates bundle shapes/formats and writes relative paths. Dataset-specific count requirements are checked during extraction through the JSON specifications. A fixed-statistics row bootstraps only the generated side. A real-feature row bootstraps both sides, sharing the reference draw across models. The bootstrap is conditional on the available pools; see [PROTOCOLS.md](PROTOCOLS.md) before interpreting percentile columns.

## 4. Generated Sample Size

Reuse the same cached generated features; do not regenerate 50K images for each trial:

```bash
python evaluation/subsample_fid_stability.py outputs/manifests/imagenet.csv \
  --output-dir outputs/generated_size --generated-per-class 5,10,25 \
  --reference-per-class none --trials 20 --seed 2026 --backend torch --device cuda
```

The listed sizes are **per class**: 5/10/25 become 5K/10K/25K for ImageNet. Each subset is drawn without replacement. Rankings use each protocol's full-pool ordering as baseline. The 50K baseline is evaluated once internally.

## 5. Real-Reference Sample Size

Create another manifest with generated pools unchanged but `--reference train_pool=features/imagenet_train`, and **no** fixed-reference row. For example:

```bash
python scripts/make_manifest.py --output outputs/manifests/reference_size.csv \
  --generated BAR-L=features/BAR-L --generated RAE-DDT=features/RAE-DDT \
  --reference train_pool=features/imagenet_train
python evaluation/subsample_fid_stability.py outputs/manifests/reference_size.csv \
  --output-dir outputs/reference_size --generated-per-class none \
  --reference-per-class 50,100,250,500,732 --trials 20 --seed 2026 \
  --backend torch --device cuda
```

The ranking baseline is computed from these full training features. It is not the supplied OpenAI statistics. To create a single balanced reference bundle for a sensitivity card:

```bash
python evaluation/create_balanced_feature_subset.py features/imagenet_train \
  features/imagenet_balanced732 --per-class 732 --seed 2026
python evaluation/compute_reference_stats.py features/imagenet_balanced732 \
  --output outputs/balanced732_stats.npz
```

Repeated balanced-50 estimates should average scores over recorded draws; averaging feature statistics is a different operation. Do not replace the unavailable historical balanced-50 seed list with the seeds of the new resampling study.

## 6. CIFAR-10 and Alternate Backbones

Use the model sources, commits, hashes, and samplers in `data/provenance/cifar10_models.json` to prepare your own balanced pools. This repository does not run or redistribute the upstream generators. The JSON spec `configs/cifar10_generated.json` expects a labeled NPZ with 10K native 32x32 RGB images.

```bash
python evaluation/extract_backbone_features.py \
  --spec-json configs/cifar10_generated.json --backbone clip-vit-b-16 \
  --output-dir features/cifar10/clip/Neon-EDM --device cuda
```

Use `dinov2-vit-b-14` for DINOv2, or the Inception extractor for the third backbone. Repeat extraction for four models and all real references, using a separate directory tree for each backbone. Full train has 50K images; balanced train subsets and test have 10K. Construct balanced training subsets at 1000/class with seeds 2026 and 2027. In the manifest name the references `train_full`, `train_10k_seed2026`, `train_10k_seed2027`, and `test_10k`, all as **feature references**. Run 200 bootstrap replicates with seed 2026 for each backbone. Keep generated images identical across backbones.

Source identity is part of deterministic balanced-subset seed derivation. Independently reconstructed bundles can therefore yield different subsets even with the same master seed. Exact reuse requires the original selected indices or matching source identity, not just the seed integer.

## Resources, Resume, and Safety

Feature-level resampling avoids repeated generation, but float64 covariance eigendecomposition is still expensive at 2048 dimensions. NumPy is suitable for the tiny smoke test; a CUDA backend is recommended for the full experiment. Alternate backbones require their own extraction and storage.

Bootstrap/subsampling resume requires matching manifest bytes, master seed, backend, device, and chunk size. Extraction also checks source/settings metadata. Those checks do not fully content-hash every image or feature array; keep inputs immutable and start a new output directory if their contents change. Reusing a path after modifying its feature data is not a valid resume.

`--no-resume` and legacy `--overwrite` flags do not erase existing output. Nonempty fresh destinations are rejected. Matching resumable runs update only their own managed checkpoint/result files; interrupted partial rows are reconstructed. Do not run two writers against the same output directory. Tests retain their small fixtures, and no cleanup or SSH command is provided.
