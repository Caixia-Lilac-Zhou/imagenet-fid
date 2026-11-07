# Provenance and Release Scope

## Included Evidence

| Material | Location | Scope |
| --- | --- | --- |
| Original multi-reference/multi-backbone scores | `data/original/` | Aggregate historical audit, including plotting subset and full 47-label table |
| Eight-family bootstrap and size experiments | `data/imagenet/` | Raw replicate/subset scores and numerical summaries |
| Four-model CIFAR-10 study | `data/cifar10/` | Inception, DINOv2, CLIP bootstrap records and cross-backbone summaries |
| Original 47-label source registry | `data/provenance/imagenet_original_models.json`, [model-source table](MODEL_SOURCES.md) | Author-confirmed standard-recipe policy; verified official source/configuration links, with MeanFlow sourced from AlphaFlow and pMF from the official PyTorch `torch` branch |
| Eight-family source descriptions | `data/provenance/imagenet_eight_families.csv` | Available upstream identifiers and generation seeds |
| CIFAR model registry | `data/provenance/cifar10_models.json` | Repository commits, checkpoint URLs/hashes, sampler settings |
| Input integrity | `data/SHA256SUMS.json` | SHA-256 for every released data file |
| Code/data lineage | `SOURCE_MANIFEST.json` | Relative source names and pre-adaptation hashes |

Raw result CSVs are copied without numerical changes. The ImageNet provenance CSV retains generated-model rows and omits private filesystem columns. Unknown archive-generation seeds remain unknown; seed 2026 is not assigned retroactively to author-released images. The AFM archive permutation seed is not its generation seed.

On 2026-11-06, the authors confirmed that generated images used standard configurations from the original papers' GitHub repositories, with MeanFlow obtained from the AlphaFlow repository. The original-model registry now maps all 47 audit labels to 15 families and their upstream recipes. This establishes source-level attribution, not a complete per-run manifest: current README parameters are labeled as upstream documentation, and historical commits, hashes, seeds and unrecorded overrides are not guessed. The separate eight-family study retains its recorded author-released archives and generation settings.

The authors additionally confirmed that the audited pMF models used the official PyTorch implementation on the `torch` branch. Its branch-specific evaluation documentation is now linked in the registry. This resolves the pMF implementation-branch question; it does not establish a particular historical commit or recover missing run-level fields.

## Not Included

- ImageNet/CIFAR image datasets, generated image pools, feature caches, and reference-statistics archives.
- Generator training or inference implementations and pretrained weights. Use the recorded official upstream releases.
- Shared-server job schedulers, SSH/proxy settings, account information, internal reviewer discussions, or unpublished response drafts.
- Exact checkpoint/archive hashes, sampling seeds, and preprocessing settings for every pool in the historical 47-label audit. Those records are not fully recoverable from aggregate scores.
- The historical balanced-50 repeat count and seed list. The separate 20-draw reference-size experiment must not be used to fill that historical gap.

These exclusions mean result-level reproduction is available from the repository alone, but identical image-level reruns require additional assets and, for the historical audit, additional provenance. No missing metadata have been guessed.

## Release Adaptations

The FD estimators, stratified resampling, stable seeds, and rank calculations are retained from the experiment code. The release adds portable entry points, table reproduction, and explicit input/output checks. Fresh-run flags no longer delete previous results. Output collisions fail or resume only through the existing metadata checks. Tests retain their fixtures rather than automatically deleting temporary directories.

Figure rendering uses the camera-ready plotting implementation. Numerical equivalence is checked at the CSV/fact level; PDF bytes can differ with rendering libraries and timestamps. The repository has not been uploaded or assigned a public URL by the preparation process.
