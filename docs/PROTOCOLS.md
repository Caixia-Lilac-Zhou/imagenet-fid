# Evaluation Protocols

## Original Audit vs Resampling Study

The historical audit contains 47 model/run labels, not 47 independent model families. Guidance-specific pools are separate counting units. For continuity with the paper, the plotting subset has 38 labels and 34 CFG pools common to all reference comparisons; the complete table has 40 matched CFG pools. The nine omitted labels are DiT-XL-2, four SiT variants, and four LlamaGen variants. Missing cells are not imputed.

The main reference scatter's broad views retain scores at most 8 on both axes; its focused column shows the nine lowest conventional CFG scores. The backbone scatter has 28 competitive pools in the score rows and 34 common pools in the rank row. These visual filters do not change the full inventory.

The eight-family uncertainty study uses separately recorded generated pools. Its point scores must not be substituted into the original audit or presented as re-evaluations of identical historical samples.

## Reference Definitions

| Name | Definition | Important distinction |
| --- | --- | --- |
| Conventional train | Supplied OpenAI training statistics | The statistics archive is fixed; its exact construction is not reconstructed from code |
| Recomputed train | Full training images processed through the JiT/ADM-style crop in our pipeline | Not identical to the supplied statistics |
| Balanced-50 | 50 real training images per class | Historical aggregate averages over repeated subsets; historical repeat count/seed list remain unavailable |
| Balanced-732 | 732 real training images per class | The least-populated training class determines the common count |
| Validation | 50 held-out validation images per class | Changes the target split as well as sample size relative to full training |

## Features and FD

ImageNet real images use repeated BOX downsampling until the shorter edge is below 512, bicubic resizing of the shorter edge to 256, and a 256x256 center crop. Already prepared generated images receive no additional pre-crop. Inception uses torch-fidelity's TensorFlow-1-compatible `inception-v3-compat` 2048-D representation, internal resizing to 299 and normalization `(x-128)/128`.

FD is computed from feature means and covariances in float64, using a symmetric matrix-square-root formulation. DINOv2 ViT-B/14 uses 768-D features and CLIP ViT-B/16 uses 512-D features. Raw FD magnitudes are only meaningful within the same feature space. The historical VGG16 scores are included for plotting; a VGG16 feature-extraction runner is not included in this release.

## Conditional Bootstrap

The eight-family study has 50K generated images per model, balanced at 50 images per class. Each bootstrap replicate samples the same count with replacement within each class. Master seed: 2026; replicates: 200.

- Conventional train: resample generated features; the supplied real statistics remain fixed.
- Validation: resample generated and real features; the same real-reference draw is shared across models.
- The full-pool score is the point estimate. `w95 = percentile97.5 - percentile2.5` is the full distribution width, not a standard deviation or a symmetric error bar.
- Legacy output columns `ci_2.5`, `ci_97.5`, and `ci_width` name bootstrap percentiles. They do not establish calibrated population coverage. Bootstrap centers can be substantially biased relative to full-pool FD.
- These replicates do not capture independent training, generation, checkpoint selection, or uncertainty in inherited fixed statistics.

## Sample-Size Axes

Generated-size analysis draws 20 class-balanced subsets without replacement at 5, 10, or 25 images per class (5K, 10K, 25K). Each protocol's own full-50K ordering is its baseline. The full pool is evaluated once, not 20 times. Repeated subsets can overlap; they are not independent image-generation runs.

Real-reference analysis fixes generated pools at 50K and draws 20 training subsets at 50, 100, 250, 500, or 732 images per class. Its baseline is **recomputed full-training statistics**, not the supplied OpenAI archive. Full-training class imbalance and balanced-subset construction differ even at the largest subset size.

Winner counts and Spearman correlations describe conditional stability under these choices. They are not evidence of a universally superior generator or of statistical equivalence between methods.

## CIFAR-10

Four conditional generators: Neon-EDM, EDM, StyleGAN2-ADA, and C2OT. Each generated pool has 10K images (1000/class). References: full 50K train, two balanced 10K train subsets (seeds 2026 and 2027), and the 10K test set. Native RGB images receive no extra pre-crop. Inception, DINOv2, and CLIP reuse the same generated pools. Both sides are bootstrapped within class for 200 replicates.

Neon-EDM wins every replicate in every released setting. Inception reverses the order of StyleGAN2-ADA and C2OT relative to DINOv2 and CLIP. This boundary study does not reproduce the ImageNet reference-induced winner reversal and does not establish cross-domain universality or human preference alignment.
