# Generator Sources and Standard Configurations

Verified against official upstream sources on 2026-11-06.

The authors confirm that generated images used the standard configurations supplied by the original papers' GitHub repositories. **MeanFlow is the exception in implementation provenance: it uses the MeanFlow models released in the AlphaFlow repository.** The table below identifies the source repository and the model-specific sampling/evaluation recipe for all **47 original audit labels across 15 families**.

The numerical recipe summaries describe the currently inspected upstream documentation. They are not a recovered log of every historical launch. A repository's current default is not evidence of a historical seed, commit, file hash, or preprocessing override. For machine-readable model-to-source mappings, see [imagenet_original_models.json](../data/provenance/imagenet_original_models.json).

## Source and Configuration Table

"Standard configuration" means the published evaluation recipe for the corresponding model, not one universal CLI default for a whole family. Values in the last column are guided examples unless stated otherwise.

| Family | Model IDs in the original audit | Source repository | Standard configuration reference |
| --- | --- | --- | --- |
| AlphaFlow | `AlphaFlow-B-2-1NFE`, `AlphaFlow-B-2-2NFE`, `AlphaFlow-B-2-cfg-1NFE`, `AlphaFlow-B-2-cfg-2NFE`, `AlphaFlow-XL-2-cfg-1NFE`, `AlphaFlow-XL-2-cfg-2NFE`, `AlphaFlow-XL-2-plus-cfg-1NFE`, `AlphaFlow-XL-2-plus-cfg-2NFE` | [snap-research/alphaflow](https://github.com/snap-research/alphaflow) | [Recipe](https://github.com/snap-research/alphaflow/blob/main/scripts/evaluate.sh); `scripts/generate.py`. Use the checkpoint and 1/2-NFE command matching the label. The XL CFG two-step recipe uses a custom time grid; XL+ also enables consistency sampling. |
| BAR | `BAR-B`, `BAR-B-2`, `BAR-B-4`, `BAR-L` | [amazon-far/BAR](https://github.com/amazon-far/BAR) | [Recipe](https://github.com/amazon-far/BAR/blob/main/README.md); `sample_imagenet.py`. Use the matching configs/generator YAML and generator/tokenizer pair. Published guided examples use scales B/4=9.6, B/2=5.5, B=5.0, L=5.3; temperature and bit-allocation settings also differ. |
| DiT | `DiT-XL-2` | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | [Recipe](https://github.com/facebookresearch/DiT/blob/main/README.md); `sample_ddp.py`. ImageNet-256 DiT-XL/2 sampling with the official checkpoint and 50K evaluation entry point. |
| FlowAR | `FlowAR-S`, `FlowAR-L`, `FlowAR-H` | [OliverRensu/FlowAR](https://github.com/OliverRensu/FlowAR) | [Recipe](https://github.com/OliverRensu/FlowAR/blob/main/README.md); `eval.py`. S: 25 flow steps, CFG 4.3, guidance 0.9; L: 25 steps, CFG 2.4, guidance 0.9; H: 50 steps, CFG 2.4, guidance 0.7. Use each model's DiffLoss dimensions. |
| JiT | `JiT-B-16`, `JiT-L-16`, `JiT-H-16` | [LTH14/JiT](https://github.com/LTH14/JiT) | [Recipe](https://github.com/LTH14/JiT/blob/main/README.md); `main_jit.py`. ImageNet-256 evaluation recipes use B/16 CFG 3.0, L/16 CFG 2.4, H/16 CFG 2.2 and guidance interval [0.1, 1.0]. Use Evaluation, not the online-training example. |
| LlamaGen | `LlamaGen-B-256`, `LlamaGen-B-384`, `LlamaGen-XXL`, `LlamaGen-3B` | [FoundationVision/LlamaGen](https://github.com/FoundationVision/LlamaGen) | [Recipe](https://github.com/FoundationVision/LlamaGen/blob/main/GETTING_STARTED.md); `scripts/autoregressive/sample_c2i.sh`. Use the class-conditional recipe and matching B/XXL/3B checkpoint. The guide distinguishes native generation size from 256 evaluation size; XXL and 3B use the FSDP loading option. |
| MAR | `MAR-B`, `MAR-L`, `MAR-H` | [LTH14/mar](https://github.com/LTH14/mar) | [Recipe](https://github.com/LTH14/mar/blob/main/README.md); `main_mar.py`. B/L/H use CFG 2.9/3.0/3.2, respectively; 256 autoregressive iterations, 100 diffusion steps, linear guidance schedule and temperature 1.0. DiffLoss dimensions differ by size. |
| MeanFlow | `MeanFlow-B-2-1NFE`, `MeanFlow-B-2-2NFE`, `MeanFlow-B-2-cfg-1NFE`, `MeanFlow-B-2-cfg-2NFE`, `MeanFlow-XL-2-cfg-1NFE`, `MeanFlow-XL-2-cfg-2NFE` | [snap-research/alphaflow](https://github.com/snap-research/alphaflow) | [Recipe](https://github.com/snap-research/alphaflow/blob/main/scripts/evaluate.sh); `scripts/generate.py`. Use the meanflow checkpoint entries and corresponding 1/2-NFE commands in AlphaFlow's evaluation script, not a separate MeanFlow implementation. |
| RAE-DDT | `RAE-DDT` | [bytetriper/RAE](https://github.com/bytetriper/RAE) | [Recipe](https://github.com/bytetriper/RAE/tree/main/configs/stage2/sampling/ImageNet256); `src/sample_ddp.py`. The ImageNet256 directory provides DiTDHXL-DINOv2-B.yaml and its _AG autoguidance variant. The latter specifies 50 Euler steps and autoguidance scale 1.42. |
| REPA | `REPA-XL-2` | [sihyun-yu/REPA](https://github.com/sihyun-yu/REPA) | [Recipe](https://github.com/sihyun-yu/REPA/blob/main/README.md); `generate.py`. Published SiT-XL/2 evaluation: linear path, SDE, 250 steps, CFG 1.8, upper guidance boundary 0.7; released final checkpoint trained for 4M iterations. |
| REPA-E | `REPA-E` | [End2End-Diffusion/REPA-E](https://github.com/End2End-Diffusion/REPA-E) | [Recipe](https://github.com/End2End-Diffusion/REPA-E/blob/main/README.md); `generate.py`. Use the published generation/evaluation recipe with its tokenizer and checkpoint variant. The model zoo includes joint tuning and frozen E2E-VAE variants. |
| SiT | `SiT-S-2`, `SiT-B-2`, `SiT-L-2`, `SiT-XL-2` | [willisma/SiT](https://github.com/willisma/SiT) | [Recipe](https://github.com/willisma/SiT/blob/main/README.md); `sample_ddp.py`. Use the matching S/B/L/XL architecture and checkpoint. The evaluation example uses ODE; the repository also supports SDE. The historical solver/checkpoint revision is not pinned here. |
| LightningDiT | `VAVAE-64ep`, `VAVAE-800ep` | [hustvl/LightningDiT](https://github.com/hustvl/LightningDiT) | [Recipe](https://github.com/hustvl/LightningDiT/tree/main/configs/reproductions); `run_inference.sh`. Both reproduction YAMLs use 250 Euler steps. The 64ep recipe uses CFG 10.0 and interval start 0.10; 800ep uses CFG 6.7 and interval start 0.125. [64ep YAML](https://github.com/hustvl/LightningDiT/blob/main/configs/reproductions/lightningdit_xl_vavae_f16d32_64ep_cfg.yaml), [800ep YAML](https://github.com/hustvl/LightningDiT/blob/main/configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml). |
| pMF | `pMF-B-16`, `pMF-L-16`, `pMF-H-16` | [Lyy-iiis/pMF, torch branch](https://github.com/Lyy-iiis/pMF/tree/torch) | [Recipe](https://github.com/Lyy-iiis/pMF/blob/torch/README.md); `evaluate.py evaluate`. Author-confirmed official PyTorch implementation. Use the matching B/16, L/16 or H/16 checkpoint and model-specific guidance settings. |
| xAR | `xAR-B`, `xAR-L`, `xAR-H` | [OliverRensu/xAR](https://github.com/OliverRensu/xAR) | [Recipe](https://github.com/OliverRensu/xAR/blob/main/README.md); `eval.py`. All use 256 autoregressive iterations. B: 40 diffusion steps and CFG 2.7; L: 50 steps and CFG 2.3; H: 50 steps and CFG 2.2. |

## Important Model Mappings

- **MeanFlow:** all six MeanFlow labels map to `snap-research/alphaflow`, with the `meanflow-*.pt` entries in its [evaluation script](https://github.com/snap-research/alphaflow/blob/main/scripts/evaluate.sh). Method attribution remains MeanFlow; implementation/checkpoint attribution is AlphaFlow.
- **LightningDiT:** `VAVAE-64ep` and `VAVAE-800ep` are legacy CSV labels for the corresponding LightningDiT generators, not two standalone VAE-only generators. The JSON retains these exact join keys.
- **LlamaGen:** preserve the B-256/B-384 distinction. The official [model zoo](https://github.com/FoundationVision/LlamaGen) lists `c2i_B_256.pt`, `c2i_B_384.pt`, `c2i_XXL_384.pt` and `c2i_3B_384.pt`. The sampling guide distinguishes native generation size and evaluation size; do not silently label all native outputs as 256.
- **RAE-DDT:** the [ImageNet256 recipes](https://github.com/bytetriper/RAE/tree/main/configs/stage2/sampling/ImageNet256) distinguish unguided and autoguided sampling. A generic CFG column is not sufficient to recover the exact historical guidance mechanism.
- **REPA-E:** the upstream model zoo has several tokenizer/checkpoint variants. The exact archive in the separate eight-family study is recorded in [its registry](../data/provenance/imagenet_eight_families.csv); this is not proof that the original audit used that same archive.
- **SiT:** architecture identifiers and the official sampling entry point are recorded for all four sizes; exact historical checkpoint hashes and ODE/SDE selection are not reconstructed from their scores.
- **pMF:** the authors confirmed that the audited models used the [official torch branch](https://github.com/Lyy-iiis/pMF/tree/torch), implemented in PyTorch, rather than the JAX implementation. This confirms the implementation branch, not the exact historical commit, seed or file hash. The torch README documents an image-retention option (`--save-samples`); upstream scripts must be inspected for cleanup behavior before use.

## What This Table Establishes

| Field | Evidence |
| --- | --- |
| Use of original-paper standard configurations | Author confirmation |
| MeanFlow implementation source | Explicit author confirmation: AlphaFlow |
| pMF implementation branch | Explicit author confirmation: official PyTorch implementation, `torch` branch |
| Repository and configuration-document locations | Official upstream source inspection |
| Model-label coverage | Exactly the 47 non-reference labels in the full original Inception CSV |
| Historical commit, checkpoint/pool hashes and generation seeds | Not recovered by looking up current upstream defaults |
| Full no-CFG commands and model-specific preprocessing overrides | Not established for every historical pool |

The paper's 50K class-balanced generated-pool protocol is separate from upstream model hyperparameters. This table does not assert that every unmodified upstream launcher produces balanced labels automatically. CFG and no-CFG observations remain distinct pools; do not treat a guided recipe as a complete specification of both.

This registry improves source-level provenance. It does not close every run-level provenance item described in [PROVENANCE.md](PROVENANCE.md). The [eight-family registry](../data/provenance/imagenet_eight_families.csv) and [CIFAR-10 registry](../data/provenance/cifar10_models.json) remain separate, including any author-released archives.

No generator code, checkpoint or image archive is downloaded or executed by this table. Upstream assets retain their original terms; links do not grant redistribution rights.
