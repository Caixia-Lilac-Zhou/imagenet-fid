# Third-Party Materials

The repository contains evaluation/analysis code and small numerical result tables. It does not vendor generator implementations, pretrained weights, ImageNet/CIFAR images, or generated image archives.

| Component | Use | Upstream |
| --- | --- | --- |
| torch-fidelity 0.4.0 | Inception, DINOv2, and CLIP extraction wrappers | https://github.com/toshas/torch-fidelity |
| PyTorch / torchvision | Tensor computation and image loading | https://github.com/pytorch/pytorch |
| NumPy, Matplotlib, Pillow | Numerical analysis, plotting, image decoding | Respective installed packages |
| ADM / JiT | Reference center-crop procedure reproduced by the evaluation pipeline | https://github.com/openai/guided-diffusion and https://github.com/LTH14/JiT |
| Generator releases | Source checkpoints or sample archives, not redistributed here | `data/provenance/` |

The reference crop follows the standard ADM/JiT procedure (BOX downsampling, bicubic short-edge resize, centered square crop). The upstream MIT notices are retained in `licenses/ADM_JiT_MIT.txt`, following the [OpenAI license](https://github.com/openai/guided-diffusion/blob/main/LICENSE) and [JiT license](https://github.com/LTH14/JiT/blob/main/LICENSE). No upstream generator source tree is included. Citation of an upstream project does not grant permission to redistribute its weights or data. Users must obtain datasets and weights under the applicable terms.

The MIT grant covers this project's own code and documentation. It does not override third-party licenses or dataset access restrictions. Redistribution of cached per-image features or sample archives requires a separate review of the underlying terms.
