# Depth Pro: faster inference fork

> **You are on `portable`.** Research fork of [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) (`main` mirrors upstream). Two branches:
> - [`portable`](https://github.com/fastforks/ml-depth-pro/tree/portable): `src/depth_pro/fast_inference.py`, `DepthProFast(model)`: export-friendly inference graph (token gathers instead of crops, merge by index, composed deconv/1x1 and head weights, FOV head skipped when `f_px` is given, channels-last convs after the ViTs); plain PyTorch, no custom kernels, no build step. With fp32 on CPU it reproduces the upstream forward (max abs 1e-6).
> - [`cuda-kernels`](https://github.com/fastforks/ml-depth-pro/tree/cuda-kernels): `portable` plus `src/depth_pro/fast`, `DepthProMax(DepthProFast(model))`: CUTLASS GEMMs / convs with fused epilogues (bias, LayerScale, residual, GELU, ReLU) and fp16 accumulation, fp16-accumulate flash attention. Prebuilt for torch 2.14 + CUDA 13 (SASS sm_80/86/89/120 + compute_80 PTX); for another torch build, rebuild once with nvcc + CUTLASS headers (`CUTLASS_DIR=... python -m depth_pro.fast.dpfused`).
>
> Licence: the Apple licence of upstream carries over unchanged (see `LICENSE`; read it before reuse). The pretrained weights are Apple's and are not included.

`dinov2l16_384`, 1536x1536 network input, fp16 weights, RTX 5060 Ti 16 GB, torch 2.14.1+cu130, 2026-10-23. Median of 10 runs after 3 warm-ups; memory = torch peak allocated / reserved; AbsRel against Middlebury-H laser GT (6 scenes, non-occluded, short check, f_px not given for raw); TensorRT: not measured (no working recipe in this release).

| version | KITTI ms | Midd-H ms | speed-up at H (vs flags / vs upstream) | peak MB at KITTI (alloc / reserved) | peak MB at H (alloc / reserved) | startup / first call at H | Midd-H AbsRel raw / median-scaled (6 scenes) | max / mean rel. diff vs upstream at H |
|---|---|---|---|---|---|---|---|---|
| upstream (fp16 weights, as upstream CLI) | 526.8 | 532.2 | 1.00x | 3816 / 5270 | 3821 / 5322 | 15.4 s / 0.82 s | 0.1481 / 0.0609 | - |
| upstream + flags (inference_mode, cudnn.benchmark, TF32) | 522.0 | 528.1 | 1.01x | 3815 / 5276 | 3821 / 4898 | 17.0 s / 3.86 s | 0.1480 / 0.0609 | 0.003 / 0.00008 |
| **portable** (`DepthProFast`) | 520.4 | 519.5 | 1.02x / 1.02x | 3529 / 4304 | 3532 / 4218 | 13.5 s / 0.79 s | 0.1482 / 0.0609 | 0.00305 / 0.000169 |
| **cuda-kernels** (`DepthProMax`) | 282.8 | 282.3 | 1.87x / 1.89x | 3632 / 4516 | 3635 / 4516 | 13.6 s / 0.54 s | 0.1481 / 0.0609 | 0.0079 / 0.000839 |

upstream + flags = unmodified upstream code (already fp16 weights from `create_model_and_transforms(precision=torch.half)`; the CLI default), driver-side only: torch.inference_mode (infer already uses it), cudnn.benchmark=True, TF32 allowed; autocast / channels_last / .half() not applicable (fp16 already; channels_last cannot be set without editing the network); torch.compile not part of the row.

---

## Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

This software project accompanies the research paper:
**[Depth Pro: Sharp Monocular Metric Depth in Less Than a Second](https://arxiv.org/abs/2410.02073)**, 
*Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun*.

![](data/depth-pro-teaser.jpg)

We present a foundation model for zero-shot metric monocular depth estimation. Our model, Depth Pro, synthesizes high-resolution depth maps with unparalleled sharpness and high-frequency details. The predictions are metric, with absolute scale, without relying on the availability of metadata such as camera intrinsics. And the model is fast, producing a 2.25-megapixel depth map in 0.3 seconds on a standard GPU. These characteristics are enabled by a number of technical contributions, including an efficient multi-scale vision transformer for dense prediction, a training protocol that combines real and synthetic datasets to achieve high metric accuracy alongside fine boundary tracing, dedicated evaluation metrics for boundary accuracy in estimated depth maps, and state-of-the-art focal length estimation from a single image.


The model in this repository is a reference implementation, which has been re-trained. Its performance is close to the model reported in the paper but does not match it exactly.

## Getting Started

We recommend setting up a virtual environment. Using e.g. miniconda, the `depth_pro` package can be installed via:

```bash
conda create -n depth-pro -y python=3.9
conda activate depth-pro

pip install -e .
```

To download pretrained checkpoints follow the code snippet below:
```bash
source get_pretrained_models.sh   # Files will be downloaded to `checkpoints` directory.
```

### Running from commandline

We provide a helper script to directly run the model on a single image:
```bash
# Run prediction on a single image:
depth-pro-run -i ./data/example.jpg
# Run `depth-pro-run -h` for available options.
```

### Running from python

```python
from PIL import Image
import depth_pro

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
```


### Evaluation (boundary metrics) 

Our boundary metrics can be found under `eval/boundary_metrics.py` and used as follows:

```python
# for a depth-based dataset
boundary_f1 = SI_boundary_F1(predicted_depth, target_depth)

# for a mask-based dataset (image matting / segmentation) 
boundary_recall = SI_boundary_Recall(predicted_depth, target_mask)
```


## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  booktitle  = {International Conference on Learning Representations},
  year       = {2025},
  url        = {https://arxiv.org/abs/2410.02073},
}
```

## License
This sample code is released under the [LICENSE](LICENSE) terms.

The model weights are released under the [LICENSE](LICENSE) terms.

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [Acknowledgements](ACKNOWLEDGEMENTS.md) for more details.

Please check the paper for a complete list of references and datasets used in this work.
