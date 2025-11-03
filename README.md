# EfRLFN real-time Super-Resolution

This repository contains the implementation and inference code for EfRLFN, a deep learning model for single image super-resolution.

## Features
- Efficient residual architecture for high-quality image upscaling
- Supports multiple upscaling factors
- Fast inference on CUDA-enabled GPUs

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Inference

To upscale an image using a trained model:
```bash
python inference.py -w [WEIGHTS_PATH] -s [SCALE_FACTOR] -i [INPUT_IMAGE] -o [OUTPUT_IMAGE]
```
### Arguments:

    -w/--weights: Path to the pretrained model weights (.pt or .ckpt format)

    -s/--scale: Upscaling factor (e.g., 2, 4)

    -i/--input: Path to input image

    -o/--output: Path to save the output image

### Example
```bash
python inference.py -w weights/EfRLFN-4x-model.ph -s 4 -i images/low_res.jpg -o images/high_res.jpg
```
## Model Weights

Pretrained weights are available for different scale factors:

[EfRLFN x2](https://drive.google.com/file/d/1VeoW94hN1X-8kxGXQSyR53YzRqF1htKQ/view?usp=sharingm)

[EfRLFN x4](https://drive.google.com/file/d/1vJgrsz62IAMeS9i2ChDhQGO6UO1ZUXhr/view?usp=sharing)

## Dataset
The proposed dataset can be downloaded here:

[Train dataset](https://drive.google.com/file/d/1yPSpvyIn-g1xywUmZXeurbUCHD5ptNE5/view?usp=sharing)

[Test dataset](https://drive.google.com/file/d/1CWqsZxyn31_m67n8TFx8K-Nt9YzreGfj/view?usp=sharing)