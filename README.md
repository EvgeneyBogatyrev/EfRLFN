# EfRLFN: Efficient Real-time Super-Resolution

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation and inference code for **EfRLFN**, a deep learning model for single image super-resolution. Our paper has been accepted to **ICLR 2026** 🎉

> **Note:** This repository is the official implementation of the ICLR 2026 paper *"Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content"*. If you use this code, please cite our paper (see [Citation](#citation)).

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Inference](#inference)
  - [Arguments](#arguments)
  - [Example](#example)
- [Model Weights](#model-weights)
- [Dataset](#dataset)
- [License](#license)
- [Citation](#citation)

## Features
- ✅ Efficient residual architecture for high-quality image upscaling
- ✅ Supports multiple upscaling factors (2×, 4×, etc.)
- ✅ Fast inference on CUDA-enabled GPUs
- ✅ Lightweight and real-time performance
- ✅ Pretrained weights available

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/EfRLFN.git
cd EfRLFN
pip install -r requirements.txt
```


**Requirements** (see `requirements.txt`):
- PyTorch ≥ 2.0
- torchvision
- opencv-python
- numpy
- Pillow

## Usage

### Inference

To upscale a single image using a pretrained model:

```python
python inference.py -w [WEIGHTS_PATH] -s [SCALE_FACTOR] -i [INPUT_IMAGE] -o [OUTPUT_IMAGE]
```

### Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--weights` | `-w` | Path to the pretrained model weights (`.pt` or `.ckpt` format) |
| `--scale`  | `-s` | Upscaling factor (e.g., `2`, `4`) |
| `--input`  | `-i` | Path to input image |
| `--output` | `-o` | Path to save the output image |

### Example
```python
python inference.py -w weights/EfRLFN-4x-model.pt -s 4 -i images/low_res.jpg -o images/high_res.jpg
```

## Model Weights

Pretrained weights for different scale factors can be downloaded from the links below:

| Scale | Download Link |
|-------|----------------|
| 2×    | [EfRLFN x2](https://drive.google.com/file/d/1VeoW94hN1X-8kxGXQSyR53YzRqF1htKQ/view?usp=sharing) |
| 4×    | [EfRLFN x4](https://drive.google.com/file/d/1vJgrsz62IAMeS9i2ChDhQGO6UO1ZUXhr/view?usp=sharing) |

Place the downloaded weights in the `weights/` directory (create it if missing).

## Dataset

The dataset used for training and evaluation is publicly available:

| Dataset | Link |
|---------|------|
| Train   | [Download train dataset](https://drive.google.com/file/d/1yPSpvyIn-g1xywUmZXeurbUCHD5ptNE5/view?usp=sharing) |
| Test    | [Download test dataset](https://drive.google.com/file/d/1CWqsZxyn31_m67n8TFx8K-Nt9YzreGfj/view?usp=sharing) |

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
You are free to use, modify, and distribute this code for academic and commercial purposes, provided that the original copyright notice is retained.

## Citation

If you find this work useful for your research, please cite our ICLR 2026 paper:

ArXiv:
```bibtex
@article{bogatyrev2026exploring,
  title={Exploring Real-Time Super-Resolution: Benchmarking and Fine-Tuning for Streaming Content},
  author={Bogatyrev, Evgeney and Abud, Khaled and Molodetskikh, Ivan and Alutis, Nikita and Vatolin, Dmitriy},
  journal={arXiv preprint arXiv:2602.11339},
  year={2026}
}
```

ICLR:<br>
TODO

