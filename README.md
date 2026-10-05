# 🔬 NanoDes: RepAF-Denoise Net

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kilugit/nanodes/blob/main/NanoDes_Colab.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-kilugit%2Fnanodes-181717.svg?logo=github)](https://github.com/kilugit/nanodes)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **NanoDes** is an ultra-fast, lightweight image denoising and artifact-removal model architecture designed for edge devices and real-time processing. Powered by **RepAF-Denoise Net** (Re-parameterized Asymmetric Feature Denoise Net) with 2D Haar Wavelet Transforms and Multi-Branch Structural Re-parameterization.

---

## ⚡ Quickstart on Google Colab

Run everything in the cloud with free GPU acceleration (no local setup required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kilugit/nanodes/blob/main/NanoDes_Colab.ipynb)

The [`NanoDes_Colab.ipynb`](NanoDes_Colab.ipynb) notebook replaces both desktop GUIs (`gui.py` and `export_gui.py`) with an interactive, form-enabled interface:
- **Interactive Denoising & Inference**: Test on sample images or upload your own photos; simulate progressive JPEG compression and sensor noise.
- **Tiled Processing**: Hann-window weighted tiling handles 4K/8K images without VRAM limits.
- **ONNX Export Suite**: Full structural re-parameterization, FP32/FP16 export, and QDQ static INT8 quantization.
- **Training Pipeline**: Two-stage progressive training with live metric curves.

---

## 🏗️ Model Architecture & Key Innovations

1. **2D Haar Wavelet Decomposition**:
   - Parameter-free Discrete Wavelet Transform (`haar_dwt`) splits feature spaces into 4 frequency sub-bands (LL, LH, HL, HH) to isolate high-frequency noise from structural edges.
2. **Structural Re-parameterization (`RepConv2d`)**:
   - **Training mode**: Multi-branch topology ($3\times 3 + 1\times 1 + \text{Identity}$) for superior gradient flow.
   - **Deployment mode**: Mathematically collapsed into a single $3\times 3$ convolution kernel with zero inference overhead:
     $$W_{\text{deploy}} = W_{3\times 3} + \text{pad}(W_{1\times 1}) + \text{pad}(I)$$
3. **Two-Stage Progressive Training**:
   - **Stage 1**: Progressive patch scaling ($128 \times 128 \to 256 \times 256$) with `Stage1Loss` (Charbonnier + Sobel gradient loss).
   - **Stage 2**: Perceptual fine-tuning with $512 \times 512$ patches and direct `PSNRLoss`.

---

## 🚀 Performance Benchmarks

Inference latency benchmark on standard $512 \times 512$ input:

| Backend / Format | Precision | Latency (ms) | Speedup vs Baseline | Output Status |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch (Multi-branch)** | FP32 | ~110 ms | 1.00x *(Baseline)* | Training Architecture |
| **PyTorch (Deploy Fused)** | FP32 | ~58 ms | **1.90x** | Fused Single-Conv |
| **ONNX Runtime (CPU)** | FP32 | ~35 ms | **3.14x** | Optimized Graph |
| **ONNX Runtime (INT8 QDQ)** | INT8 | ~18 ms | **6.11x** | Static Calibrated |

---

## 💻 Local Setup & Execution

### 1. Requirements
```bash
git clone https://github.com/kilugit/nanodes.git
cd nanodes
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install torch torchvision onnx onnxruntime pillow matplotlib PyQt6
```

### 2. Launch Local Desktop GUI
```bash
# Main Denoise & Training GUI
run.bat        # or: python gui.py

# ONNX Export & Quantization GUI
run_export.bat # or: python export_gui.py
```

### 3. Command Line Interface (CLI)
```bash
# Verify structural re-parameterization equivalence
python -c "from models import test_reparameterization_equivalence; test_reparameterization_equivalence('cpu')"

# Export to FP32 ONNX and INT8 QDQ ONNX
python export.py --model "pytorch models/best_model.pth" --test --int8

# Run CLI denoising inference
python -c "from inference import run_denoise; img, ms = run_denoise('samples/image-544.png', 'onnx models/repaf_denoise_net.onnx', tiled=True); img.save('outputs/result.png')"

# Run two-stage training
python train.py --stage1-epochs 100 --stage2-epochs 30 --batch-size-stage1 32 --batch-size-stage2 8
```

---

## 📂 Project Repository Structure

```
nanodes/
├── NanoDes_Colab.ipynb         # Interactive Google Colab Notebook
├── models.py                   # RepAFDenoiseNet, RepConv2d, Haar DWT/IWT
├── inference.py                # Tiled & full inference (PyTorch & ONNX)
├── export.py                   # ONNX FP32/FP16 & INT8 QDQ exporter
├── export_gui.py               # PyQt6 Model Export & Quantization GUI
├── gui.py                      # PyQt6 Main Desktop GUI
├── train.py                    # Two-stage progressive training pipeline
├── dataset.py                  # DenoisingDataset & memory-efficient caching
├── losses.py                   # Stage1Loss (Charbonnier + Sobel) & PSNRLoss
├── metrics.py                  # Analytical PSNR & SSIM evaluation
├── process_samples.py          # Synthetic degradation & crop generator
├── pytorch models/             # Saved PyTorch checkpoints (.pth)
├── onnx models/                # Exported ONNX graphs (.onnx)
└── samples/                    # High-resolution sample images
```

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
