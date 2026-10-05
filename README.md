# 🔬 NanoDes: RepAF-Denoise Net

[![GitHub Repository](https://img.shields.io/badge/GitHub-kilugit%2Fnanodes-181717.svg?logo=github)](https://github.com/kilugit/nanodes)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **NanoDes** is an ultra-fast, lightweight image denoising and artifact-removal model architecture designed for edge devices and real-time processing. Powered by **RepAF-Denoise Net** (Re-parameterized Asymmetric Feature Denoise Net) with 2D Haar Wavelet Transforms and Multi-Branch Structural Re-parameterization.


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
This project is open-source and licensed under the [Apache 2.0 License](LICENSE).
