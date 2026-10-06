# 🔬 NanoDes: RepAF-Denoise Net

[![GitHub Repository](https://img.shields.io/badge/GitHub-kilugit%2Fnanodes-181717.svg?logo=github)](https://github.com/kilugit/nanodes)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **NanoDes** is an lightweight image denoising and artifact-removal model. Powered by **RepAF-Denoise Net** (Re-parameterized Asymmetric Feature Denoise Net) with 2D Haar Wavelet Transforms and Multi-Branch Structural Re-parameterization.


---

## 💻 Local Setup & Installation

### 1. Prerequisites
- **Python:** 3.12
- **Git**

---

### 2. Environment Setup & Dependency Installation

#### A. Clone & Create Virtual Environment
```bash
# Clone repository
git clone https://github.com/kilugit/nanodes.git
cd nanodes

# Create virtual environment with Python 3.12:
py -3.12 -m venv venv

# Activate virtual environment:
# On Windows (PowerShell / Command Prompt):
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
```

#### B. Install Core Dependencies
Install packages from `requirements.txt` (note: this repository uses `onnxruntime-windowsml` instead of `onnxruntime-directml`):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### C. Hardware Acceleration (PyTorch & ROCm / CUDA)
Install the optimized PyTorch and ONNX Runtime backend for your hardware:

- **AMD GPU (Native ROCm on Windows):**
  Note: (device-gfx1200)=RX 9060XT.
  ```bash
  python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,device-gfx1200]==10.0.0"
  python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1200]==2.13.0+rocm10.0.0" "torchvision[device-gfx1200]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0"
  ```

- **ONNX Runtime (Windows ML):**
  This repository uses `onnxruntime-windowsml` (latest version, providing `DmlExecutionProvider` and `CPUExecutionProvider`):
  ```bash
  pip install onnxruntime-windowsml
  ```

- **NVIDIA GPU (CUDA):**
  ```bash
  # PyTorch with CUDA 12.1:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

  # ONNX Runtime with CUDA / TensorRT:
  pip install onnxruntime-gpu
  ```

- **CPU-Only / Headless Server (No Desktop GUI):**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install onnx onnxruntime pillow numpy matplotlib
  ```

#### D. Verify Installation
Run this quick check to confirm your environment and acceleration status:
```bash
python -c "import torch, onnxruntime as ort; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}'); print(f'ONNX Runtime: {ort.__version__} | Providers: {ort.get_available_providers()}')"
```

---

### 3. Launch Local Desktop GUI
```bash
# Main Denoise & Training GUI
run.bat        # or: python gui.py

# ONNX Export & Quantization GUI
run_export.bat # or: python export_gui.py
```

### 4. Command Line Interface (CLI)
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
├── requirements.txt            # Python dependencies specification
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
