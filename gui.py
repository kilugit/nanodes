import io
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from dataset import DenoisingDataset, set_max_cache_items
from inference import run_denoise
from models import RepAFDenoiseNet, load_pretrained_weights
from process_samples import (
    add_gaussian_noise,
    apply_blur,
    apply_downsample_upsample,
    apply_jpeg_compression,
    process_all_samples,
)

# Startup directories verification
REQUIRED_DIRS = ["samples", "outputs", "pytorch models", "onnx models"]
for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)


def pil_to_qpixmap(pil_img: Image.Image, max_size: Tuple[int, int] = (900, 650)) -> QPixmap:
    w, h = pil_img.size
    if w > max_size[0] or h > max_size[1]:
        scale = min(max_size[0] / w, max_size[1] / h)
        preview = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    else:
        preview = pil_img

    if preview.mode == "RGBA":
        data = preview.tobytes("raw", "RGBA")
        qimg = QImage(data, preview.width, preview.height, preview.width * 4, QImage.Format.Format_RGBA8888)
    else:
        if preview.mode != "RGB":
            preview = preview.convert("RGB")
        data = preview.tobytes("raw", "RGB")
        qimg = QImage(data, preview.width, preview.height, preview.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# -----------------------------------------------------------------------------
# Background Worker Threads
# -----------------------------------------------------------------------------
class InferenceWorker(QThread):
    finished = pyqtSignal(object, float)
    failed = pyqtSignal(str)

    def __init__(self, image: Image.Image, model_path: str, tiled: bool, tile_size: int, device: str):
        super().__init__()
        self.image = image
        self.model_path = model_path
        self.tiled = tiled
        self.tile_size = tile_size
        self.device = device

    def run(self):
        try:
            denoised_img, elapsed_ms = run_denoise(
                self.image,
                self.model_path,
                tiled=self.tiled,
                tile_size=self.tile_size,
                device=self.device,
            )
            self.finished.emit(denoised_img, elapsed_ms)
        except Exception as e:
            self.failed.emit(str(e))


class DataGenWorker(QThread):
    progress = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(
        self,
        samples_dir: str,
        clean_dir: str,
        noisy_dir: str,
        qualities: Optional[list] = None,
        quality_range: Optional[tuple] = None,
        num_random_versions: int = 0,
        num_crops: int = 3,
        p_blur: float = 0.5,
        blur_type: str = "random",
        blur_range: Tuple[float, float] = (0.5, 2.5),
    ):
        super().__init__()
        self.samples_dir = samples_dir
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.qualities = qualities
        self.quality_range = quality_range
        self.num_random_versions = num_random_versions
        self.num_crops = num_crops
        self.p_blur = p_blur
        self.blur_type = blur_type
        self.blur_range = blur_range

    def run(self):
        try:
            self.log_signal.emit("Scanning 'samples' folder for high-resolution sources...")
            count = process_all_samples(
                samples_dir=self.samples_dir,
                output_clean_dir=self.clean_dir,
                output_noisy_dir=self.noisy_dir,
                jpeg_qualities=self.qualities,
                quality_range=self.quality_range,
                num_random_versions=self.num_random_versions,
                num_crops=self.num_crops,
                p_blur=self.p_blur,
                blur_type=self.blur_type,
                blur_range=self.blur_range,
                progress_callback=lambda c: self.progress.emit(c),
            )
            self.finished.emit(count)
        except Exception as e:
            self.log_signal.emit(f"Error during data generation: {e}")
            self.finished.emit(0)


class TrainingWorker(QThread):
    epoch_progress = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        try:
            import torch
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            from torch.utils.data import DataLoader
            from devices import clear_device_cache, create_grad_scaler, get_amp_context, get_device, get_device_name
            from losses import Stage1Loss, PSNRLoss, Stage2SharpLoss
            from metrics import calculate_psnr, calculate_ssim
            from train import get_progressive_patch_size

            device = get_device(self.config.get("device", "auto"))
            self.log_signal.emit(f"Initializing RepAF-Denoise Net on {device} ({get_device_name(device)})...")
            model = RepAFDenoiseNet(c=40).to(device)

            num_workers = self.config.get("num_workers", 0)
            eval_interval = self.config.get("eval_interval", 5)
            preload_ram = self.config.get("preload_ram", False)
            set_max_cache_items(self.config.get("max_cache_items", 2048))

            train_dataset = DenoisingDataset(
                noisy_dir=self.config.get("train_noisy_dir"),
                clean_dir=self.config.get("train_clean_dir"),
                patch_size=128,
                is_train=True,
                num_synthetic_samples=self.config.get("synthetic_samples", 64),
                cache=not preload_ram,
                preload_to_ram=preload_ram,
            )
            val_dataset = DenoisingDataset(
                noisy_dir=self.config.get("val_noisy_dir"),
                clean_dir=self.config.get("val_clean_dir"),
                patch_size=256,
                is_train=False,
                num_synthetic_samples=32,
                cache=not preload_ram,
                preload_to_ram=preload_ram,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=16,
                shuffle=False,
                pin_memory=(device.type in ("cuda", "xpu")),
            )

            best_psnr = -float("inf")
            save_dir = "pytorch models"
            os.makedirs(save_dir, exist_ok=True)
            bf16 = self.config.get("bf16", False)
            fp16 = self.config.get("fp16", False)
            amp_dtype = torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
            amp_enabled = (fp16 or bf16) and device.type in ("cuda", "xpu", "cpu")
            scaler = create_grad_scaler(device, enabled=(fp16 and device.type in ("cuda", "xpu")))

            is_finetune = self.config.get("is_finetune", False)
            pretrained_path = self.config.get("pretrained_path")
            if is_finetune and pretrained_path and os.path.exists(pretrained_path):
                ckpt_info = load_pretrained_weights(model, pretrained_path, device=device)
                best_psnr = ckpt_info.get("best_psnr", -float("inf"))
                self.log_signal.emit(f"Loaded pre-trained weights from: {pretrained_path}")

            if self.config.get("freeze_stem", False):
                for p in model.stem.parameters():
                    p.requires_grad = False
                self.log_signal.emit("Stem feature extractor frozen.")

            if is_finetune:
                self.log_signal.emit("Evaluating baseline validation performance...")
                model.eval()
                total_psnr = torch.zeros(1, device=device)
                total_ssim = torch.zeros(1, device=device)
                count = 0
                with torch.inference_mode():
                    for noisy, clean in val_loader:
                        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                        with get_amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
                            pred = torch.clamp(model(noisy), 0.0, 1.0)
                        bs = noisy.size(0)
                        total_psnr += calculate_psnr(pred, clean, as_tensor=True) * bs
                        total_ssim += calculate_ssim(pred, clean, as_tensor=True) * bs
                        count += bs
                base_psnr = (total_psnr / count).item() if count > 0 else 0.0
                base_ssim = (total_ssim / count).item() if count > 0 else 0.0
                if base_psnr > best_psnr:
                    best_psnr = base_psnr
                self.log_signal.emit(f"Pre-trained Baseline -> Val PSNR: {base_psnr:.2f} dB | SSIM: {base_ssim:.4f}")
            else:
                # Stage 1
                s1_epochs = self.config["stage1_epochs"]
                self.log_signal.emit(f"\n--- Starting Stage 1 ({s1_epochs} epochs) ---")
                optimizer_s1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
                scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=s1_epochs, eta_min=1e-6)
                criterion_s1 = Stage1Loss(eps=1e-3, lambda_grad=0.05).to(device)

                loader_s1 = DataLoader(
                    train_dataset,
                    batch_size=self.config["batch_size_stage1"],
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=(device.type in ("cuda", "xpu")),
                    persistent_workers=(num_workers > 0),
                    drop_last=(len(train_dataset) >= self.config["batch_size_stage1"]),
                )

                for epoch in range(1, s1_epochs + 1):
                    if self._is_stopped:
                        self.log_signal.emit("Training cancelled by user.")
                        self.finished.emit("Cancelled")
                        return

                    patch_size = get_progressive_patch_size(epoch - 1, s1_epochs, min_size=128, max_size=256)
                    train_dataset.set_patch_size(patch_size)

                    model.train()
                    epoch_loss = 0.0
                    valid_count = 0
                    for noisy, clean in loader_s1:
                        if self._is_stopped:
                            break
                        noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                        optimizer_s1.zero_grad(set_to_none=True)
                        with get_amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
                            loss = criterion_s1(model(noisy), clean)
                        if not torch.isfinite(loss) or loss.abs().item() > 100.0:
                            continue
                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer_s1)
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            if torch.isfinite(grad_norm):
                                scaler.step(optimizer_s1)
                                scaler.update()
                                epoch_loss += loss.item() * noisy.size(0)
                                valid_count += noisy.size(0)
                            else:
                                optimizer_s1.zero_grad(set_to_none=True)
                                scaler.update()
                        else:
                            loss.backward()
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            if torch.isfinite(grad_norm):
                                optimizer_s1.step()
                                epoch_loss += loss.item() * noisy.size(0)
                                valid_count += noisy.size(0)
                            else:
                                optimizer_s1.zero_grad(set_to_none=True)

                    if self._is_stopped:
                        self.log_signal.emit("Training cancelled by user.")
                        self.finished.emit("Cancelled")
                        return

                    scheduler_s1.step()
                    mean_loss = epoch_loss / valid_count if valid_count > 0 else float("nan")

                    should_eval = (epoch % eval_interval == 0) or (epoch == s1_epochs)
                    if should_eval:
                        model.eval()
                        total_psnr = torch.zeros(1, device=device)
                        total_ssim = torch.zeros(1, device=device)
                        count = 0
                        with torch.inference_mode():
                            for noisy, clean in val_loader:
                                if self._is_stopped:
                                    self.log_signal.emit("Training cancelled by user.")
                                    self.finished.emit("Cancelled")
                                    return
                                noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                                with get_amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
                                    pred = torch.clamp(model(noisy), 0.0, 1.0)
                                if not torch.isfinite(pred).all():
                                    continue
                                bs = noisy.size(0)
                                total_psnr += calculate_psnr(pred, clean, as_tensor=True) * bs
                                total_ssim += calculate_ssim(pred, clean, as_tensor=True) * bs
                                count += bs
                        val_psnr = (total_psnr / count).item() if count > 0 else 0.0
                        val_ssim = (total_ssim / count).item() if count > 0 else 0.0

                        is_best = val_psnr > best_psnr
                        if is_best:
                            best_psnr = val_psnr
                            torch.save(
                                {"state_dict": model.state_dict(), "best_psnr": best_psnr, "stage": 1},
                                os.path.join(save_dir, "best_model.pth"),
                            )
                        val_str = f"Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}"
                    else:
                        val_str = "Val: (skipped)"

                    self.epoch_progress.emit(epoch, s1_epochs + self.config["stage2_epochs"])
                    self.log_signal.emit(
                        f"S1 [Epoch {epoch:03d}/{s1_epochs:03d}] Patch: {patch_size}x{patch_size} | "
                        f"Loss: {mean_loss:.4f} | {val_str}"
                    )

                del optimizer_s1, scheduler_s1, criterion_s1, loader_s1
                clear_device_cache(device)

                best_ckpt = os.path.join(save_dir, "best_model.pth")
                if os.path.exists(best_ckpt):
                    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt["state_dict"])
                    best_psnr = ckpt.get("best_psnr", best_psnr)

            # Stage 2 / Fine-Tuning
            s2_epochs = self.config.get("finetune_epochs", 30) if is_finetune else self.config["stage2_epochs"]
            s2_lr = self.config.get("finetune_lr", 5e-5) if is_finetune else 1e-4
            stage_tag = "Fine-Tune" if is_finetune else "S2"
            total_epochs = s2_epochs if is_finetune else s1_epochs + s2_epochs

            self.log_signal.emit(f"\n--- Starting {stage_tag} ({s2_epochs} epochs) ---")
            train_dataset.set_patch_size(256)
            loader_s2 = DataLoader(
                train_dataset,
                batch_size=self.config["batch_size_stage2"],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device.type in ("cuda", "xpu")),
                persistent_workers=(num_workers > 0),
                drop_last=(len(train_dataset) >= self.config["batch_size_stage2"]),
            )
            optimizer_s2 = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=s2_lr, betas=(0.9, 0.999), weight_decay=1e-4)
            scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=s2_epochs, eta_min=1e-6)
            criterion_s2 = Stage2SharpLoss(eps=1e-6).to(device)

            for epoch in range(1, s2_epochs + 1):
                if self._is_stopped:
                    self.log_signal.emit("Training cancelled by user.")
                    self.finished.emit("Cancelled")
                    return

                model.train()
                epoch_loss = 0.0
                valid_count = 0
                for noisy, clean in loader_s2:
                    if self._is_stopped:
                        break
                    noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                    optimizer_s2.zero_grad(set_to_none=True)
                    with get_amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
                        loss = criterion_s2(model(noisy), clean)
                    if not torch.isfinite(loss) or loss.abs().item() > 100.0:
                        continue
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer_s2)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if torch.isfinite(grad_norm):
                            scaler.step(optimizer_s2)
                            scaler.update()
                            epoch_loss += loss.item() * noisy.size(0)
                            valid_count += noisy.size(0)
                        else:
                            optimizer_s2.zero_grad(set_to_none=True)
                            scaler.update()
                    else:
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if torch.isfinite(grad_norm):
                            optimizer_s2.step()
                            epoch_loss += loss.item() * noisy.size(0)
                            valid_count += noisy.size(0)
                        else:
                            optimizer_s2.zero_grad(set_to_none=True)

                if self._is_stopped:
                    self.log_signal.emit("Training cancelled by user.")
                    self.finished.emit("Cancelled")
                    return

                scheduler_s2.step()
                mean_loss = epoch_loss / valid_count if valid_count > 0 else float("nan")

                should_eval = (epoch % eval_interval == 0) or (epoch == s2_epochs)
                if should_eval:
                    model.eval()
                    total_psnr = torch.zeros(1, device=device)
                    total_ssim = torch.zeros(1, device=device)
                    count = 0
                    with torch.inference_mode():
                        for noisy, clean in val_loader:
                            if self._is_stopped:
                                self.log_signal.emit("Training cancelled by user.")
                                self.finished.emit("Cancelled")
                                return
                            noisy, clean = noisy.to(device, non_blocking=True), clean.to(device, non_blocking=True)
                            with get_amp_context(device, enabled=amp_enabled, dtype=amp_dtype):
                                pred = torch.clamp(model(noisy), 0.0, 1.0)
                            if not torch.isfinite(pred).all():
                                continue
                            bs = noisy.size(0)
                            total_psnr += calculate_psnr(pred, clean, as_tensor=True) * bs
                            total_ssim += calculate_ssim(pred, clean, as_tensor=True) * bs
                            count += bs
                    val_psnr = (total_psnr / count).item() if count > 0 else 0.0
                    val_ssim = (total_ssim / count).item() if count > 0 else 0.0

                    is_best = val_psnr > best_psnr
                    if is_best:
                        best_psnr = val_psnr
                        torch.save(
                            {"state_dict": model.state_dict(), "best_psnr": best_psnr, "stage": 2 if not is_finetune else "finetune"},
                            os.path.join(save_dir, "best_model.pth"),
                        )
                        if is_finetune:
                            torch.save(
                                {"state_dict": model.state_dict(), "best_psnr": best_psnr, "stage": "finetune"},
                                os.path.join(save_dir, "finetuned_best.pth"),
                            )
                    val_str = f"Val PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}"
                else:
                    val_str = "Val: (skipped)"

                current_epoch = epoch if is_finetune else s1_epochs + epoch
                self.epoch_progress.emit(current_epoch, total_epochs)
                self.log_signal.emit(
                    f"{stage_tag} [Epoch {epoch:02d}/{s2_epochs:02d}] Patch: 256x256 | "
                    f"PSNR Loss: {mean_loss:.4f} | {val_str}"
                )

            # Export deployed models for immediate inference use
            try:
                from export import _prepare_cpu_deploy_model, export_onnx
                best_ckpt = os.path.join(save_dir, "best_model.pth")
                if os.path.exists(best_ckpt):
                    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
                    model.load_state_dict(ckpt["state_dict"])
                deploy_model = _prepare_cpu_deploy_model(model)
                torch.save(deploy_model.state_dict(), os.path.join(save_dir, "repaf_denoise_net_deployed.pth"))
                export_onnx(deploy_model, save_path="onnx models/repaf_denoise_net.onnx")
                self.log_signal.emit("Auto-exported deployed PyTorch and ONNX models for inference.")
            except Exception as exp_err:
                self.log_signal.emit(f"Note: Deployed export: {exp_err}")

            self.log_signal.emit(f"\nTraining Complete! Best PSNR: {best_psnr:.2f} dB")
            self.finished.emit("Success")
        except Exception as e:
            self.log_signal.emit(f"Error: {e}")
            self.finished.emit(f"Error: {e}")
        finally:
            from dataset import clear_image_cache
            clear_image_cache()
            clear_device_cache(device if "device" in locals() else None)


# -----------------------------------------------------------------------------
# Main Application GUI
# -----------------------------------------------------------------------------
class RepAFDenoiseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NanoDes")
        self.resize(1300, 850)

        self.loaded_image: Optional[Image.Image] = None
        self.noisy_image: Optional[Image.Image] = None
        self.denoised_image: Optional[Image.Image] = None
        self.image_path: Optional[str] = None

        self._init_ui()
        self._apply_theme()
        self._refresh_models()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_inference = QWidget()
        self.tab_training = QWidget()
        self.tab_datagen = QWidget()

        self.tabs.addTab(self.tab_inference, "Denoise & Inference")
        self.tabs.addTab(self.tab_training, "Model Training")
        self.tabs.addTab(self.tab_datagen, "Sample Data Processing")

        self._setup_inference_tab()
        self._setup_training_tab()
        self._setup_datagen_tab()

        # Status Bar
        self.status_label = QLabel("System ready. Directories verified: samples/, outputs/, pytorch models/, onnx models/")
        self.statusBar().addWidget(self.status_label)

    # -------------------------------------------------------------------------
    # Tab 1: Inference & Interactive Denoising
    # -------------------------------------------------------------------------
    def _setup_inference_tab(self):
        layout = QHBoxLayout(self.tab_inference)

        # Control Panel (Left)
        control_panel = QWidget()
        control_panel.setFixedWidth(340)
        c_layout = QVBoxLayout(control_panel)
        c_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Model Selection Box
        box_model = QGroupBox("Model & Execution Backend")
        b_layout = QFormLayout(box_model)

        self.combo_model = QComboBox()
        self.btn_refresh_models = QPushButton("Refresh Models")
        self.btn_refresh_models.clicked.connect(self._refresh_models)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["Auto", "CUDA / ROCm", "Intel XPU", "CPU"])

        self.chk_tiled = QCheckBox("Tiled Mode")
        self.combo_tile_size = QComboBox()
        self.combo_tile_size.addItems(["512", "1024", "2048"])
        self.combo_tile_size.setCurrentIndex(1)

        b_layout.addRow("Model:", self.combo_model)
        b_layout.addRow("", self.btn_refresh_models)
        b_layout.addRow("Backend:", self.combo_device)
        b_layout.addRow(self.chk_tiled)
        b_layout.addRow("Tile Size:", self.combo_tile_size)
        c_layout.addWidget(box_model)

        # 2. Image Loading & Test Artifact Injection
        box_img = QGroupBox("Image Source & Simulation")
        i_layout = QVBoxLayout(box_img)

        self.btn_open_img = QPushButton("Open Image File...")
        self.btn_open_img.clicked.connect(self._open_image_dialog)

        self.combo_sample_img = QComboBox()
        self._refresh_sample_images()
        self.btn_load_sample = QPushButton("Load Selected Sample")
        self.btn_load_sample.clicked.connect(self._load_sample_image)

        # Artifact Simulation tool
        lbl_sim = QLabel("Simulate Degradations & Noise:")
        self.slider_quality = QSlider(Qt.Orientation.Horizontal)
        self.slider_quality.setRange(10, 95)
        self.slider_quality.setValue(35)
        self.lbl_quality_val = QLabel("JPEG Quality: 35")
        self.slider_quality.valueChanged.connect(
            lambda v: self.lbl_quality_val.setText(f"JPEG Quality: {v}")
        )

        self.slider_blur = QSlider(Qt.Orientation.Horizontal)
        self.slider_blur.setRange(0, 50)
        self.slider_blur.setValue(0)
        self.lbl_blur_val = QLabel("Blur Radius: 0.0")
        self.slider_blur.valueChanged.connect(
            lambda v: self.lbl_blur_val.setText(f"Blur Radius: {v / 10.0:.1f}")
        )

        self.btn_inject_noise = QPushButton("Inject Degradation to Image")
        self.btn_inject_noise.clicked.connect(self._inject_artifacts_to_image)

        i_layout.addWidget(self.btn_open_img)
        i_layout.addWidget(QLabel("Or choose from 'samples/':"))
        i_layout.addWidget(self.combo_sample_img)
        i_layout.addWidget(self.btn_load_sample)
        i_layout.addSpacing(8)
        i_layout.addWidget(lbl_sim)
        i_layout.addWidget(self.lbl_quality_val)
        i_layout.addWidget(self.slider_quality)
        i_layout.addWidget(self.lbl_blur_val)
        i_layout.addWidget(self.slider_blur)
        i_layout.addWidget(self.btn_inject_noise)
        c_layout.addWidget(box_img)

        # 3. Action Buttons
        self.btn_run_denoise = QPushButton("Denoise Image")
        self.btn_run_denoise.setFixedHeight(45)
        self.btn_run_denoise.setStyleSheet("font-size: 14px; font-weight: bold; background-color: #2e7d32; color: white;")
        self.btn_run_denoise.clicked.connect(self._execute_denoise)

        self.btn_save_output = QPushButton("Save Output (PNG)")
        self.btn_save_output.clicked.connect(self._save_output_image)

        c_layout.addWidget(self.btn_run_denoise)
        c_layout.addWidget(self.btn_save_output)
        c_layout.addStretch()

        # Display Panel (Right: Side-by-Side Images)
        display_panel = QWidget()
        d_layout = QHBoxLayout(display_panel)
        d_layout.setContentsMargins(5, 5, 5, 5)

        # Left View: Input Noisy
        left_box = QGroupBox("Input Image")
        left_v = QVBoxLayout(left_box)
        self.lbl_input_info = QLabel("No image loaded")
        self.lbl_input_info.setStyleSheet("color: #90caf9; font-weight: bold;")
        self.scroll_input = QScrollArea()
        self.scroll_input.setWidgetResizable(True)
        self.view_input = QLabel("Load an image to preview")
        self.view_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_input.setWidget(self.view_input)
        left_v.addWidget(self.lbl_input_info)
        left_v.addWidget(self.scroll_input)

        # Right View: Output Denoised
        right_box = QGroupBox("Denoised Output")
        right_v = QVBoxLayout(right_box)
        self.lbl_output_info = QLabel("Denoising not yet run")
        self.lbl_output_info.setStyleSheet("color: #a5d6a7; font-weight: bold;")
        self.scroll_output = QScrollArea()
        self.scroll_output.setWidgetResizable(True)
        self.view_output = QLabel("Denoised result will appear here")
        self.view_output.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_output.setWidget(self.view_output)
        right_v.addWidget(self.lbl_output_info)
        right_v.addWidget(self.scroll_output)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_box)
        splitter.addWidget(right_box)
        d_layout.addWidget(splitter)

        layout.addWidget(control_panel)
        layout.addWidget(display_panel)

    # -------------------------------------------------------------------------
    # Tab 2: Model Training
    # -------------------------------------------------------------------------
    def _setup_training_tab(self):
        layout = QVBoxLayout(self.tab_training)

        mode_box = QGroupBox("Fine-Tuning Mode")
        m_layout = QVBoxLayout(mode_box)

        self.chk_finetune_mode = QCheckBox("Fine-Tune Pre-trained Model (Skip Stage 1 coarse training)")
        self.chk_finetune_mode.setStyleSheet("font-weight: bold; color: #64b5f6;")
        self.chk_finetune_mode.toggled.connect(self._on_finetune_mode_toggled)

        ft_row = QHBoxLayout()
        ft_row.addWidget(QLabel("Checkpoint:"))
        self.combo_pretrained_model = QComboBox()
        self.btn_browse_pretrained = QPushButton("Browse...")
        self.btn_browse_pretrained.clicked.connect(self._browse_pretrained_model)
        self.btn_refresh_checkpoints = QPushButton("Refresh")
        self.btn_refresh_checkpoints.clicked.connect(self._refresh_checkpoint_models)
        ft_row.addWidget(self.combo_pretrained_model, 1)
        ft_row.addWidget(self.btn_browse_pretrained)
        ft_row.addWidget(self.btn_refresh_checkpoints)

        ft_opts_row = QHBoxLayout()
        self.spin_ft_epochs = QSpinBox()
        self.spin_ft_epochs.setRange(1, 10000)
        self.spin_ft_epochs.setValue(30)

        self.spin_ft_lr = QDoubleSpinBox()
        self.spin_ft_lr.setRange(0.000001, 0.01)
        self.spin_ft_lr.setDecimals(6)
        self.spin_ft_lr.setSingleStep(0.00001)
        self.spin_ft_lr.setValue(0.00005)

        self.chk_freeze_stem = QCheckBox("Freeze Stem")

        ft_opts_row.addWidget(QLabel("Fine-Tune Epochs:"))
        ft_opts_row.addWidget(self.spin_ft_epochs)
        ft_opts_row.addSpacing(15)
        ft_opts_row.addWidget(QLabel("Learning Rate:"))
        ft_opts_row.addWidget(self.spin_ft_lr)
        ft_opts_row.addSpacing(15)
        ft_opts_row.addWidget(self.chk_freeze_stem)
        ft_opts_row.addStretch()

        m_layout.addWidget(self.chk_finetune_mode)
        m_layout.addLayout(ft_row)
        m_layout.addLayout(ft_opts_row)
        layout.addWidget(mode_box)

        cfg_box = QGroupBox("Training Settings")
        grid = QGridLayout(cfg_box)

        self.spin_s1_epochs = QSpinBox()
        self.spin_s1_epochs.setRange(1, 100000)
        self.spin_s1_epochs.setValue(100)

        self.spin_s2_epochs = QSpinBox()
        self.spin_s2_epochs.setRange(1, 100000)
        self.spin_s2_epochs.setValue(30)

        self.spin_s1_batch = QSpinBox()
        self.spin_s1_batch.setRange(1, 1000000)
        self.spin_s1_batch.setValue(32)

        self.spin_s2_batch = QSpinBox()
        self.spin_s2_batch.setRange(1, 1000000)
        self.spin_s2_batch.setValue(16)

        grid.addWidget(QLabel("Stage 1 Epochs:"), 0, 0)
        grid.addWidget(self.spin_s1_epochs, 0, 1)
        grid.addWidget(QLabel("Stage 1 Batch Size:"), 0, 2)
        grid.addWidget(self.spin_s1_batch, 0, 3)

        grid.addWidget(QLabel("Stage 2 Epochs:"), 1, 0)
        grid.addWidget(self.spin_s2_epochs, 1, 1)
        grid.addWidget(QLabel("Stage 2 Batch Size:"), 1, 2)
        grid.addWidget(self.spin_s2_batch, 1, 3)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 16)
        self.spin_workers.setValue(2)

        self.spin_eval_interval = QSpinBox()
        self.spin_eval_interval.setRange(1, 100)
        self.spin_eval_interval.setValue(5)

        grid.addWidget(QLabel("Workers:"), 2, 0)
        grid.addWidget(self.spin_workers, 2, 1)
        grid.addWidget(QLabel("Eval Interval:"), 2, 2)
        grid.addWidget(self.spin_eval_interval, 2, 3)

        self.spin_max_cache = QSpinBox()
        self.spin_max_cache.setRange(0, 100000)
        self.spin_max_cache.setValue(2048)
        self.spin_max_cache.setSingleStep(256)
        self.spin_max_cache.valueChanged.connect(set_max_cache_items)

        self.chk_preload_ram = QCheckBox("Preload Data to RAM")
        self.chk_preload_ram.setChecked(False)
        self.chk_preload_ram.toggled.connect(lambda c: self.spin_max_cache.setEnabled(not c))

        grid.addWidget(QLabel("Max Cache Items:"), 3, 0)
        grid.addWidget(self.spin_max_cache, 3, 1)
        grid.addWidget(self.chk_preload_ram, 3, 2, 1, 2)

        self.chk_fp16 = QCheckBox("FP16 Mixed Precision")
        self.chk_fp16.setChecked(True)
        self.chk_bf16 = QCheckBox("BF16 Mixed Precision")
        self.chk_bf16.setChecked(False)
        self.chk_fp16.toggled.connect(lambda c: self.chk_bf16.setChecked(False) if c else None)
        self.chk_bf16.toggled.connect(lambda c: self.chk_fp16.setChecked(False) if c else None)
        grid.addWidget(self.chk_fp16, 4, 0)
        grid.addWidget(self.chk_bf16, 4, 1)

        self.combo_train_device = QComboBox()
        self.combo_train_device.addItems(["Auto", "CUDA / ROCm", "Intel XPU", "CPU"])
        grid.addWidget(QLabel("Device:"), 5, 0)
        grid.addWidget(self.combo_train_device, 5, 1)

        self.chk_shutdown = QCheckBox("Shutdown PC after training")
        grid.addWidget(self.chk_shutdown, 5, 2, 1, 2)

        layout.addWidget(cfg_box)

        # Buttons & Progress
        btn_box = QHBoxLayout()
        self.btn_start_train = QPushButton("Start Training Pipeline")
        self.btn_start_train.setStyleSheet("QPushButton:enabled { font-weight: bold; background-color: #1976d2; color: white; }")
        self.btn_start_train.clicked.connect(self._start_training)

        self.btn_stop_train = QPushButton("Stop Training")
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self._stop_training)

        btn_box.addWidget(self.btn_start_train)
        btn_box.addWidget(self.btn_stop_train)
        layout.addLayout(btn_box)

        self.train_progress = QProgressBar()
        layout.addWidget(self.train_progress)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setStyleSheet("background-color: #121212; color: #76ff03; font-family: Consolas, monospace;")
        layout.addWidget(self.train_log)

        self._refresh_checkpoint_models()
        self._on_finetune_mode_toggled(False)

    # -------------------------------------------------------------------------
    # Tab 3: Sample Data Generation
    # -------------------------------------------------------------------------
    def _setup_datagen_tab(self):
        layout = QVBoxLayout(self.tab_datagen)

        box_info = QGroupBox("Sample Processing Settings")
        v = QVBoxLayout(box_info)
        desc = QLabel(
            "Scans high-resolution images in 'samples/' and generates paired clean-noisy training\n"
            "data with non-aggressive cropping, progressive JPEG compression, and noise."
        )
        v.addWidget(desc)

        form = QFormLayout()
        self.spin_crops = QSpinBox()
        self.spin_crops.setRange(0, 10)
        self.spin_crops.setValue(3)

        self.edit_qualities = QLineEdit("30 50 75")
        self.edit_qualities.setPlaceholderText("e.g. 15 20 40 60 65 80")

        self.chk_random = QCheckBox("Random")
        self.spin_random_versions = QSpinBox()
        self.spin_random_versions.setRange(1, 50)
        self.spin_random_versions.setValue(5)

        self.spin_q_min = QSpinBox()
        self.spin_q_min.setRange(1, 100)
        self.spin_q_min.setValue(20)

        self.spin_q_max = QSpinBox()
        self.spin_q_max.setRange(1, 100)
        self.spin_q_max.setValue(80)

        random_layout = QHBoxLayout()
        random_layout.addWidget(self.chk_random)
        random_layout.addSpacing(12)
        random_layout.addWidget(QLabel("Versions:"))
        random_layout.addWidget(self.spin_random_versions)
        random_layout.addSpacing(12)
        random_layout.addWidget(QLabel("Range:"))
        random_layout.addWidget(self.spin_q_min)
        random_layout.addWidget(QLabel("to"))
        random_layout.addWidget(self.spin_q_max)
        random_layout.addStretch()

        self.chk_random.toggled.connect(self._on_random_quality_toggled)
        self._on_random_quality_toggled(False)

        self.chk_blur = QCheckBox("Enable Blur")
        self.chk_blur.setChecked(True)
        self.spin_p_blur = QDoubleSpinBox()
        self.spin_p_blur.setRange(0.0, 1.0)
        self.spin_p_blur.setSingleStep(0.05)
        self.spin_p_blur.setValue(0.50)

        self.combo_blur_type = QComboBox()
        self.combo_blur_type.addItems(["random", "gaussian", "motion", "defocus", "box"])

        self.spin_blur_min = QDoubleSpinBox()
        self.spin_blur_min.setRange(0.1, 10.0)
        self.spin_blur_min.setValue(0.5)

        self.spin_blur_max = QDoubleSpinBox()
        self.spin_blur_max.setRange(0.1, 10.0)
        self.spin_blur_max.setValue(2.5)

        blur_layout = QHBoxLayout()
        blur_layout.addWidget(self.chk_blur)
        blur_layout.addSpacing(12)
        blur_layout.addWidget(QLabel("Prob:"))
        blur_layout.addWidget(self.spin_p_blur)
        blur_layout.addSpacing(12)
        blur_layout.addWidget(QLabel("Type:"))
        blur_layout.addWidget(self.combo_blur_type)
        blur_layout.addSpacing(12)
        blur_layout.addWidget(QLabel("Range:"))
        blur_layout.addWidget(self.spin_blur_min)
        blur_layout.addWidget(QLabel("to"))
        blur_layout.addWidget(self.spin_blur_max)
        blur_layout.addStretch()

        form.addRow("Crops per Image:", self.spin_crops)
        form.addRow("JPEG Quality Levels:", self.edit_qualities)
        form.addRow("Random Quality Mode:", random_layout)
        form.addRow("Blurring Degradation:", blur_layout)
        v.addLayout(form)

        self.btn_run_datagen = QPushButton("Generate Training Pairs")
        self.btn_run_datagen.setStyleSheet("font-weight: bold; background-color: #f57c00; color: white;")
        self.btn_run_datagen.clicked.connect(self._run_data_generation)
        v.addWidget(self.btn_run_datagen)

        self.datagen_progress = QProgressBar()
        v.addWidget(self.datagen_progress)

        self.datagen_log = QTextEdit()
        self.datagen_log.setReadOnly(True)
        self.datagen_log.setStyleSheet("background-color: #121212; color: #ffb74d; font-family: Consolas, monospace;")
        v.addWidget(self.datagen_log)

        layout.addWidget(box_info)

    def _on_random_quality_toggled(self, checked: bool):
        self.edit_qualities.setEnabled(not checked)
        self.spin_random_versions.setEnabled(checked)
        self.spin_q_min.setEnabled(checked)
        self.spin_q_max.setEnabled(checked)

    # -------------------------------------------------------------------------
    # UI Logic & Event Handlers
    # -------------------------------------------------------------------------
    def _apply_theme(self):
        """Sets a sleek modern dark palette for the application."""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; }
            QTabWidget::pane { border: 1px solid #333333; background-color: #252526; }
            QTabBar::tab { background: #2d2d2d; padding: 10px 20px; color: #cccccc; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #1e1e1e; color: #ffffff; font-weight: bold; border-bottom: 2px solid #007acc; }
            QGroupBox { border: 1px solid #3e3e42; border-radius: 6px; margin-top: 10px; font-weight: bold; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4fc3f7; }
            QPushButton { background-color: #333333; border: 1px solid #444444; border-radius: 4px; padding: 6px 14px; font-size: 13px; }
            QPushButton:hover { background-color: #444444; }
            QPushButton:pressed { background-color: #222222; }
            QLineEdit, QComboBox, QSpinBox, QSlider { background-color: #2d2d2d; border: 1px solid #444444; border-radius: 4px; padding: 4px; color: #e0e0e0; }
            QLineEdit:disabled, QSpinBox:disabled { background-color: #1e1e1e; color: #666666; border-color: #333333; }
            QScrollArea { border: 1px solid #333333; background-color: #181818; }
            QProgressBar { border: 1px solid #444444; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #007acc; }
        """)

    def _refresh_models(self):
        self.combo_model.clear()
        models = []
        if os.path.exists("pytorch models"):
            for f in os.listdir("pytorch models"):
                if f.endswith(".pth"):
                    models.append(os.path.join("pytorch models", f))
        if os.path.exists("onnx models"):
            for f in os.listdir("onnx models"):
                if f.endswith(".onnx"):
                    models.append(os.path.join("onnx models", f))

        self.combo_model.addItems(models)
        if len(models) == 0:
            self.combo_model.addItem("No models found")

    def _refresh_sample_images(self):
        self.combo_sample_img.clear()
        valid = (".png", ".jpg", ".jpeg", ".bmp")
        if os.path.exists("samples"):
            for f in os.listdir("samples"):
                if f.lower().endswith(valid):
                    self.combo_sample_img.addItem(os.path.join("samples", f))

    def _open_image_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "samples", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self._set_active_image(file_path)

    def _load_sample_image(self):
        selected = self.combo_sample_img.currentText()
        if selected and os.path.exists(selected):
            self._set_active_image(selected)

    def _set_active_image(self, file_path: str):
        self.image_path = file_path
        with Image.open(file_path) as raw_img:
            if raw_img.mode in ("RGBA", "LA") or (raw_img.mode == "P" and "transparency" in raw_img.info):
                self.loaded_image = raw_img.convert("RGBA")
            else:
                self.loaded_image = raw_img.convert("RGB")
        self.noisy_image = self.loaded_image.copy()
        self.denoised_image = None

        w, h = self.loaded_image.size
        self.lbl_input_info.setText(f"Resolution: {w} x {h} (Source: {os.path.basename(file_path)})")
        self.view_input.setPixmap(pil_to_qpixmap(self.noisy_image))
        self.lbl_output_info.setText("Ready to denoise.")
        self.view_output.setText("Click 'Denoise Image' to execute.")

    def _inject_artifacts_to_image(self):
        if self.loaded_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        q = self.slider_quality.value()
        blur_val = self.slider_blur.value() / 10.0
        degraded = apply_downsample_upsample(self.loaded_image, scale=0.85)
        if blur_val > 0.0:
            degraded = apply_blur(degraded, blur_type="gaussian", strength=blur_val)
        degraded = apply_jpeg_compression(degraded, quality=q)
        degraded = add_gaussian_noise(degraded, sigma=0.02)

        self.noisy_image = degraded
        w, h = self.noisy_image.size
        blur_str = f" + Blur r={blur_val:.1f}" if blur_val > 0.0 else ""
        self.lbl_input_info.setText(f"Resolution: {w} x {h} (JPEG Q={q}{blur_str} + Noise Injected)")
        self.view_input.setPixmap(pil_to_qpixmap(self.noisy_image))
        self.status_label.setText(f"Injected JPEG compression (Q={q}){blur_str} and sensor noise.")

    def _execute_denoise(self):
        if self.noisy_image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        model_path = self.combo_model.currentText()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Invalid Model", "Selected model file does not exist.")
            return

        tiled = self.chk_tiled.isChecked()
        tile_size = int(self.combo_tile_size.currentText())

        dev_map = {"auto": "auto", "cuda / rocm": "cuda", "intel xpu": "xpu", "gpu": "auto", "cpu": "cpu"}
        device = dev_map.get(self.combo_device.currentText().strip().lower(), "auto")

        self.btn_run_denoise.setEnabled(False)
        self.status_label.setText(f"Running inference on {self.noisy_image.size[0]}x{self.noisy_image.size[1]} image...")

        self.inf_worker = InferenceWorker(self.noisy_image, model_path, tiled, tile_size, device)
        self.inf_worker.finished.connect(self._on_denoise_finished)
        self.inf_worker.failed.connect(self._on_denoise_failed)
        self.inf_worker.start()

    def _on_denoise_finished(self, result_img: Image.Image, elapsed_ms: float):
        self.btn_run_denoise.setEnabled(True)
        self.denoised_image = result_img
        w, h = result_img.size

        self.lbl_output_info.setText(f"Resolution: {w} x {h} | Latency: {elapsed_ms:.1f}ms")
        self.view_output.setPixmap(pil_to_qpixmap(result_img))
        self.status_label.setText(f"Denoising complete ({w}x{h} in {elapsed_ms:.1f}ms).")

    def _on_denoise_failed(self, err_msg: str):
        self.btn_run_denoise.setEnabled(True)
        QMessageBox.critical(self, "Inference Failed", f"Denoising encountered an error:\n{err_msg}")
        self.status_label.setText("Inference failed.")

    def _save_output_image(self):
        if self.denoised_image is None:
            QMessageBox.warning(self, "No Result", "No denoised image to save.")
            return

        os.makedirs("outputs", exist_ok=True)
        default_name = "denoised_output.png"
        if self.image_path:
            base = os.path.splitext(os.path.basename(self.image_path))[0]
            default_name = f"{base}_denoised.png"

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Denoised PNG", os.path.join("outputs", default_name), "PNG Images (*.png)"
        )
        if save_path:
            self.denoised_image.save(save_path, format="PNG")
            QMessageBox.information(self, "Saved", f"Denoised image saved to:\n{save_path}")

    # -------------------------------------------------------------------------
    # Training Tab Actions
    # -------------------------------------------------------------------------
    def _on_finetune_mode_toggled(self, checked: bool):
        self.combo_pretrained_model.setEnabled(checked)
        self.btn_browse_pretrained.setEnabled(checked)
        self.btn_refresh_checkpoints.setEnabled(checked)
        self.spin_ft_epochs.setEnabled(checked)
        self.spin_ft_lr.setEnabled(checked)
        self.chk_freeze_stem.setEnabled(checked)

        self.spin_s1_epochs.setEnabled(not checked)
        self.spin_s1_batch.setEnabled(not checked)
        self.btn_start_train.setText("Start Fine-Tuning Pipeline" if checked else "Start Training Pipeline")

    def _refresh_checkpoint_models(self):
        current = self.combo_pretrained_model.currentText()
        self.combo_pretrained_model.clear()
        found = []
        for d in ["checkpoints", "pytorch models"]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith(".pth"):
                        found.append(os.path.normpath(os.path.join(d, f)))
        for p in sorted(set(found)):
            self.combo_pretrained_model.addItem(p)
        if current and self.combo_pretrained_model.findText(current) >= 0:
            self.combo_pretrained_model.setCurrentText(current)

    def _browse_pretrained_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-trained Checkpoint", "", "PyTorch Models (*.pth *.pt);;All Files (*.*)"
        )
        if path:
            norm_path = os.path.normpath(path)
            idx = self.combo_pretrained_model.findText(norm_path)
            if idx < 0:
                self.combo_pretrained_model.addItem(norm_path)
                idx = self.combo_pretrained_model.count() - 1
            self.combo_pretrained_model.setCurrentIndex(idx)

    def _start_training(self):
        if hasattr(self, "train_worker") and self.train_worker.isRunning():
            return

        is_finetune = self.chk_finetune_mode.isChecked()
        pretrained_path = self.combo_pretrained_model.currentText().strip() if is_finetune else None
        if is_finetune and (not pretrained_path or not os.path.exists(pretrained_path)):
            QMessageBox.warning(self, "Invalid Checkpoint", "Please select a valid pre-trained .pth checkpoint to fine-tune.")
            return

        dev_map = {"auto": "auto", "cuda / rocm": "cuda", "intel xpu": "xpu", "gpu": "auto", "cpu": "cpu"}
        train_dev = dev_map.get(self.combo_train_device.currentText().strip().lower(), "auto")

        config = {
            "device": train_dev,
            "is_finetune": is_finetune,
            "pretrained_path": pretrained_path,
            "finetune_epochs": self.spin_ft_epochs.value(),
            "finetune_lr": self.spin_ft_lr.value(),
            "freeze_stem": self.chk_freeze_stem.isChecked(),
            "stage1_epochs": self.spin_s1_epochs.value(),
            "stage2_epochs": self.spin_s2_epochs.value(),
            "batch_size_stage1": self.spin_s1_batch.value(),
            "batch_size_stage2": self.spin_s2_batch.value(),
            "num_workers": self.spin_workers.value(),
            "eval_interval": self.spin_eval_interval.value(),
            "preload_ram": self.chk_preload_ram.isChecked(),
            "max_cache_items": self.spin_max_cache.value(),
            "bf16": self.chk_bf16.isChecked(),
            "fp16": self.chk_fp16.isChecked(),
            "train_clean_dir": "samples/processed/train/clean" if os.path.exists("samples/processed/train/clean") else "samples/processed/clean",
            "train_noisy_dir": "samples/processed/train/noisy" if os.path.exists("samples/processed/train/noisy") else "samples/processed/noisy",
            "val_clean_dir": "samples/processed/val/clean" if os.path.exists("samples/processed/val/clean") else "samples/processed/clean",
            "val_noisy_dir": "samples/processed/val/noisy" if os.path.exists("samples/processed/val/noisy") else "samples/processed/noisy",
            "synthetic_samples": 64,
        }

        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.train_log.clear()
        self.train_progress.setValue(0)

        self.train_worker = TrainingWorker(config)
        self.train_worker.log_signal.connect(lambda msg: self.train_log.append(msg))
        self.train_worker.epoch_progress.connect(lambda curr, total: self.train_progress.setValue(int(curr / total * 100)))
        self.train_worker.finished.connect(self._on_training_finished)
        self.train_worker.start()

    def _stop_training(self):
        if hasattr(self, "train_worker") and self.train_worker.isRunning():
            self.train_worker.stop()
            self.btn_stop_train.setEnabled(False)
            self.status_label.setText("Stopping training...")

    def _on_training_finished(self, status: str):
        if hasattr(self, "train_worker"):
            self.train_worker.wait()
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self._refresh_models()
        self.status_label.setText(f"Training finished: {status}")

        if self.chk_shutdown.isChecked() and status == "Success":
            self.train_log.append("Shutting down computer in 60 seconds...")
            if sys.platform == "win32":
                os.system("shutdown /s /t 60")
            else:
                os.system("shutdown -h now")

    # -------------------------------------------------------------------------
    # Data Generation Actions
    # -------------------------------------------------------------------------
    def _run_data_generation(self):
        import re

        is_random = self.chk_random.isChecked()
        qualities = None
        quality_range = None
        num_random_versions = 0

        if is_random:
            num_random_versions = self.spin_random_versions.value()
            q_min = self.spin_q_min.value()
            q_max = self.spin_q_max.value()
            quality_range = (min(q_min, q_max), max(q_min, q_max))
        else:
            raw_text = self.edit_qualities.text().strip()
            parsed = [int(x) for x in re.findall(r"\d+", raw_text)]
            qualities = list(dict.fromkeys([max(1, min(100, q)) for q in parsed]))
            if not qualities:
                QMessageBox.warning(
                    self,
                    "Invalid Qualities",
                    "Please enter at least one valid JPEG quality (1-100), e.g. '15 20 40 60'.",
                )
                return

        self.btn_run_datagen.setEnabled(False)
        self.datagen_log.clear()
        self.datagen_progress.setValue(0)

        if is_random:
            self.datagen_log.append(
                f"Starting data generation: {num_random_versions} random versions in range [{quality_range[0]}, {quality_range[1]}] per image..."
            )
        else:
            self.datagen_log.append(
                f"Starting data generation with quality levels: {qualities} ({len(qualities)} versions per image/crop)..."
            )

        p_blur = self.spin_p_blur.value() if self.chk_blur.isChecked() else 0.0
        blur_type = self.combo_blur_type.currentText()
        blur_range = (self.spin_blur_min.value(), self.spin_blur_max.value())

        self.datagen_worker = DataGenWorker(
            samples_dir="samples",
            clean_dir="samples/processed/clean",
            noisy_dir="samples/processed/noisy",
            qualities=qualities,
            quality_range=quality_range,
            num_random_versions=num_random_versions,
            num_crops=self.spin_crops.value(),
            p_blur=p_blur,
            blur_type=blur_type,
            blur_range=blur_range,
        )
        self.datagen_worker.log_signal.connect(lambda msg: self.datagen_log.append(msg))
        self.datagen_worker.progress.connect(lambda c: self.datagen_log.append(f"Generated pair #{c}"))
        self.datagen_worker.finished.connect(self._on_datagen_finished)
        self.datagen_worker.start()

    def _on_datagen_finished(self, total: int):
        self.btn_run_datagen.setEnabled(True)
        self.datagen_progress.setValue(100)
        self.datagen_log.append(f"\nProcessing complete! {total} clean-noisy training pairs generated.")
        QMessageBox.information(self, "Data Generation Complete", f"Successfully generated {total} paired training samples!")


def main():
    app = QApplication(sys.argv)
    window = RepAFDenoiseGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
