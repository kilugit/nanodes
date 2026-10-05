import os
import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from export import (
    export_int8_quantization,
    export_onnx,
    export_onnx_int8,
    load_checkpoint_model,
)
from models import test_reparameterization_equivalence


class ExportTaskWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, task_type: str, model_path: str, out_dir: str):
        super().__init__()
        self.task_type = task_type
        self.model_path = model_path
        self.out_dir = out_dir

    def run(self):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            if self.task_type == "test":
                self.log_signal.emit("Executing Structural Re-parameterization Unit Test...")
                err = test_reparameterization_equivalence(device="cpu", tol=1e-5)
                self.log_signal.emit(f"PASSED! Max numerical difference: {err:.8e} (threshold: 1e-5)")
                self.finished_signal.emit(True, f"Analytical equivalence verified!\nMax error: {err:.8e} < 1e-5")

            elif self.task_type == "fp32":
                self.log_signal.emit(f"Loading checkpoint: {self.model_path}")
                net = load_checkpoint_model(self.model_path)
                out_path = os.path.join(self.out_dir, "repaf_denoise_net.onnx")
                self.log_signal.emit(f"Exporting FP32 ONNX -> {out_path}...")
                export_onnx(net, save_path=out_path, fp16=False)
                self.log_signal.emit(f"Successfully exported FP32 model to {out_path}")
                self.finished_signal.emit(True, f"FP32 ONNX model saved to:\n{out_path}")

            elif self.task_type == "fp16":
                self.log_signal.emit(f"Loading checkpoint: {self.model_path}")
                net = load_checkpoint_model(self.model_path)
                out_path = os.path.join(self.out_dir, "repaf_denoise_net_fp16.onnx")
                self.log_signal.emit(f"Exporting FP16 ONNX -> {out_path}...")
                export_onnx(net, save_path=out_path, fp16=True)
                self.log_signal.emit(f"Successfully exported FP16 model to {out_path}")
                self.finished_signal.emit(True, f"FP16 ONNX model saved to:\n{out_path}")

            elif self.task_type == "int8":
                self.log_signal.emit(f"Loading checkpoint: {self.model_path}")
                net = load_checkpoint_model(self.model_path)
                out_path = os.path.join(self.out_dir, "repaf_denoise_net_int8.onnx")
                self.log_signal.emit(f"Performing QDQ static INT8 quantization with sample calibration -> {out_path}...")
                export_onnx_int8(net, save_path=out_path, samples_dir="samples")
                pth_out = "pytorch models/repaf_denoise_net_int8.pth"
                export_int8_quantization(net, save_path=pth_out)
                self.log_signal.emit(f"Successfully exported INT8 ONNX: {out_path}")
                self.log_signal.emit(f"Successfully exported INT8 PyTorch: {pth_out}")
                self.finished_signal.emit(True, f"INT8 quantization complete!\nONNX: {out_path}\nPyTorch: {pth_out}")

        except Exception as e:
            self.log_signal.emit(f"ERROR: {e}")
            self.finished_signal.emit(False, str(e))
        finally:
            import gc
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


class ModelExportGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NanoDes - Model Export & Quantization")
        self.resize(780, 560)
        self.worker: Optional[ExportTaskWorker] = None

        self._init_ui()
        self._apply_theme()

    def _default_checkpoint(self) -> str:
        candidates = [
            "pytorch models/repaf_denoise_net_deployed.pth",
            "pytorch models/best_model.pth",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        if os.path.exists("pytorch models"):
            for f in os.listdir("pytorch models"):
                if f.endswith(".pth"):
                    return os.path.join("pytorch models", f)
        return "pytorch models/best_model.pth"

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 1. Model Configuration Group
        box_cfg = QGroupBox("Model & Target Configuration")
        form = QFormLayout(box_cfg)

        model_row = QHBoxLayout()
        self.edit_model = QLineEdit(self._default_checkpoint())
        self.btn_browse_model = QPushButton("Browse...")
        self.btn_browse_model.clicked.connect(self._browse_model)
        model_row.addWidget(self.edit_model)
        model_row.addWidget(self.btn_browse_model)

        out_row = QHBoxLayout()
        self.edit_out_dir = QLineEdit("onnx models")
        self.btn_browse_out = QPushButton("Browse...")
        self.btn_browse_out.clicked.connect(self._browse_output_dir)
        out_row.addWidget(self.edit_out_dir)
        out_row.addWidget(self.btn_browse_out)

        form.addRow("PyTorch Checkpoint (.pth):", model_row)
        form.addRow("Output Directory:", out_row)
        layout.addWidget(box_cfg)

        # 2. Operations Group
        box_ops = QGroupBox("Export & Quantization Operations")
        v_ops = QVBoxLayout(box_ops)

        self.btn_test = QPushButton("Run Re-parameterization Equivalence Test")
        self.btn_test.clicked.connect(lambda: self._start_task("test"))

        btn_row = QHBoxLayout()
        self.btn_fp32 = QPushButton("Export ONNX (FP32)")
        self.btn_fp32.setStyleSheet("background-color: #0288d1; color: white; font-weight: bold; padding: 10px;")
        self.btn_fp32.clicked.connect(lambda: self._start_task("fp32"))

        self.btn_fp16 = QPushButton("Export ONNX (FP16)")
        self.btn_fp16.setStyleSheet("background-color: #00897b; color: white; font-weight: bold; padding: 10px;")
        self.btn_fp16.clicked.connect(lambda: self._start_task("fp16"))

        self.btn_int8 = QPushButton("Quantize to INT8 (ONNX QDQ)")
        self.btn_int8.setStyleSheet("background-color: #6a1b9a; color: white; font-weight: bold; padding: 10px;")
        self.btn_int8.clicked.connect(lambda: self._start_task("int8"))

        btn_row.addWidget(self.btn_fp32)
        btn_row.addWidget(self.btn_fp16)
        btn_row.addWidget(self.btn_int8)

        v_ops.addWidget(self.btn_test)
        v_ops.addLayout(btn_row)
        layout.addWidget(box_ops)

        # 3. Log Console
        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setStyleSheet("background-color: #121212; color: #80d8ff; font-family: Consolas, monospace;")
        layout.addWidget(self.export_log)

        self.statusBar().showMessage("Ready.")

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { color: #e0e0e0; font-family: 'Segoe UI', Arial, sans-serif; }
            QGroupBox { border: 1px solid #3e3e42; border-radius: 6px; margin-top: 10px; font-weight: bold; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4fc3f7; }
            QLineEdit { background-color: #2d2d2d; border: 1px solid #444444; border-radius: 4px; padding: 6px; color: #e0e0e0; }
            QPushButton { background-color: #333333; border: 1px solid #444444; border-radius: 4px; padding: 7px 14px; font-size: 13px; }
            QPushButton:hover { background-color: #444444; }
            QPushButton:pressed { background-color: #222222; }
            QPushButton:disabled { background-color: #252526; color: #666666; border-color: #333333; }
        """)

    def _browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PyTorch Checkpoint", "pytorch models", "PyTorch Models (*.pth *.pt)"
        )
        if file_path:
            self.edit_model.setText(file_path)

    def _browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.edit_out_dir.text())
        if folder:
            self.edit_out_dir.setText(folder)

    def _set_buttons_enabled(self, enabled: bool):
        self.btn_test.setEnabled(enabled)
        self.btn_fp32.setEnabled(enabled)
        self.btn_fp16.setEnabled(enabled)
        self.btn_int8.setEnabled(enabled)
        self.btn_browse_model.setEnabled(enabled)
        self.btn_browse_out.setEnabled(enabled)

    def _start_task(self, task_type: str):
        model_path = self.edit_model.text().strip()
        out_dir = self.edit_out_dir.text().strip() or "onnx models"

        if task_type != "test" and not os.path.exists(model_path):
            reply = QMessageBox.question(
                self,
                "Checkpoint Not Found",
                f"Checkpoint file '{model_path}' does not exist.\nProceed with initialized weights?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._set_buttons_enabled(False)
        self.statusBar().showMessage(f"Running operation: {task_type.upper()}...")

        self.worker = ExportTaskWorker(task_type=task_type, model_path=model_path, out_dir=out_dir)
        self.worker.log_signal.connect(lambda msg: self.export_log.append(msg))
        self.worker.finished_signal.connect(self._on_task_finished)
        self.worker.start()

    def _on_task_finished(self, success: bool, message: str):
        self._set_buttons_enabled(True)
        if success:
            self.statusBar().showMessage("Operation completed successfully.")
            QMessageBox.information(self, "Success", message)
        else:
            self.statusBar().showMessage("Operation failed.")
            QMessageBox.critical(self, "Error", f"Operation failed:\n{message}")


def main():
    app = QApplication(sys.argv)
    window = ModelExportGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
