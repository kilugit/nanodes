import os
import unittest
from PIL import Image
import numpy as np
import torch

import devices
from losses import Stage1Loss, Stage2SharpLoss
from metrics import calculate_psnr, calculate_ssim
from models import RepAFDenoiseNet, test_reparameterization_equivalence
from inference import run_denoise


class BackendCompatibilityTests(unittest.TestCase):
    def test_device_resolution(self):
        self.assertEqual(devices.get_device("cpu").type, "cpu")
        auto_dev = devices.get_device("auto")
        self.assertIn(auto_dev.type, ("cuda", "xpu", "mps", "xla", "cpu"))
        self.assertTrue(len(devices.get_device_name(auto_dev)) > 0)

        # XPU fallback or resolution
        xpu_dev = devices.get_device("xpu")
        self.assertIn(xpu_dev.type, ("xpu", "cpu"))

    def test_amp_and_scaler(self):
        cpu_dev = torch.device("cpu")
        scaler_cpu = devices.create_grad_scaler(cpu_dev, enabled=False)
        self.assertFalse(scaler_cpu.is_enabled())

        with devices.get_amp_context(cpu_dev, enabled=False):
            x = torch.ones(1)
            y = x * 2.0
            self.assertEqual(y.item(), 2.0)

        auto_dev = devices.get_device("auto")
        if auto_dev.type in ("cuda", "xpu"):
            scaler_gpu = devices.create_grad_scaler(auto_dev, enabled=True)
            self.assertTrue(scaler_gpu.is_enabled())
            with devices.get_amp_context(auto_dev, enabled=True, dtype=torch.float16):
                x = torch.ones(2, device=auto_dev)
                self.assertEqual(x.sum().item(), 2.0)

    def test_reparameterization_cpu_and_auto(self):
        err_cpu = test_reparameterization_equivalence(device="cpu", tol=1e-5)
        self.assertLess(err_cpu, 1e-5)

        auto_dev = devices.get_device("auto")
        err_auto = test_reparameterization_equivalence(device=auto_dev, tol=1e-5)
        self.assertLess(err_auto, 1e-5)

    def test_losses_and_metrics(self):
        for dev_str in ("cpu", "auto"):
            dev = devices.get_device(dev_str)
            p = torch.rand(2, 3, 64, 64, device=dev)
            t = torch.rand(2, 3, 64, 64, device=dev)

            l1 = Stage1Loss().to(dev)
            loss_val = l1(p, t)
            self.assertTrue(torch.isfinite(loss_val))

            l2 = Stage2SharpLoss().to(dev)
            loss_val2 = l2(p, t)
            self.assertTrue(torch.isfinite(loss_val2))

            psnr = calculate_psnr(p, t)
            self.assertTrue(np.isfinite(psnr))
            ssim = calculate_ssim(p, t)
            self.assertTrue(np.isfinite(ssim))

    def test_inference_pipeline(self):
        os.makedirs("outputs", exist_ok=True)
        dummy_img = "outputs/test_backend_sample.png"
        Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)).save(dummy_img)

        candidate_pth = [
            "checkpoints/repaf_denoise_net_deployed.pth",
            "pytorch models/repaf_denoise_net_deployed.pth",
            "checkpoints/best_model.pth",
        ]
        model_path = next((c for c in candidate_pth if os.path.exists(c)), None)

        if model_path:
            out_cpu, ms_cpu = run_denoise(dummy_img, model_path, device="cpu")
            self.assertIsInstance(out_cpu, Image.Image)
            self.assertGreater(ms_cpu, 0.0)

            out_auto, ms_auto = run_denoise(dummy_img, model_path, device="auto")
            self.assertIsInstance(out_auto, Image.Image)
            self.assertGreater(ms_auto, 0.0)

        onnx_path = "exports/repaf_denoise_net.onnx"
        if not os.path.exists(onnx_path):
            onnx_path = "onnx models/repaf_denoise_net.onnx"
        if os.path.exists(onnx_path):
            out_onnx, ms_onnx = run_denoise(dummy_img, onnx_path, device="auto")
            self.assertIsInstance(out_onnx, Image.Image)
            self.assertGreater(ms_onnx, 0.0)


if __name__ == "__main__":
    unittest.main()
