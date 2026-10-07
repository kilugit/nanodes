import glob
import os
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


def _prepare_cpu_deploy_model(model: nn.Module) -> nn.Module:
    from models import RepAFDenoiseNet

    c = getattr(model, "c", 40)
    cpu_model = RepAFDenoiseNet(c=c).eval()
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if getattr(model, "is_deployed", False) or "stem.conv_deploy.weight" in cpu_state:
        cpu_model.switch_to_deploy()
        cpu_model.load_state_dict(cpu_state)
    else:
        cpu_model.load_state_dict(cpu_state)
        cpu_model.switch_to_deploy()
    return cpu_model


def load_checkpoint_model(model_path: Optional[str] = None) -> nn.Module:
    from models import RepAFDenoiseNet

    model = RepAFDenoiseNet(c=40).eval()
    target_path = model_path
    if not target_path or not os.path.exists(target_path):
        for candidate in ["pytorch models/repaf_denoise_net_deployed.pth", "pytorch models/best_model.pth"]:
            if os.path.exists(candidate):
                target_path = candidate
                break

    if target_path and os.path.exists(target_path):
        ckpt = torch.load(target_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        if isinstance(ckpt, dict) and ckpt.get("quantized_format") == "symmetric_int8_per_channel":
            dequant = {}
            for k, v in state.items():
                if k.endswith("_scale"):
                    continue
                if f"{k}_scale" in state:
                    dequant[k] = v.float() * state[f"{k}_scale"]
                else:
                    dequant[k] = v
            state = dequant

        is_deployed = (
            getattr(model, "is_deployed", False)
            or (isinstance(ckpt, dict) and ckpt.get("is_deployed", False))
            or "stem.conv_deploy.weight" in state
        )
        if is_deployed:
            model.switch_to_deploy()
            model.load_state_dict(state)
        else:
            model.load_state_dict(state)
            model.switch_to_deploy()
        print(f"Loaded weights from: {target_path}")
    else:
        model.switch_to_deploy()
        print("Initialized model in deployment mode (no checkpoint found).")

    return model


def export_onnx(
    model: nn.Module,
    save_path: str = "onnx models/repaf_denoise_net.onnx",
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    opset_version: int = 17,
    fp16: bool = False,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    deploy_model = _prepare_cpu_deploy_model(model)

    if fp16:
        deploy_model = deploy_model.half()
        dummy_input = torch.randn(*input_shape, dtype=torch.float16)
    else:
        dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    torch.onnx.export(
        deploy_model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["noisy_image"],
        output_names=["denoised_image"],
        dynamic_axes={
            "noisy_image": {0: "batch_size", 2: "height", 3: "width"},
            "denoised_image": {0: "batch_size", 2: "height", 3: "width"},
        },
        dynamo=False,
    )

    try:
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"ONNX verification note: {e}")

    dtype_str = "FP16" if fp16 else "FP32"
    print(f"ONNX {dtype_str} export saved: {save_path}")
    return save_path


class _ImageCalibrationDataReader:
    def __init__(self, samples_dir: str = "samples", input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256), num_samples: int = 8):
        self.data: List[dict] = []
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        paths: List[str] = []
        if os.path.exists(samples_dir):
            for ext in exts:
                paths.extend(glob.glob(os.path.join(samples_dir, ext)))

        from process_samples import composite_rgba_to_rgb

        for p in paths[:num_samples]:
            try:
                with Image.open(p) as raw_img:
                    img = composite_rgba_to_rgb(raw_img).resize((input_shape[3], input_shape[2]))
                arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                self.data.append({"noisy_image": np.expand_dims(arr, axis=0)})
            except Exception:
                pass

        while len(self.data) < max(2, num_samples):
            self.data.append({"noisy_image": np.random.rand(*input_shape).astype(np.float32)})

        self.enum = iter(self.data)

    def get_next(self):
        return next(self.enum, None)


def export_onnx_int8(
    model_or_path: Union[nn.Module, str],
    save_path: str = "onnx models/repaf_denoise_net_int8.onnx",
    samples_dir: str = "samples",
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
) -> str:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    temp_fp32 = None

    if isinstance(model_or_path, str) and model_or_path.lower().endswith(".onnx"):
        source_onnx = model_or_path
    else:
        temp_fp32 = save_path + ".tmp_fp32.onnx"
        export_onnx(model_or_path, save_path=temp_fp32, input_shape=input_shape, fp16=False)
        source_onnx = temp_fp32

    class CalibBridge(CalibrationDataReader):
        def __init__(self, reader):
            self.reader = reader
        def get_next(self):
            return self.reader.get_next()

    raw_reader = _ImageCalibrationDataReader(samples_dir=samples_dir, input_shape=input_shape, num_samples=8)
    quantize_static(
        model_input=source_onnx,
        model_output=save_path,
        calibration_data_reader=CalibBridge(raw_reader),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    if temp_fp32 and os.path.exists(temp_fp32):
        try:
            os.remove(temp_fp32)
        except OSError:
            pass

    # Verify session execution
    session = ort.InferenceSession(save_path, providers=["CPUExecutionProvider"])
    dummy = np.random.rand(*input_shape).astype(np.float32)
    session.run(None, {"noisy_image": dummy})

    print(f"ONNX INT8 QDQ model successfully exported: {save_path}")
    return save_path


def export_int8_quantization(
    model: nn.Module,
    save_path: str = "pytorch models/repaf_denoise_net_int8.pth",
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 8,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    deploy_model = _prepare_cpu_deploy_model(model)

    int8_checkpoint = {"is_deployed": True, "quantized_format": "symmetric_int8_per_channel"}
    state_dict = deploy_model.state_dict()
    quantized_weights = {}

    for name, param in state_dict.items():
        if "weight" in name and param.dim() >= 2:
            reduce_dims = list(range(1, param.dim()))
            scale = torch.clamp(param.abs().amax(dim=reduce_dims, keepdim=True) / 127.0, min=1e-8)
            q_weight = torch.clamp(torch.round(param / scale), -128, 127).to(torch.int8)

            quantized_weights[name] = q_weight
            quantized_weights[f"{name}_scale"] = scale
        else:
            quantized_weights[name] = param

    int8_checkpoint["state_dict"] = quantized_weights
    torch.save(int8_checkpoint, save_path)
    print(f"PyTorch INT8 model saved: {save_path}")
    return save_path


if __name__ == "__main__":
    import argparse
    from models import RepAFDenoiseNet, test_reparameterization_equivalence

    parser = argparse.ArgumentParser(description="Export RepAFDenoiseNet to ONNX (FP32/FP16) and INT8")
    parser.add_argument("--model", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--onnx-path", type=str, default="onnx models/repaf_denoise_net.onnx")
    parser.add_argument("--fp16", action="store_true", help="Export ONNX in FP16 format")
    parser.add_argument("--int8", action="store_true", help="Quantize to INT8 ONNX and PyTorch formats")
    parser.add_argument("--int8-path", type=str, default="onnx models/repaf_denoise_net_int8.onnx")
    parser.add_argument("--test", action="store_true", help="Run equivalence test before export")
    args = parser.parse_args()

    if args.test:
        print("Running re-parameterization equivalence test...")
        err = test_reparameterization_equivalence(device="cpu")
        print(f"Equivalence test passed (max error: {err:.8e})")

    net = load_checkpoint_model(args.model)

    if args.fp16:
        save_path = args.onnx_path.replace(".onnx", "_fp16.onnx") if not args.onnx_path.endswith("_fp16.onnx") else args.onnx_path
        export_onnx(net, save_path=save_path, fp16=True)
    else:
        export_onnx(net, save_path=args.onnx_path, fp16=False)

    if args.int8:
        export_onnx_int8(net, save_path=args.int8_path)
        export_int8_quantization(net, save_path="pytorch models/repaf_denoise_net_int8.pth")
