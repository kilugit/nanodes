import os
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models import RepAFDenoiseNet

_TORCH_MODEL_CACHE: Dict[Tuple[str, str, float], RepAFDenoiseNet] = {}
_ONNX_SESSION_CACHE: Dict[Tuple[str, str, float], object] = {}
_HANN_WINDOW_CACHE: Dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
_NP_HANN_CACHE: Dict[int, np.ndarray] = {}


def load_pytorch_model(model_path: str, device: torch.device) -> RepAFDenoiseNet:
    mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0.0
    cache_key = (os.path.abspath(model_path), str(device), mtime)
    if cache_key in _TORCH_MODEL_CACHE:
        return _TORCH_MODEL_CACHE[cache_key]

    model = RepAFDenoiseNet(c=40).to(device).eval()
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    if isinstance(ckpt, dict) and ckpt.get("quantized_format") == "symmetric_int8_per_channel":
        dequant = {}
        for k, v in state_dict.items():
            if k.endswith("_scale"):
                continue
            if f"{k}_scale" in state_dict:
                dequant[k] = v.float() * state_dict[f"{k}_scale"]
            else:
                dequant[k] = v
        state_dict = dequant

    is_deployed = ckpt.get("is_deployed", False) or "stem.conv_deploy.weight" in state_dict
    if is_deployed:
        model.switch_to_deploy()
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
        model.switch_to_deploy()

    _TORCH_MODEL_CACHE[cache_key] = model
    return model


def get_hann_window_torch(tile_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (tile_size, device, dtype)
    mask = _HANN_WINDOW_CACHE.get(key)
    if mask is None:
        w_y = torch.hann_window(tile_size, device=device, dtype=dtype).view(tile_size, 1)
        w_x = torch.hann_window(tile_size, device=device, dtype=dtype).view(1, tile_size)
        mask = torch.clamp((w_y * w_x).view(1, 1, tile_size, tile_size), min=1e-4)
        _HANN_WINDOW_CACHE[key] = mask
    return mask


def get_hann_window_numpy(tile_size: int) -> np.ndarray:
    mask = _NP_HANN_CACHE.get(tile_size)
    if mask is None:
        wy = np.hanning(tile_size).reshape(tile_size, 1)
        wx = np.hanning(tile_size).reshape(1, tile_size)
        mask = np.clip((wy * wx).reshape(1, 1, tile_size, tile_size).astype(np.float32), 1e-4, 1.0)
        _NP_HANN_CACHE[tile_size] = mask
    return mask


def tiled_denoise_pytorch(
    model: torch.nn.Module,
    x: torch.Tensor,
    tile_size: int = 1024,
    overlap: int = 64,
) -> torch.Tensor:
    b, c, h, w = x.shape
    tile_size = min(tile_size, max(h, w))
    tile_size -= tile_size % 2

    if h <= tile_size and w <= tile_size:
        return model(x)

    stride = tile_size - overlap
    output = torch.zeros_like(x)
    weights = torch.zeros(1, 1, h, w, device=x.device, dtype=x.dtype)
    tile_mask = get_hann_window_torch(tile_size, x.device, x.dtype)

    h_starts = list(range(0, h - tile_size + 1, stride))
    if len(h_starts) == 0 or h_starts[-1] + tile_size < h:
        h_starts.append(max(0, h - tile_size))

    w_starts = list(range(0, w - tile_size + 1, stride))
    if len(w_starts) == 0 or w_starts[-1] + tile_size < w:
        w_starts.append(max(0, w - tile_size))

    for top in h_starts:
        for left in w_starts:
            patch = x[:, :, top : top + tile_size, left : left + tile_size]
            with torch.no_grad():
                pred = model(patch)
            output[:, :, top : top + tile_size, left : left + tile_size] += pred * tile_mask
            weights[:, :, top : top + tile_size, left : left + tile_size] += tile_mask

    output /= torch.clamp(weights, min=1e-8)
    return output


def tiled_denoise_onnx(
    session,
    x_np: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 64,
) -> np.ndarray:
    b, c, h, w = x_np.shape
    tile_size = min(tile_size, max(h, w))
    tile_size -= tile_size % 2

    input_name = session.get_inputs()[0].name
    if h <= tile_size and w <= tile_size:
        return session.run(None, {input_name: x_np})[0]

    stride = tile_size - overlap
    output = np.zeros((b, c, h, w), dtype=np.float32)
    weights = np.zeros((1, 1, h, w), dtype=np.float32)
    tile_mask = get_hann_window_numpy(tile_size)

    h_starts = list(range(0, h - tile_size + 1, stride))
    if len(h_starts) == 0 or h_starts[-1] + tile_size < h:
        h_starts.append(max(0, h - tile_size))

    w_starts = list(range(0, w - tile_size + 1, stride))
    if len(w_starts) == 0 or w_starts[-1] + tile_size < w:
        w_starts.append(max(0, w - tile_size))

    for top in h_starts:
        for left in w_starts:
            patch = x_np[:, :, top : top + tile_size, left : left + tile_size]
            pred = session.run(None, {input_name: patch})[0].astype(np.float32)
            output[:, :, top : top + tile_size, left : left + tile_size] += pred * tile_mask
            weights[:, :, top : top + tile_size, left : left + tile_size] += tile_mask

    output /= np.clip(weights, 1e-8, None)
    return output


def run_denoise(
    image: Union[str, Image.Image],
    model_path: str,
    tiled: bool = False,
    tile_size: int = 1024,
    tile_overlap: int = 64,
    device: Optional[str] = None,
) -> Tuple[Image.Image, float]:
    pil_img = Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
    w, h = pil_img.size
    is_onnx = model_path.lower().endswith(".onnx")
    t0 = time.time()

    if is_onnx:
        import onnxruntime as ort

        dev_name = "cpu" if device == "cpu" else "gpu"
        mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0.0
        cache_key = (os.path.abspath(model_path), dev_name, mtime)

        session = _ONNX_SESSION_CACHE.get(cache_key)
        if session is None:
            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"] if device == "cpu" else (
                ["DmlExecutionProvider", "CPUExecutionProvider"]
                if "DmlExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
            session = ort.InferenceSession(model_path, providers=providers)
            _ONNX_SESSION_CACHE[cache_key] = session

        input_meta = session.get_inputs()[0]
        is_fp16 = "float16" in input_meta.type.lower()
        input_dtype = np.float16 if is_fp16 else np.float32

        arr = np.array(pil_img, dtype=input_dtype).transpose(2, 0, 1) / 255.0
        x_np = np.expand_dims(arr, axis=0)

        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x_np = np.pad(x_np, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        if tiled:
            out_np = tiled_denoise_onnx(session, x_np, tile_size=tile_size, overlap=tile_overlap)
        else:
            out_np = session.run(None, {input_meta.name: x_np})[0]

        if pad_h > 0 or pad_w > 0:
            out_np = out_np[:, :, :h, :w]

        out_arr = np.clip(np.round(out_np[0].transpose(1, 2, 0).astype(np.float32) * 255.0), 0.0, 255.0).astype(np.uint8)
        denoised_img = Image.fromarray(out_arr)

    else:
        dev = torch.device("cpu") if device == "cpu" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_pytorch_model(model_path, dev)

        # Transfer uint8 to GPU, then float/div on device for reduced PCIe bandwidth
        arr = np.array(pil_img)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)
        x = x.float().div_(255.0)

        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            out = tiled_denoise_pytorch(model, x, tile_size=tile_size, overlap=tile_overlap) if tiled else model(x)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        # Convert on device before host transfer
        out_uint8 = out[0].mul(255.0).add_(0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        denoised_img = Image.fromarray(out_uint8)

    elapsed_ms = (time.time() - t0) * 1000.0
    return denoised_img, elapsed_ms
