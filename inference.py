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


def clear_inference_cache():
    """Clears cached models, sessions, and window buffers to reclaim RAM and VRAM."""
    _TORCH_MODEL_CACHE.clear()
    _ONNX_SESSION_CACHE.clear()
    _HANN_WINDOW_CACHE.clear()
    _NP_HANN_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pytorch_model(model_path: str, device: torch.device) -> RepAFDenoiseNet:
    mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0.0
    cache_key = (os.path.abspath(model_path), str(device), mtime)
    if cache_key in _TORCH_MODEL_CACHE:
        return _TORCH_MODEL_CACHE[cache_key]

    model = RepAFDenoiseNet(c=40).to(device).eval()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
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


def get_hann_window_torch(
    tile_size: int, device: torch.device, dtype: torch.dtype, tw: Optional[int] = None, h_tiled: bool = True, w_tiled: bool = True
) -> torch.Tensor:
    width = tw if tw is not None else tile_size
    key = (tile_size, width, str(device), dtype, h_tiled, w_tiled)
    mask = _HANN_WINDOW_CACHE.get(key)
    if mask is None:
        if getattr(device, "type", None) == "xla":
            w_y = torch.hann_window(tile_size, dtype=dtype).to(device).view(tile_size, 1) if h_tiled else torch.ones(tile_size, 1, device=device, dtype=dtype)
            w_x = torch.hann_window(width, dtype=dtype).to(device).view(1, width) if w_tiled else torch.ones(1, width, device=device, dtype=dtype)
        else:
            w_y = torch.hann_window(tile_size, device=device, dtype=dtype).view(tile_size, 1) if h_tiled else torch.ones(tile_size, 1, device=device, dtype=dtype)
            w_x = torch.hann_window(width, device=device, dtype=dtype).view(1, width) if w_tiled else torch.ones(1, width, device=device, dtype=dtype)
        mask = torch.clamp((w_y * w_x).view(1, 1, tile_size, width), min=1e-4)
        _HANN_WINDOW_CACHE[key] = mask
    return mask


def get_hann_window_numpy(
    tile_size: int, tw: Optional[int] = None, h_tiled: bool = True, w_tiled: bool = True
) -> np.ndarray:
    width = tw if tw is not None else tile_size
    key = (tile_size, width, h_tiled, w_tiled)
    mask = _NP_HANN_CACHE.get(key)
    if mask is None:
        wy = np.hanning(tile_size).reshape(tile_size, 1) if h_tiled else np.ones((tile_size, 1), dtype=np.float32)
        wx = np.hanning(width).reshape(1, width) if w_tiled else np.ones((1, width), dtype=np.float32)
        mask = np.clip((wy * wx).reshape(1, 1, tile_size, width).astype(np.float32), 1e-4, 1.0)
        _NP_HANN_CACHE[key] = mask
    return mask


def tiled_denoise_pytorch(
    model: torch.nn.Module,
    x: torch.Tensor,
    tile_size: int = 1024,
    overlap: int = 64,
) -> torch.Tensor:
    b, c, h, w = x.shape
    if h <= tile_size and w <= tile_size:
        return model(x)

    th = min(tile_size, h)
    tw = min(tile_size, w)
    th -= th % 2
    tw -= tw % 2
    stride_h = max(1, th - overlap)
    stride_w = max(1, tw - overlap)

    h_starts = list(range(0, h - th + 1, stride_h)) if h > th else [0]
    if not h_starts or h_starts[-1] + th < h:
        h_starts.append(max(0, h - th))

    w_starts = list(range(0, w - tw + 1, stride_w)) if w > tw else [0]
    if not w_starts or w_starts[-1] + tw < w:
        w_starts.append(max(0, w - tw))

    tile_mask = get_hann_window_torch(th, x.device, x.dtype, tw=tw, h_tiled=(h > th), w_tiled=(w > tw))
    output = torch.zeros_like(x)
    weights = torch.zeros(1, 1, h, w, device=x.device, dtype=x.dtype)

    for top in h_starts:
        for left in w_starts:
            patch = x[:, :, top : top + th, left : left + tw]
            pred = model(patch)
            output[:, :, top : top + th, left : left + tw] += pred * tile_mask
            weights[:, :, top : top + th, left : left + tw] += tile_mask

    output.div_(torch.clamp(weights, min=1e-8))
    return output


def tiled_denoise_onnx(
    session,
    x_np: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 64,
) -> np.ndarray:
    b, c, h, w = x_np.shape
    input_name = session.get_inputs()[0].name
    if h <= tile_size and w <= tile_size:
        return session.run(None, {input_name: x_np})[0]

    th = min(tile_size, h)
    tw = min(tile_size, w)
    th -= th % 2
    tw -= tw % 2
    stride_h = max(1, th - overlap)
    stride_w = max(1, tw - overlap)

    h_starts = list(range(0, h - th + 1, stride_h)) if h > th else [0]
    if not h_starts or h_starts[-1] + th < h:
        h_starts.append(max(0, h - th))

    w_starts = list(range(0, w - tw + 1, stride_w)) if w > tw else [0]
    if not w_starts or w_starts[-1] + tw < w:
        w_starts.append(max(0, w - tw))

    tile_mask = get_hann_window_numpy(th, tw=tw, h_tiled=(h > th), w_tiled=(w > tw))
    output = np.zeros((b, c, h, w), dtype=np.float32)
    weights = np.zeros((1, 1, h, w), dtype=np.float32)

    for top in h_starts:
        for left in w_starts:
            patch = x_np[:, :, top : top + th, left : left + tw]
            pred = session.run(None, {input_name: patch})[0].astype(np.float32)
            output[:, :, top : top + th, left : left + tw] += pred * tile_mask
            weights[:, :, top : top + th, left : left + tw] += tile_mask

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
    if isinstance(image, str):
        with Image.open(image) as raw_img:
            input_img = raw_img.copy()
    else:
        input_img = image

    has_alpha = input_img.mode in ("RGBA", "LA") or (input_img.mode == "P" and "transparency" in input_img.info)
    alpha_channel = None
    if has_alpha:
        rgba = input_img.convert("RGBA")
        alpha_channel = rgba.split()[-1]
        pil_img = rgba.convert("RGB")
    else:
        pil_img = input_img.convert("RGB")

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
        if isinstance(device, torch.device):
            dev = device
        elif device == "cpu":
            dev = torch.device("cpu")
        elif "xla" in str(device):
            try:
                import torch_xla.core.xla_model as xm
                dev = xm.xla_device()
            except Exception:
                dev = torch.device("cpu")
        elif device == "cuda" or (device != "cpu" and torch.cuda.is_available()):
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        model = load_pytorch_model(model_path, dev)

        # Transfer uint8 to GPU/TPU, then float/div on device for reduced host bandwidth
        arr = np.array(pil_img)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)
        x = x.float().div_(255.0)

        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.inference_mode():
            out = tiled_denoise_pytorch(model, x, tile_size=tile_size, overlap=tile_overlap) if tiled else model(x)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        # Convert on device before host transfer
        out_uint8 = out[0].mul(255.0).add_(0.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        denoised_img = Image.fromarray(out_uint8)
        del x, out
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        elif dev.type == "xla":
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            except Exception:
                pass

    if has_alpha and alpha_channel is not None:
        denoised_img = Image.merge("RGBA", (*denoised_img.split(), alpha_channel))

    elapsed_ms = (time.time() - t0) * 1000.0
    return denoised_img, elapsed_ms
