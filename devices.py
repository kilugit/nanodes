import contextlib
import gc
from typing import Optional, Union
import torch


def get_device(hint: Optional[Union[str, torch.device]] = "auto") -> torch.device:
    if isinstance(hint, torch.device):
        return hint
    name = (hint or "auto").strip().lower()

    if name == "cpu":
        return torch.device("cpu")

    if name == "xpu":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    if name in ("cuda", "rocm"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name == "xla":
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except Exception:
            return torch.device("cpu")

    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except Exception:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name(device: Union[str, torch.device]) -> str:
    dev = device if isinstance(device, torch.device) else get_device(device)
    if dev.type == "cuda":
        name = torch.cuda.get_device_name(dev.index or 0)
        hip = getattr(torch.version, "hip", None)
        return f"{name} (ROCm {hip})" if hip else f"{name} (CUDA)"
    if dev.type == "xpu":
        return torch.xpu.get_device_name(dev.index or 0)
    if dev.type == "mps":
        return "Apple Silicon (MPS)"
    if dev.type == "xla":
        return "Cloud TPU (XLA)"
    return "CPU"


def clear_device_cache(device: Optional[Union[str, torch.device]] = None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def get_amp_context(device: torch.device, enabled: bool = True, dtype: torch.dtype = torch.float16):
    if not enabled:
        return contextlib.nullcontext()
    if device.type in ("cuda", "xpu", "cpu"):
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
    return contextlib.nullcontext()


def create_grad_scaler(device: torch.device, enabled: bool = True):
    if enabled and device.type in ("cuda", "xpu"):
        return torch.amp.GradScaler(device.type, enabled=True)
    return torch.amp.GradScaler("cpu", enabled=False)
