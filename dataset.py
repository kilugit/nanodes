from collections import OrderedDict
import glob
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def paired_augment(noisy: torch.Tensor, clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if random.random() > 0.5:
        noisy = torch.flip(noisy, dims=[-1])
        clean = torch.flip(clean, dims=[-1])

    if random.random() > 0.5:
        noisy = torch.flip(noisy, dims=[-2])
        clean = torch.flip(clean, dims=[-2])

    k = random.randint(0, 3)
    if k > 0:
        noisy = torch.rot90(noisy, k, dims=[-2, -1])
        clean = torch.rot90(clean, k, dims=[-2, -1])

    return noisy, clean


def paired_crop(
    noisy: torch.Tensor, clean: torch.Tensor, patch_size: int, is_train: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, h, w = noisy.shape
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        pad_mode = "reflect" if (pad_w < w and pad_h < h) else "replicate"
        noisy = F.pad(noisy, (0, pad_w, 0, pad_h), mode=pad_mode)
        clean = F.pad(clean, (0, pad_w, 0, pad_h), mode=pad_mode)
        _, h, w = noisy.shape

    if is_train:
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
    else:
        top = (h - patch_size) // 2
        left = (w - patch_size) // 2

    noisy_patch = noisy[:, top : top + patch_size, left : left + patch_size]
    clean_patch = clean[:, top : top + patch_size, left : left + patch_size]
    return noisy_patch, clean_patch


_IMAGE_CACHE: OrderedDict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
_MAX_CACHE_ITEMS: int = 2048


def set_max_cache_items(n: int):
    global _MAX_CACHE_ITEMS
    _MAX_CACHE_ITEMS = max(0, int(n))
    while len(_IMAGE_CACHE) > _MAX_CACHE_ITEMS:
        _IMAGE_CACHE.popitem(last=False)


def clear_image_cache():
    """Clears in-memory cached tensors to reclaim RAM."""
    _IMAGE_CACHE.clear()


class DenoisingDataset(Dataset):
    """Paired image dataset for denoising training and validation."""

    def __init__(
        self,
        noisy_dir: Optional[str] = None,
        clean_dir: Optional[str] = None,
        patch_size: int = 128,
        is_train: bool = True,
        num_synthetic_samples: int = 128,
        cache: bool = True,
        preload_to_ram: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.is_train = is_train
        self.preload_to_ram = preload_to_ram
        self.cache = cache and not preload_to_ram
        self.cached_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

        self.paired_files: List[Tuple[str, str]] = []
        if noisy_dir is None and clean_dir is None:
            split_subdir = "train" if self.is_train else "val"
            split_clean = os.path.join("samples", "processed", split_subdir, "clean")
            split_noisy = os.path.join("samples", "processed", split_subdir, "noisy")
            if os.path.exists(split_clean) and os.path.exists(split_noisy):
                clean_dir = split_clean
                noisy_dir = split_noisy
            else:
                default_clean = "samples/processed/clean"
                default_noisy = "samples/processed/noisy"
                if os.path.exists(default_clean) and os.path.exists(default_noisy):
                    clean_dir = default_clean
                    noisy_dir = default_noisy
        elif clean_dir == "samples/processed/clean" and (not os.path.exists(clean_dir) or not glob.glob(os.path.join(clean_dir, "*.png"))):
            split_subdir = "train" if self.is_train else "val"
            split_clean = os.path.join("samples", "processed", split_subdir, "clean")
            split_noisy = os.path.join("samples", "processed", split_subdir, "noisy")
            if os.path.exists(split_clean) and os.path.exists(split_noisy):
                clean_dir = split_clean
                noisy_dir = split_noisy

        if noisy_dir and clean_dir and os.path.exists(noisy_dir) and os.path.exists(clean_dir):
            extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
            clean_paths = []
            for ext in extensions:
                clean_paths.extend(glob.glob(os.path.join(clean_dir, ext)))
            clean_paths.sort()

            for cp in clean_paths:
                base = os.path.basename(cp)
                np_path = os.path.join(noisy_dir, base)
                if not os.path.exists(np_path) and "_clean" in base:
                    noisy_base = base.replace("_clean.", "_noisy.")
                    np_path = os.path.join(noisy_dir, noisy_base)

                if os.path.exists(np_path):
                    self.paired_files.append((np_path, cp))

        self.use_synthetic = len(self.paired_files) == 0
        self.num_synthetic_samples = num_synthetic_samples

        if self.preload_to_ram and not self.use_synthetic:
            for np_path, cp in self.paired_files:
                try:
                    with Image.open(np_path) as n_img:
                        n_arr = np.array(n_img.convert("RGB"))
                    with Image.open(cp) as c_img:
                        c_arr = np.array(c_img.convert("RGB"))
                    self.cached_pairs.append((
                        torch.from_numpy(n_arr).permute(2, 0, 1),
                        torch.from_numpy(c_arr).permute(2, 0, 1),
                    ))
                except (OSError, FileNotFoundError):
                    continue

    def set_patch_size(self, patch_size: int):
        self.patch_size = patch_size

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.num_synthetic_samples
        return len(self.paired_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_synthetic:
            base_size = max(512, self.patch_size)
            clean = torch.rand(3, base_size, base_size, dtype=torch.float32)
            noise_std = random.uniform(0.02, 0.10)
            noisy = clean
            if random.random() < 0.5:
                k = random.choice([3, 5])
                noisy = F.avg_pool2d(noisy.unsqueeze(0), kernel_size=k, stride=1, padding=k // 2).squeeze(0)
            noisy = torch.clamp(noisy + torch.randn_like(noisy) * noise_std, 0.0, 1.0)
            if self.patch_size:
                noisy, clean = paired_crop(noisy, clean, self.patch_size, is_train=self.is_train)
            if self.is_train:
                noisy, clean = paired_augment(noisy, clean)
            return noisy, clean

        if self.preload_to_ram and self.cached_pairs:
            noisy, clean = self.cached_pairs[idx % len(self.cached_pairs)]
        else:
            noisy_path, clean_path = self.paired_files[idx]
            cache_key = (noisy_path, clean_path)
            if self.cache and _MAX_CACHE_ITEMS > 0 and cache_key in _IMAGE_CACHE:
                noisy, clean = _IMAGE_CACHE[cache_key]
                _IMAGE_CACHE.move_to_end(cache_key)
            else:
                try:
                    with Image.open(noisy_path) as n_img:
                        noisy_arr = np.array(n_img.convert("RGB"))
                    with Image.open(clean_path) as c_img:
                        clean_arr = np.array(c_img.convert("RGB"))
                    noisy = torch.from_numpy(noisy_arr).permute(2, 0, 1)
                    clean = torch.from_numpy(clean_arr).permute(2, 0, 1)
                    if self.cache and _MAX_CACHE_ITEMS > 0:
                        while len(_IMAGE_CACHE) >= _MAX_CACHE_ITEMS:
                            _IMAGE_CACHE.popitem(last=False)
                        _IMAGE_CACHE[cache_key] = (noisy, clean)
                except (OSError, FileNotFoundError):
                    return self.__getitem__((idx + 1) % len(self.paired_files))

        if self.patch_size:
            noisy, clean = paired_crop(noisy, clean, self.patch_size, is_train=self.is_train)
        if self.is_train:
            noisy, clean = paired_augment(noisy, clean)

        return noisy.float().div_(255.0), clean.float().div_(255.0)
