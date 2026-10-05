from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import glob
import io
import json
import os
import random
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def apply_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_downsample_upsample(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    down_w = max(16, int(w * scale))
    down_h = max(16, int(h * scale))
    down = img.resize((down_w, down_h), Image.Resampling.BICUBIC)
    return down.resize((w, h), Image.Resampling.BICUBIC)


def add_realistic_noise(
    img: Image.Image,
    noise_type: str = "sensor",
    sigma: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    if sigma <= 0.0 or noise_type == "none":
        return img

    gen = rng if rng is not None else np.random.default_rng()
    arr = np.array(img, dtype=np.float32) / 255.0

    if noise_type == "gaussian":
        noise = gen.normal(0.0, sigma, arr.shape)
    elif noise_type == "poisson":
        shot_std = sigma * np.sqrt(np.maximum(arr, 1e-4))
        noise = gen.normal(0.0, 1.0, arr.shape) * shot_std
    elif noise_type == "sensor":
        shot_var = (sigma ** 2) * np.maximum(arr, 1e-4)
        read_var = (sigma * 0.5) ** 2
        channel_gains = gen.uniform(0.85, 1.15, size=(1, 1, 3)).astype(np.float32)
        total_std = np.sqrt(shot_var + read_var) * channel_gains
        noise = gen.normal(0.0, 1.0, arr.shape) * total_std
    else:
        noise = gen.normal(0.0, sigma, arr.shape)

    noisy_arr = np.clip((arr + noise) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


def add_gaussian_noise(img: Image.Image, sigma: float, rng: Optional[np.random.Generator] = None) -> Image.Image:
    return add_realistic_noise(img, noise_type="gaussian", sigma=sigma, rng=rng)


def extract_patches(
    img: Image.Image,
    patch_size: int = 256,
    num_crops: int = 3,
    rng: Optional[random.Random] = None,
) -> List[Tuple[Tuple[int, int], Image.Image]]:
    w, h = img.size
    if w < patch_size or h < patch_size:
        return []

    r = rng if rng is not None else random.Random()
    patches: List[Tuple[Tuple[int, int], Image.Image]] = []

    for _ in range(num_crops):
        best_left, best_top = 0, 0
        best_patch = None
        best_std = -1.0

        for _ in range(8):
            left = r.randint(0, w - patch_size)
            top = r.randint(0, h - patch_size)
            candidate = img.crop((left, top, left + patch_size, top + patch_size))
            std_val = float(np.std(np.array(candidate, dtype=np.float32)))
            if std_val > 5.0:
                best_left, best_top, best_patch = left, top, candidate
                break
            if std_val > best_std:
                best_std = std_val
                best_left, best_top, best_patch = left, top, candidate

        if best_patch is None:
            best_patch = img.crop((best_left, best_top, best_left + patch_size, best_top + patch_size))

        patches.append(((best_left, best_top), best_patch))

    return patches


def split_source_images(
    image_paths: List[str],
    split_ratios: Tuple[float, float, float] = (0.8, 0.2, 0.0),
    seed: int = 42,
) -> Dict[str, List[str]]:
    sorted_paths = sorted(image_paths)
    r = random.Random(seed)
    shuffled = list(sorted_paths)
    r.shuffle(shuffled)

    n = len(shuffled)
    r_train, r_val, r_test = split_ratios
    total_ratio = r_train + r_val + r_test
    if total_ratio <= 0 or n <= 1:
        return {"train": shuffled, "val": [], "test": []}

    r_train /= total_ratio
    r_val /= total_ratio
    r_test /= total_ratio

    if r_val == 0.0 and r_test == 0.0:
        return {"train": shuffled, "val": [], "test": []}

    n_val = int(round(n * r_val))
    n_test = int(round(n * r_test))
    if r_val > 0 and n_val == 0 and n >= 2:
        n_val = 1
    if r_test > 0 and n_test == 0 and n >= 3:
        n_test = 1

    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def process_sample_image(
    image_path: str,
    output_clean_dir: str,
    output_noisy_dir: str,
    split: str = "train",
    patch_size: int = 256,
    num_crops: int = 3,
    num_versions: int = 1,
    jpeg_qualities: Optional[List[int]] = None,
    quality_range: Optional[Tuple[int, int]] = (20, 80),
    task_mode: str = "restoration",
    noise_type: str = "sensor",
    noise_range: Tuple[float, float] = (0.01, 0.04),
    p_noise: float = 1.0,
    p_jpeg: float = 0.7,
    p_downsample: float = 0.4,
    downsample_range: Tuple[float, float] = (0.70, 0.95),
    seed: int = 42,
    progress_callback=None,
    num_random_versions: int = 0,
) -> List[Dict]:
    if not os.path.exists(image_path):
        return []

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Failed to open image {image_path}: {e}")
        return []

    if img.width < patch_size or img.height < patch_size:
        print(f"Warning: Skipping {image_path}: size {img.size} is smaller than patch size {patch_size}x{patch_size}")
        return []

    os.makedirs(output_clean_dir, exist_ok=True)
    os.makedirs(output_noisy_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    py_rng = random.Random(seed)

    patches = extract_patches(img, patch_size=patch_size, num_crops=num_crops, rng=py_rng)
    if not patches:
        return []

    if num_random_versions > 0:
        actual_versions = num_random_versions
    elif jpeg_qualities is not None and len(jpeg_qualities) > 0:
        actual_versions = len(jpeg_qualities)
    else:
        actual_versions = max(1, num_versions)

    is_pure_denoise = (task_mode == "denoise")
    effective_p_jpeg = 0.0 if is_pure_denoise else p_jpeg
    effective_p_down = 0.0 if is_pure_denoise else p_downsample
    effective_p_noise = 1.0 if is_pure_denoise else p_noise

    generated_metadata: List[Dict] = []

    for crop_idx, ((left, top), clean_patch) in enumerate(patches, start=1):
        if clean_patch.size != (patch_size, patch_size):
            raise ValueError(f"Crop size {clean_patch.size} does not match expected patch size {(patch_size, patch_size)}")

        for v_idx in range(1, actual_versions + 1):
            sample_seed = (seed * 31 + crop_idx * 101 + v_idx * 1009) & 0x7FFFFFFF
            s_rng = random.Random(sample_seed)
            s_np_rng = np.random.default_rng(sample_seed)

            noisy_patch = clean_patch

            apply_down = s_rng.random() < effective_p_down
            apply_jpeg = s_rng.random() < effective_p_jpeg
            apply_noise = s_rng.random() < effective_p_noise

            if not is_pure_denoise and not (apply_down or apply_jpeg or apply_noise):
                apply_noise = True

            resize_scale = None
            if apply_down:
                resize_scale = round(s_rng.uniform(downsample_range[0], downsample_range[1]), 3)
                noisy_patch = apply_downsample_upsample(noisy_patch, resize_scale)

            jpeg_quality = None
            if apply_jpeg:
                if jpeg_qualities and len(jpeg_qualities) > 0:
                    if v_idx - 1 < len(jpeg_qualities) and num_random_versions == 0:
                        jpeg_quality = int(jpeg_qualities[v_idx - 1])
                    else:
                        jpeg_quality = int(s_rng.choice(jpeg_qualities))
                elif quality_range:
                    jpeg_quality = s_rng.randint(min(quality_range), max(quality_range))
                else:
                    jpeg_quality = s_rng.randint(20, 80)
                noisy_patch = apply_jpeg_compression(noisy_patch, quality=jpeg_quality)

            noise_sigma = 0.0
            actual_noise_type = "none"
            if apply_noise:
                actual_noise_type = noise_type
                noise_sigma = round(float(s_rng.uniform(noise_range[0], noise_range[1])), 4)
                noisy_patch = add_realistic_noise(
                    noisy_patch,
                    noise_type=actual_noise_type,
                    sigma=noise_sigma,
                    rng=s_np_rng,
                )

            if noisy_patch.size != clean_patch.size:
                raise ValueError(f"Noisy patch size {noisy_patch.size} does not match clean patch size {clean_patch.size}")

            q_tag = f"_q{jpeg_quality}" if jpeg_quality is not None else "_nojpeg"
            pair_id = f"{base_name}_p{crop_idx:02d}_v{v_idx:02d}{q_tag}"
            clean_filename = f"{pair_id}_clean.png"
            noisy_filename = f"{pair_id}_noisy.png"

            clean_out = os.path.join(output_clean_dir, clean_filename)
            noisy_out = os.path.join(output_noisy_dir, noisy_filename)

            clean_patch.save(clean_out, format="PNG")
            noisy_patch.save(noisy_out, format="PNG")

            if not (os.path.exists(clean_out) and os.path.getsize(clean_out) > 0):
                raise IOError(f"Failed to write output to {clean_out}")
            if not (os.path.exists(noisy_out) and os.path.getsize(noisy_out) > 0):
                raise IOError(f"Failed to write output to {noisy_out}")

            meta = {
                "pair_id": pair_id,
                "source_id": base_name,
                "split": split,
                "crop_position": [left, top],
                "crop_size": [patch_size, patch_size],
                "noise_type": actual_noise_type,
                "noise_sigma": noise_sigma,
                "jpeg_quality": jpeg_quality,
                "resize_scale": resize_scale,
                "seed": sample_seed,
                "clean_file": clean_filename,
                "noisy_file": noisy_filename,
            }
            generated_metadata.append(meta)

            if progress_callback:
                progress_callback(1)

    return generated_metadata


def process_all_samples(
    samples_dir: str = "samples",
    output_dir: str = "samples/processed",
    output_clean_dir: Optional[str] = None,
    output_noisy_dir: Optional[str] = None,
    split_ratios: Tuple[float, float, float] = (0.8, 0.2, 0.0),
    patch_size: int = 256,
    num_crops: int = 3,
    num_random_versions: int = 0,
    jpeg_qualities: Optional[List[int]] = None,
    quality_range: Optional[Tuple[int, int]] = (20, 80),
    task_mode: str = "restoration",
    noise_type: str = "sensor",
    noise_range: Tuple[float, float] = (0.01, 0.04),
    p_noise: float = 1.0,
    p_jpeg: float = 0.7,
    p_downsample: float = 0.4,
    downsample_range: Tuple[float, float] = (0.70, 0.95),
    seed: int = 42,
    progress_callback=None,
    max_workers: Optional[int] = None,
) -> int:
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(samples_dir, ext)))

    if not image_paths:
        print(f"No valid images found in '{samples_dir}'.")
        return 0

    use_splits = split_ratios is not None and (split_ratios[1] > 0 or split_ratios[2] > 0)
    base_out = output_dir
    if output_clean_dir is not None and output_dir == "samples/processed":
        base_out = os.path.dirname(output_clean_dir) or output_dir

    split_map = split_source_images(image_paths, split_ratios=split_ratios, seed=seed) if use_splits else {"train": sorted(image_paths)}

    num_versions = num_random_versions if num_random_versions > 0 else (len(jpeg_qualities) if jpeg_qualities else 1)
    expected_samples = len(image_paths) * num_crops * num_versions

    workers = max_workers or min(8, os.cpu_count() or 4)
    total_pairs = 0
    lock = threading.Lock()

    def wrapped_cb(count: int):
        nonlocal total_pairs
        with lock:
            total_pairs += count
            if progress_callback:
                progress_callback(total_pairs)

    tasks = []
    for split_name, paths in split_map.items():
        if not paths:
            continue
        if use_splits:
            s_clean_dir = os.path.join(base_out, split_name, "clean")
            s_noisy_dir = os.path.join(base_out, split_name, "noisy")
        else:
            s_clean_dir = output_clean_dir or os.path.join(base_out, "clean")
            s_noisy_dir = output_noisy_dir or os.path.join(base_out, "noisy")

        for idx, p in enumerate(paths):
            img_seed = (seed * 10007 + idx * 997) & 0x7FFFFFFF
            tasks.append((p, s_clean_dir, s_noisy_dir, split_name, img_seed))

    all_metadata: List[Dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_sample_image,
                p,
                clean_dir,
                noisy_dir,
                split=split_name,
                patch_size=patch_size,
                num_crops=num_crops,
                num_versions=num_versions,
                jpeg_qualities=jpeg_qualities,
                quality_range=quality_range,
                task_mode=task_mode,
                noise_type=noise_type,
                noise_range=noise_range,
                p_noise=p_noise,
                p_jpeg=p_jpeg,
                p_downsample=p_downsample,
                downsample_range=downsample_range,
                seed=img_seed,
                progress_callback=wrapped_cb if progress_callback else None,
                num_random_versions=num_random_versions,
            )
            for p, clean_dir, noisy_dir, split_name, img_seed in tasks
        ]
        for f in as_completed(futures):
            all_metadata.extend(f.result())

    if len(all_metadata) != expected_samples:
        print(f"Note: Generated {len(all_metadata)} samples (expected up to {expected_samples}; skipped if images smaller than patch size).")

    split_counts: Dict[str, int] = {}
    for item in all_metadata:
        s = item.get("split", "train")
        split_counts[s] = split_counts.get(s, 0) + 1

    metadata_payload = {
        "dataset_version": "1.0.0",
        "config_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "task_mode": task_mode,
            "seed": seed,
            "patch_size": patch_size,
            "num_crops": num_crops,
            "num_versions": num_versions,
            "split_ratios": list(split_ratios) if split_ratios else [1.0, 0.0, 0.0],
            "noise_type": noise_type,
            "noise_range": list(noise_range),
            "p_noise": p_noise,
            "p_jpeg": p_jpeg,
            "jpeg_range": list(quality_range) if quality_range else None,
            "p_downsample": p_downsample,
            "downsample_range": list(downsample_range),
        },
        "stats": {
            "expected_samples": expected_samples,
            "total_samples": len(all_metadata),
            "by_split": split_counts,
        },
        "samples": all_metadata,
    }

    os.makedirs(base_out, exist_ok=True)
    meta_path = os.path.join(base_out, "dataset_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

    return len(all_metadata)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process sample images into clean-noisy training pairs")
    parser.add_argument("--samples-dir", default="samples", help="Input directory containing source images")
    parser.add_argument("--output-dir", default="samples/processed", help="Directory where processed datasets will be saved")
    parser.add_argument("--patch-size", type=int, default=256, help="Fixed crop patch size (e.g. 128, 192, 256)")
    parser.add_argument("--crops", type=int, default=3, help="Number of fixed-size patches per image")
    parser.add_argument("--random", type=int, default=0, help="Number of random versions per patch (0 for quality list)")
    parser.add_argument("--qualities", nargs="+", type=int, default=None, help="Discrete JPEG quality levels (e.g. 15 20 40 60)")
    parser.add_argument("--range", nargs=2, type=int, default=[20, 80], help="JPEG quality range (min max)")
    parser.add_argument("--task-mode", default="restoration", choices=["restoration", "denoise"], help="Task mode: restoration or pure denoise")
    parser.add_argument("--noise-type", default="sensor", choices=["sensor", "gaussian", "poisson"], help="Noise model type")
    parser.add_argument("--noise-range", nargs=2, type=float, default=[0.01, 0.04], help="Noise sigma range (min max)")
    parser.add_argument("--p-noise", type=float, default=1.0, help="Probability of applying noise (restoration mode)")
    parser.add_argument("--p-jpeg", type=float, default=0.7, help="Probability of applying JPEG compression (restoration mode)")
    parser.add_argument("--p-downsample", type=float, default=0.4, help="Probability of downsample/upsample degradation")
    parser.add_argument("--downsample-range", nargs=2, type=float, default=[0.70, 0.95], help="Downsample scale range (min max)")
    parser.add_argument("--split-ratios", nargs=3, type=float, default=[0.8, 0.2, 0.0], help="Train/val/test split ratios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=None, help="Max worker threads")
    args = parser.parse_args()

    q_range = tuple(args.range) if args.range else (20, 80)
    print("Processing samples directory...")
    n = process_all_samples(
        samples_dir=args.samples_dir,
        output_dir=args.output_dir,
        split_ratios=tuple(args.split_ratios),
        patch_size=args.patch_size,
        num_crops=args.crops,
        num_random_versions=args.random,
        jpeg_qualities=args.qualities,
        quality_range=q_range,
        task_mode=args.task_mode,
        noise_type=args.noise_type,
        noise_range=tuple(args.noise_range),
        p_noise=args.p_noise,
        p_jpeg=args.p_jpeg,
        p_downsample=args.p_downsample,
        downsample_range=tuple(args.downsample_range),
        seed=args.seed,
        max_workers=args.workers,
    )
    print(f"Generated {n} paired training samples.")
