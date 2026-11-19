"""Modul caching satu-kali (resumable) untuk citra I_224 (float32) dan lung mask M_224 (uint8 biner).

Mengacu pada:
- `docs/2.md` Aturan 7: U-Net dijalankan pada SEMUA citra (train/val/test) SATU kali;
  hasilnya di-cache di `data/processed/`.
- `docs/1.md` baris 103 & AMBIG-17: cache `I_224` sebagai `float32` [0,1] (tanpa galat kuantisasi uint8)
  dan `M_224` sebagai `uint8` biner murni `{0, 1}`.
- Verifikasi integritas split: memeriksa checksum SHA-256 `split_patientwise.csv` sebelum berjalan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.data.split import verify_locked_split
from src.data.transforms import load_gray01, resize_image_to_224
from src.segmentation.lung_mask import (
    binarize_probability_map,
    compute_r_paru,
    postprocess_lung_mask,
    resize_mask_to_224,
)
from src.segmentation.unet import predict_lung_probability


@dataclass(frozen=True)
class PreprocessCacheConfig:
    """Konfigurasi terpusat untuk pipeline caching U-Net + resize 224x224 (tanpa magic number)."""

    image_size: int
    binarize_threshold: float
    keep_largest_components: int
    hole_filling: bool
    r_paru_extreme_threshold: float

    @classmethod
    def from_yaml_cfg(cls, cfg: dict) -> "PreprocessCacheConfig":
        lm = cfg["locked"]["lung_mask"]
        pp = cfg["locked"]["preprocess"]
        return cls(
            image_size=int(pp["image_size"]["value"]),
            binarize_threshold=float(lm["binarize_threshold"]["value"]),
            keep_largest_components=int(lm["keep_largest_components"]["value"]),
            hole_filling=bool(lm["hole_filling"]["value"]),
            r_paru_extreme_threshold=float(lm["r_paru_extreme_threshold"]["value"]),
        )


def run_full_lung_mask_cache(
    unet_model: Any,
    split_csv_path: Path,
    kermany_root: Path,
    processed_dir: Path,
    cache_cfg: PreprocessCacheConfig,
    device: Any = None,
    verify_split_checksum: bool = True,
    max_samples: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Jalankan inferensi U-Net beku & resize 224x224 secara resumable untuk seluruh citra Kermany.

    Menyimpan per citra:
    - `processed_dir/images_224/<image_id>.npy` : `float32` (224, 224) rentang [0.0, 1.0] (bilinear)
    - `processed_dir/masks_224/<image_id>.npy`  : `uint8` (224, 224) nilai `{0, 1}` (nearest)
    - `processed_dir/lung_mask_metadata.csv`    : tabel metadata & `r_paru` lengkap seluruh citra.
    """
    if verify_split_checksum:
        verify_locked_split(split_csv_path)

    split_df = pd.read_csv(split_csv_path)
    if max_samples is not None:
        split_df = split_df.head(max_samples).copy()

    img_out_dir = processed_dir / "images_224"
    mask_out_dir = processed_dir / "masks_224"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    meta_csv = processed_dir / "lung_mask_metadata.csv"
    existing_meta: Dict[str, Dict[str, Any]] = {}
    if meta_csv.exists():
        old_df = pd.read_csv(meta_csv)
        for rec in old_df.to_dict(orient="records"):
            existing_meta[str(rec["image_id"])] = rec

    target_hw = (cache_cfg.image_size, cache_cfg.image_size)
    records: List[Dict[str, Any]] = []

    iterator = split_df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(split_df), desc="Caching I_224 & M_224 (U-Net)")

    for _, row in iterator:
        image_id = str(row["image_id"])
        img_npy = img_out_dir / f"{image_id}.npy"
        mask_npy = mask_out_dir / f"{image_id}.npy"

        # Resumable: lewati jika kedua berkas cache valid & metadata sudah tercatat
        if img_npy.exists() and mask_npy.exists() and image_id in existing_meta:
            records.append(existing_meta[image_id])
            continue

        img_path = kermany_root / str(row["rel_path"])
        assert img_path.exists(), f"Citra Kermany tidak ditemukan: {img_path}"

        with Image.open(img_path) as im:
            orig_w, orig_h = im.size
            gray01 = load_gray01(im)
            i224 = resize_image_to_224(gray01, target_size=target_hw)

            prob_map = predict_lung_probability(
                unet_model, im, device=device, return_original_size=True
            )
            bin_raw = binarize_probability_map(prob_map, threshold=cache_cfg.binarize_threshold)
            pred_mask_orig = postprocess_lung_mask(
                bin_raw,
                keep_components=cache_cfg.keep_largest_components,
                fill_holes=cache_cfg.hole_filling,
            )
            m224 = resize_mask_to_224(pred_mask_orig, target_size=target_hw)

        # Validasi kontrak tipe & nilai sebelum tulis ke disk
        assert i224.dtype == np.float32 and i224.shape == target_hw
        assert m224.dtype == np.uint8 and m224.shape == target_hw
        assert set(np.unique(m224)).issubset({0, 1}), f"Mask {image_id} bukan biner!"

        np.save(img_npy, i224)
        np.save(mask_npy, m224)

        r_orig = compute_r_paru(pred_mask_orig)
        r_224 = compute_r_paru(m224)

        rec = {
            "image_id": image_id,
            "rel_path": str(row["rel_path"]),
            "split": str(row["split"]),
            "label": str(row["label"]),
            "label_id": int(row["label_id"]),
            "patient_id": str(row["patient_id"]),
            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
            "r_paru_orig": float(r_orig),
            "r_paru_224": float(r_224),
            "extreme_failure": bool(r_224 < cache_cfg.r_paru_extreme_threshold),
            "image_224_path": f"images_224/{image_id}.npy",
            "mask_224_path": f"masks_224/{image_id}.npy",
        }
        records.append(rec)

    meta_df = pd.DataFrame(records)
    meta_df.to_csv(meta_csv, index=False, encoding="utf-8")
    return meta_df


def verify_cache_integrity(
    split_csv_path: Path,
    processed_dir: Path,
    sample_check_n: int = 20,
    seed: int = 42,
    expected_size: int = 224,
) -> Dict[str, Any]:
    """Verifikasi kelengkapan & konsistensi cache I_224 & M_224 terhadap split terkunci."""
    split_df = pd.read_csv(split_csv_path)
    meta_csv = processed_dir / "lung_mask_metadata.csv"

    if not meta_csv.exists():
        return {
            "status": "FAIL",
            "reason": f"Metadata cache tidak ditemukan di {meta_csv}",
            "n_expected": int(len(split_df)),
            "n_cached": 0,
        }

    meta_df = pd.read_csv(meta_csv)
    expected_ids = set(split_df["image_id"].astype(str))
    cached_ids = set(meta_df["image_id"].astype(str))
    missing_ids = sorted(expected_ids - cached_ids)

    missing_files = 0
    for _, r in meta_df.iterrows():
        if not (processed_dir / r["image_224_path"]).exists() or not (processed_dir / r["mask_224_path"]).exists():
            missing_files += 1

    # Cek mendalam tipe data, rentang, binaritas, & alignment pada sampel acak (seed tetap)
    check_df = meta_df.sample(n=min(sample_check_n, len(meta_df)), random_state=seed)
    dtype_ok = True
    binary_ok = True
    shape_ok = True
    range_ok = True

    for _, r in check_df.iterrows():
        i224 = np.load(processed_dir / r["image_224_path"])
        m224 = np.load(processed_dir / r["mask_224_path"])
        if i224.dtype != np.float32 or m224.dtype != np.uint8:
            dtype_ok = False
        if i224.shape != (expected_size, expected_size) or m224.shape != (expected_size, expected_size):
            shape_ok = False
        if float(i224.min()) < 0.0 or float(i224.max()) > 1.0:
            range_ok = False
        if not set(np.unique(m224)).issubset({0, 1}):
            binary_ok = False

    passed = (
        len(missing_ids) == 0
        and missing_files == 0
        and dtype_ok
        and binary_ok
        and shape_ok
        and range_ok
    )

    return {
        "status": "PASS" if passed else "FAIL",
        "n_expected": int(len(expected_ids)),
        "n_cached": int(len(cached_ids)),
        "n_missing_ids": int(len(missing_ids)),
        "n_missing_files": int(missing_files),
        "sample_checked": int(len(check_df)),
        "dtype_ok": dtype_ok,
        "shape_ok": shape_ok,
        "range_ok": range_ok,
        "binary_ok": binary_ok,
    }
