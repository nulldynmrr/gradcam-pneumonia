"""Skrip ekspor overlay kontur batas paru U-Net & daftar 50 sampel uji Kermany (L3(2)).

Mengacu pada Bab III (§3.2.2, L73) dan `docs/1.md` L3(2):
- Memilih 50 citra dari `D_test` (25 Normal + 25 Pneumonia) dengan seed tetap dan menyimpan daftarnya
  ke `data/interim/kermany_qc_50_list.csv`.
- Mengekspor overlay kontur hijau U-Net ke `outputs/figures/lung_mask_overlays/` untuk inspeksi visual
  serta membantu pembuatan kontur acuan manual penulis di `data/raw/kermany_manual_masks/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.paths import ensure_output_dirs, load_config
from src.segmentation.lung_mask import (
    binarize_probability_map,
    compute_r_paru,
    postprocess_lung_mask,
)
from src.segmentation.qc import select_kermany_qc_subset
from src.segmentation.unet import load_frozen_unet, predict_lung_probability


def create_contour_overlay(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Tumpuk garis kontur mask biner di atas citra grayscale CXR."""
    if isinstance(image, Image.Image):
        im_arr = np.array(image.convert("RGB"))
    else:
        if image.ndim == 2:
            if image.dtype != np.uint8:
                image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            im_arr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            im_arr = image.copy()

    if im_arr.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (im_arr.shape[1], im_arr.shape[0]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = im_arr.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def run_overlay_export(
    n_samples_per_class: int | None = None,
    output_dir: Path | None = None,
    unet_model: Any | None = None,
    device: Any | None = None,
    seed: int | None = None,
) -> List[Path]:
    """Jalankan ekstraksi kontur U-Net dan ekspor overlay untuk sampel D_test Kermany."""
    cfg = load_config()
    paths = ensure_output_dirs(cfg=cfg)
    lm_cfg = cfg["locked"]["lung_mask"]

    if n_samples_per_class is None:
        n_samples_per_class = int(lm_cfg["qc_kermany_normal"]["value"])
    if seed is None:
        seed = int(cfg["assumed"]["qc_kermany_selection_seed"]["value"])
    if output_dir is None:
        output_dir = paths["outputs"] / "figures" / "lung_mask_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    if unet_model is None:
        weights_path = paths["unet_frozen"] / "weights.pth"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Bobot U-Net tidak ditemukan di: {weights_path}. "
                "Jalankan: python scripts/download_unet_weights.py"
            )
        unet_model = load_frozen_unet(weights_path, device=device)

    split_csv = paths["interim"] / "split_patientwise.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Berkas split tidak ditemukan di: {split_csv}")

    split_df = pd.read_csv(split_csv)
    subset_csv = paths["interim"] / "kermany_qc_50_list.csv"
    subset_df = select_kermany_qc_subset(
        split_df, n_per_class=n_samples_per_class, seed=seed, out_csv=subset_csv
    )

    kermany_root = paths["kermany_chest_xray"]
    bin_thr = float(lm_cfg["binarize_threshold"]["value"])
    keep_comp = int(lm_cfg["keep_largest_components"]["value"])
    fill_holes = bool(lm_cfg["hole_filling"]["value"])

    saved_paths: List[Path] = []
    for _, row in subset_df.iterrows():
        img_p = kermany_root / row["rel_path"]
        with Image.open(img_p) as im:
            prob = predict_lung_probability(unet_model, im, device=device, return_original_size=True)
            bin_mask = binarize_probability_map(prob, threshold=bin_thr)
            clean_mask = postprocess_lung_mask(bin_mask, keep_components=keep_comp, fill_holes=fill_holes)
            r_paru = compute_r_paru(clean_mask)

            overlay = create_contour_overlay(im, clean_mask, color=(0, 255, 0), thickness=2)
            out_path = output_dir / f"{row['label']}_{row['image_id']}_r{r_paru:.3f}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            saved_paths.append(out_path)

    print(f"[OK] {len(saved_paths)} citra overlay kontur paru diekspor ke: {output_dir}")
    print(f"[OK] Daftar subset terkunci tersimpan di: {subset_csv}")
    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekspor overlay kontur batas paru U-Net (50 sampel D_test)")
    parser.add_argument("--n", type=int, default=25, help="Jumlah sampel visualisasi per kelas (default: 25)")
    args = parser.parse_args()
    run_overlay_export(n_samples_per_class=args.n)
