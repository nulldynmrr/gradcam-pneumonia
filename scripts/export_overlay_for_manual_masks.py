"""Skrip visualisasi kontur batas paru U-Net (overlay) pada citra rontgen CXR.

Menghasilkan citra overlay visualisasi batas kontur paru (hijau) di atas citra rontgen dada
untuk verifikasi visual dokter / penulis skripsi mengacu pada Bab III (§3.2.2, L73).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.paths import ensure_output_dirs, get_project_paths, load_config
from src.segmentation.lung_mask import (
    binarize_probability_map,
    compute_r_paru,
    postprocess_lung_mask,
    resize_mask_to_224,
)
from src.segmentation.unet import load_frozen_unet, predict_lung_probability


def create_contour_overlay(
    image: np.ndarray | Image.Image,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),  # Hijau
    thickness: int = 2,
) -> np.ndarray:
    """Tumpuk garis kontur mask biner di atas citra grayscale CXR."""
    if isinstance(image, Image.Image):
        im_arr = np.array(image.convert("RGB"))
    else:
        if image.ndim == 2:
            im_arr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            im_arr = image.copy()

    # Pastikan ukuran citra dan mask sama
    if im_arr.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (im_arr.shape[1], im_arr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Temukan kontur
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = im_arr.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay


def run_overlay_export(
    n_samples_per_class: int = 10,
    output_dir: Path | None = None,
) -> None:
    """Jalankan ekstraksi kontur U-Net dan ekspor overlay untuk sampel data uji Kermany."""
    cfg = load_config()
    paths = ensure_output_dirs(cfg=cfg)
    project_root = paths["project_root"]

    if output_dir is None:
        output_dir = project_root / "outputs" / "figures" / "lung_mask_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = paths["unet_frozen"] / "weights.pth"
    if not weights_path.exists():
        print(f"[ERROR] Bobot U-Net tidak ditemukan di: {weights_path}")
        print("Silakan unduh terlebih dahulu dengan: python scripts/download_unet_weights.py")
        return

    print("Memuat model U-Net beku...")
    model = load_frozen_unet(weights_path)

    # Baca file split
    split_csv = paths["interim"] / "split_patientwise.csv"
    if not split_csv.exists():
        print(f"[ERROR] Berkas split tidak ditemukan di: {split_csv}")
        return

    split_df = pd.read_csv(split_csv)
    test_df = split_df[split_df["split"] == "test"]

    kermany_root = paths["kermany_chest_xray"]
    print(f"Mengekspor {n_samples_per_class} citra per kelas ke {output_dir}...")

    for label in ["Normal", "Pneumonia"]:
        sub = test_df[test_df["label"] == label].sample(n=min(n_samples_per_class, len(test_df[test_df["label"] == label])), random_state=42)
        for _, row in sub.iterrows():
            img_p = kermany_root / row["rel_path"]
            with Image.open(img_p) as im:
                prob = predict_lung_probability(model, im, return_original_size=True)
                bin_mask = binarize_probability_map(prob, threshold=0.5)
                clean_mask = postprocess_lung_mask(bin_mask, keep_components=2, fill_holes=True)
                r_paru = compute_r_paru(clean_mask)

                overlay = create_contour_overlay(im, clean_mask, color=(0, 255, 0), thickness=2)

                # Simpan citra overlay
                out_name = f"{label}_{row['image_id']}_r{r_paru:.3f}.png"
                cv2.imwrite(str(output_dir / out_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"[OK] Seluruh citra visualisasi overlay kontur paru berhasil diekspor ke: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekspor overlay kontur batas paru U-Net")
    parser.add_argument("--n", type=int, default=10, help="Jumlah sampel visualisasi per kelas")
    args = parser.parse_args()
    run_overlay_export(n_samples_per_class=args.n)
