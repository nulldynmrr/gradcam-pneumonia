"""Modul Quality Control (QC) segmentasi paru U-Net mengacu pada Bab III (§3.2.2, L71-74).

Target QC proposal:
- Rata-rata per kelas: DSC >= 0.85 dan IoU >= 0.75.
- Evaluasi pada 85 citra RSUA Validated (32 Normal, 53 Pneumonia).
- Evaluasi pada subset citra uji Kermany (>= 50 citra: 25 Normal, 25 Pneumonia).
- Pencatatan kegagalan deteksi paru ekstrem (r_paru < 0.05).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.segmentation.lung_mask import (
    binarize_probability_map,
    compute_r_paru,
    postprocess_lung_mask,
    resize_mask_to_224,
)
from src.segmentation.unet import predict_lung_probability


def compute_dice_coefficient(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Hitung Dice Similarity Coefficient (DSC): 2 * |P cap G| / (|P| + |G| + eps)."""
    p = (pred_mask > 0).astype(np.float32)
    g = (gt_mask > 0).astype(np.float32)
    intersection = np.sum(p * g)
    sum_areas = np.sum(p) + np.sum(g)
    if sum_areas == 0:
        return 1.0  # dua-duanya kosong dianggap cocok sempurna
    return float((2.0 * intersection + eps) / (sum_areas + eps))


def compute_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Hitung Intersection over Union (IoU): |P cap G| / (|P cup G| + eps)."""
    p = (pred_mask > 0).astype(np.float32)
    g = (gt_mask > 0).astype(np.float32)
    intersection = np.sum(p * g)
    union = np.sum(np.clip(p + g, 0, 1))
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def evaluate_rsua_dataset(
    unet_model: Any,
    rsua_validated_root: Path,
    device: Any = None,
) -> Dict[str, Any]:
    """Evaluasi kuantitatif U-Net pada 85 citra RSUA Validated (32 Normal, 53 Pneumonia).

    Args:
        unet_model: Model U-Net beku.
        rsua_validated_root: Direktori Data Chest X-Ray RSUA (Validated).
        device: Device torch.

    Returns:
        Dict hasil evaluasi DSC, IoU per kelas, dan status kelulusan target proposal.
    """
    classes_info = [
        ("Normal", "Non_Covid", 32),
        ("Pneumonia", "Non_Covid_Pneumonia", 53),
    ]

    records: List[Dict[str, Any]] = []

    for label_name, dir_name, expected_count in classes_info:
        img_dir = rsua_validated_root / dir_name / "images"
        mask_dir = rsua_validated_root / dir_name / "masks"

        assert img_dir.exists(), f"Direktori citra RSUA tidak ditemukan: {img_dir}"
        assert mask_dir.exists(), f"Direktori mask RSUA tidak ditemukan: {mask_dir}"

        img_files = sorted(list(img_dir.glob("*.bmp")))
        assert len(img_files) == expected_count, (
            f"Jumlah citra RSUA {label_name} ({len(img_files)}) tidak sesuai target ({expected_count})!"
        )

        for img_p in img_files:
            # Mask bersesuaian: CXR_Images_X.bmp -> CXR_Mask_X.bmp
            mask_filename = img_p.name.replace("CXR_Images_", "CXR_Mask_")
            mask_p = mask_dir / mask_filename
            assert mask_p.exists(), f"Mask rujukan dokter tidak ditemukan: {mask_p}"

            # Baca citra asli dan mask GT dokter
            im = Image.open(img_p).convert("L")
            gt_arr = np.array(Image.open(mask_p).convert("L"), dtype=np.uint8)
            gt_bin = (gt_arr > 0).astype(np.uint8)

            # Inferensi U-Net
            prob_map = predict_lung_probability(unet_model, im, device=device, return_original_size=True)
            bin_raw = binarize_probability_map(prob_map, threshold=0.5)
            pred_mask = postprocess_lung_mask(bin_raw, keep_components=2, fill_holes=True)

            dsc = compute_dice_coefficient(pred_mask, gt_bin)
            iou = compute_iou(pred_mask, gt_bin)
            r_paru = compute_r_paru(pred_mask)

            records.append({
                "dataset": "RSUA",
                "filename": img_p.name,
                "label": label_name,
                "dice": dsc,
                "iou": iou,
                "r_paru": r_paru,
            })

    df = pd.DataFrame(records)

    # Hitung metrik ringkasan per kelas
    summary_by_class: Dict[str, Dict[str, float]] = {}
    for cls in ["Normal", "Pneumonia"]:
        sub = df[df["label"] == cls]
        summary_by_class[cls] = {
            "n_samples": int(len(sub)),
            "dice_mean": float(sub["dice"].mean()),
            "dice_std": float(sub["dice"].std()),
            "dice_min": float(sub["dice"].min()),
            "iou_mean": float(sub["iou"].mean()),
            "iou_std": float(sub["iou"].std()),
            "iou_min": float(sub["iou"].min()),
            "r_paru_mean": float(sub["r_paru"].mean()),
        }

    # Cek kriteria proposal (Bab III L14, L73): DSC >= 0.85 dan IoU >= 0.75 per kelas
    normal_pass = (summary_by_class["Normal"]["dice_mean"] >= 0.85) and (summary_by_class["Normal"]["iou_mean"] >= 0.75)
    pneumonia_pass = (summary_by_class["Pneumonia"]["dice_mean"] >= 0.85) and (summary_by_class["Pneumonia"]["iou_mean"] >= 0.75)
    overall_pass = normal_pass and pneumonia_pass

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "criteria": {"dsc_min": 0.85, "iou_min": 0.75},
        "normal_pass": normal_pass,
        "pneumonia_pass": pneumonia_pass,
        "summary_by_class": summary_by_class,
        "details_df": df,
    }


def evaluate_kermany_subset_r_paru(
    unet_model: Any,
    manifest_df: pd.DataFrame,
    kermany_root: Path,
    device: Any = None,
    n_per_class: int = 25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Pemeriksaan langsung pada citra anak Kermany dkk. (>= 50 citra: 25 Normal, 25 Pneumonia).

    Menghitung sebaran r_paru, mendeteksi kegagalan ekstrem (r_paru < 0.05), dan mengukur
    selisih rata-rata luas paru (|delta_r_paru| > 0.05) mengacu pada Bab III L73 & L77.
    """
    # Filter data uji jika kolom split tersedia, jika tidak ambil dari manifest
    test_df = manifest_df[manifest_df["split"] == "test"] if "split" in manifest_df.columns else manifest_df

    normal_samples = test_df[test_df["label"] == "Normal"].sample(n=n_per_class, random_state=seed)
    pneumonia_samples = test_df[test_df["label"] == "Pneumonia"].sample(n=n_per_class, random_state=seed)
    selected_samples = pd.concat([normal_samples, pneumonia_samples]).reset_index(drop=True)

    records: List[Dict[str, Any]] = []

    for _, row in selected_samples.iterrows():
        img_p = kermany_root / row["rel_path"]
        with Image.open(img_p) as im:
            prob_map = predict_lung_probability(unet_model, im, device=device, return_original_size=True)
            bin_raw = binarize_probability_map(prob_map, threshold=0.5)
            pred_mask = postprocess_lung_mask(bin_raw, keep_components=2, fill_holes=True)
            m224 = resize_mask_to_224(pred_mask)

            r_orig = compute_r_paru(pred_mask)
            r_224 = compute_r_paru(m224)

            records.append({
                "image_id": row.get("image_id", img_p.stem),
                "rel_path": row["rel_path"],
                "label": row["label"],
                "patient_id": row.get("patient_id", ""),
                "r_paru_orig": r_orig,
                "r_paru_224": r_224,
                "extreme_failure": r_224 < 0.05,
            })

    res_df = pd.DataFrame(records)

    mean_r_norm = float(res_df[res_df["label"] == "Normal"]["r_paru_224"].mean())
    mean_r_pneu = float(res_df[res_df["label"] == "Pneumonia"]["r_paru_224"].mean())
    delta_r = abs(mean_r_norm - mean_r_pneu)
    extreme_failures = int(res_df["extreme_failure"].sum())

    return {
        "n_evaluated": len(res_df),
        "normal_evaluated": int((res_df["label"] == "Normal").sum()),
        "pneumonia_evaluated": int((res_df["label"] == "Pneumonia").sum()),
        "mean_r_paru_normal": mean_r_norm,
        "mean_r_paru_pneumonia": mean_r_pneu,
        "delta_r_paru": delta_r,
        "triggers_acr_norm": delta_r > 0.05,
        "extreme_failure_count": extreme_failures,
        "details_df": res_df,
    }
