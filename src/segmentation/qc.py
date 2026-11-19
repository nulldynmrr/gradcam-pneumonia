"""Modul Quality Control (QC) segmentasi paru U-Net mengacu pada Bab III (§3.2.2, L71-77).

Target QC proposal (terkunci di `configs/config.yaml::locked.lung_mask`):
- Rata-rata per kelas: DSC >= 0.85 dan IoU >= 0.75.
- Evaluasi pada 85 citra RSUA Validated (32 Normal, 53 Pneumonia; 207 COVID-19 dikeluarkan).
- Evaluasi pada subset citra uji Kermany (>= 50 citra: 25 Normal, 25 Pneumonia, seed tetap)
  terhadap kontur manual penulis di `data/raw/kermany_manual_masks/*.png`.
- Pencatatan kegagalan deteksi paru ekstrem (`r_paru < 0.05`) per kelas dan pengujian selisih
  rata-rata `r_paru` antarkelas (`|delta r_paru| > 0.05`) untuk memicu pelaporan `ACR_norm`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from src.segmentation.lung_mask import (
    binarize_probability_map,
    check_class_lung_ratio_diff,
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
        return 1.0
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
    dsc_min: float = 0.85,
    iou_min: float = 0.75,
    binarize_thr: float = 0.5,
    keep_components: int = 2,
    fill_holes: bool = True,
    expected_normal: int = 32,
    expected_pneumonia: int = 53,
    r_paru_extreme_thr: float = 0.05,
) -> Dict[str, Any]:
    """Evaluasi kuantitatif U-Net pada 85 citra RSUA Validated (32 Normal, 53 Pneumonia)."""
    classes_info = [
        ("Normal", "Non_Covid", expected_normal),
        ("Pneumonia", "Non_Covid_Pneumonia", expected_pneumonia),
    ]

    records: List[Dict[str, Any]] = []

    for label_name, dir_name, expected_count in classes_info:
        img_dir = rsua_validated_root / dir_name / "images"
        mask_dir = rsua_validated_root / dir_name / "masks"

        assert img_dir.exists(), f"Direktori citra RSUA tidak ditemukan: {img_dir}"
        assert mask_dir.exists(), f"Direktori mask RSUA tidak ditemukan: {mask_dir}"

        img_files = sorted([p for p in img_dir.glob("*.bmp") if not p.name.startswith(".")])
        assert len(img_files) == expected_count, (
            f"Jumlah citra RSUA {label_name} ({len(img_files)}) tidak sesuai target ({expected_count})!"
        )

        for img_p in img_files:
            mask_filename = img_p.name.replace("CXR_Images_", "CXR_Mask_")
            mask_p = mask_dir / mask_filename
            assert mask_p.exists(), f"Mask rujukan dokter tidak ditemukan: {mask_p}"

            with Image.open(img_p) as im_raw:
                im = im_raw.convert("L")
                with Image.open(mask_p) as mk_raw:
                    gt_arr = np.array(mk_raw.convert("L"), dtype=np.uint8)
                gt_bin = (gt_arr > 0).astype(np.uint8)

                prob_map = predict_lung_probability(
                    unet_model, im, device=device, return_original_size=True
                )
                bin_raw = binarize_probability_map(prob_map, threshold=binarize_thr)
                pred_mask = postprocess_lung_mask(
                    bin_raw, keep_components=keep_components, fill_holes=fill_holes
                )

                # AMBIG-23: pastikan resolusi prediksi sama dengan GT 256x256
                if pred_mask.shape != gt_bin.shape:
                    pred_mask = resize_mask_to_224(
                        pred_mask, target_size=(gt_bin.shape[1], gt_bin.shape[0])
                    )

                dsc = compute_dice_coefficient(pred_mask, gt_bin)
                iou = compute_iou(pred_mask, gt_bin)
                r_paru = compute_r_paru(pred_mask)
                r_paru_gt = compute_r_paru(gt_bin)

                reasons: List[str] = []
                if dsc < dsc_min:
                    reasons.append(f"DSC<{dsc_min:.2f}")
                if iou < iou_min:
                    reasons.append(f"IoU<{iou_min:.2f}")
                if r_paru < r_paru_extreme_thr:
                    reasons.append(f"r_paru<{r_paru_extreme_thr:.2f}")

                records.append({
                    "dataset": "RSUA",
                    "filename": img_p.name,
                    "img_path": str(img_p),
                    "gt_mask_path": str(mask_p),
                    "label": label_name,
                    "dice": dsc,
                    "iou": iou,
                    "r_paru": r_paru,
                    "r_paru_gt": r_paru_gt,
                    "sample_pass": len(reasons) == 0,
                    "fail_reason": "; ".join(reasons),
                })

    df = pd.DataFrame(records)

    summary_by_class: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    for cls in ["Normal", "Pneumonia"]:
        sub = df[df["label"] == cls]
        cls_pass = (float(sub["dice"].mean()) >= dsc_min) and (float(sub["iou"].mean()) >= iou_min)
        info = {
            "n_samples": int(len(sub)),
            "dice_mean": float(sub["dice"].mean()),
            "dice_std": float(sub["dice"].std()),
            "dice_min": float(sub["dice"].min()),
            "iou_mean": float(sub["iou"].mean()),
            "iou_std": float(sub["iou"].std()),
            "iou_min": float(sub["iou"].min()),
            "r_paru_mean": float(sub["r_paru"].mean()),
            "r_paru_gt_mean": float(sub["r_paru_gt"].mean()),
            "n_sample_below_thr": int((~sub["sample_pass"]).sum()),
            "class_status": "PASS" if cls_pass else "FAIL",
        }
        summary_by_class[cls] = info
        summary_rows.append({"kelas": cls, **info})

    normal_pass = summary_by_class["Normal"]["class_status"] == "PASS"
    pneumonia_pass = summary_by_class["Pneumonia"]["class_status"] == "PASS"
    overall_pass = normal_pass and pneumonia_pass

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "criteria": {"dsc_min": dsc_min, "iou_min": iou_min},
        "normal_pass": normal_pass,
        "pneumonia_pass": pneumonia_pass,
        "summary_by_class": summary_by_class,
        "summary_df": pd.DataFrame(summary_rows),
        "details_df": df,
    }


def select_kermany_qc_subset(
    split_df: pd.DataFrame,
    n_per_class: int = 25,
    seed: int = 42,
    out_csv: Path | None = None,
) -> pd.DataFrame:
    """Pilih subset seimbang dari D_test Kermany (25 Normal + 25 Pneumonia) dengan seed tetap (L3(2))."""
    test_df = split_df[split_df["split"] == "test"].copy() if "split" in split_df.columns else split_df.copy()
    test_df = test_df.sort_values(by=["label", "image_id"]).reset_index(drop=True)

    normal_pool = test_df[test_df["label"] == "Normal"]
    pneu_pool = test_df[test_df["label"] == "Pneumonia"]

    n_norm = min(n_per_class, len(normal_pool))
    n_pneu = min(n_per_class, len(pneu_pool))

    normal_samples = normal_pool.sample(n=n_norm, random_state=seed)
    pneumonia_samples = pneu_pool.sample(n=n_pneu, random_state=seed)
    selected = pd.concat([normal_samples, pneumonia_samples], ignore_index=True)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(out_csv, index=False, encoding="utf-8")

    return selected


def evaluate_kermany_subset_r_paru(
    unet_model: Any,
    manifest_df: pd.DataFrame,
    kermany_root: Path,
    device: Any = None,
    n_per_class: int = 25,
    seed: int = 42,
    binarize_thr: float = 0.5,
    keep_components: int = 2,
    fill_holes: bool = True,
    r_paru_extreme_thr: float = 0.05,
    r_paru_diff_thr: float = 0.05,
    subset_list_csv: Path | None = None,
) -> Dict[str, Any]:
    """Pemeriksaan langsung pada subset citra uji Kermany (25 Normal, 25 Pneumonia)."""
    selected_samples = select_kermany_qc_subset(
        manifest_df, n_per_class=n_per_class, seed=seed, out_csv=subset_list_csv
    )

    records: List[Dict[str, Any]] = []
    for _, row in selected_samples.iterrows():
        img_p = kermany_root / row["rel_path"]
        with Image.open(img_p) as im:
            prob_map = predict_lung_probability(
                unet_model, im, device=device, return_original_size=True
            )
            bin_raw = binarize_probability_map(prob_map, threshold=binarize_thr)
            pred_mask = postprocess_lung_mask(
                bin_raw, keep_components=keep_components, fill_holes=fill_holes
            )
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
                "extreme_failure": r_224 < r_paru_extreme_thr,
            })

    res_df = pd.DataFrame(records)
    mean_r_norm = float(res_df[res_df["label"] == "Normal"]["r_paru_224"].mean())
    mean_r_pneu = float(res_df[res_df["label"] == "Pneumonia"]["r_paru_224"].mean())
    triggers_acr_norm, delta_r = check_class_lung_ratio_diff(
        mean_r_norm, mean_r_pneu, threshold=r_paru_diff_thr
    )
    extreme_failures = int(res_df["extreme_failure"].sum())

    return {
        "n_evaluated": len(res_df),
        "normal_evaluated": int((res_df["label"] == "Normal").sum()),
        "pneumonia_evaluated": int((res_df["label"] == "Pneumonia").sum()),
        "mean_r_paru_normal": mean_r_norm,
        "mean_r_paru_pneumonia": mean_r_pneu,
        "delta_r_paru": delta_r,
        "triggers_acr_norm": triggers_acr_norm,
        "extreme_failure_count": extreme_failures,
        "details_df": res_df,
    }


def evaluate_kermany_manual_masks(
    unet_model: Any,
    subset_df: pd.DataFrame,
    kermany_root: Path,
    manual_masks_dir: Path,
    device: Any = None,
    dsc_min: float = 0.85,
    iou_min: float = 0.75,
    binarize_thr: float = 0.5,
    keep_components: int = 2,
    fill_holes: bool = True,
) -> Dict[str, Any]:
    """Evaluasi DSC & IoU keluaran U-Net terhadap kontur manual penulis pada subset Kermany (L3(2)).

    Membaca berkas `data/raw/kermany_manual_masks/<image_id>.png` (resolusi citra asli, AMBIG-23).
    Jika kontur manual belum tersedia, mengembalikan status `PENDING_MANUAL_MASKS` tanpa crash.
    """
    manual_masks_dir = Path(manual_masks_dir)
    available_masks: Dict[str, Path] = {}
    if manual_masks_dir.exists():
        for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg"):
            for p in manual_masks_dir.glob(ext):
                if not p.name.startswith("."):
                    available_masks[p.stem] = p

    matched_rows = [
        (row, available_masks[str(row["image_id"])])
        for _, row in subset_df.iterrows()
        if str(row["image_id"]) in available_masks
    ]

    if not matched_rows:
        return {
            "status": "PENDING_MANUAL_MASKS",
            "n_expected": int(len(subset_df)),
            "n_found": 0,
            "summary_by_class": {},
            "summary_df": pd.DataFrame(),
            "details_df": pd.DataFrame(),
            "note": (
                f"Belum ada berkas kontur manual di {manual_masks_dir}. "
                "Buat kontur manual (<image_id>.png) untuk 50 sampel pada kermany_qc_50_list.csv "
                "lalu jalankan ulang evaluasi ini."
            ),
        }

    records: List[Dict[str, Any]] = []
    for row, gt_path in matched_rows:
        img_p = kermany_root / row["rel_path"]
        with Image.open(img_p) as im_raw, Image.open(gt_path) as mk_raw:
            im = im_raw.convert("L")
            gt_arr = np.array(mk_raw.convert("L"), dtype=np.uint8)
            gt_bin = (gt_arr > 0).astype(np.uint8)

            prob_map = predict_lung_probability(
                unet_model, im, device=device, return_original_size=True
            )
            bin_raw = binarize_probability_map(prob_map, threshold=binarize_thr)
            pred_mask = postprocess_lung_mask(
                bin_raw, keep_components=keep_components, fill_holes=fill_holes
            )

            assert pred_mask.shape == gt_bin.shape, (
                f"Ukuran mask manual {gt_path.name} ({gt_bin.shape}) != ukuran citra asli ({pred_mask.shape})!"
            )

            dsc = compute_dice_coefficient(pred_mask, gt_bin)
            iou = compute_iou(pred_mask, gt_bin)
            records.append({
                "dataset": "Kermany_Manual",
                "image_id": row["image_id"],
                "label": row["label"],
                "dice": dsc,
                "iou": iou,
                "sample_pass": (dsc >= dsc_min) and (iou >= iou_min),
            })

    df = pd.DataFrame(records)
    summary_by_class: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    all_pass = True
    for cls in ("Normal", "Pneumonia"):
        sub = df[df["label"] == cls]
        if sub.empty:
            all_pass = False
            continue
        cls_pass = (float(sub["dice"].mean()) >= dsc_min) and (float(sub["iou"].mean()) >= iou_min)
        all_pass = all_pass and cls_pass
        info = {
            "n_samples": int(len(sub)),
            "dice_mean": float(sub["dice"].mean()),
            "dice_std": float(sub["dice"].std()),
            "dice_min": float(sub["dice"].min()),
            "iou_mean": float(sub["iou"].mean()),
            "iou_std": float(sub["iou"].std()),
            "iou_min": float(sub["iou"].min()),
            "class_status": "PASS" if cls_pass else "FAIL",
        }
        summary_by_class[cls] = info
        summary_rows.append({"kelas": cls, **info})

    complete = len(matched_rows) >= len(subset_df)
    status = ("PASS" if all_pass else "FAIL") if complete else "INCOMPLETE_MANUAL_MASKS"
    return {
        "status": status,
        "n_expected": int(len(subset_df)),
        "n_found": int(len(matched_rows)),
        "summary_by_class": summary_by_class,
        "summary_df": pd.DataFrame(summary_rows),
        "details_df": df,
    }


def summarize_full_r_paru(
    metadata_df: pd.DataFrame,
    r_paru_extreme_thr: float = 0.05,
    r_paru_diff_thr: float = 0.05,
) -> Dict[str, Any]:
    """Ringkas distribusi r_paru lintas seluruh dataset & per partisi (L5, docs/3.md butir 8)."""
    df = metadata_df.copy()
    df["extreme_failure"] = df["r_paru_224"] < r_paru_extreme_thr

    per_split_class = (
        df.groupby(["split", "label"])
        .agg(
            n_images=("image_id", "count"),
            r_paru_mean=("r_paru_224", "mean"),
            r_paru_std=("r_paru_224", "std"),
            r_paru_median=("r_paru_224", "median"),
            r_paru_min=("r_paru_224", "min"),
            r_paru_max=("r_paru_224", "max"),
            n_extreme_fail=("extreme_failure", "sum"),
        )
        .reset_index()
    )

    overall_by_class = (
        df.groupby("label")
        .agg(
            n_images=("image_id", "count"),
            r_paru_mean=("r_paru_224", "mean"),
            r_paru_std=("r_paru_224", "std"),
            r_paru_min=("r_paru_224", "min"),
            n_extreme_fail=("extreme_failure", "sum"),
        )
        .reset_index()
    )

    mean_norm_all = float(df[df["label"] == "Normal"]["r_paru_224"].mean())
    mean_pneu_all = float(df[df["label"] == "Pneumonia"]["r_paru_224"].mean())
    triggers_all, delta_all = check_class_lung_ratio_diff(
        mean_norm_all, mean_pneu_all, threshold=r_paru_diff_thr
    )

    test_df = df[df["split"] == "test"]
    mean_norm_test = float(test_df[test_df["label"] == "Normal"]["r_paru_224"].mean()) if not test_df.empty else float("nan")
    mean_pneu_test = float(test_df[test_df["label"] == "Pneumonia"]["r_paru_224"].mean()) if not test_df.empty else float("nan")
    triggers_test, delta_test = (
        check_class_lung_ratio_diff(mean_norm_test, mean_pneu_test, threshold=r_paru_diff_thr)
        if not test_df.empty
        else (False, float("nan"))
    )

    return {
        "n_total": int(len(df)),
        "mean_r_paru_normal_all": mean_norm_all,
        "mean_r_paru_pneumonia_all": mean_pneu_all,
        "delta_r_paru_all": delta_all,
        "triggers_acr_norm_all": triggers_all,
        "mean_r_paru_normal_test": mean_norm_test,
        "mean_r_paru_pneumonia_test": mean_pneu_test,
        "delta_r_paru_test": delta_test,
        "triggers_acr_norm_test": triggers_test,
        "total_extreme_failures": int(df["extreme_failure"].sum()),
        "test_extreme_failures": int(test_df["extreme_failure"].sum()) if not test_df.empty else 0,
        "per_split_class_df": per_split_class,
        "overall_by_class_df": overall_by_class,
    }


def build_failed_qc_table(
    rsua_details_df: pd.DataFrame,
    kermany_meta_df: pd.DataFrame,
    dsc_min: float = 0.85,
    iou_min: float = 0.75,
    r_paru_extreme_thr: float = 0.05,
) -> pd.DataFrame:
    """Gabungkan daftar sampel yang gagal kriteria QC (RSUA & Kermany) beserta alasannya."""
    rows: List[Dict[str, Any]] = []

    if not rsua_details_df.empty:
        rsua_fail = rsua_details_df[
            (rsua_details_df["dice"] < dsc_min)
            | (rsua_details_df["iou"] < iou_min)
            | (rsua_details_df["r_paru"] < r_paru_extreme_thr)
        ]
        for _, r in rsua_fail.iterrows():
            reasons = []
            if r["dice"] < dsc_min:
                reasons.append(f"DSC={r['dice']:.4f} < {dsc_min}")
            if r["iou"] < iou_min:
                reasons.append(f"IoU={r['iou']:.4f} < {iou_min}")
            if r["r_paru"] < r_paru_extreme_thr:
                reasons.append(f"r_paru={r['r_paru']:.4f} < {r_paru_extreme_thr}")
            rows.append({
                "dataset": "RSUA",
                "split": "qc_rsua",
                "sample_id": r["filename"],
                "label": r["label"],
                "dice": round(float(r["dice"]), 4),
                "iou": round(float(r["iou"]), 4),
                "r_paru": round(float(r["r_paru"]), 4),
                "alasan_gagal": "; ".join(reasons),
            })

    if not kermany_meta_df.empty:
        kerm_fail = kermany_meta_df[kermany_meta_df["r_paru_224"] < r_paru_extreme_thr]
        for _, r in kerm_fail.iterrows():
            rows.append({
                "dataset": "Kermany",
                "split": r.get("split", ""),
                "sample_id": r["image_id"],
                "label": r["label"],
                "dice": np.nan,
                "iou": np.nan,
                "r_paru": round(float(r["r_paru_224"]), 4),
                "alasan_gagal": f"Kegagalan deteksi ekstrem: r_paru_224={r['r_paru_224']:.4f} < {r_paru_extreme_thr}",
            })

    cols = ["dataset", "split", "sample_id", "label", "dice", "iou", "r_paru", "alasan_gagal"]
    return pd.DataFrame(rows, columns=cols)
