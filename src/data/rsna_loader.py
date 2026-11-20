"""Modul pemuat, penyaring biner, dan harmonisasi format dataset RSNA Pneumonia Detection Challenge.

Aturan sesuai Proposal Bab III (§3.2.1):
- Menggunakan `stage_2_train_labels.csv` dan `stage_2_detailed_class_info.csv`.
- Mengagregasi baris multi-bounding-box ke tingkat citra unik (26.684 citra pada data penuh).
- Hanya mengambil kelas `Normal` (label_id=0) dan `Lung Opacity` (Pneumonia, label_id=1);
  kelas `No Lung Opacity / Not Normal` (11.821 citra) DIKELUARKAN secara ketat.
- Memeriksa tag DICOM `PhotometricInterpretation`: jika `MONOCHROME1`, intensitas dibalik
  agar tulang/opasitas selalu terang dan ruang udara gelap (setara `MONOCHROME2` pada Kermany).
- Mengekstrak metadata `ViewPosition` (AP/PA) dan `PatientSex` bila tersedia.
- Menyeragamkan seluruh citra (baik RSNA maupun Kermany) ke PNG grayscale 8-bit lossless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


RSNA_ALLOWED_CLASSES = {
    "Normal": ("Normal", 0),
    "Lung Opacity": ("Pneumonia", 1),
}
RSNA_EXCLUDED_CLASS = "No Lung Opacity / Not Normal"


def standardize_dicom_array(
    pixel_array: np.ndarray,
    photometric_interpretation: str = "MONOCHROME2",
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
) -> np.ndarray:
    """Konversi array mentah DICOM menjadi uint8 [0, 255] berpolaritas MONOCHROME2.

    Args:
        pixel_array: Array piksel 2D mentah dari berkas DICOM.
        photometric_interpretation: Tag DICOM `PhotometricInterpretation`.
            Jika `MONOCHROME1`, piksel dibalik (`max - arr`).
        rescale_slope: Nilai `RescaleSlope` DICOM (default 1.0).
        rescale_intercept: Nilai `RescaleIntercept` DICOM (default 0.0).

    Returns:
        np.ndarray: Citra grayscale 2D `uint8` pada rentang `[0, 255]`.
    """
    arr = np.asarray(pixel_array, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1).astype(np.float32)

    arr = arr * float(rescale_slope) + float(rescale_intercept)

    photo = str(photometric_interpretation or "MONOCHROME2").strip().upper()
    if photo == "MONOCHROME1":
        arr = float(arr.max()) - arr

    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def read_dicom_standardized_uint8(dcm_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Baca satu berkas DICOM RSNA, koreksi polaritas MONOCHROME1, dan ekstrak metadata."""
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "Pustaka 'pydicom' belum terpasang untuk membaca berkas DICOM RSNA. "
            "Jalankan di Colab/env: pip install pydicom"
        ) from exc

    ds = pydicom.dcmread(str(dcm_path))
    photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip().upper()
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    view_pos = str(getattr(ds, "ViewPosition", "UNKNOWN")).strip().upper() or "UNKNOWN"
    sex = str(getattr(ds, "PatientSex", "UNKNOWN")).strip().upper() or "UNKNOWN"
    age = str(getattr(ds, "PatientAge", "UNKNOWN")).strip() or "UNKNOWN"

    img_u8 = standardize_dicom_array(
        ds.pixel_array,
        photometric_interpretation=photo,
        rescale_slope=slope,
        rescale_intercept=intercept,
    )
    meta = {
        "photometric": photo,
        "view_position": view_pos,
        "sex": sex,
        "age": age,
        "width": int(img_u8.shape[1]),
        "height": int(img_u8.shape[0]),
        "mode": "L",
    }
    return img_u8, meta


def harmonize_image_to_lossless_png(
    src_path: Path,
    dst_png_path: Path,
) -> Dict[str, Any]:
    """Konversi satu citra (DICOM, JPEG, PNG, BMP) ke PNG grayscale 8-bit lossless."""
    dst_png_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = src_path.suffix.lower()

    if suffix == ".dcm":
        img_u8, meta = read_dicom_standardized_uint8(src_path)
        Image.fromarray(img_u8, mode="L").save(dst_png_path, format="PNG", compress_level=1)
        return meta

    with Image.open(src_path) as im:
        orig_mode = im.mode
        gray = im.convert("L")
        w, h = gray.size
        gray.save(dst_png_path, format="PNG", compress_level=1)
        return {
            "photometric": "MONOCHROME2",
            "view_position": "UNKNOWN",
            "sex": "UNKNOWN",
            "age": "UNKNOWN",
            "width": int(w),
            "height": int(h),
            "mode": orig_mode,
        }


def parse_rsna_metadata_tables(
    labels_csv: Path,
    class_info_csv: Path,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Agregasi tabel CSV RSNA ke tingkat citra unik dan saring ke kelas biner bersih.

    Returns:
        Tuple `(binary_df, audit_counts)`:
        - `binary_df`: 1 baris per `patientId` unik yang berkelas Normal atau Lung Opacity.
        - `audit_counts`: Ringkasan jumlah baris mentah, pasien unik, kelas diterima & dibuang.
    """
    if not labels_csv.exists():
        raise FileNotFoundError(f"Berkas label RSNA tidak ditemukan: {labels_csv}")
    if not class_info_csv.exists():
        raise FileNotFoundError(f"Berkas detailed class info RSNA tidak ditemukan: {class_info_csv}")

    labels_df = pd.read_csv(labels_csv)
    cls_df = pd.read_csv(class_info_csv)

    raw_label_rows = len(labels_df)
    raw_cls_rows = len(cls_df)

    # Pastikan tiap patientId hanya punya 1 kelas unik di detailed_class_info
    cls_nunique = cls_df.groupby("patientId")["class"].nunique()
    bad_cls = cls_nunique[cls_nunique > 1]
    if not bad_cls.empty:
        raise ValueError(
            f"Ditemukan patientId RSNA dengan lebih dari satu kelas: {bad_cls.index.tolist()[:5]}"
        )

    cls_unique = cls_df.drop_duplicates(subset=["patientId"])[["patientId", "class"]].copy()

    # Agregasi bounding box per patientId
    has_target = "Target" in labels_df.columns
    agg_spec: Dict[str, Any] = {}
    if "x" in labels_df.columns:
        agg_spec["n_bboxes"] = ("x", lambda s: int(s.notna().sum()))
    elif has_target:
        agg_spec["n_bboxes"] = ("Target", lambda s: int((s == 1).sum()))
    else:
        agg_spec["n_bboxes"] = ("patientId", "count")

    if has_target:
        agg_spec["target_max"] = ("Target", "max")

    bbox_agg = labels_df.groupby("patientId", as_index=False).agg(**agg_spec)
    merged = cls_unique.merge(bbox_agg, on="patientId", how="left")
    merged["n_bboxes"] = merged["n_bboxes"].fillna(0).astype(int)
    merged["has_bbox"] = merged["n_bboxes"] > 0

    class_counts = merged["class"].value_counts().to_dict()
    n_excluded = int(class_counts.get(RSNA_EXCLUDED_CLASS, 0))

    # Filter ketat hanya Normal dan Lung Opacity
    binary_df = merged[merged["class"].isin(RSNA_ALLOWED_CLASSES.keys())].copy()
    binary_df["label"] = binary_df["class"].map(lambda c: RSNA_ALLOWED_CLASSES[c][0])
    binary_df["label_id"] = binary_df["class"].map(lambda c: RSNA_ALLOWED_CLASSES[c][1]).astype(int)
    binary_df = binary_df.sort_values("patientId").reset_index(drop=True)

    # Validasi konsistensi Target vs Lung Opacity jika kolom Target tersedia
    if "target_max" in binary_df.columns:
        mismatch = binary_df[
            ((binary_df["label_id"] == 1) & (binary_df["target_max"] != 1))
            | ((binary_df["label_id"] == 0) & (binary_df["target_max"] != 0))
        ]
        if not mismatch.empty:
            raise AssertionError(
                f"Inkonsistensi label RSNA antara class_info dan Target pada {len(mismatch)} baris!"
            )

    audit_counts = {
        "raw_label_rows": int(raw_label_rows),
        "raw_class_info_rows": int(raw_cls_rows),
        "unique_total_images": int(len(merged)),
        "n_normal": int((binary_df["label"] == "Normal").sum()),
        "n_lung_opacity": int((binary_df["label"] == "Pneumonia").sum()),
        "n_excluded_not_normal": int(n_excluded),
        "n_binary_kept": int(len(binary_df)),
    }
    return binary_df, audit_counts


def _locate_rsna_image_file(images_dir: Path, patient_id: str) -> Optional[Path]:
    """Cari berkas citra RSNA berdasarkan patientId (.dcm terlebih dahulu, lalu .png/.jpg)."""
    for ext in (".dcm", ".png", ".jpg", ".jpeg"):
        cand = images_dir / f"{patient_id}{ext}"
        if cand.exists():
            return cand
    return None


def build_rsna_manifest(
    rsna_root: Path,
    harmonized_dir: Optional[Path] = None,
    convert_to_png: bool = False,
    extract_dicom_meta: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Bangun manifest tabular terstandar untuk dataset RSNA (Normal + Lung Opacity).

    Struktur direktori RSNA yang didukung di `rsna_root`:
    - `stage_2_train_labels.csv`
    - `stage_2_detailed_class_info.csv`
    - `stage_2_train_images/<patientId>.dcm` (atau `.png` bila sudah dikonversi / sintetis)
    """
    if not rsna_root.exists():
        raise FileNotFoundError(f"Direktori RSNA tidak ditemukan di: {rsna_root}")

    labels_csv = rsna_root / "stage_2_train_labels.csv"
    class_info_csv = rsna_root / "stage_2_detailed_class_info.csv"
    images_dir = rsna_root / "stage_2_train_images"
    if not images_dir.exists():
        # Fallback jika folder gambar dinamai 'images'
        alt_dir = rsna_root / "images"
        if alt_dir.exists():
            images_dir = alt_dir
        else:
            raise FileNotFoundError(f"Direktori citra RSNA tidak ditemukan di: {images_dir}")

    binary_df, audit_counts = parse_rsna_metadata_tables(labels_csv, class_info_csv)

    rows: List[Dict[str, Any]] = []
    missing_files: List[str] = []

    for _, row in binary_df.iterrows():
        pid = str(row["patientId"])
        img_path = _locate_rsna_image_file(images_dir, pid)
        if img_path is None:
            missing_files.append(pid)
            continue

        meta: Dict[str, Any] = {
            "photometric": "MONOCHROME2",
            "view_position": str(row["view_position"]) if "view_position" in row and pd.notna(row["view_position"]) else "UNKNOWN",
            "sex": str(row["sex"]) if "sex" in row and pd.notna(row["sex"]) else "UNKNOWN",
            "width": None,
            "height": None,
            "mode": None,
        }

        if convert_to_png:
            if harmonized_dir is None:
                raise ValueError("`harmonized_dir` wajib diisi bila `convert_to_png=True`!")
            dst_png = harmonized_dir / "rsna" / f"rsna_{pid}.png"
            if not dst_png.exists():
                meta = harmonize_image_to_lossless_png(img_path, dst_png)
            else:
                with Image.open(dst_png) as im_cached:
                    meta["width"], meta["height"] = im_cached.size
                    meta["mode"] = im_cached.mode
            rel_path = dst_png.relative_to(harmonized_dir).as_posix()
        else:
            if extract_dicom_meta and img_path.suffix.lower() == ".dcm":
                _, meta = read_dicom_standardized_uint8(img_path)
            elif extract_dicom_meta:
                with Image.open(img_path) as im:
                    meta["width"], meta["height"] = im.size
                    meta["mode"] = im.mode
            rel_path = img_path.relative_to(rsna_root).as_posix()

        rows.append({
            "image_id": f"rsna_{pid}",
            "source": "rsna",
            "rel_path": rel_path,
            "orig_split": "rsna_train",
            "folder_class": str(row["class"]).upper(),
            "raw_cls": str(row["class"]),
            "label": str(row["label"]),
            "label_id": int(row["label_id"]),
            "patient_id": pid,
            "group_id": f"rsna_{pid}",
            "seq_num": 1,
            "filename": img_path.name,
            "view_position": meta["view_position"],
            "sex": meta["sex"],
            "photometric": meta["photometric"],
            "n_bboxes": int(row["n_bboxes"]),
            "has_bbox": bool(row["has_bbox"]),
            "width": meta["width"],
            "height": meta["height"],
            "mode": meta["mode"],
        })

    if missing_files:
        raise FileNotFoundError(
            f"Ditemukan {len(missing_files)} citra RSNA yang hilang di {images_dir}! "
            f"Contoh: {missing_files[:5]}"
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError(f"Tidak ada citra RSNA valid yang ditemukan di {rsna_root}")

    return out_df, audit_counts
