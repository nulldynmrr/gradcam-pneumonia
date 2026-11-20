"""Modul penyeimbangan kelas per sumber (1:1) dan pembagian data patient-wise stratified (70:15:15).

Aturan sesuai Proposal Bab III (§3.2.1):
- Penyeimbangan kelas 1:1 dilakukan DI DALAM masing-masing sumber (Kermany & RSNA) pada tingkat pasien
  dengan seed tetap (default: 42) agar rasio kelas identik lintas sumber.
- Pembagian data patient-wise stratified 70:15:15:
  D_train ∩ D_val = ∅, D_train ∩ D_test = ∅, D_val ∩ D_test = ∅ pada tingkat pasien/group_id & citra.
- Stratifikasi: `source × label` (dan `view_position` bila tersedia dengan sampel mencukupi).
- Hasil pembagian disimpan ke `data/interim/split_patientwise.csv` dan dikunci dengan SHA-256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _resolve_group_col(df: pd.DataFrame) -> str:
    """Gunakan `group_id` jika tersedia (multi-sumber), fallback ke `patient_id`."""
    if "group_id" in df.columns and df["group_id"].notna().all():
        return "group_id"
    if "patient_id" in df.columns:
        return "patient_id"
    raise ValueError("DataFrame wajib memuat kolom `group_id` atau `patient_id`!")


def source_balanced_subsample(
    df: pd.DataFrame,
    seed: int = 42,
    source_col: str = "source",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Seimbangkan kelas Normal vs Pneumonia menjadi 1:1 di dalam tiap sumber pada tingkat pasien.

    Kelas minoritas pada setiap sumber dipertahankan 100%, sedangkan kelas mayoritas disubsampling
    secara deterministik berbasis kelompok pasien hingga total citranya sedekat mungkin dengan kelas
    minoritas tanpa memotong pasien menjadi separuh.
    """
    group_col = _resolve_group_col(df)
    work = df.copy()
    if source_col not in work.columns:
        work[source_col] = "single_source"

    rng = np.random.default_rng(seed)
    kept_parts: List[pd.DataFrame] = []
    report_rows: List[Dict] = []

    for src_name in sorted(work[source_col].unique()):
        src_df = work[work[source_col] == src_name].copy()
        cls_counts = src_df["label"].value_counts()
        if len(cls_counts) < 2:
            raise ValueError(f"Sumber '{src_name}' tidak memiliki dua kelas (Normal & Pneumonia)!")

        target_n = int(cls_counts.min())

        for cls_name in sorted(src_df["label"].unique()):
            cls_df = src_df[src_df["label"] == cls_name].copy()
            orig_n = len(cls_df)

            if orig_n <= target_n:
                selected_df = cls_df
            else:
                # Pilih kelompok pasien secara deterministik hingga mencapai target_n
                grp_sizes = (
                    cls_df.groupby(group_col)
                    .size()
                    .reset_index(name="n_images")
                    .sort_values(group_col)
                    .reset_index(drop=True)
                )
                perm = rng.permutation(len(grp_sizes))
                grp_sizes = grp_sizes.iloc[perm].reset_index(drop=True)

                chosen_groups: List[str] = []
                acc = 0
                for _, grow in grp_sizes.iterrows():
                    cnt = int(grow["n_images"])
                    if acc + cnt <= target_n:
                        chosen_groups.append(grow[group_col])
                        acc += cnt
                    elif acc < target_n and (acc + cnt - target_n) < (target_n - acc):
                        # Ambil pasien terakhir bila lebih dekat ke target_n
                        chosen_groups.append(grow[group_col])
                        acc += cnt
                        break
                    if acc == target_n:
                        break

                selected_df = cls_df[cls_df[group_col].isin(set(chosen_groups))].copy()

            kept_parts.append(selected_df)
            report_rows.append({
                "source": src_name,
                "label": cls_name,
                "orig_images": int(orig_n),
                "target_images": int(target_n),
                "kept_images": int(len(selected_df)),
                "dropped_images": int(orig_n - len(selected_df)),
                "kept_groups": int(selected_df[group_col].nunique()),
            })

    balanced_df = pd.concat(kept_parts, ignore_index=True)
    report_df = pd.DataFrame(report_rows)
    return balanced_df, report_df


def verify_source_class_balance(
    df: pd.DataFrame,
    max_prevalence_gap: float = 0.02,
    source_col: str = "source",
) -> Dict[str, float]:
    """Verifikasi bahwa proporsi kelas Pneumonia seragam lintas sumber (mencegah source shortcut)."""
    if source_col not in df.columns:
        prev = float((df["label"] == "Pneumonia").mean())
        return {"single_source": prev, "max_gap": 0.0}

    prevalences: Dict[str, float] = {}
    for src_name, sub in df.groupby(source_col):
        prevalences[str(src_name)] = float((sub["label"] == "Pneumonia").mean())

    vals = list(prevalences.values())
    gap = float(max(vals) - min(vals)) if len(vals) >= 2 else 0.0
    if gap > max_prevalence_gap:
        raise AssertionError(
            f"Ketimpangan proporsi Pneumonia antarsumber melebihi batas ({gap:.4f} > {max_prevalence_gap})! "
            f"Detail: {prevalences}"
        )
    return {**prevalences, "max_gap": gap}


def patientwise_stratified_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    stratify_by_view: bool = True,
) -> pd.DataFrame:
    """Bagi DataFrame citra menjadi partisi train, val, dan test secara patient-wise & stratified.

    Stratifikasi otomatis menggunakan:
    - `source × label` (plus `view_position` untuk RSNA bila >= 20 pasien per strata),
    - atau `label` saja jika dataset satu sumber tanpa kolom `source`.
    """
    assert len(ratios) == 3 and np.isclose(sum(ratios), 1.0), "Rasio harus berupa tuple 3 angka dengan jumlah 1.0"
    train_r, val_r, test_r = ratios
    group_col = _resolve_group_col(df)

    # Validasi satu pasien/group hanya punya 1 kelas
    patient_classes = df.groupby(group_col)["label"].nunique()
    if (patient_classes > 1).any():
        bad = patient_classes[patient_classes > 1].index.tolist()
        raise AssertionError(f"Pasien memiliki lebih dari 1 kelas: {bad[:5]}")

    work = df.copy()
    has_source = "source" in work.columns and work["source"].notna().all()
    if has_source:
        strata_base = work["source"].astype(str) + "__" + work["label"].astype(str)
    else:
        strata_base = work["label"].astype(str)

    # Tambahkan stratifikasi ViewPosition (AP/PA) bila tersedia dengan jumlah memadai
    if stratify_by_view and "view_position" in work.columns:
        vp = work["view_position"].fillna("UNKNOWN").astype(str).str.upper()
        cand_strata = np.where(vp.isin(["AP", "PA"]), strata_base + "__" + vp, strata_base)
        # Hanya pakai sub-strata view jika tiap sub-strata punya >= 10 grup pasien
        grp_tmp = pd.DataFrame({group_col: work[group_col], "_s": cand_strata}).drop_duplicates(group_col)
        counts_s = grp_tmp["_s"].value_counts()
        valid_view_strata = set(counts_s[counts_s >= 10].index)
        work["_strata"] = [s if s in valid_view_strata else sb for s, sb in zip(cand_strata, strata_base)]
    else:
        work["_strata"] = strata_base

    # Satu grup masuk ke strata dominannya (deterministik)
    grp_strata = (
        work.groupby(group_col)["_strata"]
        .agg(lambda s: sorted(s.mode().tolist())[0])
        .reset_index(name="strata")
    )
    grp_counts = work.groupby(group_col).size().reset_index(name="n_images")
    patient_info = grp_strata.merge(grp_counts, on=group_col, how="inner")

    rng = np.random.default_rng(seed)
    split_assignment: Dict[str, str] = {}

    for strata_name in sorted(patient_info["strata"].unique()):
        cls_patients = (
            patient_info[patient_info["strata"] == strata_name]
            .sort_values(by=group_col)
            .reset_index(drop=True)
        )
        perm = rng.permutation(len(cls_patients))
        cls_patients = cls_patients.iloc[perm].sort_values(by="n_images", ascending=False).reset_index(drop=True)

        total_cls_images = int(cls_patients["n_images"].sum())
        target_counts = {
            "train": int(round(train_r * total_cls_images)),
            "val": int(round(val_r * total_cls_images)),
            "test": int(round(test_r * total_cls_images)),
        }
        curr_sum = sum(target_counts.values())
        if curr_sum < total_cls_images:
            target_counts["train"] += total_cls_images - curr_sum
        elif curr_sum > total_cls_images:
            target_counts["train"] -= curr_sum - total_cls_images

        alloc_counts = {"train": 0, "val": 0, "test": 0}

        for _, row in cls_patients.iterrows():
            gid = row[group_col]
            cnt = int(row["n_images"])

            best_part = None
            max_deficit = -float("inf")
            for part in ("train", "val", "test"):
                t = target_counts[part]
                c = alloc_counts[part]
                deficit = (t - c) / max(1, t)
                if deficit > max_deficit:
                    max_deficit = deficit
                    best_part = part

            split_assignment[gid] = best_part
            alloc_counts[best_part] += cnt

    out_df = df.copy()
    out_df["split"] = out_df[group_col].map(split_assignment)

    # Verifikasi ketat sifat mutually exclusive pada tingkat grup pasien DAN citra
    train_gids = set(out_df[out_df["split"] == "train"][group_col])
    val_gids = set(out_df[out_df["split"] == "val"][group_col])
    test_gids = set(out_df[out_df["split"] == "test"][group_col])

    assert len(train_gids & val_gids) == 0, "Irisan grup pasien train & val tidak kosong!"
    assert len(train_gids & test_gids) == 0, "Irisan grup pasien train & test tidak kosong!"
    assert len(val_gids & test_gids) == 0, "Irisan grup pasien val & test tidak kosong!"

    train_imgs = set(out_df[out_df["split"] == "train"]["image_id"])
    val_imgs = set(out_df[out_df["split"] == "val"]["image_id"])
    test_imgs = set(out_df[out_df["split"] == "test"]["image_id"])

    assert len(train_imgs & val_imgs) == 0, "Irisan citra train & val tidak kosong!"
    assert len(train_imgs & test_imgs) == 0, "Irisan citra train & test tidak kosong!"
    assert len(val_imgs & test_imgs) == 0, "Irisan citra val & test tidak kosong!"

    return out_df


def save_and_lock_split(split_df: pd.DataFrame, out_path: Path) -> str:
    """Simpan DataFrame split ke CSV dan kembalikan SHA-256 checksum-nya."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sort_cols = [c for c in ["split", "source", "label", "group_id", "patient_id", "image_id"] if c in split_df.columns]
    sorted_df = split_df.sort_values(by=sort_cols).reset_index(drop=True)
    sorted_df.to_csv(out_path, index=False, encoding="utf-8")

    h = hashlib.sha256()
    with open(out_path, "rb") as fh:
        h.update(fh.read())
    digest = h.hexdigest()

    checksum_file = out_path.with_suffix(".csv.sha256")
    checksum_file.write_text(digest, encoding="utf-8")
    return digest


def verify_locked_split(csv_path: Path, expected_sha256: Optional[str] = None) -> None:
    """Verifikasi bahwa berkas split terkunci dan tidak mengalami perubahan."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Berkas split tidak ditemukan di: {csv_path}")

    h = hashlib.sha256()
    with open(csv_path, "rb") as fh:
        h.update(fh.read())
    actual_digest = h.hexdigest()

    checksum_file = csv_path.with_suffix(".csv.sha256")
    if expected_sha256 is None and checksum_file.exists():
        expected_sha256 = checksum_file.read_text(encoding="utf-8").strip()

    if expected_sha256 and actual_digest != expected_sha256:
        raise ValueError(
            f"Checksum split TIDAK COCOK! Berkas telah termodifikasi.\n"
            f"Expected: {expected_sha256}\nActual:   {actual_digest}"
        )
