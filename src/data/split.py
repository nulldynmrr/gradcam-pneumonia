"""Modul pembagian data patient-wise stratified group split (70% train, 15% val, 15% test).

Aturan:
- Proposal §3.2.1 (Bab III L28-36):
  D_train ∩ D_val = ∅, D_train ∩ D_test = ∅, D_val ∩ D_test = ∅.
- Grup = patient_id; Strata = kelas (Normal / Pneumonia).
- Deterministik: dipandu seed split (default: 42, AMBIG-3).
- Hasil pembagian disimpan ke `data/interim/split_patientwise.csv` dan diverifikasi
  menggunakan checksum SHA-256 agar DIKUNCI sama untuk seluruh konfigurasi dan seed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def patientwise_stratified_split(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> pd.DataFrame:
    """Bagi DataFrame citra menjadi partisi train, val, dan test secara patient-wise & stratified.

    Args:
        df: Manifest citra (wajib memuat kolom 'patient_id', 'label', 'image_id').
        ratios: Tuple rasio target (train, val, test), jumlah = 1.0.
        seed: Random seed untuk determinisme pengacakan urutan pasien.

    Returns:
        pd.DataFrame dengan tambahan kolom 'split' ('train', 'val', 'test').
    """
    assert len(ratios) == 3 and np.isclose(sum(ratios), 1.0), "Rasio harus berupa tuple 3 angka dengan jumlah 1.0"
    train_r, val_r, test_r = ratios

    # Validasi satu pasien hanya punya 1 kelas
    patient_classes = df.groupby("patient_id")["label"].nunique()
    if (patient_classes > 1).any():
        bad = patient_classes[patient_classes > 1].index.tolist()
        raise AssertionError(f"Pasien memiliki lebih dari 1 kelas: {bad[:5]}")

    patient_info = df.groupby(["patient_id", "label"]).size().reset_index(name="n_images")

    rng = np.random.default_rng(seed)
    split_assignment: Dict[str, str] = {}

    for cls_name in sorted(patient_info["label"].unique()):
        cls_patients = patient_info[patient_info["label"] == cls_name].copy()
        # Acak urutan sebelum diurutkan menurun berdasarkan n_images agar deterministik dan adil
        perm = rng.permutation(len(cls_patients))
        cls_patients = cls_patients.iloc[perm].sort_values(by="n_images", ascending=False).reset_index(drop=True)

        total_cls_images = cls_patients["n_images"].sum()
        target_counts = {
            "train": int(round(train_r * total_cls_images)),
            "val": int(round(val_r * total_cls_images)),
            "test": int(round(test_r * total_cls_images)),
        }
        # Sesuaikan jika pembulatan selisih 1
        curr_sum = sum(target_counts.values())
        if curr_sum < total_cls_images:
            target_counts["train"] += (total_cls_images - curr_sum)
        elif curr_sum > total_cls_images:
            target_counts["train"] -= (curr_sum - total_cls_images)

        alloc_counts = {"train": 0, "val": 0, "test": 0}

        for _, row in cls_patients.iterrows():
            pid = row["patient_id"]
            cnt = row["n_images"]

            # Pilih partisi dengan defisit relatif terbesar (target - current) / target
            best_part = None
            max_deficit = -float("inf")
            for part in ("train", "val", "test"):
                t = target_counts[part]
                c = alloc_counts[part]
                deficit = (t - c) / max(1, t)
                if deficit > max_deficit:
                    max_deficit = deficit
                    best_part = part

            split_assignment[pid] = best_part
            alloc_counts[best_part] += cnt

    out_df = df.copy()
    out_df["split"] = out_df["patient_id"].map(split_assignment)

    # Verifikasi ketat sifat mutually exclusive
    train_pids = set(out_df[out_df["split"] == "train"]["patient_id"])
    val_pids = set(out_df[out_df["split"] == "val"]["patient_id"])
    test_pids = set(out_df[out_df["split"] == "test"]["patient_id"])

    assert len(train_pids & val_pids) == 0, "Irisan train & val tidak kosong!"
    assert len(train_pids & test_pids) == 0, "Irisan train & test tidak kosong!"
    assert len(val_pids & test_pids) == 0, "Irisan val & test tidak kosong!"

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
    # Urutkan deterministik sebelum menyimpan
    sorted_df = split_df.sort_values(by=["split", "label", "patient_id", "image_id"]).reset_index(drop=True)
    sorted_df.to_csv(out_path, index=False, encoding="utf-8")

    h = hashlib.sha256()
    with open(out_path, "rb") as fh:
        h.update(fh.read())
    digest = h.hexdigest()

    checksum_file = out_path.with_suffix(".csv.sha256")
    checksum_file.write_text(digest, encoding="utf-8")
    return digest


def verify_locked_split(csv_path: Path, expected_sha256: str | None = None) -> None:
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
