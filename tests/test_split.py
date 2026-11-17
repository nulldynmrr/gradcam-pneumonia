"""Unit test untuk prosedur stratified group split patient-wise."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import (
    patientwise_stratified_split,
    save_and_lock_split,
    verify_locked_split,
)


def test_split_no_patient_or_image_leakage() -> None:
    rows = []
    # 20 pasien Normal
    for i in range(20):
        pid = f"N_{i}"
        n_img = 1 if i % 2 == 0 else 2
        for s in range(n_img):
            rows.append({"image_id": f"img_N_{i}_{s}", "patient_id": pid, "label": "Normal"})

    # 40 pasien Pneumonia
    for i in range(40):
        pid = f"P_{i}"
        n_img = 1 if i % 3 == 0 else 2
        for s in range(n_img):
            rows.append({"image_id": f"img_P_{i}_{s}", "patient_id": pid, "label": "Pneumonia"})

    df = pd.DataFrame(rows)
    split_df = patientwise_stratified_split(df, ratios=(0.70, 0.15, 0.15), seed=42)

    parts = {"train", "val", "test"}
    assert set(split_df["split"].unique()) == parts

    # Validasi D_train ∩ D_val = ∅, D_train ∩ D_test = ∅, D_val ∩ D_test = ∅
    for p1 in ("train", "val", "test"):
        for p2 in ("train", "val", "test"):
            if p1 >= p2:
                continue
            p1_pids = set(split_df[split_df["split"] == p1]["patient_id"])
            p2_pids = set(split_df[split_df["split"] == p2]["patient_id"])
            assert len(p1_pids & p2_pids) == 0, f"Leakage pasien antara {p1} dan {p2}!"

            p1_imgs = set(split_df[split_df["split"] == p1]["image_id"])
            p2_imgs = set(split_df[split_df["split"] == p2]["image_id"])
            assert len(p1_imgs & p2_imgs) == 0, f"Leakage citra antara {p1} dan {p2}!"


def test_split_locking_and_verification(tmp_path) -> None:
    df = pd.DataFrame([
        {"image_id": "img1", "patient_id": "p1", "label": "Normal"},
        {"image_id": "img2", "patient_id": "p2", "label": "Pneumonia"},
    ])
    split_df = patientwise_stratified_split(df, ratios=(0.5, 0.25, 0.25), seed=42)

    csv_path = tmp_path / "split.csv"
    digest = save_and_lock_split(split_df, csv_path)

    # Verifikasi normal harus sukses
    verify_locked_split(csv_path, expected_sha256=digest)

    # Modifikasi berkas secara ilegal -> wajib raise ValueError
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\nTAMBAHAN_ILEGAL", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum split TIDAK COCOK"):
        verify_locked_split(csv_path, expected_sha256=digest)
