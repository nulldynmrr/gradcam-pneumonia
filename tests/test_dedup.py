"""Unit test untuk prosedur deduplikasi citra identik."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.dedup import run_deduplication


def test_dedup_detects_byte_and_pixel_duplicates(tmp_path) -> None:
    # Buat direktori tiruan
    root = tmp_path / "kermany"
    (root / "train" / "NORMAL").mkdir(parents=True)
    (root / "train" / "PNEUMONIA").mkdir(parents=True)

    # 1. Citra A unik
    arr_a = np.full((100, 100), 50, dtype=np.uint8)
    Image.fromarray(arr_a).save(root / "train" / "NORMAL" / "NORMAL-1-0001.jpeg")

    # 2. Citra B identik dengan A (byte sama persis)
    (root / "train" / "NORMAL" / "NORMAL-2-0001.jpeg").write_bytes(
        (root / "train" / "NORMAL" / "NORMAL-1-0001.jpeg").read_bytes()
    )

    # 3. Citra C unik pneumonia
    arr_c = np.full((100, 100), 120, dtype=np.uint8)
    Image.fromarray(arr_c).save(root / "train" / "PNEUMONIA" / "BACTERIA-3-0001.jpeg")

    manifest = pd.DataFrame([
        {"image_id": "NORMAL-1-0001", "rel_path": "train/NORMAL/NORMAL-1-0001.jpeg", "label": "Normal", "patient_id": "1"},
        {"image_id": "NORMAL-2-0001", "rel_path": "train/NORMAL/NORMAL-2-0001.jpeg", "label": "Normal", "patient_id": "2"},
        {"image_id": "BACTERIA-3-0001", "rel_path": "train/PNEUMONIA/BACTERIA-3-0001.jpeg", "label": "Pneumonia", "patient_id": "3"},
    ])

    clean_df, dedup_log = run_deduplication(manifest, root, check_pixel_hash=True)

    assert len(clean_df) == 2
    assert "NORMAL-1-0001" in set(clean_df["image_id"])
    assert "BACTERIA-3-0001" in set(clean_df["image_id"])
    assert "NORMAL-2-0001" not in set(clean_df["image_id"])

    dropped = dedup_log[dedup_log["status"] == "DROPPED"]
    assert len(dropped) == 1
    assert dropped.iloc[0]["image_id"] == "NORMAL-2-0001"


def test_dedup_raises_on_cross_class_conflict(tmp_path) -> None:
    root = tmp_path / "kermany"
    (root / "train" / "NORMAL").mkdir(parents=True)
    (root / "train" / "PNEUMONIA").mkdir(parents=True)

    # Citra sama persis tetapi satu dilabeli Normal dan satu Pneumonia
    arr = np.full((50, 50), 80, dtype=np.uint8)
    p_norm = root / "train" / "NORMAL" / "NORMAL-10-0001.jpeg"
    p_pneu = root / "train" / "PNEUMONIA" / "BACTERIA-20-0001.jpeg"
    Image.fromarray(arr).save(p_norm)
    p_pneu.write_bytes(p_norm.read_bytes())

    manifest = pd.DataFrame([
        {"image_id": "NORMAL-10-0001", "rel_path": "train/NORMAL/NORMAL-10-0001.jpeg", "label": "Normal", "patient_id": "10"},
        {"image_id": "BACTERIA-20-0001", "rel_path": "train/PNEUMONIA/BACTERIA-20-0001.jpeg", "label": "Pneumonia", "patient_id": "20"},
    ])

    with pytest.raises(ValueError, match="label kelas BERBEDA"):
        run_deduplication(manifest, root, check_pixel_hash=True)
