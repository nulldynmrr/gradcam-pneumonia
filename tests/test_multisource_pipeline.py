"""Unit & integration tests untuk pipa multi-sumber (Kermany + RSNA), dedup 2-tahap,
penyeimbangan 1:1 per sumber, QC keseimbangan paru kiri/kanan (r_LR), & shortcut audit.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from exploratory.shortcut_audit import run_shortcut_audit_probes
from scripts.make_synthetic_data import generate_synthetic_mirror
from src.data.dedup import compute_dhash64, compute_phash64, hamming_distance64, run_deduplication
from src.data.manifest import build_multisource_manifest
from src.data.rsna_loader import parse_rsna_metadata_tables, standardize_dicom_array
from src.data.split import (
    patientwise_stratified_split,
    source_balanced_subsample,
    verify_source_class_balance,
)
from src.paths import get_repo_root, load_config
from src.segmentation.cache import PreprocessCacheConfig, run_full_lung_mask_cache
from src.segmentation.lung_mask import compute_lr_lung_balance
from src.segmentation.qc import summarize_full_r_paru


class _DummyFrozenUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = torch.zeros((b, 1, h, w), dtype=torch.float32, device=x.device)
        out[:, :, h // 5 : 4 * h // 5, w // 6 : 5 * w // 12] = 0.9
        out[:, :, h // 5 : 4 * h // 5, 7 * w // 12 : 5 * w // 6] = 0.9
        return out


def test_standardize_dicom_array_fixes_monochrome1_inversion() -> None:
    raw = np.array([[0, 1000], [2000, 4095]], dtype=np.uint16)
    m2 = standardize_dicom_array(raw, photometric_interpretation="MONOCHROME2")
    m1 = standardize_dicom_array(raw, photometric_interpretation="MONOCHROME1")

    assert m2.dtype == np.uint8 and m1.dtype == np.uint8
    assert m2[0, 0] == 0 and m2[1, 1] == 255
    # MONOCHROME1 harus terbalik sempurna agar setara MONOCHROME2
    assert m1[0, 0] == 255 and m1[1, 1] == 0
    assert np.allclose(m1.astype(int) + m2.astype(int), 255, atol=1)


def test_parse_rsna_metadata_aggregates_bboxes_and_drops_not_normal(tmp_path: Path) -> None:
    labels_csv = tmp_path / "stage_2_train_labels.csv"
    cls_csv = tmp_path / "stage_2_detailed_class_info.csv"

    pd.DataFrame([
        {"patientId": "p_norm", "x": np.nan, "y": np.nan, "width": np.nan, "height": np.nan, "Target": 0},
        {"patientId": "p_opac", "x": 10.0, "y": 20.0, "width": 50.0, "height": 60.0, "Target": 1},
        {"patientId": "p_opac", "x": 100.0, "y": 40.0, "width": 40.0, "height": 50.0, "Target": 1},
        {"patientId": "p_excl", "x": np.nan, "y": np.nan, "width": np.nan, "height": np.nan, "Target": 0},
    ]).to_csv(labels_csv, index=False)

    pd.DataFrame([
        {"patientId": "p_norm", "class": "Normal"},
        {"patientId": "p_opac", "class": "Lung Opacity"},
        {"patientId": "p_opac", "class": "Lung Opacity"},
        {"patientId": "p_excl", "class": "No Lung Opacity / Not Normal"},
    ]).to_csv(cls_csv, index=False)

    binary_df, audit = parse_rsna_metadata_tables(labels_csv, cls_csv)
    assert audit["raw_label_rows"] == 4
    assert audit["unique_total_images"] == 3
    assert audit["n_normal"] == 1
    assert audit["n_lung_opacity"] == 1
    assert audit["n_excluded_not_normal"] == 1
    assert len(binary_df) == 2
    opac_row = binary_df[binary_df["patientId"] == "p_opac"].iloc[0]
    assert opac_row["label"] == "Pneumonia"
    assert int(opac_row["n_bboxes"]) == 2
    assert bool(opac_row["has_bbox"]) is True


def test_perceptual_dedup_detects_cross_source_near_duplicate(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    # Buat citra broad-spectrum mirip struktur paru (bukan gelombang sinus tunggal yang spektrum DCT-nya nol)
    yy, xx = np.mgrid[0:128, 0:128]
    left = (((xx - 42) / 22.0) ** 2 + ((yy - 66) / 38.0) ** 2) <= 1.0
    right = (((xx - 86) / 24.0) ** 2 + ((yy - 64) / 40.0) ** 2) <= 1.0
    grad = ((xx * 0.6 + yy * 0.4) % 90).astype(np.float32)
    base = np.where(left | right, 45.0 + grad * 0.4, 155.0 + grad * 0.5).clip(0, 255).astype(np.uint8)
    # Near-duplicate realistis: pergeseran kecerahan ringan + re-kompresi JPEG (byte & pixel hash beda)
    near = np.clip(base.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    diff = np.where((((xx - 64) / 45.0) ** 2 + ((yy - 30) / 18.0) ** 2) <= 1.0, 220, 20).astype(np.uint8)
    diff = np.clip(diff.astype(np.int16) + rng.integers(-15, 16, size=diff.shape), 0, 255).astype(np.uint8)

    p1 = tmp_path / "kermany_1.png"
    p2 = tmp_path / "rsna_1.jpg"
    p3 = tmp_path / "rsna_2.png"
    Image.fromarray(base, mode="L").save(p1)
    Image.fromarray(near, mode="L").save(p2, quality=85)
    Image.fromarray(diff, mode="L").save(p3)

    assert hamming_distance64(compute_phash64(p1), compute_phash64(p2)) <= 5
    assert compute_dhash64(p1) >= 0

    manifest = pd.DataFrame([
        {"image_id": "k1", "source": "kermany", "rel_path": "kermany_1.png", "label": "Normal", "patient_id": "1", "group_id": "kermany_1"},
        {"image_id": "r1", "source": "rsna", "rel_path": "rsna_1.jpg", "label": "Normal", "patient_id": "1", "group_id": "rsna_1"},
        {"image_id": "r2", "source": "rsna", "rel_path": "rsna_2.png", "label": "Pneumonia", "patient_id": "2", "group_id": "rsna_2"},
    ])
    clean_df, dedup_log = run_deduplication(
        manifest, tmp_path, check_pixel_hash=True, check_perceptual_hash=True, max_hamming=5
    )
    assert len(clean_df) == 2
    dropped = dedup_log[dedup_log["status"] == "DROPPED"]
    assert len(dropped) == 1
    assert "perceptual" in dropped.iloc[0]["match_type"]

    # Konflik label pada near-duplicate lintas sumber wajib raise ValueError
    manifest_conflict = manifest.copy()
    manifest_conflict.loc[manifest_conflict["image_id"] == "r1", "label"] = "Pneumonia"
    with pytest.raises(ValueError, match="label kelas BERBEDA"):
        run_deduplication(
            manifest_conflict, tmp_path, check_pixel_hash=True, check_perceptual_hash=True, max_hamming=5
        )

    # Dua citra pasien berbeda dengan anatomi makro serupa TIDAK boleh digabung sebagai duplikat
    left2 = (((xx - 46) / 20.0) ** 2 + ((yy - 62) / 34.0) ** 2) <= 1.0
    right2 = (((xx - 82) / 21.0) ** 2 + ((yy - 68) / 36.0) ** 2) <= 1.0
    other_patient = np.where(left2 | right2, 55.0 + grad * 0.35, 165.0 + grad * 0.45).clip(0, 255).astype(np.uint8)
    p4 = tmp_path / "rsna_other_patient.png"
    Image.fromarray(other_patient, mode="L").save(p4)
    manifest_distinct = pd.DataFrame([
        {"image_id": "k1", "source": "kermany", "rel_path": "kermany_1.png", "label": "Normal", "patient_id": "1", "group_id": "kermany_1"},
        {"image_id": "r3", "source": "rsna", "rel_path": "rsna_other_patient.png", "label": "Pneumonia", "patient_id": "3", "group_id": "rsna_3"},
    ])
    clean_distinct, _ = run_deduplication(
        manifest_distinct, tmp_path, check_pixel_hash=True, check_perceptual_hash=True, max_hamming=5
    )
    assert len(clean_distinct) == 2


def test_lr_lung_balance_qc_detects_unilateral_collapse() -> None:
    # Mask bilateral sehat (kiri & kanan relatif seimbang)
    bilateral = np.zeros((224, 224), dtype=np.uint8)
    bilateral[40:180, 30:95] = 1
    bilateral[40:180, 125:195] = 1
    res_ok = compute_lr_lung_balance(bilateral, min_lr_ratio=0.30)
    assert res_ok["n_components"] == 2
    assert res_ok["lr_ratio"] > 0.85
    assert res_ok["lr_imbalance_flag"] is False

    # Mask gagal satu sisi (hanya paru kanan)
    unilateral = np.zeros((224, 224), dtype=np.uint8)
    unilateral[40:180, 125:195] = 1
    res_bad = compute_lr_lung_balance(unilateral, min_lr_ratio=0.30)
    assert res_bad["n_components"] == 1
    assert res_bad["lr_ratio"] < 0.10
    assert res_bad["lr_imbalance_flag"] is True


def test_end_to_end_multisource_synthetic_pipeline_and_shortcut_audit(tmp_path: Path) -> None:
    cfg = load_config()
    generate_synthetic_mirror(tmp_path, n_patients_per_class=12, seed=42)

    kermany_root = tmp_path / "data" / "raw" / "kermany" / "chest_xray"
    rsna_root = tmp_path / "data" / "raw" / "rsna"
    harmonized_dir = tmp_path / "data" / "interim" / "harmonized_png"
    processed_dir = tmp_path / "data" / "processed"

    manifest_df, audit = build_multisource_manifest(
        kermany_root=kermany_root,
        rsna_root=rsna_root,
        harmonized_dir=harmonized_dir,
        convert_to_png=True,
    )
    assert set(manifest_df["source"].unique()) == {"kermany", "rsna"}
    assert audit["rsna_audit"]["n_excluded_not_normal"] > 0

    clean_df, dedup_log = run_deduplication(
        manifest_df, harmonized_dir, check_pixel_hash=True, check_perceptual_hash=True, max_hamming=5
    )
    assert (dedup_log["status"] == "DROPPED").sum() >= 1

    balanced_df, sub_report = source_balanced_subsample(clean_df, seed=42)
    assert len(sub_report) == 4
    bal_stats = verify_source_class_balance(balanced_df, max_prevalence_gap=0.05)
    assert bal_stats["max_gap"] <= 0.05

    split_df = patientwise_stratified_split(balanced_df, ratios=(0.70, 0.15, 0.15), seed=42)
    # Cek 0 kebocoran group_id lintas split
    for s1 in ("train", "val", "test"):
        for s2 in ("train", "val", "test"):
            if s1 >= s2:
                continue
            g1 = set(split_df[split_df["split"] == s1]["group_id"])
            g2 = set(split_df[split_df["split"] == s2]["group_id"])
            assert len(g1 & g2) == 0

    split_csv = tmp_path / "data" / "interim" / "split_patientwise.csv"
    split_df.to_csv(split_csv, index=False)

    cache_cfg = PreprocessCacheConfig.from_yaml_cfg(cfg)
    meta_df = run_full_lung_mask_cache(
        unet_model=_DummyFrozenUNet().eval(),
        split_csv_path=split_csv,
        kermany_root=harmonized_dir,
        processed_dir=processed_dir,
        cache_cfg=cache_cfg,
        device=torch.device("cpu"),
        verify_split_checksum=False,
        show_progress=False,
    )
    assert "lr_ratio" in meta_df.columns
    assert "lr_imbalance_flag" in meta_df.columns
    assert (meta_df["lr_imbalance_flag"] == False).all()

    r_sum = summarize_full_r_paru(meta_df)
    assert set(r_sum["per_source_class_df"]["source"].unique()) == {"kermany", "rsna"}

    audit_res = run_shortcut_audit_probes(meta_df, processed_dir, seed=42)
    assert len(audit_res["summary_df"]) == 4


def test_hybridgnet_pure_pytorch_architecture_and_dispatch() -> None:
    from src.segmentation.hybridgnet import create_hybridgnet_model, get_hybridgnet_spec
    from src.segmentation.unet import predict_lung_probability

    spec = get_hybridgnet_spec()
    assert spec["frozen"] is True
    assert spec["input_size_hw"] == [1024, 1024]

    model = create_hybridgnet_model(device="cpu")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    dummy_img = np.full((256, 256), 120, dtype=np.uint8)
    prob = predict_lung_probability(model, dummy_img, device="cpu", return_original_size=True)
    assert prob.shape == (256, 256)
    assert prob.dtype == np.float32
    assert set(np.unique(prob)).issubset({0.0, 1.0})


def test_updated_notebooks_valid_python_syntax() -> None:
    repo = get_repo_root()
    forbidden_md_tokens = ("Gambar 3.1", "AMBIG-", "docs/1.md", "docs/2.md", "docs/3.md", "G3(4)")
    for nb_path in sorted((repo / "notebooks").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        assert nb["cells"][0]["cell_type"] == "markdown", f"{nb_path.name}: sel pertama harus markdown"
        h1 = "".join(nb["cells"][0]["source"]).splitlines()[0]
        assert h1.startswith("# "), f"{nb_path.name}: harus diawali judul H1"
        assert "Gambar 3.1" not in h1, f"{nb_path.name}: judul H1 tidak boleh memuat 'Gambar 3.1'"

        for idx, cell in enumerate(nb["cells"]):
            text = "".join(cell["source"])
            if cell["cell_type"] == "markdown":
                for tok in forbidden_md_tokens:
                    assert tok not in text, f"{nb_path.name} cell {idx} masih memuat token internal: {tok}"
            elif cell["cell_type"] == "code":
                cleaned = [
                    (line[: len(line) - len(line.lstrip())] + "pass\n") if line.lstrip().startswith("!") else line
                    for line in cell["source"]
                ]
                ast.parse("".join(cleaned), filename=f"{nb_path.name}:cell_{idx}")
