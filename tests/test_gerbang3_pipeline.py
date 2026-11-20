"""Unit & integration tests untuk modul Gerbang 3 (seed, cache, QC lanjutan, visualisasi, & struktur notebook 02-03)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from src.data.input_schemes import build_baseline_3ch, build_fusion_4ch, build_hardmask_3ch
from src.data.transforms import PairwiseAffineTransform, load_gray01, resize_pair
from src.paths import get_repo_root, load_config
from src.seed import get_env_snapshot, set_global_seed
from src.segmentation.cache import (
    PreprocessCacheConfig,
    run_full_lung_mask_cache,
    verify_cache_integrity,
)
from src.segmentation.qc import (
    build_failed_qc_table,
    evaluate_kermany_manual_masks,
    select_kermany_qc_subset,
    summarize_full_r_paru,
)
from src.segmentation.unet import get_unet_spec
from src.viz.qc_plots import (
    plot_kermany_overlay_grid,
    plot_pairwise_augmentation_grid,
    plot_r_paru_histogram,
    plot_rsua_metrics_distribution,
    plot_three_input_schemes,
)


class _DummyFrozenUNet(nn.Module):
    """Model segmentasi sintetis (bobot beku) untuk uji CPU lokal tanpa mengunduh 208 MB bobot Zenodo."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = torch.zeros((b, 1, h, w), dtype=torch.float32, device=x.device)
        # Dua kotak paru kiri-kanan sintetis
        out[:, :, h // 5 : 4 * h // 5, w // 6 : 5 * w // 12] = 0.9
        out[:, :, h // 5 : 4 * h // 5, 7 * w // 12 : 5 * w // 6] = 0.9
        return out


def test_seed_reproducibility_and_snapshot() -> None:
    set_global_seed(42, deterministic=True)
    a_np = np.random.rand(5)
    a_th = torch.rand(5)

    set_global_seed(42, deterministic=True)
    b_np = np.random.rand(5)
    b_th = torch.rand(5)

    assert np.allclose(a_np, b_np)
    assert torch.equal(a_th, b_th)

    snap = get_env_snapshot(seed=42)
    assert snap["seed"] == 42
    assert "torch" in snap["packages"]
    assert "segmentation_models_pytorch" in snap["packages"]


def test_unet_spec_verified_fields() -> None:
    spec = get_unet_spec()
    assert spec["doi"] == "10.5281/zenodo.8393556"
    assert spec["weights_license"] == "CC-BY-4.0"
    assert spec["encoder_name"] == "se_resnext101_32x4d"
    assert spec["input_size_hw"] == [384, 384]
    assert spec["in_channels"] == 3
    assert spec["activation"] == "sigmoid"
    assert spec["frozen"] is True


def test_load_gray01_and_resize_pair_from_config() -> None:
    cfg = load_config()
    rgb_img = np.full((300, 400, 3), 128, dtype=np.uint8)
    gray01 = load_gray01(rgb_img)
    assert gray01.shape == (300, 400)
    assert gray01.dtype == np.float32
    assert 0.0 <= float(gray01.min()) <= float(gray01.max()) <= 1.0

    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[50:250, 60:180] = 1
    i224, m224 = resize_pair(gray01, mask, target_size=(224, 224))
    assert i224.shape == (224, 224) and i224.dtype == np.float32
    assert m224.shape == (224, 224) and m224.dtype == np.uint8
    assert set(np.unique(m224)).issubset({0, 1})

    transform = PairwiseAffineTransform.from_config(cfg)
    assert transform.rotation_deg == 15.0
    assert transform.hflip_p == 0.5
    assert transform.translate_frac == 0.10
    assert transform.zoom_range == (0.9, 1.1)
    assert transform.fill_value == 0


def test_synthetic_end_to_end_cache_qc_and_plots(tmp_path: Path) -> None:
    cfg = load_config()
    kermany_root = tmp_path / "kermany"
    processed_dir = tmp_path / "processed"
    fig_dir = tmp_path / "figures"
    manual_dir = tmp_path / "manual_masks"

    rows = []
    for sp, n_per_cls in [("train", 4), ("val", 2), ("test", 3)]:
        for cls_idx, (lbl, prefix) in enumerate([("Normal", "NORMAL"), ("Pneumonia", "BACTERIA")]):
            folder = kermany_root / sp / lbl.upper()
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_cls):
                pid = f"{sp}_{cls_idx}_{i}"
                img_id = f"{prefix}-{1000 + i}-{sp}"
                fname = f"{img_id}.jpeg"
                arr = np.full((256, 256), 120 + i * 5, dtype=np.uint8)
                Image.fromarray(arr, mode="L").save(folder / fname)
                rows.append({
                    "image_id": img_id,
                    "rel_path": f"{sp}/{lbl.upper()}/{fname}",
                    "split": sp,
                    "label": lbl,
                    "label_id": cls_idx,
                    "patient_id": pid,
                })

    split_csv = tmp_path / "split_patientwise.csv"
    split_df = pd.DataFrame(rows)
    split_df.to_csv(split_csv, index=False)

    model = _DummyFrozenUNet().eval()
    cache_cfg = PreprocessCacheConfig.from_yaml_cfg(cfg)

    meta_df = run_full_lung_mask_cache(
        unet_model=model,
        split_csv_path=split_csv,
        kermany_root=kermany_root,
        processed_dir=processed_dir,
        cache_cfg=cache_cfg,
        device=torch.device("cpu"),
        verify_split_checksum=False,
        show_progress=False,
    )
    assert len(meta_df) == len(split_df)

    # Cek integritas cache & resumable
    check = verify_cache_integrity(split_csv, processed_dir, sample_check_n=10, seed=42)
    assert check["status"] == "PASS"
    assert check["n_cached"] == len(split_df)

    # Ringkasan r_paru & manual masks (pending lalu dengan 1 mask valid)
    r_summary = summarize_full_r_paru(meta_df)
    assert r_summary["n_total"] == len(split_df)

    subset_df = select_kermany_qc_subset(split_df, n_per_class=2, seed=42, out_csv=tmp_path / "subset.csv")
    assert len(subset_df) == 4
    pending_res = evaluate_kermany_manual_masks(model, subset_df, kermany_root, manual_dir, device="cpu")
    assert pending_res["status"] == "PENDING_MANUAL_MASKS"

    # Buat semua mask manual sintetis cocok ukuran
    manual_dir.mkdir(parents=True, exist_ok=True)
    for _, r in subset_df.iterrows():
        gt = np.zeros((256, 256), dtype=np.uint8)
        gt[51:204, 42:106] = 255
        gt[51:204, 149:213] = 255
        Image.fromarray(gt, mode="L").save(manual_dir / f"{r['image_id']}.png")

    manual_res = evaluate_kermany_manual_masks(model, subset_df, kermany_root, manual_dir, device="cpu")
    assert manual_res["status"] == "PASS"
    assert manual_res["n_found"] == 4

    # Tabel sampel gagal QC
    rsua_dummy = pd.DataFrame([
        {"filename": "CXR_Images_1.bmp", "label": "Normal", "dice": 0.92, "iou": 0.86, "r_paru": 0.30},
        {"filename": "CXR_Images_2.bmp", "label": "Pneumonia", "dice": 0.70, "iou": 0.55, "r_paru": 0.25},
    ])
    failed_df = build_failed_qc_table(rsua_dummy, meta_df)
    assert len(failed_df) == 1
    assert failed_df.iloc[0]["sample_id"] == "CXR_Images_2.bmp"

    # Uji seluruh fungsi plotting menulis PNG valid tanpa crash
    p1 = plot_rsua_metrics_distribution(rsua_dummy, save_path=fig_dir / "rsua_dist.png", show=False)
    p2 = plot_r_paru_histogram(meta_df, save_path=fig_dir / "r_paru_hist.png", show=False)
    p3 = plot_kermany_overlay_grid(meta_df, processed_dir, n_random=2, n_worst=2, save_path=fig_dir / "kerm_ovl.png", show=False)

    r0 = meta_df.iloc[0]
    i224 = np.load(processed_dir / r0["image_224_path"])
    m224 = np.load(processed_dir / r0["mask_224_path"])
    payload = [(
        r0["image_id"],
        r0["label"],
        i224,
        m224,
        {
            "baseline_3ch": build_baseline_3ch(i224).numpy(),
            "hardmask_3ch": build_hardmask_3ch(i224, m224).numpy(),
            "fusion_4ch": build_fusion_4ch(i224, m224).numpy(),
        },
    )]
    p4 = plot_three_input_schemes(payload, save_path=fig_dir / "schemes.png", show=False)

    tf = PairwiseAffineTransform.from_config(cfg)
    aug_pairs = [tf(i224, m224) for _ in range(3)]
    p5 = plot_pairwise_augmentation_grid(i224, m224, aug_pairs, sample_title="test", save_path=fig_dir / "aug.png", show=False)

    for p in (p1, p2, p3, p4, p5):
        assert p is not None and p.exists() and p.stat().st_size > 1000


def test_notebooks_02_and_03_valid_syntax_and_gate_check() -> None:
    repo = get_repo_root()
    specs = [
        ("02_lung_mask_and_qc.ipynb", "# Tahap 1 —"),
        ("03_preprocess_sanity_check.ipynb", "# Tahap 2 & Tahap 3 —"),
    ]
    for fname, expected_title_prefix in specs:
        nb = json.loads((repo / "notebooks" / fname).read_text(encoding="utf-8"))
        cells = nb["cells"]
        assert cells[0]["cell_type"] == "markdown"
        first_line = "".join(cells[0]["source"]).splitlines()[0]
        assert first_line.startswith(expected_title_prefix), f"{fname}: judul tidak sesuai docs/2.md Aturan 1"

        # Sel terakhir wajib memuat GATE CHECK
        last_code = [c for c in cells if c["cell_type"] == "code"][-1]
        last_src = "".join(last_code["source"])
        assert "GATE CHECK" in last_src, f"{fname}: sel akhir wajib memuat GATE CHECK!"

        # Semua sel kode wajib valid Python syntax (abaikan baris !pip magic Colab)
        for idx, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            cleaned_lines = [
                line if not line.lstrip().startswith("!") else "pass"
                for line in cell["source"]
            ]
            ast.parse("".join(cleaned_lines), filename=f"{fname}:cell_{idx}")
