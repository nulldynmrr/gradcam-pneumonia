"""Modul visualisasi Quality Control (QC) lung mask & sanity check prapemrosesan.

Memenuhi spesifikasi:
- Tahap 2a-b: semua figur tampil inline (`plt.show()`) DAN tersimpan ke Drive (`outputs/figures/`).
- `docs/2.md` Aturan 2: label & warna persis mengikuti `configs/pipeline_stages.yaml`:
  * Baseline 3-channel       : abu-abu (#7F8C8D)
  * Hard Masking 3-channel   : merah muda (#E8A0BF)
  * Fusi 4-Channel (Usulan)  : hijau mint (#76C893)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.segmentation.lung_mask import binarize_probability_map, postprocess_lung_mask
from src.segmentation.unet import predict_lung_probability

SCHEME_DISPLAY_LABELS: Dict[str, str] = {
    "baseline_3ch": "Baseline 3-channel",
    "hardmask_3ch": "Hard Masking 3-channel",
    "fusion_4ch": "Fusi 4-Channel (Usulan)",
}

SCHEME_COLORS: Dict[str, str] = {
    "baseline_3ch": "#7F8C8D",   # abu-abu
    "hardmask_3ch": "#E8A0BF",   # merah muda
    "fusion_4ch": "#76C893",     # hijau mint
    "model_output": "#FFF3B0",   # kuning muda
}

CLASS_COLORS: Dict[str, str] = {
    "Normal": "#5A6B7C",
    "Pneumonia": "#2A9D8F",
}


def _finalize_fig(fig: plt.Figure, save_path: Path | None, show: bool = True, dpi: int = 150) -> Path | None:
    """Simpan figur ke disk (jika diminta) lalu tampilkan atau tutup bersih."""
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_rsua_metrics_distribution(
    rsua_details_df: pd.DataFrame,
    dsc_min: float = 0.85,
    iou_min: float = 0.75,
    save_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot distribusi DSC dan IoU per kelas pada 85 citra RSUA beserta garis ambang proposal."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    classes = ["Normal", "Pneumonia"]

    for ax, metric, thr, title in [
        (axes[0], "dice", dsc_min, f"Distribusi DSC RSUA (Target ≥ {dsc_min:.2f})"),
        (axes[1], "iou", iou_min, f"Distribusi IoU RSUA (Target ≥ {iou_min:.2f})"),
    ]:
        data = [rsua_details_df[rsua_details_df["label"] == c][metric].values for c in classes]
        bp = ax.boxplot(data, tick_labels=classes, patch_artist=True, widths=0.45)
        for patch, c in zip(bp["boxes"], classes):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.65)

        rng = np.random.default_rng(42)
        for i, vals in enumerate(data, start=1):
            jitter = rng.uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, color="#222222", s=14, alpha=0.55)

        ax.axhline(thr, color="#D62828", linestyle="--", linewidth=1.4, label=f"Ambang Proposal ({thr:.2f})")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(metric.upper())
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Quality Control U-Net Beku pada Dataset RSUA (n=85: 32 Normal, 53 Pneumonia)", fontsize=11, fontweight="bold")
    return _finalize_fig(fig, save_path, show=show)


def plot_rsua_overlay_grid(
    rsua_details_df: pd.DataFrame,
    unet_model: object,
    device: object,
    n_random: int = 4,
    n_worst: int = 4,
    seed: int = 42,
    binarize_thr: float = 0.5,
    keep_components: int = 2,
    fill_holes: bool = True,
    save_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Grid overlay RSUA: baris atas = sampel acak (seed tetap), baris bawah = kasus DSC terburuk.

    Kontur biru/cyan = Ground Truth radiolog; kontur hijau = prediksi U-Net beku.
    """
    rand_df = rsua_details_df.sample(n=min(n_random, len(rsua_details_df)), random_state=seed)
    worst_df = rsua_details_df.sort_values(by="dice", ascending=True).head(min(n_worst, len(rsua_details_df)))

    ncols = max(len(rand_df), len(worst_df), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 8.2), dpi=140)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for row_idx, (sub_df, row_label) in enumerate([(rand_df, "Sampel Acak (Seed Tetap)"), (worst_df, "Kasus Terburuk (Lowest DSC)")]):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(sub_df):
                ax.axis("off")
                continue
            r = sub_df.iloc[col_idx]
            with Image.open(r["img_path"]) as im_raw, Image.open(r["gt_mask_path"]) as mk_raw:
                im_l = im_raw.convert("L")
                im_arr = np.array(im_l, dtype=np.float32) / 255.0
                gt_bin = (np.array(mk_raw.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)

                prob = predict_lung_probability(unet_model, im_l, device=device, return_original_size=True)
                pred_bin = postprocess_lung_mask(
                    binarize_probability_map(prob, threshold=binarize_thr),
                    keep_components=keep_components,
                    fill_holes=fill_holes,
                )

            ax.imshow(im_arr, cmap="gray")
            if gt_bin.max() > 0:
                ax.contour(gt_bin, levels=[0.5], colors=["#00D4FF"], linewidths=1.5)
            if pred_bin.max() > 0:
                ax.contour(pred_bin, levels=[0.5], colors=["#00FF66"], linewidths=1.5)
            ax.set_title(
                f"{row_label}\n{r['label']} | {r['filename']}\nDSC={r['dice']:.3f}, IoU={r['iou']:.3f}",
                fontsize=8.5,
            )
            ax.axis("off")

    fig.suptitle("QC Visual RSUA — Kontur Cyan: GT Radiolog | Kontur Hijau: Prediksi U-Net Beku", fontsize=11, fontweight="bold")
    return _finalize_fig(fig, save_path, show=show)


def plot_r_paru_histogram(
    metadata_df: pd.DataFrame,
    r_paru_extreme_thr: float = 0.05,
    r_paru_diff_thr: float = 0.05,
    save_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Histogram distribusi rasio luas paru (r_paru_224) per kelas beserta ambang kegagalan ekstrem."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    subsets = [("Seluruh Dataset Kermany (Pasca-Dedup)", metadata_df)]
    if "split" in metadata_df.columns and (metadata_df["split"] == "test").any():
        subsets.append(("Himpunan Uji (D_test)", metadata_df[metadata_df["split"] == "test"]))
    else:
        subsets.append(("Distribusi Luas Piksel Mask (piksel)", metadata_df))

    for idx, (ax, (title, sub_df)) in enumerate(zip(axes, subsets)):
        if idx == 1 and "split" not in metadata_df.columns:
            # Plot luas dalam piksel (M224)
            for cls in ("Normal", "Pneumonia"):
                vals = sub_df[sub_df["label"] == cls]["r_paru_224"].values * (224 * 224)
                ax.hist(vals, bins=25, alpha=0.6, color=CLASS_COLORS[cls], label=f"{cls} (n={len(vals)})")
            ax.set_xlabel("Luas Area Paru pada M_224 (piksel)")
            ax.set_ylabel("Jumlah Citra")
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            continue

        means = {}
        for cls in ("Normal", "Pneumonia"):
            vals = sub_df[sub_df["label"] == cls]["r_paru_224"].values
            if len(vals) == 0:
                continue
            means[cls] = float(np.mean(vals))
            ax.hist(
                vals,
                bins=30,
                range=(0.0, 0.8),
                alpha=0.55,
                color=CLASS_COLORS[cls],
                label=f"{cls} (n={len(vals)}, μ={means[cls]:.3f})",
            )
            ax.axvline(means[cls], color=CLASS_COLORS[cls], linestyle="-", linewidth=1.5)

        ax.axvline(
            r_paru_extreme_thr,
            color="#D62828",
            linestyle="--",
            linewidth=1.5,
            label=f"Ambang Gagal Ekstrem ({r_paru_extreme_thr:.2f})",
        )
        if "Normal" in means and "Pneumonia" in means:
            delta = abs(means["Normal"] - means["Pneumonia"])
            flag = "WAJIB ACR_norm" if delta > r_paru_diff_thr else "≤ 0.05"
            ax.text(
                0.97,
                0.72,
                f"|Δr̄_paru| = {delta:.4f}\n({flag})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round", facecolor="#FFF3B0", alpha=0.85),
            )
        ax.set_xlabel("Rasio Luas Paru (r_paru pada M_224)")
        ax.set_ylabel("Frekuensi Citra")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Distribusi Rasio Luas Paru (r_paru) & Pemeriksaan Keseimbangan Antarkelas", fontsize=11, fontweight="bold")
    return _finalize_fig(fig, save_path, show=show)


def plot_kermany_overlay_grid(
    metadata_df: pd.DataFrame,
    processed_dir: Path,
    n_random: int = 4,
    n_worst: int = 4,
    seed: int = 42,
    save_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Grid overlay Kermany dari cache (I_224, M_224): acak (seed tetap) + kasus r_paru terkecil."""
    test_pool = metadata_df[metadata_df["split"] == "test"] if "split" in metadata_df.columns else metadata_df
    if test_pool.empty:
        test_pool = metadata_df

    rand_df = test_pool.sample(n=min(n_random, len(test_pool)), random_state=seed)
    worst_df = test_pool.sort_values(by="r_paru_224", ascending=True).head(min(n_worst, len(test_pool)))

    ncols = max(len(rand_df), len(worst_df), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 8.2), dpi=140)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for row_idx, (sub_df, row_label) in enumerate([(rand_df, "Sampel Acak D_test (Seed Tetap)"), (worst_df, "Kasus Terburuk (Lowest r_paru)")]):
        for col_idx in range(ncols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(sub_df):
                ax.axis("off")
                continue
            r = sub_df.iloc[col_idx]
            i224 = np.load(processed_dir / r["image_224_path"])
            m224 = np.load(processed_dir / r["mask_224_path"])

            ax.imshow(i224, cmap="gray")
            if m224.max() > 0:
                ax.contour(m224, levels=[0.5], colors=["#00FF66"], linewidths=1.5)
            status_tag = " [EKSTREM]" if r["r_paru_224"] < 0.05 else ""
            ax.set_title(
                f"{row_label}\n{r['label']} | {r['image_id']}\nr_paru={r['r_paru_224']:.3f}{status_tag}",
                fontsize=8.5,
            )
            ax.axis("off")

    fig.suptitle("QC Visual Kermany (Cache I_224 + Kontur Hijau M_224)", fontsize=11, fontweight="bold")
    return _finalize_fig(fig, save_path, show=show)


def plot_three_input_schemes(
    samples: Sequence[Tuple[str, str, np.ndarray, np.ndarray, Dict[str, np.ndarray]]],
    save_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Visualisasi perbandingan 3 skema masukan dari citra & mask NYATA (label & warna Gambar 3.1).

    Args:
        samples: List tuple `(image_id, label, i224, m224, tensors_dict)` dengan
                 `tensors_dict` berisi key `'baseline_3ch'`, `'hardmask_3ch'`, `'fusion_4ch'`.
    """
    nrows = len(samples)
    fig, axes = plt.subplots(nrows, 5, figsize=(19, 3.8 * nrows), dpi=140)
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_specs = [
        ("Citra I_224 + Kontur M_224", "#ECEFF1"),
        (f"{SCHEME_DISPLAY_LABELS['baseline_3ch']}\n(Ch-0 Ternormalisasi)", SCHEME_COLORS["baseline_3ch"]),
        (f"{SCHEME_DISPLAY_LABELS['hardmask_3ch']}\n(Ch-0 Ternormalisasi)", SCHEME_COLORS["hardmask_3ch"]),
        (f"{SCHEME_DISPLAY_LABELS['fusion_4ch']}\n(Ch-0..2 CXR Ternormalisasi)", SCHEME_COLORS["fusion_4ch"]),
        (f"{SCHEME_DISPLAY_LABELS['fusion_4ch']}\n(Ch-4 Mask Biner {{0,1}})", SCHEME_COLORS["fusion_4ch"]),
    ]

    for row_idx, (img_id, label, i224, m224, tensors) in enumerate(samples):
        base_ch0 = tensors["baseline_3ch"][0]
        hard_ch0 = tensors["hardmask_3ch"][0]
        fusi_ch0 = tensors["fusion_4ch"][0]
        fusi_ch4 = tensors["fusion_4ch"][3]

        axes[row_idx, 0].imshow(i224, cmap="gray")
        if m224.max() > 0:
            axes[row_idx, 0].contour(m224, levels=[0.5], colors=["#00FF66"], linewidths=1.3)
        axes[row_idx, 0].set_ylabel(f"{label}\n{img_id}", fontsize=9, fontweight="bold")

        axes[row_idx, 1].imshow(base_ch0, cmap="gray")
        axes[row_idx, 2].imshow(hard_ch0, cmap="gray")
        axes[row_idx, 3].imshow(fusi_ch0, cmap="gray")
        axes[row_idx, 4].imshow(fusi_ch4, cmap="viridis", vmin=0.0, vmax=1.0)

        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                title_txt, bg_col = col_specs[col_idx]
                ax.set_title(
                    title_txt,
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_col, alpha=0.75),
                )

    fig.suptitle("Sanity Check Visual — Tiga Konfigurasi Masukan EfficientNet-B0 (Tahap 3 Gambar 3.1)", fontsize=12, fontweight="bold")
    return _finalize_fig(fig, save_path, show=show)


def plot_pairwise_augmentation_grid(
    image_224: np.ndarray,
    mask_224: np.ndarray,
    augmented_pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    sample_title: str = "",
    save_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Visualisasi citra & kontur mask sebelum vs sesudah augmentasi berpasangan (P5)."""
    ncols = 1 + len(augmented_pairs)
    fig, axes = plt.subplots(1, ncols, figsize=(3.8 * ncols, 4.0), dpi=140)

    axes[0].imshow(image_224, cmap="gray")
    if mask_224.max() > 0:
        axes[0].contour(mask_224, levels=[0.5], colors=["#00FF66"], linewidths=1.5)
    axes[0].set_title(f"Asli (Tanpa Augmentasi)\nMask unik: {sorted(np.unique(mask_224).tolist())}", fontsize=9)
    axes[0].axis("off")

    for i, (aug_im, aug_mk) in enumerate(augmented_pairs, start=1):
        axes[i].imshow(aug_im, cmap="gray")
        if aug_mk.max() > 0:
            axes[i].contour(aug_mk, levels=[0.5], colors=["#00FF66"], linewidths=1.5)
        axes[i].set_title(
            f"Augmentasi Trial #{i}\nMask unik: {sorted(np.unique(aug_mk).tolist())}",
            fontsize=9,
        )
        axes[i].axis("off")

    fig.suptitle(f"Verifikasi Keselarasan Augmentasi Berpasangan (I_224, M_224) — {sample_title}", fontsize=11, fontweight="bold")
    return _finalize_fig(fig, save_path, show=show)
