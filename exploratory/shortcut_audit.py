"""[DI LUAR PROPOSAL / AUDIT DATA] Probe diagnostik shortcut learning sebelum pelatihan utama.

Dua pemeriksaan ringan untuk mengaudit integritas dataset gabungan (Kermany + RSNA):
1. Probe 1 (Tebak Sumber): Mengukur sejauh mana asal dataset (Kermany vs RSNA) dapat dibedakan
   dari citra 224x224. Nilai tinggi mengonfirmasi perlunya penyamaan rasio kelas 1:1 per sumber.
2. Probe 2 (Area di Luar Paru, I * (1 - M)): Mengukur AUC prediksi kelas Normal vs Pneumonia
   HANYA menggunakan piksel di luar lung mask. Nilai AUC > 0.50 secara bermakna menunjukkan
   keberadaan sinyal latar belakang / shortcut di luar paru (Zech dkk., 2018; DeGrave dkk., 2021).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _extract_compact_features(
    meta_df: pd.DataFrame,
    processed_dir: Path,
    mode: str = "full",
    downsample_hw: Tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Ekstrak fitur ringkas (downsample 32x32 + statistik tepi) dari cache I_224 & M_224."""
    feats = []
    h_ds, w_ds = downsample_hw
    for _, row in meta_df.iterrows():
        i224 = np.load(processed_dir / row["image_224_path"]).astype(np.float32)
        if mode == "outside_lung":
            m224 = np.load(processed_dir / row["mask_224_path"]).astype(np.float32)
            arr = i224 * (1.0 - m224)
        else:
            arr = i224

        im_ds = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).resize(
            (w_ds, h_ds), Image.Resampling.BILINEAR
        )
        vec = np.asarray(im_ds, dtype=np.float32).ravel() / 255.0
        # Tambah statistik tepi (4 sudut & border) untuk menangkap confounder bingkai
        border_vals = np.concatenate([arr[:16, :].ravel(), arr[-16:, :].ravel(), arr[:, :16].ravel(), arr[:, -16:].ravel()])
        extra = np.array([float(arr.mean()), float(arr.std()), float(border_vals.mean()), float(border_vals.std())], dtype=np.float32)
        feats.append(np.concatenate([vec, extra]))
    return np.vstack(feats)


def _fit_eval_binary_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"auc_roc": float("nan"), "accuracy": float("nan"), "n_train": int(len(y_train)), "n_test": int(len(y_test))}

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, C=0.5, random_state=seed),
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def run_shortcut_audit_probes(
    meta_df: pd.DataFrame,
    processed_dir: Path,
    seed: int = 42,
    max_train_samples: int | None = 3000,
) -> Dict[str, Any]:
    """Jalankan Probe 1 (Tebak Sumber) dan Probe 2 (Area di Luar Paru) pada data cache."""
    df = meta_df.copy()
    if "source" not in df.columns:
        df["source"] = "kermany"

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    if max_train_samples is not None and len(train_df) > max_train_samples:
        train_df = train_df.sample(n=max_train_samples, random_state=seed).reset_index(drop=True)

    # Probe 1: Tebak Sumber (hanya bila >= 2 sumber tersedia)
    unique_sources = sorted(df["source"].unique().tolist())
    if len(unique_sources) >= 2:
        X_tr_full = _extract_compact_features(train_df, processed_dir, mode="full")
        X_te_full = _extract_compact_features(test_df, processed_dir, mode="full")
        y_tr_src = (train_df["source"] == unique_sources[1]).astype(int).to_numpy()
        y_te_src = (test_df["source"] == unique_sources[1]).astype(int).to_numpy()
        probe1_res = _fit_eval_binary_probe(X_tr_full, y_tr_src, X_te_full, y_te_src, seed=seed)
        probe1_res["sources"] = unique_sources
    else:
        probe1_res = {"skipped": True, "reason": "Hanya satu sumber dataset tersedia", "auc_roc": float("nan"), "accuracy": float("nan")}

    # Probe 2: Prediksi Kelas hanya dari piksel DI LUAR PARU I * (1 - M)
    X_tr_out = _extract_compact_features(train_df, processed_dir, mode="outside_lung")
    X_te_out = _extract_compact_features(test_df, processed_dir, mode="outside_lung")
    y_tr_cls = train_df["label_id"].astype(int).to_numpy()
    y_te_cls = test_df["label_id"].astype(int).to_numpy()
    probe2_overall = _fit_eval_binary_probe(X_tr_out, y_tr_cls, X_te_out, y_te_cls, seed=seed)

    probe2_per_source: Dict[str, Dict[str, Any]] = {}
    for src in unique_sources:
        tr_mask = (train_df["source"] == src).to_numpy()
        te_mask = (test_df["source"] == src).to_numpy()
        probe2_per_source[src] = _fit_eval_binary_probe(
            X_tr_out[tr_mask], y_tr_cls[tr_mask], X_te_out[te_mask], y_te_cls[te_mask], seed=seed
        )

    summary_rows = [
        {
            "probe": "Probe 1: Tebak Sumber (Full Image)",
            "subset": "overall",
            "auc_roc": probe1_res.get("auc_roc", float("nan")),
            "accuracy": probe1_res.get("accuracy", float("nan")),
            "n_train": probe1_res.get("n_train", 0),
            "n_test": probe1_res.get("n_test", 0),
        },
        {
            "probe": "Probe 2: Tebak Kelas Luar Paru I*(1-M)",
            "subset": "overall",
            "auc_roc": probe2_overall["auc_roc"],
            "accuracy": probe2_overall["accuracy"],
            "n_train": probe2_overall["n_train"],
            "n_test": probe2_overall["n_test"],
        },
    ]
    for src, res in probe2_per_source.items():
        summary_rows.append({
            "probe": "Probe 2: Tebak Kelas Luar Paru I*(1-M)",
            "subset": src,
            "auc_roc": res["auc_roc"],
            "accuracy": res["accuracy"],
            "n_train": res["n_train"],
            "n_test": res["n_test"],
        })

    return {
        "probe1_source_predictability": probe1_res,
        "probe2_outside_lung_overall": probe2_overall,
        "probe2_outside_lung_per_source": probe2_per_source,
        "summary_df": pd.DataFrame(summary_rows),
    }
