"""Utilitas resolusi path proyek untuk Google Colab dan lingkungan lokal.

Aturan:
- Jika dijalankan di Google Colab (`google.colab` di `sys.modules`), gunakan
  `/content/drive/MyDrive/LAB_TA_GradCAM_Pneumonia` (atau `PROJECT_ROOT` jika diset).
- Jika di lokal, gunakan environment variable `PROJECT_ROOT` bila ada,
  atau default `<repo_root>/local_mirror`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import yaml

DEFAULT_COLAB_ROOT = Path("/content/drive/MyDrive/LAB_TA_GradCAM_Pneumonia")
DEFAULT_COLAB_WORK = Path("/content/work")


def is_colab() -> bool:
    """Deteksi apakah kode sedang berjalan di runtime Google Colab."""
    return "google.colab" in sys.modules


def get_repo_root() -> Path:
    """Kembalikan root direktori repositori kode `gradcam-pneumonia`."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path | None = None) -> dict:
    """Muat berkas konfigurasi `configs/config.yaml`."""
    if config_path is None:
        config_path = get_repo_root() / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_project_root(cfg: dict | None = None) -> Path:
    """Tentukan `PROJECT_ROOT` tempat `data/`, `models/`, dan `outputs/` berada."""
    env_key = "PROJECT_ROOT"
    if cfg is not None:
        env_key = cfg.get("runtime", {}).get("project_root_env", "PROJECT_ROOT")
    env_val = os.environ.get(env_key)
    if env_val:
        return Path(env_val).expanduser().resolve()

    if is_colab():
        if cfg is not None:
            colab_p = cfg.get("runtime", {}).get("project_root_colab")
            if colab_p:
                return Path(colab_p)
        return DEFAULT_COLAB_ROOT

    if cfg is not None:
        local_rel = cfg.get("runtime", {}).get("project_root_local_default", "./local_mirror")
        return (get_repo_root() / local_rel).resolve()
    return (get_repo_root() / "local_mirror").resolve()


def get_project_paths(project_root: Path | None = None, cfg: dict | None = None) -> Dict[str, Path]:
    """Kembalikan kamus path standar di bawah `PROJECT_ROOT`."""
    if cfg is None:
        cfg = load_config()
    if project_root is None:
        project_root = get_project_root(cfg)

    rel = cfg.get("runtime", {}).get("rel_paths", {})
    kermany_rel = rel.get("kermany_raw", "data/raw/kermany/chest_xray")
    rsua_val_rel = rel.get(
        "rsua_validated",
        "data/raw/rsua/Data Chest X-Ray RSUA (Validated)-20230618T025557Z-001/Data Chest X-Ray RSUA (Validated)",
    )
    manual_rel = rel.get("kermany_manual_masks", "data/raw/kermany_manual_masks")
    seg_pub_rel = rel.get("seg_public", "data/raw/seg_public")
    interim_rel = rel.get("interim", "data/interim")
    processed_rel = rel.get("processed", "data/processed")
    unet_rel = rel.get("unet_weights_dir", "models/unet_frozen")
    outputs_rel = rel.get("outputs", "outputs")

    return {
        "project_root": project_root,
        "data_raw": project_root / "data" / "raw",
        "kermany_root": project_root / "data" / "raw" / "kermany",
        "kermany_chest_xray": project_root / kermany_rel,
        "rsua_root": project_root / "data" / "raw" / "rsua",
        "rsua_validated": project_root / rsua_val_rel,
        "kermany_manual_masks": project_root / manual_rel,
        "seg_public": project_root / seg_pub_rel,
        "interim": project_root / interim_rel,
        "processed": project_root / processed_rel,
        "unet_frozen": project_root / unet_rel,
        "outputs": project_root / outputs_rel,
    }


def ensure_output_dirs(project_root: Path | None = None, cfg: dict | None = None) -> Dict[str, Path]:
    """Buat seluruh direktori keluaran/interim (TIDAK menyentuh `data/raw` karena read-only)."""
    paths = get_project_paths(project_root, cfg)
    for key in ("interim", "processed", "unet_frozen", "outputs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    for sub in (
        "checkpoints",
        "logs",
        "predictions",
        "gradcam",
        "metrics",
        "stats",
        "figures",
        "figures/eda",
        "figures/qc",
        "figures/preprocess",
        "figures/lung_mask_overlays",
        "tables",
        "tables/eda",
        "tables/qc",
        "tables/preprocess",
    ):
        (paths["outputs"] / sub).mkdir(parents=True, exist_ok=True)
    return paths
