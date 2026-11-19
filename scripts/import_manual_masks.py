"""Skrip impor & validasi kontur paru manual penulis untuk subset 50 citra D_test Kermany (L3(2)).

Memeriksa berkas di `data/raw/kermany_manual_masks/<image_id>.png`:
- Cocok dengan daftar terkunci `data/interim/kermany_qc_50_list.csv`.
- Ukuran mask sama persis dengan ukuran citra asli Kermany (AMBIG-23).
- Nilai mask dapat dikonversi bersih ke biner {0, 1}.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from src.paths import get_project_paths, load_config


def validate_manual_masks(
    subset_csv: Path,
    kermany_root: Path,
    manual_masks_dir: Path,
) -> Dict[str, Any]:
    """Validasi kelengkapan dan integritas seluruh kontur manual penulis di `manual_masks_dir`."""
    if not subset_csv.exists():
        raise FileNotFoundError(
            f"Daftar subset 50 citra belum ada: {subset_csv}. Jalankan notebook 02 terlebih dahulu."
        )

    subset_df = pd.read_csv(subset_csv)
    manual_masks_dir.mkdir(parents=True, exist_ok=True)

    found: Dict[str, Path] = {}
    for ext in ("*.png", "*.bmp", "*.jpg", "*.jpeg"):
        for p in manual_masks_dir.glob(ext):
            if not p.name.startswith("."):
                found[p.stem] = p

    missing: List[str] = []
    size_mismatch: List[str] = []
    empty_masks: List[str] = []
    valid_count = 0

    for _, row in subset_df.iterrows():
        img_id = str(row["image_id"])
        if img_id not in found:
            missing.append(img_id)
            continue

        img_p = kermany_root / str(row["rel_path"])
        mk_p = found[img_id]
        with Image.open(img_p) as im, Image.open(mk_p) as mk:
            if im.size != mk.size:
                size_mismatch.append(f"{img_id}: img={im.size} vs mask={mk.size}")
                continue
            arr = np.array(mk.convert("L"), dtype=np.uint8)
            if (arr > 0).sum() == 0:
                empty_masks.append(img_id)
                continue
            valid_count += 1

    status = "PASS" if (valid_count == len(subset_df) and not size_mismatch and not empty_masks) else "INCOMPLETE"
    return {
        "status": status,
        "n_expected": int(len(subset_df)),
        "n_valid": int(valid_count),
        "n_missing": int(len(missing)),
        "missing_sample_ids": missing[:10],
        "size_mismatch": size_mismatch,
        "empty_masks": empty_masks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validasi impor mask manual Kermany (50 sampel)")
    parser.parse_args()
    cfg = load_config()
    paths = get_project_paths(cfg=cfg)
    report = validate_manual_masks(
        subset_csv=paths["interim"] / "kermany_qc_50_list.csv",
        kermany_root=paths["kermany_chest_xray"],
        manual_masks_dir=paths["kermany_manual_masks"],
    )
    print("=== VALIDASI MASK MANUAL KERMANY ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
