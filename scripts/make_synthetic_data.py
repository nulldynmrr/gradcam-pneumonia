"""Generator dataset sintetis di `local_mirror/` untuk verifikasi pipeline lokal (CPU).

Menghasilkan struktur folder persis seperti Google Drive (`LAB_TA_GradCAM_Pneumonia`):
- `data/raw/kermany/chest_xray/{train,test}/{NORMAL,PNEUMONIA}` (pola Mendeley + 1 duplikat).
- `data/raw/rsna/` (`stage_2_train_labels.csv`, `stage_2_detailed_class_info.csv`, `stage_2_train_images/*.png`,
  termasuk baris multi-bbox dan kelas `No Lung Opacity / Not Normal` untuk uji filter biner).
- `data/raw/rsua/Data Chest X-Ray RSUA (Validated).../{Covid,Non_Covid,Non_Covid_Pneumonia}/{images,masks}` (256x256).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _make_lung_like_pair(height: int, width: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Bangkitkan pasangan (citra grayscale uint8, mask paru biner uint8 {0, 255}) sintetis."""
    yy, xx = np.mgrid[0:height, 0:width]
    cy = height * 0.52
    lx = width * 0.33
    rx = width * 0.67
    ry = height * 0.30
    rx_rad = width * 0.16

    left = (((xx - lx) / rx_rad) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    right = (((xx - rx) / rx_rad) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    mask = (left | right).astype(np.uint8) * 255

    bg = rng.integers(90, 170, size=(height, width), dtype=np.uint8)
    lung_tex = rng.integers(30, 110, size=(height, width), dtype=np.uint8)
    img = np.where(mask > 0, lung_tex, bg).astype(np.uint8)
    return img, mask


def generate_synthetic_mirror(out_root: Path, n_patients_per_class: int = 20, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    kermany_root = out_root / "data" / "raw" / "kermany" / "chest_xray"
    for split in ("train", "test"):
        for cls in ("NORMAL", "PNEUMONIA"):
            (kermany_root / split / cls).mkdir(parents=True, exist_ok=True)

    # 1. Buat citra Kermany sintetis dengan pola Mendeley (timpang sengaja untuk uji 1:1 subsampling)
    for i in range(n_patients_per_class):
        pid_norm = 1000000 + i
        split = "train" if i < int(0.8 * n_patients_per_class) else "test"
        n_imgs = 1 if (i % 3 != 0) else 2
        for seq in range(1, n_imgs + 1):
            img, _ = _make_lung_like_pair(300, 360, rng)
            fn = f"NORMAL-{pid_norm}-{seq:04d}.jpeg"
            Image.fromarray(img, mode="L").save(kermany_root / split / "NORMAL" / fn, quality=92)

    for i in range(int(n_patients_per_class * 1.5)):
        pid_pneu = 2000000 + i
        split = "train" if i < int(0.8 * n_patients_per_class * 1.5) else "test"
        sub = "BACTERIA" if (i % 2 == 0) else "VIRUS"
        n_imgs = 1 if (i % 3 != 0) else 2
        for seq in range(1, n_imgs + 1):
            img, _ = _make_lung_like_pair(310, 350, rng)
            fn = f"{sub}-{pid_pneu}-{seq:04d}.jpeg"
            Image.fromarray(img, mode="L").save(kermany_root / split / "PNEUMONIA" / fn, quality=92)

    # Tambahkan 1 salinan byte-identik untuk uji dedup
    sample_src = next((kermany_root / "train" / "NORMAL").glob("*.jpeg"))
    dup_dst = kermany_root / "train" / "NORMAL" / "NORMAL-1999999-0001.jpeg"
    dup_dst.write_bytes(sample_src.read_bytes())

    # 2. Buat RSNA sintetis (Normal, Lung Opacity multi-bbox, & No Lung Opacity / Not Normal)
    rsna_root = out_root / "data" / "raw" / "rsna"
    rsna_img_dir = rsna_root / "stage_2_train_images"
    rsna_img_dir.mkdir(parents=True, exist_ok=True)

    rsna_label_rows = []
    rsna_cls_rows = []

    n_rsna_norm = int(n_patients_per_class * 1.5)
    n_rsna_opac = n_patients_per_class
    n_rsna_excl = max(6, n_patients_per_class // 2)

    for i in range(n_rsna_norm):
        pid = f"rsna-norm-{i:04d}"
        img, _ = _make_lung_like_pair(256, 256, rng)
        Image.fromarray(img, mode="L").save(rsna_img_dir / f"{pid}.png")
        rsna_label_rows.append({"patientId": pid, "x": np.nan, "y": np.nan, "width": np.nan, "height": np.nan, "Target": 0})
        rsna_cls_rows.append({"patientId": pid, "class": "Normal"})

    for i in range(n_rsna_opac):
        pid = f"rsna-opac-{i:04d}"
        img, _ = _make_lung_like_pair(256, 256, rng)
        Image.fromarray(img, mode="L").save(rsna_img_dir / f"{pid}.png")
        # Sebagian punya 2 bounding box untuk menguji agregasi multi-baris
        n_boxes = 2 if (i % 3 == 0) else 1
        for b in range(n_boxes):
            rsna_label_rows.append({
                "patientId": pid,
                "x": 40.0 + b * 80.0,
                "y": 60.0,
                "width": 50.0,
                "height": 70.0,
                "Target": 1,
            })
            rsna_cls_rows.append({"patientId": pid, "class": "Lung Opacity"})

    for i in range(n_rsna_excl):
        pid = f"rsna-excl-{i:04d}"
        img, _ = _make_lung_like_pair(256, 256, rng)
        Image.fromarray(img, mode="L").save(rsna_img_dir / f"{pid}.png")
        rsna_label_rows.append({"patientId": pid, "x": np.nan, "y": np.nan, "width": np.nan, "height": np.nan, "Target": 0})
        rsna_cls_rows.append({"patientId": pid, "class": "No Lung Opacity / Not Normal"})

    pd.DataFrame(rsna_label_rows).to_csv(rsna_root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(rsna_cls_rows).to_csv(rsna_root / "stage_2_detailed_class_info.csv", index=False)

    # 3. Buat RSUA Validated sintetis (ukuran 256x256)
    rsua_val = (
        out_root
        / "data"
        / "raw"
        / "rsua"
        / "Data Chest X-Ray RSUA (Validated)-20230618T025557Z-001"
        / "Data Chest X-Ray RSUA (Validated)"
    )
    rsua_counts = {"Covid": 6, "Non_Covid": 6, "Non_Covid_Pneumonia": 8}
    idx = 1
    for folder_name, cnt in rsua_counts.items():
        img_d = rsua_val / folder_name / "images"
        msk_d = rsua_val / folder_name / "masks"
        img_d.mkdir(parents=True, exist_ok=True)
        msk_d.mkdir(parents=True, exist_ok=True)
        for _ in range(cnt):
            img, mask = _make_lung_like_pair(256, 256, rng)
            Image.fromarray(img, mode="L").save(img_d / f"CXR_Images_{idx}.bmp")
            Image.fromarray(mask, mode="L").save(msk_d / f"CXR_Mask_{idx}.bmp")
            idx += 1

    (out_root / "data" / "raw" / "kermany_manual_masks").mkdir(parents=True, exist_ok=True)
    (out_root / "data" / "raw" / "seg_public").mkdir(parents=True, exist_ok=True)
    (out_root / "models" / "unet_frozen").mkdir(parents=True, exist_ok=True)
    (out_root / "data" / "interim" / "harmonized_png").mkdir(parents=True, exist_ok=True)
    (out_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (out_root / "outputs").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangkitkan local_mirror sintetis (Kermany + RSNA + RSUA).")
    parser.add_argument("--out", type=Path, default=Path("local_mirror"), help="Folder tujuan")
    parser.add_argument("--n-patients-per-class", type=int, default=20)
    args = parser.parse_args()
    generate_synthetic_mirror(args.out.resolve(), n_patients_per_class=args.n_patients_per_class)
    print(f"Dataset sintetis siap di: {args.out.resolve()}")


if __name__ == "__main__":
    main()
