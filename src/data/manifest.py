"""Modul parser dan penyusun manifest dataset mentah (Kermany CXR & Multi-Sumber Kermany + RSNA).

Aturan:
- Mengacu pada docs/DATA_CONTRACT.md, Proposal Bab III (§3.2.1), dan pola regex Mendeley terverifikasi.
- Mengekstrak class label, patient ID, group_id berawalan sumber, dan sequence number.
- Wajib memvalidasi bahwa tidak ada berkas citra Kermany yang gagal diparsing (assert unmatched == 0).
- Memastikan tidak ada kebocoran ID pasien antar-kelas pada tiap sumber (assert overlap == 0).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

from src.data.rsna_loader import build_rsna_manifest, harmonize_image_to_lossless_png

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}

DEFAULT_MENDELEY_REGEX = re.compile(
    r"^(?P<cls>NORMAL|BACTERIA|VIRUS)-(?P<pid>\d+)-(?P<seq>\d+)\.(jpeg|jpg|png)$",
    re.IGNORECASE,
)


def parse_kermany_filename(
    filename: str, pattern: Optional[re.Pattern] = None
) -> Optional[Dict[str, Any]]:
    """Parsing nama berkas citra Kermany untuk mengekstrak class, patient ID, dan sequence."""
    if pattern is None:
        pattern = DEFAULT_MENDELEY_REGEX

    m = pattern.match(filename)
    if not m:
        return None

    raw_cls = m.group("cls").upper()
    pid = m.group("pid")
    seq = int(m.group("seq"))

    if raw_cls == "NORMAL":
        label = "Normal"
        label_id = 0
    elif raw_cls in ("BACTERIA", "VIRUS"):
        label = "Pneumonia"
        label_id = 1
    else:
        return None

    return {
        "raw_cls": raw_cls,
        "label": label,
        "label_id": label_id,
        "patient_id": pid,
        "seq_num": seq,
    }


def build_kermany_manifest(
    kermany_root: Path,
    regex_pattern: Optional[re.Pattern] = None,
    verify_images: bool = False,
    harmonized_dir: Optional[Path] = None,
    convert_to_png: bool = False,
) -> pd.DataFrame:
    """Bangun manifest tabular seluruh citra Kermany CXR.

    Jika `convert_to_png=True`, seluruh citra Kermany dikonversi ke PNG grayscale 8-bit lossless
    di `harmonized_dir/kermany/` agar seragam formatnya dengan RSNA (Proposal §3.2.1).
    """
    if not kermany_root.exists():
        raise FileNotFoundError(f"Direktori Kermany tidak ditemukan di: {kermany_root}")

    rows: List[Dict[str, Any]] = []
    unmatched_files: List[str] = []

    for split_dir in sorted(kermany_root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name.lower()

        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            folder_class = cls_dir.name.upper()

            for img_path in sorted(cls_dir.iterdir()):
                if not img_path.is_file():
                    continue
                if img_path.name.lower() in IGNORED_NAMES:
                    continue
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                parsed = parse_kermany_filename(img_path.name, pattern=regex_pattern)
                if parsed is None:
                    unmatched_files.append(str(img_path))
                    continue

                width, height, mode = None, None, None
                image_id = img_path.stem

                if convert_to_png:
                    if harmonized_dir is None:
                        raise ValueError("`harmonized_dir` wajib diisi bila `convert_to_png=True`!")
                    dst_png = harmonized_dir / "kermany" / f"{image_id}.png"
                    if not dst_png.exists():
                        meta = harmonize_image_to_lossless_png(img_path, dst_png)
                        width, height, mode = meta["width"], meta["height"], meta["mode"]
                    else:
                        with Image.open(dst_png) as im_cached:
                            width, height = im_cached.size
                            mode = im_cached.mode
                    rel_path = dst_png.relative_to(harmonized_dir).as_posix()
                else:
                    if verify_images:
                        with Image.open(img_path) as im:
                            width, height = im.size
                            mode = im.mode
                    rel_path = img_path.relative_to(kermany_root).as_posix()

                pid = str(parsed["patient_id"])
                rows.append({
                    "image_id": image_id,
                    "source": "kermany",
                    "rel_path": rel_path,
                    "orig_split": split_name,
                    "folder_class": folder_class,
                    "raw_cls": parsed["raw_cls"],
                    "label": parsed["label"],
                    "label_id": parsed["label_id"],
                    "patient_id": pid,
                    "group_id": f"kermany_{pid}",
                    "seq_num": parsed["seq_num"],
                    "filename": img_path.name,
                    "view_position": "FRONTAL_PEDIATRIC",
                    "sex": "UNKNOWN",
                    "photometric": "MONOCHROME2",
                    "n_bboxes": 0,
                    "has_bbox": False,
                    "width": width,
                    "height": height,
                    "mode": mode,
                })

    if unmatched_files:
        raise ValueError(
            f"Ditemukan {len(unmatched_files)} berkas citra yang tidak cocok pola regex! "
            f"Contoh: {unmatched_files[:5]}"
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Tidak ada citra valid yang ditemukan di {kermany_root}")

    # Assert tidak ada kebocoran ID pasien antar-kelas
    pids_normal = set(df[df["label"] == "Normal"]["patient_id"])
    pids_pneumonia = set(df[df["label"] == "Pneumonia"]["patient_id"])
    overlap = pids_normal & pids_pneumonia
    if overlap:
        raise AssertionError(
            f"Kebocoran ID pasien antar-kelas terdeteksi! Pasien overlap: {sorted(overlap)[:10]}"
        )

    return df


def build_multisource_manifest(
    kermany_root: Path,
    rsna_root: Path,
    harmonized_dir: Path,
    convert_to_png: bool = True,
    regex_pattern: Optional[re.Pattern] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Satukan manifest Kermany dan RSNA ke satu skema seragam (Proposal §3.2.1).

    Jika `convert_to_png=True`, seluruh citra Kermany dan RSNA dikonversi ke PNG grayscale 8-bit
    lossless di bawah `harmonized_dir` sehingga `rel_path` mengacu ke satu direktori akar seragam.
    """
    kerm_df = build_kermany_manifest(
        kermany_root=kermany_root,
        regex_pattern=regex_pattern,
        verify_images=not convert_to_png,
        harmonized_dir=harmonized_dir if convert_to_png else None,
        convert_to_png=convert_to_png,
    )
    rsna_df, rsna_audit = build_rsna_manifest(
        rsna_root=rsna_root,
        harmonized_dir=harmonized_dir if convert_to_png else None,
        convert_to_png=convert_to_png,
        extract_dicom_meta=not convert_to_png,
    )

    combined = pd.concat([kerm_df, rsna_df], ignore_index=True)

    # Assert tidak ada bentrokan image_id maupun group_id lintas sumber
    assert combined["image_id"].is_unique, "Ditemukan image_id duplikat lintas sumber!"
    group_sources = combined.groupby("group_id")["source"].nunique()
    assert (group_sources == 1).all(), "Ditemukan bentrokan group_id lintas sumber!"
    group_labels = combined.groupby("group_id")["label"].nunique()
    assert (group_labels == 1).all(), "Ditemukan group_id dengan lebih dari satu kelas!"

    audit = {
        "kermany_raw_count": int(len(kerm_df)),
        "rsna_audit": rsna_audit,
        "combined_raw_count": int(len(combined)),
        "harmonized_root": str(harmonized_dir) if convert_to_png else "",
    }
    return combined, audit
