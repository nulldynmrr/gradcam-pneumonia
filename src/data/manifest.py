"""Modul parser dan penyusun manifest dataset mentah (Kermany CXR).

Aturan:
- Mengacu pada docs/DATA_CONTRACT.md dan pola regex Mendeley terverifikasi.
- Mengekstrak class label, patient ID, dan sequence number.
- Wajib memvalidasi bahwa tidak ada berkas citra yang gagal diparsing (assert unmatched == 0).
- Memastikan tidak ada kebocoran ID pasien antar-kelas (assert patient ID overlap == 0).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
IGNORED_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}

DEFAULT_MENDELEY_REGEX = re.compile(
    r"^(?P<cls>NORMAL|BACTERIA|VIRUS)-(?P<pid>\d+)-(?P<seq>\d+)\.(jpeg|jpg|png)$",
    re.IGNORECASE,
)


def parse_kermany_filename(
    filename: str, pattern: Optional[re.Pattern] = None
) -> Optional[Dict[str, str]]:
    """Parsing nama berkas citra Kermany untuk mengekstrak class, patient ID, dan sequence.

    Args:
        filename: Nama berkas (misal: 'NORMAL-1003233-0001.jpeg').
        pattern: Objek regex compiled (default: DEFAULT_MENDELEY_REGEX).

    Returns:
        Dict dengan kunci 'raw_cls', 'label', 'label_id', 'patient_id', 'seq_num',
        atau None jika nama berkas tidak cocok.
    """
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
) -> pd.DataFrame:
    """Bangun manifest tabular seluruh citra Kermany CXR.

    Args:
        kermany_root: Path direktori `chest_xray/` yang memuat folder `train/` dan `test/`.
        regex_pattern: Pola regex untuk parsing nama berkas.
        verify_images: Jika True, buka setiap citra dengan PIL untuk memeriksa dimensi & mode.

    Returns:
        pd.DataFrame dengan kolom:
        ['image_id', 'rel_path', 'orig_split', 'folder_class', 'raw_cls',
         'label', 'label_id', 'patient_id', 'seq_num', 'filename',
         'width', 'height', 'mode']
    """
    if not kermany_root.exists():
        raise FileNotFoundError(f"Direktori Kermany tidak ditemukan di: {kermany_root}")

    rows: List[Dict] = []
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
                if verify_images:
                    with Image.open(img_path) as im:
                        width, height = im.size
                        mode = im.mode

                image_id = img_path.stem
                rel_path = img_path.relative_to(kermany_root).as_posix()

                rows.append({
                    "image_id": image_id,
                    "rel_path": rel_path,
                    "orig_split": split_name,
                    "folder_class": folder_class,
                    "raw_cls": parsed["raw_cls"],
                    "label": parsed["label"],
                    "label_id": parsed["label_id"],
                    "patient_id": parsed["patient_id"],
                    "seq_num": parsed["seq_num"],
                    "filename": img_path.name,
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
