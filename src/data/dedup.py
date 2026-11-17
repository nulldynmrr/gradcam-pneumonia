"""Modul deduplikasi citra identik (byte-identik dan piksel-identik).

Aturan:
- Proposal §3.2.1 (Bab III L24): deduplikasi dilakukan SEBELUM pembagian data (split).
- Memeriksa SHA-256 byte berkas dan SHA-256 representasi piksel grayscale uint8 + dimensi citra.
- Jika ada duplikat dengan label kelas bertentangan: STOP and RAISE (dilarang menghapus diam-diam).
- Menghasilkan log deduplikasi lengkap ke `data/interim/dedup_log.csv`.
- Mempertahankan satu citra per kelompok duplikat secara deterministik (urutan path).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def compute_byte_hash(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """Hitung SHA-256 dari seluruh isi byte berkas."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_pixel_hash(file_path: Path) -> Tuple[str, Tuple[int, int]]:
    """Hitung SHA-256 dari array piksel grayscale uint8 beserta ukuran citra."""
    with Image.open(file_path) as im:
        im_gray = im.convert("L")
        arr = np.ascontiguousarray(np.array(im_gray, dtype=np.uint8))
        h = hashlib.sha256()
        h.update(f"{arr.shape[0]}x{arr.shape[1]}:".encode("utf-8"))
        h.update(arr.tobytes())
        return h.hexdigest(), (im.width, im.height)


def run_deduplication(
    manifest_df: pd.DataFrame,
    kermany_root: Path,
    check_pixel_hash: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan prosedur deduplikasi pada manifest citra Kermany.

    Args:
        manifest_df: DataFrame hasil `build_kermany_manifest`.
        kermany_root: Path direktori root citra.
        check_pixel_hash: Jika True, selain byte hash juga hitung hash piksel grayscale.

    Returns:
        Tuple (clean_df, dedup_log_df):
        - clean_df: Manifest bersih setelah citra duplikat dikeluarkan.
        - dedup_log_df: Catatan seluruh citra (status KEPT / DROPPED, duplicate_group_id, alasan).
    """
    df = manifest_df.copy()
    df = df.sort_values(by=["rel_path"]).reset_index(drop=True)

    has_byte = "byte_hash" in df.columns and df["byte_hash"].notna().all() and (df["byte_hash"] != "").all()
    has_sha = "sha256" in df.columns and df["sha256"].notna().all() and (df["sha256"] != "").all()

    if has_byte:
        byte_hashes = df["byte_hash"].tolist()
    elif has_sha:
        byte_hashes = df["sha256"].tolist()
    else:
        byte_hashes = []
        for _, row in df.iterrows():
            p = kermany_root / row["rel_path"]
            byte_hashes.append(compute_byte_hash(p))

    if "pixel_hash" in df.columns and df["pixel_hash"].notna().all() and (df["pixel_hash"] != "").all():
        pixel_hashes = df["pixel_hash"].tolist()
    elif check_pixel_hash:
        pixel_cache: Dict[str, str] = {}
        pixel_hashes = []
        for b_hash, (_, row) in zip(byte_hashes, df.iterrows()):
            if b_hash in pixel_cache:
                pixel_hashes.append(pixel_cache[b_hash])
            else:
                p = kermany_root / row["rel_path"]
                p_hash, _ = compute_pixel_hash(p)
                pixel_cache[b_hash] = p_hash
                pixel_hashes.append(p_hash)
    else:
        pixel_hashes = list(byte_hashes)

    df["byte_hash"] = byte_hashes
    df["pixel_hash"] = pixel_hashes

    # Grup gabungan: jika byte_hash sama ATAU pixel_hash sama, kelompokkan
    # Disjoint set / union-find untuk mengelompokkan
    parent = list(range(len(df)))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    byte_map: Dict[str, int] = {}
    pixel_map: Dict[str, int] = {}

    for idx, row in df.iterrows():
        bh = row["byte_hash"]
        ph = row["pixel_hash"]
        if bh in byte_map:
            union(idx, byte_map[bh])
        else:
            byte_map[bh] = idx

        if ph in pixel_map:
            union(idx, pixel_map[ph])
        else:
            pixel_map[ph] = idx

    group_ids = [find(i) for i in range(len(df))]
    df["dup_group_id"] = group_ids

    # Periksa apakah ada konflik kelas dalam kelompok duplikat
    group_labels = df.groupby("dup_group_id")["label"].nunique()
    conflicts = group_labels[group_labels > 1]
    if not conflicts.empty:
        conflict_ids = conflicts.index.tolist()
        conflict_rows = df[df["dup_group_id"].isin(conflict_ids)]
        raise ValueError(
            f"Ditemukan kelompok duplikat dengan label kelas BERBEDA (konflik)! "
            f"Contoh konflik:\n{conflict_rows[['image_id', 'rel_path', 'label', 'dup_group_id']]}"
        )

    # Tandai status: citra pertama dalam grup = KEPT, sisanya = DROPPED
    status_list: List[str] = []
    reason_list: List[str] = []
    seen_groups: Set[int] = set()

    for _, row in df.iterrows():
        gid = row["dup_group_id"]
        group_size = (df["dup_group_id"] == gid).sum()
        if gid not in seen_groups:
            seen_groups.add(gid)
            status_list.append("KEPT")
            if group_size > 1:
                reason_list.append("Primary instance in duplicate group")
            else:
                reason_list.append("Unique image")
        else:
            status_list.append("DROPPED")
            reason_list.append(f"Identical duplicate in group {gid}")

    df["status"] = status_list
    df["reason"] = reason_list

    clean_df = df[df["status"] == "KEPT"].drop(columns=["byte_hash", "pixel_hash", "dup_group_id", "status", "reason"]).reset_index(drop=True)
    dedup_log_df = df[["image_id", "rel_path", "label", "patient_id", "byte_hash", "pixel_hash", "dup_group_id", "status", "reason"]].copy()

    return clean_df, dedup_log_df
