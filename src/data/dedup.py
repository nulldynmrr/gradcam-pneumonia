"""Modul deduplikasi citra 2-tahap (exact SHA-256 + perceptual hash pHash/dHash lintas sumber).

Aturan sesuai Proposal Bab III (§3.2.1):
- Deduplikasi dilakukan SEBELUM subsampling & pembagian data (split).
- Tahap 1: Memeriksa SHA-256 byte berkas dan SHA-256 representasi piksel grayscale uint8 + dimensi.
- Tahap 2 (opsional / multi-sumber): Memeriksa kemiripan visual via perceptual hash 64-bit
  (pHash berbasis 2D-DCT) dengan ambang jarak Hamming <= 5 (Abdullah dkk., 2026).
- Jika ada duplikat atau near-duplicate dengan label kelas bertentangan: STOP and RAISE.
- Menghasilkan log deduplikasi lengkap ke `data/interim/dedup_log.csv`.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn


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


def compute_phash64(file_path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Hitung perceptual hash 64-bit (pHash berbasis 2D-DCT) tanpa pustaka eksternal tambahan."""
    img_size = hash_size * highfreq_factor
    with Image.open(file_path) as im:
        gray = im.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float32)

    dct_coeffs = dctn(pixels, type=2, norm="ortho")
    dct_low = dct_coeffs[:hash_size, :hash_size].ravel()
    # Abaikan koefisien DC (indeks 0) saat menghitung median agar tahan terhadap pergeseran kecerahan global
    med = float(np.median(dct_low[1:]))
    bits = (dct_low > med).astype(np.uint64)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def compute_dhash64(file_path: Path, hash_size: int = 8) -> int:
    """Hitung difference hash 64-bit (dHash berbasis gradien horizontal)."""
    with Image.open(file_path) as im:
        gray = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    val = 0
    for b in diff.ravel():
        val = (val << 1) | int(bool(b))
    return val


def hamming_distance64(h1: int, h2: int) -> int:
    """Hitung jarak Hamming antara dua hash 64-bit."""
    return int(h1 ^ h2).bit_count()


def _find_perceptual_pairs(hashes: List[int], max_hamming: int = 5) -> List[Tuple[int, int, int]]:
    """Cari seluruh pasangan (i, j, dist) dengan jarak Hamming <= max_hamming secara tepat & cepat.

    Menggunakan prinsip sarang merpati (multi-index hashing):
    64 bit dipecah menjadi `n_bands = max_hamming + 1` pita bit. Dua hash dengan jarak <= max_hamming
    DIJAMIN identik minimal pada 1 pita sehingga pencarian eksak berjalan dalam waktu hampir linier.
    """
    n = len(hashes)
    if n < 2 or max_hamming < 0:
        return []

    n_bands = max_hamming + 1
    band_bits = 64 // n_bands
    band_tables: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(n_bands)]

    for idx, h in enumerate(hashes):
        for b in range(n_bands):
            shift = b * band_bits
            width = band_bits if b < n_bands - 1 else (64 - shift)
            mask = (1 << width) - 1
            key = (h >> shift) & mask
            band_tables[b][key].append(idx)

    seen_pairs: Set[Tuple[int, int]] = set()
    matches: List[Tuple[int, int, int]] = []

    for table in band_tables:
        for bucket in table.values():
            if len(bucket) < 2:
                continue
            m = len(bucket)
            for a in range(m):
                ia = bucket[a]
                ha = hashes[ia]
                for b in range(a + 1, m):
                    ib = bucket[b]
                    pair = (ia, ib) if ia < ib else (ib, ia)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    dist = hamming_distance64(ha, hashes[ib])
                    if dist <= max_hamming:
                        matches.append((pair[0], pair[1], dist))

    return matches


def run_deduplication(
    manifest_df: pd.DataFrame,
    image_root: Path,
    check_pixel_hash: bool = True,
    check_perceptual_hash: bool = False,
    max_hamming: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan prosedur deduplikasi 1 atau 2 tahap pada manifest citra.

    Args:
        manifest_df: DataFrame manifest (Kermany maupun gabungan Kermany + RSNA).
        image_root: Path direktori akar yang menjadi basis `rel_path`.
        check_pixel_hash: Jika True, hitung juga SHA-256 array piksel grayscale uint8.
        check_perceptual_hash: Jika True, jalankan Tahap 2 pHash 64-bit lintas sumber.
        max_hamming: Ambang jarak Hamming maksimum untuk near-duplicate pHash (default: 5).

    Returns:
        Tuple `(clean_df, dedup_log_df)`:
        - `clean_df`: Manifest bersih (KEPT) tanpa duplikat identik maupun near-duplicate.
        - `dedup_log_df`: Log lengkap semua citra (KEPT/DROPPED, match_type, dup_group_id, alasan).
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
        byte_hashes = [compute_byte_hash(image_root / row["rel_path"]) for _, row in df.iterrows()]

    if "pixel_hash" in df.columns and df["pixel_hash"].notna().all() and (df["pixel_hash"] != "").all():
        pixel_hashes = df["pixel_hash"].tolist()
    elif check_pixel_hash:
        pixel_cache: Dict[str, str] = {}
        pixel_hashes = []
        for b_hash, (_, row) in zip(byte_hashes, df.iterrows()):
            if b_hash in pixel_cache:
                pixel_hashes.append(pixel_cache[b_hash])
            else:
                p_hash, _ = compute_pixel_hash(image_root / row["rel_path"])
                pixel_cache[b_hash] = p_hash
                pixel_hashes.append(p_hash)
    else:
        pixel_hashes = list(byte_hashes)

    df["byte_hash"] = byte_hashes
    df["pixel_hash"] = pixel_hashes

    # Union-Find untuk mengelompokkan exact & near-duplicate
    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    byte_map: Dict[str, int] = {}
    pixel_map: Dict[str, int] = {}
    match_types: List[str] = ["unique"] * len(df)

    for idx, row in df.iterrows():
        bh = row["byte_hash"]
        ph = row["pixel_hash"]
        if bh in byte_map:
            union(idx, byte_map[bh])
            match_types[idx] = "exact_byte"
        else:
            byte_map[bh] = idx

        if ph in pixel_map:
            union(idx, pixel_map[ph])
            if match_types[idx] == "unique":
                match_types[idx] = "exact_pixel"
        else:
            pixel_map[ph] = idx

    phash_hex_list: List[str] = [""] * len(df)
    if check_perceptual_hash:
        phash_cache: Dict[str, int] = {}
        phash_ints: List[int] = []
        for b_hash, (_, row) in zip(byte_hashes, df.iterrows()):
            if b_hash in phash_cache:
                val = phash_cache[b_hash]
            else:
                val = compute_phash64(image_root / row["rel_path"])
                phash_cache[b_hash] = val
            phash_ints.append(val)
        phash_hex_list = [f"{v:016x}" for v in phash_ints]

        near_pairs = _find_perceptual_pairs(phash_ints, max_hamming=max_hamming)
        for ia, ib, dist in near_pairs:
            if find(ia) != find(ib):
                union(ia, ib)
                if match_types[ib] == "unique":
                    match_types[ib] = f"perceptual_hamming_{dist}"

    df["phash64"] = phash_hex_list
    df["dup_group_id"] = [find(i) for i in range(len(df))]

    # Periksa konflik kelas dalam kelompok duplikat (STOP keras bila ada)
    group_labels = df.groupby("dup_group_id")["label"].nunique()
    conflicts = group_labels[group_labels > 1]
    if not conflicts.empty:
        conflict_ids = conflicts.index.tolist()
        conflict_rows = df[df["dup_group_id"].isin(conflict_ids)]
        raise ValueError(
            f"Ditemukan kelompok duplikat dengan label kelas BERBEDA (konflik)! "
            f"Contoh konflik:\n{conflict_rows[['image_id', 'rel_path', 'label', 'dup_group_id']]}"
        )

    group_sizes = df["dup_group_id"].value_counts().to_dict()
    status_list: List[str] = []
    reason_list: List[str] = []
    seen_groups: Set[int] = set()

    for idx, row in df.iterrows():
        gid = row["dup_group_id"]
        gsize = group_sizes[gid]
        if gid not in seen_groups:
            seen_groups.add(gid)
            status_list.append("KEPT")
            if gsize > 1:
                reason_list.append("Primary instance in duplicate group")
            else:
                reason_list.append("Unique image")
        else:
            status_list.append("DROPPED")
            mtype = match_types[idx]
            if mtype.startswith("perceptual"):
                reason_list.append(f"Perceptual near-duplicate ({mtype}) in group {gid}")
            else:
                reason_list.append(f"Identical duplicate ({mtype}) in group {gid}")

    df["match_type"] = match_types
    df["status"] = status_list
    df["reason"] = reason_list

    drop_meta_cols = ["byte_hash", "pixel_hash", "phash64", "dup_group_id", "match_type", "status", "reason"]
    clean_df = df[df["status"] == "KEPT"].drop(columns=drop_meta_cols).reset_index(drop=True)

    log_cols = [
        c
        for c in [
            "image_id",
            "source",
            "rel_path",
            "label",
            "patient_id",
            "group_id",
            "byte_hash",
            "pixel_hash",
            "phash64",
            "dup_group_id",
            "match_type",
            "status",
            "reason",
        ]
        if c in df.columns
    ]
    dedup_log_df = df[log_cols].copy()

    return clean_df, dedup_log_df
