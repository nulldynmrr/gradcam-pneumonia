"""Modul deduplikasi citra 2-tahap (exact SHA-256 + perceptual near-duplicate terverifikasi).

Aturan sesuai Proposal Bab III (§3.2.1):
- Deduplikasi dilakukan SEBELUM subsampling & pembagian data (split).
- Tahap 1 (Exact): Memeriksa SHA-256 byte berkas dan SHA-256 array piksel grayscale uint8 + dimensi.
- Tahap 2 (Near-Duplicate Terverifikasi lintas sumber):
  1. Penyaringan kandidat cepat dengan perceptual hash 64-bit (pHash berbasis 2D-DCT, jarak Hamming <= 5).
  2. Verifikasi sekunder (mencegah false-positive antar-pasien berbeda yang beranatomi CXR serupa):
     pasangan kandidat hanya diakui sebagai near-duplicate sejati bila juga memenuhi
     dHash Hamming <= max_dhash_hamming (default 5) DAN korelasi struktural Pearson r >= 0.985
     pada citra grayscale 64x64 (sejalan dengan prinsip ambang 0.98 Abdullah dkk., 2026).
- Jika ada duplikat sejati dengan label kelas bertentangan: STOP and RAISE.
- Menghasilkan log deduplikasi lengkap ke `data/interim/dedup_log.csv`.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from tqdm.auto import tqdm


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


def _phash64_from_pil(im_gray: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    img_size = hash_size * highfreq_factor
    resized = im_gray.resize((img_size, img_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32)
    dct_coeffs = dctn(pixels, type=2, norm="ortho")
    dct_low = dct_coeffs[:hash_size, :hash_size].ravel()
    med = float(np.median(dct_low[1:]))
    bits = (dct_low > med).astype(np.uint64)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def _dhash64_from_pil(im_gray: Image.Image, hash_size: int = 8) -> int:
    resized = im_gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    val = 0
    for b in diff.ravel():
        val = (val << 1) | int(bool(b))
    return val


def _unit_thumb_from_pil(im_gray: Image.Image, size: int = 64) -> np.ndarray:
    """Buat vektor thumbnail 64x64 ternormalisasi mean=0, norm=1 untuk korelasi Pearson cepat."""
    resized = im_gray.resize((size, size), Image.Resampling.BILINEAR)
    vec = np.asarray(resized, dtype=np.float32).ravel()
    vec = vec - float(vec.mean())
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    else:
        vec = np.zeros_like(vec, dtype=np.float32)
    return vec


def compute_phash64(file_path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Hitung perceptual hash 64-bit (pHash berbasis 2D-DCT) tanpa pustaka eksternal tambahan."""
    with Image.open(file_path) as im:
        return _phash64_from_pil(im.convert("L"), hash_size=hash_size, highfreq_factor=highfreq_factor)


def compute_dhash64(file_path: Path, hash_size: int = 8) -> int:
    """Hitung difference hash 64-bit (dHash berbasis gradien horizontal)."""
    with Image.open(file_path) as im:
        return _dhash64_from_pil(im.convert("L"), hash_size=hash_size)


def hamming_distance64(h1: int, h2: int) -> int:
    """Hitung jarak Hamming antara dua hash 64-bit."""
    return int(h1 ^ h2).bit_count()


def _find_perceptual_pairs(hashes: List[int], max_hamming: int = 5) -> List[Tuple[int, int, int]]:
    """Cari seluruh pasangan (i, j, dist) dengan jarak Hamming <= max_hamming secara tepat & cepat."""
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


def _hash_single_file(
    args: Tuple[Path, bool, bool, Optional[str], Optional[str]],
) -> Tuple[str, str, int, int, Optional[np.ndarray]]:
    """Baca satu citra SEKALI saja untuk menghitung byte hash, pixel hash, pHash, dHash, & vektor 64x64."""
    path, need_pixel, need_perceptual, existing_byte, existing_pixel = args
    b_hash = existing_byte if existing_byte else compute_byte_hash(path)

    if not need_pixel and not need_perceptual:
        return b_hash, existing_pixel or b_hash, 0, 0, None

    with Image.open(path) as im:
        im_gray = im.convert("L")
        if existing_pixel:
            p_hash = existing_pixel
        elif need_pixel:
            arr = np.ascontiguousarray(np.array(im_gray, dtype=np.uint8))
            h = hashlib.sha256()
            h.update(f"{arr.shape[0]}x{arr.shape[1]}:".encode("utf-8"))
            h.update(arr.tobytes())
            p_hash = h.hexdigest()
        else:
            p_hash = b_hash

        if need_perceptual:
            ph64 = _phash64_from_pil(im_gray)
            dh64 = _dhash64_from_pil(im_gray)
            thumb = _unit_thumb_from_pil(im_gray, size=64)
        else:
            ph64, dh64, thumb = 0, 0, None

    return b_hash, p_hash, ph64, dh64, thumb


def run_deduplication(
    manifest_df: pd.DataFrame,
    image_root: Path,
    check_pixel_hash: bool = True,
    check_perceptual_hash: bool = False,
    max_hamming: int = 5,
    max_dhash_hamming: int = 5,
    min_pearson_r: float = 0.985,
    num_workers: int = 8,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan prosedur deduplikasi 1 atau 2 tahap pada manifest citra.

    Args:
        manifest_df: DataFrame manifest (Kermany maupun gabungan Kermany + RSNA).
        image_root: Path direktori akar yang menjadi basis `rel_path`.
        check_pixel_hash: Jika True, hitung juga SHA-256 array piksel grayscale uint8.
        check_perceptual_hash: Jika True, jalankan Tahap 2 pHash + verifikasi dHash & korelasi Pearson.
        max_hamming: Ambang jarak Hamming pHash kandidat (default: 5).
        max_dhash_hamming: Ambang jarak Hamming dHash untuk verifikasi sekunder (default: 5).
        min_pearson_r: Ambang minimum korelasi Pearson 64x64 agar pasangan diakui sebagai near-duplicate
            sejati dan bukan kemiripan anatomi CXR biasa antar-pasien berbeda (default: 0.985).
        num_workers: Jumlah thread paralel untuk I/O pembacaan berkas (default: 8).
        show_progress: Tampilkan progress bar tqdm bila True.

    Returns:
        Tuple `(clean_df, dedup_log_df)`.
    """
    df = manifest_df.copy()
    df = df.sort_values(by=["rel_path"]).reset_index(drop=True)

    has_byte = "byte_hash" in df.columns and df["byte_hash"].notna().all() and (df["byte_hash"] != "").all()
    has_sha = "sha256" in df.columns and df["sha256"].notna().all() and (df["sha256"] != "").all()
    has_pixel = "pixel_hash" in df.columns and df["pixel_hash"].notna().all() and (df["pixel_hash"] != "").all()

    tasks = []
    for _, row in df.iterrows():
        ex_b = row["byte_hash"] if has_byte else (row["sha256"] if has_sha else None)
        ex_p = row["pixel_hash"] if has_pixel else None
        tasks.append((image_root / row["rel_path"], check_pixel_hash, check_perceptual_hash, ex_b, ex_p))

    if num_workers > 1 and len(tasks) > 16:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            it = pool.map(_hash_single_file, tasks)
            if show_progress:
                it = tqdm(it, total=len(tasks), desc="Hashing & Verifikasi Dedup")
            results = list(it)
    else:
        it = (_hash_single_file(t) for t in tasks)
        if show_progress:
            it = tqdm(it, total=len(tasks), desc="Hashing & Verifikasi Dedup")
        results = list(it)

    byte_hashes = [r[0] for r in results]
    pixel_hashes = [r[1] for r in results]
    phash_ints = [r[2] for r in results]
    dhash_ints = [r[3] for r in results]
    thumbs = [r[4] for r in results]

    df["byte_hash"] = byte_hashes
    df["pixel_hash"] = pixel_hashes

    # Union-Find untuk mengelompokkan exact & verified near-duplicate
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
        phash_hex_list = [f"{v:016x}" for v in phash_ints]
        cand_pairs = _find_perceptual_pairs(phash_ints, max_hamming=max_hamming)
        for ia, ib, ph_dist in cand_pairs:
            if find(ia) == find(ib):
                continue
            # Verifikasi sekunder: dHash Hamming + korelasi struktural Pearson 64x64
            dh_dist = hamming_distance64(dhash_ints[ia], dhash_ints[ib])
            if dh_dist > max_dhash_hamming:
                continue
            ta, tb = thumbs[ia], thumbs[ib]
            if ta is None or tb is None:
                continue
            pearson_r = float(np.dot(ta, tb))
            if pearson_r < min_pearson_r:
                continue
            union(ia, ib)
            if match_types[ib] == "unique":
                match_types[ib] = f"perceptual_ph{ph_dist}_dh{dh_dist}_r{pearson_r:.3f}"

    df["phash64"] = phash_hex_list
    df["dup_group_id"] = [find(i) for i in range(len(df))]
    df["match_type"] = match_types

    # Periksa konflik kelas dalam kelompok duplikat sejati (STOP keras bila ada)
    group_labels = df.groupby("dup_group_id")["label"].nunique()
    conflicts = group_labels[group_labels > 1]
    if not conflicts.empty:
        conflict_ids = conflicts.index.tolist()
        conflict_rows = df[df["dup_group_id"].isin(conflict_ids)]
        raise ValueError(
            f"Ditemukan kelompok duplikat dengan label kelas BERBEDA (konflik)! "
            f"Contoh konflik:\n{conflict_rows[['image_id', 'rel_path', 'label', 'dup_group_id', 'match_type']]}"
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
                reason_list.append(f"Verified near-duplicate ({mtype}) in group {gid}")
            else:
                reason_list.append(f"Identical duplicate ({mtype}) in group {gid}")

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
