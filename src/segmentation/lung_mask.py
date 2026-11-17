"""Modul pascapemrosesan peta probabilitas paru menjadi mask biner M dan M224.

Mengacu pada Bab III (§3.2.2, L77-79):
- Ambang tetap 0.5 untuk binarisasi: M(i,j) in {0, 1}.
- Mempertahankan dua komponen paru terbesar (paru kiri dan kanan).
- Mengisi bagian berlubang (hole filling) via scipy.ndimage.binary_fill_holes.
- Interpolasi nearest-neighbor ke 224x224 (M224) tanpa menghasilkan nilai pecahan.
- Penghitungan rasio luas paru r_paru dan verifikasi selisih antarkelas.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import binary_fill_holes, label


def binarize_probability_map(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Ubah peta probabilitas kontinu menjadi matriks biner {0, 1} menggunakan ambang tetap 0.5."""
    return (prob_map >= threshold).astype(np.uint8)


def postprocess_lung_mask(
    binary_mask: np.ndarray,
    keep_components: int = 2,
    fill_holes: bool = True,
) -> np.ndarray:
    """Terapkan pembersihan morfologi: pertahankan 2 komponen terbesar & hole filling.

    Args:
        binary_mask: Array biner {0, 1} berdimensi (H, W).
        keep_components: Jumlah komponen terhubung terbesar yang dipertahankan (default: 2, paru kiri & kanan).
        fill_holes: Apakah mengisi rongga/lubang di dalam wilayah paru (default: True).

    Returns:
        np.ndarray: Mask biner bersih {0, 1} dengan dtype uint8.
    """
    assert binary_mask.ndim == 2, f"Mask harus 2 dimensi, ditemukan dimensi {binary_mask.ndim}"
    # Pastikan mask biner
    mask = (binary_mask > 0).astype(np.uint8)

    if mask.sum() == 0:
        return mask

    # 1. Label connected components
    labeled_array, num_features = label(mask)
    if num_features == 0:
        return mask

    if num_features > keep_components:
        # Hitung ukuran piksel setiap komponen (abaikan label 0/latar belakang)
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0  # latar belakang bukan komponen paru

        # Ambil indeks komponen dengan ukuran terbesar
        top_labels = np.argsort(component_sizes)[::-1][:keep_components]

        # Buat mask baru yang hanya memuat komponen terpilih
        cleaned_mask = np.isin(labeled_array, top_labels).astype(np.uint8)
    else:
        cleaned_mask = mask

    # 2. Hole filling (mengisi lubang di dalam paru)
    if fill_holes:
        cleaned_mask = binary_fill_holes(cleaned_mask).astype(np.uint8)

    assert set(np.unique(cleaned_mask)).issubset({0, 1}), "Mask hasil pascapemrosesan harus biner {0, 1}!"
    return cleaned_mask


def resize_mask_to_224(mask: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize mask biner ke 224x224 menggunakan interpolasi nearest-neighbor.

    Menjamin M224 in {0, 1} tanpa nilai pecahan pada batas kontur (Bab III L79).
    """
    assert set(np.unique(mask)).issubset({0, 1}), "Mask masukan harus biner {0, 1}!"
    try:
        import cv2
        m224 = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    except ImportError:
        from PIL import Image
        im = Image.fromarray(mask)
        m224 = np.array(im.resize(target_size, Image.Resampling.NEAREST), dtype=np.uint8)
    assert set(np.unique(m224)).issubset({0, 1}), "M224 hasil nearest-neighbor harus strictly biner {0, 1}!"
    return m224


def compute_r_paru(mask: np.ndarray) -> float:
    """Hitung rasio luas paru terhadap luas citra: r_paru = (1 / HW) * sum(M(i,j)).

    Mengacu pada Bab III (§3.2.2 L77 & §3.4.2 L251).
    """
    assert mask.ndim == 2, "Mask harus 2 dimensi"
    total_pixels = mask.shape[0] * mask.shape[1]
    lung_pixels = np.sum(mask > 0)
    return float(lung_pixels / total_pixels)


def check_extreme_failure(r_paru: float, threshold: float = 0.05) -> bool:
    """Periksa apakah terjadi kegagalan deteksi paru ekstrem (r_paru < 0.05, Bab III L73)."""
    return r_paru < threshold


def check_class_lung_ratio_diff(
    mean_r_normal: float,
    mean_r_pneumonia: float,
    threshold: float = 0.05,
) -> Tuple[bool, float]:
    """Periksa apakah selisih rata-rata luas paru kedua kelas melebihi 5 poin persentase (|delta_r| > 0.05).

    Jika True, evaluasi model wajib melaporkan ACR_norm = ACR / r_paru (Bab III L77).
    """
    diff = abs(mean_r_normal - mean_r_pneumonia)
    triggers_ac_norm = diff > threshold
    return triggers_ac_norm, float(diff)
