"""Modul pembentukan 3 skema konfigurasi masukan EfficientNet-B0.

Mengacu pada Bab III (§3.2.2 L81, §3.3 L111-128, dan Gambar 3.1):
1. baseline_3ch: Citra CXR 3-channel (X_RGB in R^{3x224x224}), normalisasi ImageNet.
2. hardmask_3ch: Citra CXR dikalikan lung mask sebelum normalisasi (I_224 dot M_224),
   digandakan ke 3 channel (X_masked in R^{3x224x224}), normalisasi ImageNet.
3. fusion_4ch: Tiga channel X_RGB digabungkan dengan 1 channel M_224 in {0,1}
   (X_fusion in R^{4x224x224}). Channel ke-4 tetap biner dan TIDAK dinormalisasi ImageNet.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def normalize_imagenet(tensor_3ch: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Terapkan normalisasi ImageNet pada tensor 3-channel: (X - mean) / std."""
    if isinstance(tensor_3ch, np.ndarray):
        t = torch.from_numpy(tensor_3ch).float()
    else:
        t = tensor_3ch.float()

    if t.ndim == 2:
        # Replikasi ke 3 channel jika masih 2D (H, W)
        t = torch.stack([t, t, t], dim=0)
    elif t.shape[0] != 3 and t.shape[-1] == 3:
        # Konversi (H, W, 3) -> (3, H, W)
        t = t.permute(2, 0, 1)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (t - mean) / std


def build_baseline_3ch(image_224: np.ndarray) -> torch.Tensor:
    """Konfigurasi 1: Baseline 3-channel standar (X_RGB).

    Args:
        image_224: Citra grayscale 224x224 pada skala [0.0, 1.0].

    Returns:
        torch.Tensor: Tensor (3, 224, 224) ternormalisasi ImageNet.
    """
    assert image_224.shape == (224, 224), f"Citra harus 224x224, ditemukan {image_224.shape}"
    arr = image_224.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    t3 = np.stack([arr, arr, arr], axis=0)  # (3, 224, 224)
    return normalize_imagenet(t3)


def build_hardmask_3ch(image_224: np.ndarray, mask_224: np.ndarray) -> torch.Tensor:
    """Konfigurasi 2: Hard masking 3-channel (X_masked).

    Wilayah di luar paru dihitamkan (I_224 * M_224) SEBELUM normalisasi ImageNet (Bab III L81, L111).

    Args:
        image_224: Citra grayscale 224x224 pada skala [0.0, 1.0].
        mask_224: Mask biner paru 224x224 in {0, 1}.

    Returns:
        torch.Tensor: Tensor (3, 224, 224) ternormalisasi ImageNet.
    """
    assert image_224.shape == (224, 224), f"Citra harus 224x224, ditemukan {image_224.shape}"
    assert mask_224.shape == (224, 224), f"Mask harus 224x224, ditemukan {mask_224.shape}"
    assert set(np.unique(mask_224)).issubset({0, 1}), "Mask harus strictly biner {0, 1}!"

    arr = image_224.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Hadamard product (perkalian elemen per piksel)
    masked_arr = arr * mask_224.astype(np.float32)

    t3 = np.stack([masked_arr, masked_arr, masked_arr], axis=0)
    return normalize_imagenet(t3)


def build_fusion_4ch(image_224: np.ndarray, mask_224: np.ndarray) -> torch.Tensor:
    """Konfigurasi 3: Usulan fusi 4-channel berbasis segmentasi paru (X_fusion).

    Menumpuk 3-channel X_RGB dengan 1-channel M_224 in {0, 1} (Persamaan 3.1 Bab III).
    Channel ke-4 tetap biner pada skala [0, 1] dan TIDAK dinormalisasi ImageNet (Bab III L81, L130).

    Args:
        image_224: Citra grayscale 224x224 pada skala [0.0, 1.0].
        mask_224: Mask biner paru 224x224 in {0, 1}.

    Returns:
        torch.Tensor: Tensor (4, 224, 224).
    """
    assert image_224.shape == (224, 224), f"Citra harus 224x224, ditemukan {image_224.shape}"
    assert mask_224.shape == (224, 224), f"Mask harus 224x224, ditemukan {mask_224.shape}"
    assert set(np.unique(mask_224)).issubset({0, 1}), "Mask harus strictly biner {0, 1}!"

    # 3-channel pertama adalah X_RGB standar ternormalisasi ImageNet
    x_rgb = build_baseline_3ch(image_224)  # (3, 224, 224)

    # Channel ke-4 adalah mask biner murni [0, 1]
    m_ch = torch.from_numpy(mask_224.astype(np.float32)).unsqueeze(0)  # (1, 224, 224)

    # Gabungkan menjadi tensor 4-channel: [X_RGB || M_224]
    x_fusion = torch.cat([x_rgb, m_ch], dim=0)  # (4, 224, 224)
    return x_fusion


def get_input_tensor(
    scheme: str,
    image_224: np.ndarray,
    mask_224: np.ndarray | None = None,
) -> torch.Tensor:
    """Fungsi pembungkus (dispatcher) untuk menghasilkan tensor masukan sesuai skema."""
    scheme_lower = scheme.lower()
    if scheme_lower in ["baseline_3ch", "baseline", "3ch"]:
        return build_baseline_3ch(image_224)
    elif scheme_lower in ["hardmask_3ch", "hardmask", "masked_3ch"]:
        if mask_224 is None:
            raise ValueError("Skema hardmask_3ch memerlukan argumen mask_224!")
        return build_hardmask_3ch(image_224, mask_224)
    elif scheme_lower in ["fusion_4ch", "fusion", "4ch"]:
        if mask_224 is None:
            raise ValueError("Skema fusion_4ch memerlukan argumen mask_224!")
        return build_fusion_4ch(image_224, mask_224)
    else:
        raise ValueError(
            f"Skema '{scheme}' tidak dikenal! Pilih: 'baseline_3ch', 'hardmask_3ch', atau 'fusion_4ch'."
        )
