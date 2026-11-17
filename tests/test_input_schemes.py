"""Unit test untuk 3 skema konfigurasi masukan (baseline_3ch, hardmask_3ch, fusion_4ch) dan augmentasi berpasangan."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.input_schemes import (
    build_baseline_3ch,
    build_fusion_4ch,
    build_hardmask_3ch,
    get_input_tensor,
)
from src.data.transforms import PairwiseAffineTransform


def test_baseline_3ch_shape_and_normalization():
    img = np.full((224, 224), 128, dtype=np.uint8)
    t = build_baseline_3ch(img)

    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 224, 224)
    assert t.dtype == torch.float32

    # Verifikasi rumus ImageNet: (X - mean) / std per channel
    val = 128.0 / 255.0
    expected_ch0 = (val - 0.485) / 0.229
    expected_ch1 = (val - 0.456) / 0.224
    expected_ch2 = (val - 0.406) / 0.225

    assert pytest.approx(t[0, 0, 0].item(), abs=1e-4) == expected_ch0
    assert pytest.approx(t[1, 0, 0].item(), abs=1e-4) == expected_ch1
    assert pytest.approx(t[2, 0, 0].item(), abs=1e-4) == expected_ch2


def test_hardmask_3ch_outside_lung_is_blackened_before_normalization():
    img = np.full((224, 224), 128, dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[50:150, 50:150] = 1  # hanya wilayah ini yang paru

    t = build_hardmask_3ch(img, mask)

    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 224, 224)
    assert t.dtype == torch.float32

    # Di luar paru (misal koordinat 0, 0), piksel mula-mula adalah 0
    # Setelah normalisasi ImageNet: (0 - mean) / std
    mean_r = 0.485
    std_r = 0.229
    expected_outside_val_ch0 = (0.0 - mean_r) / std_r
    assert pytest.approx(t[0, 0, 0].item(), abs=1e-3) == expected_outside_val_ch0


def test_fusion_4ch_channel4_is_strictly_binary():
    img = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[30:180, 40:180] = 1

    t = build_fusion_4ch(img, mask)

    assert isinstance(t, torch.Tensor)
    assert t.shape == (4, 224, 224)

    # Channel ke-4 adalah mask biner murni [0, 1] (TIDAK ternormalisasi ImageNet)
    ch4 = t[3].numpy()
    unique_ch4 = set(np.unique(ch4))
    assert unique_ch4.issubset({0.0, 1.0}), f"Channel ke-4 harus biner {{0.0, 1.0}}, ditemukan: {unique_ch4}"
    # Piksel di dalam paru bernilai 1.0, di luar bernilai 0.0
    assert ch4[100, 100] == 1.0
    assert ch4[0, 0] == 0.0


def test_get_input_tensor_dispatcher():
    img = np.full((224, 224), 0.5, dtype=np.float32)
    mask = np.ones((224, 224), dtype=np.uint8)

    t1 = get_input_tensor("baseline_3ch", img)
    assert t1.shape == (3, 224, 224)

    t2 = get_input_tensor("hardmask_3ch", img, mask)
    assert t2.shape == (3, 224, 224)

    t3 = get_input_tensor("fusion_4ch", img, mask)
    assert t3.shape == (4, 224, 224)

    with pytest.raises(ValueError):
        get_input_tensor("unknown_scheme", img)


def test_pairwise_transform_preserves_mask_binarity():
    img = np.random.uniform(0.0, 1.0, (224, 224)).astype(np.float32)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[50:170, 50:170] = 1

    transform = PairwiseAffineTransform(
        rotation_deg=15.0,
        hflip_p=0.5,
        translate_frac=0.10,
        zoom_range=(0.9, 1.1),
    )

    for _ in range(5):
        aug_img, aug_mask = transform(img, mask)
        assert aug_img.shape == (224, 224)
        assert aug_mask.shape == (224, 224)
        assert set(np.unique(aug_mask)).issubset({0, 1}), "Mask setelah augmentasi harus tetap biner {0, 1}!"
