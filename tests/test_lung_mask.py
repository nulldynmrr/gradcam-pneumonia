"""Unit test untuk pascapemrosesan mask paru (binarisasi, 2 komponen terbesar, hole filling, r_paru)."""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.lung_mask import (
    binarize_probability_map,
    check_class_lung_ratio_diff,
    check_extreme_failure,
    compute_r_paru,
    postprocess_lung_mask,
    resize_mask_to_224,
)


def test_binarize_probability_map():
    prob = np.array([
        [0.1, 0.49, 0.50],
        [0.51, 0.99, 0.0],
    ], dtype=np.float32)

    bin_mask = binarize_probability_map(prob, threshold=0.5)
    expected = np.array([
        [0, 0, 1],
        [1, 1, 0],
    ], dtype=np.uint8)

    np.testing.assert_array_equal(bin_mask, expected)
    assert set(np.unique(bin_mask)).issubset({0, 1})


def test_postprocess_keeps_two_largest_components():
    # Buat kanvas dengan 4 komponen terpisah:
    # Komponen A (ukuran 100 px), Komponen B (ukuran 80 px),
    # Komponen C (ukuran 10 px), Komponen D (ukuran 5 px)
    canvas = np.zeros((50, 50), dtype=np.uint8)
    canvas[5:15, 5:15] = 1   # 10x10 = 100 px (A)
    canvas[25:35, 5:13] = 1  # 10x8 = 80 px (B)
    canvas[5:7, 35:40] = 1   # 2x5 = 10 px (C)
    canvas[40:41, 40:45] = 1 # 1x5 = 5 px (D)

    cleaned = postprocess_lung_mask(canvas, keep_components=2, fill_holes=False)

    # Hanya A dan B yang boleh tersisa (100 + 80 = 180 piksel)
    assert cleaned.sum() == 180
    assert cleaned[5:15, 5:15].sum() == 100
    assert cleaned[25:35, 5:13].sum() == 80
    assert cleaned[5:7, 35:40].sum() == 0   # C terhapus
    assert cleaned[40:41, 40:45].sum() == 0 # D terhapus


def test_postprocess_hole_filling():
    # Buat komponen persegi berlubang di tengahnya
    canvas = np.zeros((30, 30), dtype=np.uint8)
    canvas[5:25, 5:25] = 1
    canvas[10:20, 10:20] = 0  # lubang 10x10 = 100 piksel di tengah

    filled = postprocess_lung_mask(canvas, keep_components=2, fill_holes=True)

    # Lubang harus terisi penuh: 20x20 = 400 piksel
    assert filled[5:25, 5:25].sum() == 400
    assert filled[10:20, 10:20].sum() == 100


def test_resize_mask_nearest_strictly_binary():
    # Mask berukuran non-standar (misal 512x512) dengan kontur miring
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:400, 100:400] = 1

    m224 = resize_mask_to_224(mask, target_size=(224, 224))

    assert m224.shape == (224, 224)
    assert m224.dtype == np.uint8
    unique_vals = set(np.unique(m224))
    assert unique_vals.issubset({0, 1}), f"Nilai non-biner ditemukan: {unique_vals}"


def test_compute_r_paru_and_extreme_failure():
    # Mask dengan rasio 25% (1/4 area)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :50] = 1  # 2500 / 10000 = 0.25

    r = compute_r_paru(mask)
    assert pytest.approx(r, abs=1e-5) == 0.25
    assert not check_extreme_failure(r, threshold=0.05)

    # Mask sangat kecil (3% area) -> extreme failure
    mask_small = np.zeros((100, 100), dtype=np.uint8)
    mask_small[:30, :10] = 1  # 300 / 10000 = 0.03
    r_small = compute_r_paru(mask_small)
    assert pytest.approx(r_small, abs=1e-5) == 0.03
    assert check_extreme_failure(r_small, threshold=0.05)


def test_check_class_lung_ratio_diff():
    # Selisih 0.06 (> 0.05) -> triggers ACR_norm
    triggers, diff = check_class_lung_ratio_diff(0.28, 0.22, threshold=0.05)
    assert triggers is True
    assert pytest.approx(diff, abs=1e-5) == 0.06

    # Selisih 0.03 (<= 0.05) -> no trigger
    triggers_no, diff_no = check_class_lung_ratio_diff(0.25, 0.22, threshold=0.05)
    assert triggers_no is False
    assert pytest.approx(diff_no, abs=1e-5) == 0.03
