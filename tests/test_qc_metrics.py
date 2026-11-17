"""Unit test untuk metrik evaluasi Dice Similarity Coefficient (DSC) dan IoU."""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.qc import compute_dice_coefficient, compute_iou


def test_dice_and_iou_perfect_match():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 1

    dsc = compute_dice_coefficient(mask, mask)
    iou = compute_iou(mask, mask)

    assert pytest.approx(dsc, abs=1e-5) == 1.0
    assert pytest.approx(iou, abs=1e-5) == 1.0


def test_dice_and_iou_completely_disjoint():
    pred = np.zeros((50, 50), dtype=np.uint8)
    gt = np.zeros((50, 50), dtype=np.uint8)

    pred[0:10, 0:10] = 1
    gt[20:30, 20:30] = 1

    dsc = compute_dice_coefficient(pred, gt)
    iou = compute_iou(pred, gt)

    assert pytest.approx(dsc, abs=1e-5) == 0.0
    assert pytest.approx(iou, abs=1e-5) == 0.0


def test_dice_and_iou_partial_overlap():
    # Dua persegi 10x10 beririsan separuh (5x10)
    # Area A = 100, Area B = 100, Intersection = 50, Union = 150
    # DSC = 2*50 / (100 + 100) = 100 / 200 = 0.5
    # IoU = 50 / 150 = 1/3 ~ 0.333333
    pred = np.zeros((50, 50), dtype=np.uint8)
    gt = np.zeros((50, 50), dtype=np.uint8)

    pred[10:20, 10:20] = 1
    gt[10:20, 15:25] = 1

    dsc = compute_dice_coefficient(pred, gt)
    iou = compute_iou(pred, gt)

    assert pytest.approx(dsc, abs=1e-5) == 0.5
    assert pytest.approx(iou, abs=1e-5) == 1.0 / 3.0


def test_dice_and_iou_both_empty():
    pred = np.zeros((50, 50), dtype=np.uint8)
    gt = np.zeros((50, 50), dtype=np.uint8)

    dsc = compute_dice_coefficient(pred, gt)
    iou = compute_iou(pred, gt)

    assert pytest.approx(dsc, abs=1e-5) == 1.0
    assert pytest.approx(iou, abs=1e-5) == 1.0
