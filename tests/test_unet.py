"""Unit test untuk inisialisasi arsitektur U-Net dan prapemrosesan inferensi."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.segmentation.unet import (
    create_unet_model,
    preprocess_image_for_unet,
)


def test_create_unet_architecture_parameters():
    model = create_unet_model()
    total_params = sum(p.numel() for p in model.parameters())
    # Sesuai config.json Zenodo 8393556: num_params ~ 55.9e6
    assert total_params == 55919489, f"Parameter U-Net ({total_params}) tidak sesuai spesifikasi Zenodo!"


def test_preprocess_image_for_unet():
    img = np.full((500, 700), 100, dtype=np.uint8)
    tensor, orig_hw = preprocess_image_for_unet(img, target_size=(384, 384))

    assert orig_hw == (500, 700)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 384, 384)
    assert tensor.dtype == torch.float32

    # Nilai setelah normalisasi ImageNet
    val = 100.0 / 255.0
    expected_ch0 = (val - 0.485) / 0.229
    assert pytest.approx(tensor[0, 0, 0, 0].item(), abs=1e-3) == expected_ch0
