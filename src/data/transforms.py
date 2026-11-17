"""Modul transformasi geometris berpasangan citra dan mask paru.

Mengacu pada Bab III (§3.2.2, L83):
- Transformasi geometris pada himpunan data latih (D_train):
  1. Rotasi acak: [-15°, +15°]
  2. Horizontal flip acak: p = 0.5
  3. Translasi acak: hingga 10% (sumbu X dan Y)
  4. Zoom acak: [0.9, 1.1] (skala isotropik)
- Nilai pengisi tepi (fill_value) = 0
- Interpolasi citra = bilinear; interpolasi mask = nearest-neighbor
- Parameter transformasi identik pada pasangan (I_224, M_224)
- Data validasi dan uji (D_val, D_test) TANPA augmentasi (hanya resize dan penyesuaian tipe).
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class PairwiseAffineTransform:
    """Transformasi affine acak identik untuk pasangan (citra, mask)."""

    def __init__(
        self,
        rotation_deg: float = 15.0,
        hflip_p: float = 0.5,
        translate_frac: float = 0.10,
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        fill_value: int = 0,
    ) -> None:
        self.rotation_deg = rotation_deg
        self.hflip_p = hflip_p
        self.translate_frac = translate_frac
        self.zoom_range = zoom_range
        self.fill_value = fill_value

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Terapkan augmentasi identik pada pasangan (image, mask).

        Args:
            image: Citra float32 atau uint8 (H, W) pada rentang [0, 1] atau [0, 255].
            mask: Mask biner uint8 (H, W) in {0, 1}.

        Returns:
            Tuple (aug_image, aug_mask) dengan ukuran (H, W) yang sama.
        """
        assert image.shape[:2] == mask.shape[:2], "Ukuran citra dan mask harus sama!"
        h, w = image.shape[:2]

        # 1. Horizontal Flip acak (L83: p = 0.5)
        if random.random() < self.hflip_p:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 2. Sampel parameter affine identik
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        scale = random.uniform(self.zoom_range[0], self.zoom_range[1])
        tx = int(random.uniform(-self.translate_frac, self.translate_frac) * w)
        ty = int(random.uniform(-self.translate_frac, self.translate_frac) * h)

        # Ubah ke tensor untuk transformasi presisi torchvision
        img_t = torch.from_numpy(image).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        # Terapkan transformasi geometris identik:
        # Citra: bilinear interpolation (L83)
        aug_img_t = TF.affine(
            img_t,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=self.fill_value,
        )

        # Mask: nearest-neighbor interpolation (L83: agar tetap biner {0, 1})
        aug_mask_t = TF.affine(
            mask_t,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=self.fill_value,
        )

        aug_image = aug_img_t.squeeze(0).numpy()
        aug_mask = aug_mask_t.squeeze(0).numpy()

        # Pastikan tipe dan binaritas mask tetap terjaga
        aug_mask = (aug_mask > 0).astype(np.uint8)
        assert set(np.unique(aug_mask)).issubset({0, 1}), "Augmentasi mask harus mempertahankan sifat biner {0, 1}!"

        return aug_image.astype(image.dtype), aug_mask
