"""Modul prapemrosesan dasar (grayscale [0,1], resize 224x224) & transformasi geometris berpasangan.

Mengacu pada Bab III (§3.2.2, L79-L83) dan SPEC_FROM_PROPOSAL (D5, P1, P5):
- D5: Konversi grayscale 1-channel, skala [0.0, 1.0] bertipe float32 sebelum normalisasi.
- P1: Resize citra CXR ke 224x224 (bilinear); resize lung mask ke 224x224 (nearest-neighbor).
- P5: Augmentasi HANYA pada himpunan data latih (D_train):
  1. Rotasi acak: [-15°, +15°]
  2. Horizontal flip acak: p = 0.5
  3. Translasi acak: hingga 10% (sumbu X dan Y)
  4. Zoom acak: [0.9, 1.1] (skala isotropik)
  - Nilai pengisi tepi (fill_value) = 0
  - Interpolasi citra = bilinear; interpolasi mask = nearest-neighbor
  - Parameter transformasi identik pada pasangan (I_224, M_224)
  - Data validasi dan uji (D_val, D_test) TANPA augmentasi.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


def load_gray01(image_or_path: Path | str | Image.Image | np.ndarray) -> np.ndarray:
    """Muat citra sebagai grayscale 1-channel float32 pada skala [0.0, 1.0] (D5)."""
    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as im:
            arr = np.array(im.convert("L"), dtype=np.float32) / 255.0
    elif isinstance(image_or_path, Image.Image):
        arr = np.array(image_or_path.convert("L"), dtype=np.float32) / 255.0
    else:
        arr = np.asarray(image_or_path, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1).astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    assert arr.ndim == 2, f"Citra grayscale wajib 2D, ditemukan shape {arr.shape}"
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0, "Intensitas citra harus di [0.0, 1.0]"
    return arr


def resize_image_to_224(
    image_gray01: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Resize citra grayscale [0,1] ke 224x224 menggunakan interpolasi bilinear (P1, AMBIG-4)."""
    assert image_gray01.ndim == 2, f"Citra masukan harus 2D, ditemukan {image_gray01.shape}"
    arr = image_gray01.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    h_out, w_out = target_size
    try:
        import cv2
        resized = cv2.resize(arr, (w_out, h_out), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    except ImportError:
        im = Image.fromarray(arr)
        resized = np.array(im.resize((w_out, h_out), Image.Resampling.BILINEAR), dtype=np.float32)

    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def resize_pair(
    image_gray01: np.ndarray,
    mask_bin: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize pasangan (I, M) ke 224x224: citra bilinear float32 [0,1], mask nearest uint8 {0,1}."""
    from src.segmentation.lung_mask import resize_mask_to_224

    i224 = resize_image_to_224(image_gray01, target_size=target_size)
    m224 = resize_mask_to_224(mask_bin, target_size=(target_size[1], target_size[0]))
    assert i224.shape == m224.shape == target_size, "Dimensi pasangan hasil resize tidak sama!"
    return i224, m224


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
        self.rotation_deg = float(rotation_deg)
        self.hflip_p = float(hflip_p)
        self.translate_frac = float(translate_frac)
        self.zoom_range = (float(zoom_range[0]), float(zoom_range[1]))
        self.fill_value = int(fill_value)

    @classmethod
    def from_config(cls, cfg: dict) -> "PairwiseAffineTransform":
        """Bangun transformasi langsung dari bagian `locked.augmentation` pada `config.yaml`."""
        aug = cfg["locked"]["augmentation"]
        return cls(
            rotation_deg=aug["rotation_deg"]["value"],
            hflip_p=aug["hflip_p"]["value"],
            translate_frac=aug["translate_frac"]["value"],
            zoom_range=tuple(aug["zoom_range"]["value"]),
            fill_value=aug["fill_value"]["value"],
        )

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

        img_t = torch.from_numpy(image).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

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

        aug_mask = (aug_mask > 0).astype(np.uint8)
        assert set(np.unique(aug_mask)).issubset({0, 1}), "Augmentasi mask harus mempertahankan sifat biner {0, 1}!"

        return aug_image.astype(image.dtype), aug_mask
