"""Modul arsitektur dan inferensi model U-Net pralatih beku (frozen).

Mengacu pada:
- Proposal Bab III (§3.2.2, L68): U-Net pralatih publik dengan bobot yang dikunci sepenuhnya.
- Sumber primer terverifikasi (docs/3.md G3(4)):
  * Zenodo Record 8393556 (DOI: 10.5281/zenodo.8393556, lisensi CC-BY-4.0), berkas `Unet.zip`
    (208.426.482 byte, MD5: 9ddfcf1d5cf1927dd095b7ca333c6350).
  * Repo pendamping: https://github.com/ViacheslavDanilov/covid_scoring (lisensi MIT),
    `inference.py` baris 95-103:
    - model: `smp.Unet`, encoder: `se_resnext101_32x4d`, in_channels=3, classes=1, activation='sigmoid'.
    - input_size: (384, 384), normalisasi: ImageNet mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225).
    - data latih: Darwin + Montgomery County + Shenzhen (di luar Kermany dan RSUA).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None  # type: ignore

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
UNET_INPUT_SIZE: Tuple[int, int] = (384, 384)


def get_unet_spec() -> Dict[str, Any]:
    """Kembalikan dokumentasi resmi format masukan dan sumber model U-Net (docs/3.md G3(4))."""
    return {
        "source_name": "Zenodo Record 8393556 — Deep Learning Models for Lung Segmentation in CXR",
        "doi": "10.5281/zenodo.8393556",
        "weights_file": "Unet.zip -> weights.pth",
        "zip_size_bytes": 208426482,
        "zip_md5": "9ddfcf1d5cf1927dd095b7ca333c6350",
        "weights_license": "CC-BY-4.0",
        "companion_repo": "https://github.com/ViacheslavDanilov/covid_scoring",
        "companion_repo_license": "MIT",
        "training_datasets": ["Darwin", "Montgomery County", "Shenzhen"],
        "architecture": "smp.Unet",
        "encoder_name": "se_resnext101_32x4d",
        "in_channels": 3,
        "classes": 1,
        "activation": "sigmoid",
        "input_size_hw": list(UNET_INPUT_SIZE),
        "input_color_space": "RGB (grayscale direplikasi 3 kanal)",
        "input_normalization": {
            "range_before_norm": [0.0, 1.0],
            "mean": IMAGENET_MEAN.tolist(),
            "std": IMAGENET_STD.tolist(),
        },
        "output_range": [0.0, 1.0],
        "frozen": True,
    }


def compute_file_sha256(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """Hitung checksum SHA-256 berkas bobot untuk audit integritas."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def create_unet_model() -> nn.Module:
    """Inisialisasi arsitektur smp.Unet sesuai spesifikasi Zenodo 8393556."""
    if smp is None:
        raise ImportError(
            "Pustaka 'segmentation_models_pytorch' belum terpasang! "
            "Jalankan: pip install segmentation-models-pytorch==0.6.0"
        )
    model = smp.Unet(
        encoder_name="se_resnext101_32x4d",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    return model


def load_frozen_unet(
    weights_path: Path | str,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Muat bobot model U-Net, kunci seluruh parameter (requires_grad=False), dan set eval().

    Args:
        weights_path: Jalur menuju berkas bobot `weights.pth`.
        device: Perangkat komputasi ('cuda', 'cpu', atau `torch.device`).

    Returns:
        nn.Module: Model U-Net siap inferensi dengan bobot beku.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Berkas bobot U-Net tidak ditemukan di: {weights_path}. "
            "Jalankan skrip: python scripts/download_unet_weights.py"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = create_unet_model()

    # Muat state_dict dengan weights_only=True terlebih dahulu (keamanan PyTorch >= 2.4)
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    except Exception as exc:
        logger.warning("torch.load(weights_only=True) gagal (%s); mencoba fallback weights_only=False.", exc)
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=True)

    # KUNCI SELURUH PARAMETER (Bab III L68: bobot yang dikunci sepenuhnya)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == 0, "[FATAL] Terdeteksi parameter U-Net yang tidak beku!"

    return model


def resize_bilinear(arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize array 2D menggunakan interpolasi bilinear (cv2 dengan fallback PIL)."""
    h_out, w_out = target_hw
    try:
        import cv2
        return cv2.resize(arr, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        im = Image.fromarray(arr)
        return np.array(im.resize((w_out, h_out), Image.Resampling.BILINEAR), dtype=arr.dtype)


def preprocess_image_for_unet(
    image: Image.Image | np.ndarray,
    target_size: Tuple[int, int] = UNET_INPUT_SIZE,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Siapkan citra masukan untuk U-Net (resize 384x384, 3-ch, normalisasi ImageNet).

    Returns:
        Tuple `(tensor_batch, original_hw)`:
        - `tensor_batch`: Tensor bentuk `(1, 3, 384, 384)` bertipe `float32`.
        - `original_hw`: Tuple `(tinggi_asli, lebar_asli)`.
    """
    if isinstance(image, Image.Image):
        orig_w, orig_h = image.size
        im_gray = image.convert("L")
        arr = np.array(im_gray, dtype=np.float32)
    else:
        orig_h, orig_w = image.shape[:2]
        if image.ndim == 3:
            arr = image.mean(axis=-1).astype(np.float32)
        else:
            arr = image.astype(np.float32)

    if arr.max() > 1.0:
        arr = arr / 255.0

    resized = resize_bilinear(arr, (target_size[1], target_size[0]))
    rgb = np.stack([resized, resized, resized], axis=-1)
    norm_rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(norm_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, (orig_h, orig_w)


def predict_lung_probability(
    model: nn.Module,
    image: Image.Image | np.ndarray,
    device: torch.device | str | None = None,
    return_original_size: bool = True,
) -> np.ndarray:
    """Jalankan inferensi model segmentasi paru beku (HybridGNet maupun U-Net) untuk menghasilkan peta mask/probabilitas.

    Args:
        model: Model segmentasi beku (`HybridGNet` dari CheXmask atau `smp.Unet` legacy).
        image: Citra rontgen CXR masukan (`PIL.Image` atau `np.ndarray`).
        device: Device komputasi.
        return_original_size: Jika True, kembalikan peta pada ukuran citra asli.

    Returns:
        np.ndarray: Peta probabilitas / mask biner kontinu `[0.0, 1.0]` bertipe `float32`.
    """
    if getattr(model, "is_hybridgnet", False):
        from src.segmentation.hybridgnet import predict_lung_mask_hybridgnet
        return predict_lung_mask_hybridgnet(model, image, device=device, return_float_prob=True)

    if device is None:
        device = next(model.parameters()).device

    tensor, (orig_h, orig_w) = preprocess_image_for_unet(image, UNET_INPUT_SIZE)
    tensor = tensor.to(device)

    with torch.no_grad():
        pred_prob = model(tensor)
        prob_arr = pred_prob.squeeze().cpu().numpy().astype(np.float32)

    prob_arr = np.clip(prob_arr, 0.0, 1.0)

    if return_original_size and (orig_h != UNET_INPUT_SIZE[1] or orig_w != UNET_INPUT_SIZE[0]):
        prob_orig = resize_bilinear(prob_arr, (orig_h, orig_w))
        return np.clip(prob_orig, 0.0, 1.0)

    return prob_arr
