"""Modul arsitektur dan inferensi model U-Net pralatih beku (frozen).

Mengacu pada:
- Proposal Bab III (§3.2.2, L68): U-Net pralatih publik dengan bobot yang dikunci sepenuhnya.
- Zenodo Record 8393556: smp.Unet dengan encoder se_resnext101_32x4d.
- Format Masukan: Citra CXR 384x384, 3-channel, normalisasi ImageNet.
- Format Keluaran: Peta probabilitas kontinu [0.0, 1.0].
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None  # type: ignore

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
UNET_INPUT_SIZE = (384, 384)


def create_unet_model() -> nn.Module:
    """Inisialisasi arsitektur smp.Unet sesuai spesifikasi Zenodo 8393556."""
    if smp is None:
        raise ImportError(
            "Pustaka 'segmentation_models_pytorch' belum terpasang! "
            "Jalankan: pip install segmentation-models-pytorch"
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
        weights_path: Jalur menuju berkas bobot weights.pth.
        device: Perangkat komputasi ('cuda', 'cpu', atau torch.device).

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

    # Muat state dict
    state_dict = torch.load(weights_path, map_location=device)
    # Tangani jika tersimpan dalam struktur dict checkpoint
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)

    # KUNCI SELURUH PARAMETER (Bab III L68: bobot yang dikunci sepenuhnya)
    for name, param in model.named_parameters():
        param.requires_grad = False

    model.eval()
    model.to(device)

    # Verifikasi penguncian
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
        Tuple (tensor_batch, original_hw):
        - tensor_batch: Tensor bentuk (1, 3, 384, 384) bertipe float32.
        - original_hw: Tuple (tinggi_asli, lebar_asli).
    """
    if isinstance(image, Image.Image):
        orig_w, orig_h = image.size
        # Konversi ke L (grayscale) terlebih dahulu
        im_gray = image.convert("L")
        arr = np.array(im_gray, dtype=np.float32)
    else:
        orig_h, orig_w = image.shape[:2]
        if image.ndim == 3:
            # Grayscale mean
            arr = image.mean(axis=-1).astype(np.float32)
        else:
            arr = image.astype(np.float32)

    # Normalisasi intensitas ke skala [0.0, 1.0]
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Resize citra ke ukuran masukan U-Net (384x384) menggunakan bilinear
    resized = resize_bilinear(arr, (target_size[1], target_size[0]))

    # Gandakan menjadi 3-channel (R, G, B identik)
    rgb = np.stack([resized, resized, resized], axis=-1)  # (384, 384, 3)

    # Normalisasi statistik ImageNet: (X - mean) / std
    norm_rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

    # Ubah menjadi format PyTorch tensor (1, 3, H, W)
    tensor = torch.from_numpy(norm_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, (orig_h, orig_w)


def predict_lung_probability(
    model: nn.Module,
    image: Image.Image | np.ndarray,
    device: torch.device | str | None = None,
    return_original_size: bool = True,
) -> np.ndarray:
    """Jalankan inferensi U-Net untuk memprediksi peta probabilitas batas paru.

    Args:
        model: Model U-Net beku hasil `load_frozen_unet`.
        image: Citra rontgen CXR masukan (PIL Image atau np.ndarray).
        device: Device komputasi.
        return_original_size: Jika True, kembalikan probabilitas pada resolusi asli citra (bilinear).

    Returns:
        np.ndarray: Peta probabilitas kontinu [0.0, 1.0] dengan dtype float32.
    """
    if device is None:
        device = next(model.parameters()).device

    tensor, (orig_h, orig_w) = preprocess_image_for_unet(image, UNET_INPUT_SIZE)
    tensor = tensor.to(device)

    with torch.no_grad():
        pred_prob = model(tensor)  # model sudah memiliki aktivasi sigmoid di keluaran
        prob_arr = pred_prob.squeeze().cpu().numpy().astype(np.float32)  # (384, 384)

    # Pastikan nilai berada di rentang [0.0, 1.0]
    prob_arr = np.clip(prob_arr, 0.0, 1.0)

    if return_original_size and (orig_h != UNET_INPUT_SIZE[1] or orig_w != UNET_INPUT_SIZE[0]):
        # Resize probabilitas kembali ke ukuran citra asli menggunakan bilinear (AMBIG-1 b)
        prob_orig = resize_bilinear(prob_arr, (orig_h, orig_w))
        return np.clip(prob_orig, 0.0, 1.0)

    return prob_arr
