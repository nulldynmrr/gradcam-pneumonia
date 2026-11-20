"""Modul arsitektur dan inferensi model HybridGNet (CheXmask) pralatih beku (pure PyTorch).

Mengacu pada:
- Proposal Bab I (Batasan 3) & Bab III (§3.2.2): Model segmentasi paru pralatih HybridGNet
  (Gaggion dkk., 2024, Nature Scientific Data, DOI: 10.1038/s41597-024-03358-1;
  Gaggion dkk., 2022, IEEE TMI, DOI: 10.1109/TMI.2022.3224660) dengan bobot dibekukan sepenuhnya.
- Sumber bobot resmi:
  https://huggingface.co/spaces/ngaggion/Chest-x-ray-HybridGNet-Segmentation/resolve/main/weights/weights.pt
- Implementasi ChebConv & GraphPool ditulis ulang secara ekuivalen dalam PyTorch murni
  sehingga tidak memerlukan dependensi `torch_geometric` dan kompatibel lintas versi Python/PyTorch.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

logger = logging.getLogger(__name__)

HYBRIDGNET_WEIGHTS_URL = (
    "https://huggingface.co/spaces/ngaggion/Chest-x-ray-HybridGNet-Segmentation"
    "/resolve/main/weights/weights.pt"
)
HYBRIDGNET_INPUT_SIZE: int = 1024


def compute_file_sha256(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """Hitung checksum SHA-256 berkas bobot untuk audit integritas."""
    h = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def get_hybridgnet_spec() -> Dict[str, Any]:
    """Kembalikan dokumentasi resmi spesifikasi dan sumber bobot HybridGNet (CheXmask)."""
    return {
        "source_name": "CheXmask / HybridGNet — Landmark-Based Anatomical CXR Segmentation",
        "doi_chexmask": "10.1038/s41597-024-03358-1",
        "doi_hybridgnet": "10.1109/TMI.2022.3224660",
        "weights_url": HYBRIDGNET_WEIGHTS_URL,
        "weights_file": "weights.pt",
        "license": "CC-BY-4.0 / MIT (ngaggion/CheXmask-Database)",
        "training_datasets": ["JSRT", "Montgomery County", "Shenzhen", "PadChest"],
        "training_images_count": 911,
        "architecture": "HybridGNet (CNN Encoder + Chebyshev Graph Decoder + IGSC)",
        "landmarks": {"right_lung": 44, "left_lung": 50, "heart": 26},
        "input_size_hw": [HYBRIDGNET_INPUT_SIZE, HYBRIDGNET_INPUT_SIZE],
        "input_color_space": "Grayscale 1-channel [0.0, 1.0] padded square",
        "output_type": "Anatomical landmark polygons rasterized to binary lung mask {0, 1}",
        "frozen": True,
    }


def _m_organ(n: int) -> np.ndarray:
    sub = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sub[i, i - 1] = 1.0
        sub[i, (i + 1) % n] = 1.0
    return sub


def _m_organ_down(n: int) -> np.ndarray:
    n2 = int(np.ceil(n / 2))
    sub = np.zeros((n2, n), dtype=np.float32)
    for i in range(n2):
        if (2 * i + 1) == n:
            sub[i, 2 * i] = 1.0
        else:
            sub[i, 2 * i] = 0.5
            sub[i, 2 * i + 1] = 0.5
    return sub


def _m_organ_up(n: int) -> np.ndarray:
    n2 = int(np.ceil(n / 2))
    sub = np.zeros((n, n2), dtype=np.float32)
    for i in range(n):
        if i % 2 == 0:
            sub[i, i // 2] = 1.0
        else:
            sub[i, i // 2] = 0.5
            sub[i, (i // 2 + 1) % n2] = 0.5
    return sub


def gen_lungs_heart_matrices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bangun matriks adjacency, downsampling, dan upsampling graf paru kanan (44), paru kiri (50), jantung (26)."""
    rlung, llung, heart = 44, 50, 26
    p1, p2, p3 = rlung, rlung + llung, rlung + llung + heart
    p1_ = int(np.ceil(rlung / 2))
    p2_ = p1_ + int(np.ceil(llung / 2))
    p3_ = p2_ + int(np.ceil(heart / 2))

    A = np.zeros((p3, p3), dtype=np.float32)
    A[:p1, :p1] = _m_organ(rlung)
    A[p1:p2, p1:p2] = _m_organ(llung)
    A[p2:p3, p2:p3] = _m_organ(heart)

    AD = np.zeros((p3_, p3_), dtype=np.float32)
    AD[:p1_, :p1_] = _m_organ(int(np.ceil(rlung / 2)))
    AD[p1_:p2_, p1_:p2_] = _m_organ(int(np.ceil(llung / 2)))
    AD[p2_:p3_, p2_:p3_] = _m_organ(int(np.ceil(heart / 2)))

    D = np.zeros((p3_, p3), dtype=np.float32)
    D[:p1_, :p1] = _m_organ_down(rlung)
    D[p1_:p2_, p1:p2] = _m_organ_down(llung)
    D[p2_:p3_, p2:p3] = _m_organ_down(heart)

    U = np.zeros((p3, p3_), dtype=np.float32)
    U[:p1, :p1_] = _m_organ_up(rlung)
    U[p1:p2, p1_:p2_] = _m_organ_up(llung)
    U[p2:p3, p2_:p3_] = _m_organ_up(heart)

    return A, AD, D, U


def _scipy_to_torch_sparse(mat: sp.coo_matrix) -> torch.Tensor:
    indices = torch.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    values = torch.from_numpy(mat.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(mat.shape)).coalesce()


class ChebConv(nn.Module):
    """Implementasi PyTorch murni ekuivalen PyG ChebConv(normalization='sym') untuk bobot HybridGNet."""

    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in range(K)])
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _build_scaled_laplacian(
        edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
        adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
        mask = row != col
        adj[row[mask], col[mask]] = 1.0
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        eye = torch.eye(num_nodes, dtype=dtype, device=device)
        lap = eye - norm_adj
        lambda_max = 2.0 * lap.max()
        return (2.0 / lambda_max) * lap - eye

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(-2)
        Tx_0 = x
        out = self.lins[0](Tx_0)
        if self.K > 1:
            L_tilde = self._build_scaled_laplacian(edge_index, num_nodes, x.dtype, x.device)
            Tx_1 = torch.matmul(L_tilde, x)
            out = out + self.lins[1](Tx_1)
            for lin in self.lins[2:]:
                Tx_2 = 2.0 * torch.matmul(L_tilde, Tx_1) - Tx_0
                out = out + lin(Tx_2)
                Tx_0, Tx_1 = Tx_1, Tx_2
        if self.bias is not None:
            out = out + self.bias
        return out


class GraphPool(nn.Module):
    """Operasi upsampling graf berbasis matriks sparse HybridGNet."""

    def forward(self, x: torch.Tensor, pool_mat: torch.Tensor) -> torch.Tensor:
        mat = pool_mat.to_dense().to(dtype=x.dtype, device=x.device)
        return torch.matmul(mat, x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=False),
            )
        else:
            self.skip = None
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x) if self.skip is not None else x
        return F.relu(self.block(x) + identity)


class EncoderConv(nn.Module):
    def __init__(self, latents: int = 64, hw: int = 32):
        super().__init__()
        self.latents = latents
        self.size = 4 * np.array([2, 4, 8, 16, 32], dtype=np.int32)
        self.maxpool = nn.MaxPool2d(2)
        self.dconv_down1 = ResidualBlock(1, int(self.size[0]))
        self.dconv_down2 = ResidualBlock(int(self.size[0]), int(self.size[1]))
        self.dconv_down3 = ResidualBlock(int(self.size[1]), int(self.size[2]))
        self.dconv_down4 = ResidualBlock(int(self.size[2]), int(self.size[3]))
        self.dconv_down5 = ResidualBlock(int(self.size[3]), int(self.size[4]))
        self.dconv_down6 = ResidualBlock(int(self.size[4]), int(self.size[4]))
        self.fc_mu = nn.Linear(int(self.size[4]) * hw * hw, latents)
        self.fc_logvar = nn.Linear(int(self.size[4]) * hw * hw, latents)

    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.dconv_down1(x))
        x = self.maxpool(self.dconv_down2(x))
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        conv6 = self.dconv_down6(x)
        flat = conv6.view(conv6.size(0), -1)
        return self.fc_mu(flat), self.fc_logvar(flat), conv6, conv5


class SkipBlock(nn.Module):
    def __init__(self, in_filters: int, window: Tuple[int, int] = (3, 3)):
        super().__init__()
        self.window = window
        self.graphConv_pre = ChebConv(in_filters, 2, 1, bias=False)

    def lookup(self, pos: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        b, n = pos.shape[0], pos.shape[1]
        h = layer.shape[-1]
        pos_scaled = pos * h
        dx1 = float(self.window[0] // 2)
        dx2 = float(self.window[0] // 2 + 1)
        dy1 = float(self.window[1] // 2)
        dy2 = float(self.window[1] // 2 + 1)
        boxes = []
        for idx in range(b):
            x1 = pos_scaled[idx, :, 0:1] - dx1
            x2 = pos_scaled[idx, :, 0:1] + dx2
            y1 = pos_scaled[idx, :, 1:2] - dy1
            y2 = pos_scaled[idx, :, 1:2] + dy2
            boxes.append(torch.cat([x1, y1, x2, y2], dim=1))
        skip = roi_align(layer, boxes, output_size=(1, 1), aligned=True)
        return skip.view(b, n, -1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, conv_layer: torch.Tensor):
        pos = self.graphConv_pre(x, adj)
        skip = self.lookup(pos, conv_layer)
        return torch.cat((x, skip, pos), dim=2), pos


class HybridGNet(nn.Module):
    """Arsitektur HybridGNet (CheXmask) dengan penanda `is_hybridgnet = True`."""

    is_hybridgnet: bool = True

    def __init__(self, device: torch.device | str = "cpu"):
        super().__init__()
        A, AD, D, U = gen_lungs_heart_matrices()
        n1, n2 = A.shape[0], AD.shape[0]
        A_coo = sp.csc_matrix(A).tocoo()
        AD_coo = sp.csc_matrix(AD).tocoo()
        U_coo = sp.csc_matrix(U).tocoo()

        dev = torch.device(device)
        self.adjacency_matrices = [
            _scipy_to_torch_sparse(A_coo).to(dev),
            _scipy_to_torch_sparse(A_coo).to(dev),
            _scipy_to_torch_sparse(A_coo).to(dev),
            _scipy_to_torch_sparse(AD_coo).to(dev),
            _scipy_to_torch_sparse(AD_coo).to(dev),
            _scipy_to_torch_sparse(AD_coo).to(dev),
        ]
        self.upsample_matrices = [_scipy_to_torch_sparse(U_coo).to(dev)]

        self.z = 64
        self.encoder = EncoderConv(latents=self.z, hw=HYBRIDGNET_INPUT_SIZE // 32)
        f = 32
        self.filters = [2, f, f, f, f // 2, f // 2, f // 2]
        self.K = 6
        self.window = (3, 3)

        self.dec_lin = nn.Linear(self.z, self.filters[-1] * n2)
        self.normalization2u = nn.InstanceNorm1d(self.filters[1])
        self.normalization3u = nn.InstanceNorm1d(self.filters[2])
        self.normalization4u = nn.InstanceNorm1d(self.filters[3])
        self.normalization5u = nn.InstanceNorm1d(self.filters[4])
        self.normalization6u = nn.InstanceNorm1d(self.filters[5])

        outsize = int(self.encoder.size[4])
        self.graphConv_up6 = ChebConv(self.filters[6], self.filters[5], self.K)
        self.graphConv_up5 = ChebConv(self.filters[5], self.filters[4], self.K)
        self.SC_1 = SkipBlock(self.filters[4], self.window)
        self.graphConv_up4 = ChebConv(self.filters[4] + outsize + 2, self.filters[3], self.K)
        self.graphConv_up3 = ChebConv(self.filters[3], self.filters[2], self.K)
        self.SC_2 = SkipBlock(self.filters[2], self.window)
        self.graphConv_up2 = ChebConv(self.filters[2] + outsize + 2, self.filters[1], self.K)
        self.graphConv_up1 = ChebConv(self.filters[1], self.filters[0], 1, bias=False)
        self.pool = GraphPool()

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        dev = next(self.parameters()).device
        self.adjacency_matrices = [m.to(dev) for m in self.adjacency_matrices]
        self.upsample_matrices = [m.to(dev) for m in self.upsample_matrices]
        return res

    def forward(self, x: torch.Tensor):
        mu, _, conv6, conv5 = self.encoder(x)
        h = F.relu(self.dec_lin(mu)).reshape(x.shape[0], -1, self.filters[-1])
        h = F.relu(self.normalization6u(self.graphConv_up6(h, self.adjacency_matrices[5].indices())))
        h = F.relu(self.normalization5u(self.graphConv_up5(h, self.adjacency_matrices[4].indices())))
        h, pos1 = self.SC_1(h, self.adjacency_matrices[3].indices(), conv6)
        h = F.relu(self.normalization4u(self.graphConv_up4(h, self.adjacency_matrices[3].indices())))
        h = self.pool(h, self.upsample_matrices[0])
        h = F.relu(self.normalization3u(self.graphConv_up3(h, self.adjacency_matrices[2].indices())))
        h, pos2 = self.SC_2(h, self.adjacency_matrices[1].indices(), conv5)
        h = F.relu(self.normalization2u(self.graphConv_up2(h, self.adjacency_matrices[1].indices())))
        out = self.graphConv_up1(h, self.adjacency_matrices[0].indices())
        return out, pos1, pos2


def download_hybridgnet_weights(target_dir: Path, force_download: bool = False) -> Path:
    """Unduh bobot resmi HybridGNet (`weights.pt`) dari Hugging Face Space ngaggion."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    weights_path = target_dir / "weights.pt"
    if weights_path.exists() and not force_download:
        return weights_path
    logger.info("Mengunduh bobot HybridGNet (CheXmask) dari %s ...", HYBRIDGNET_WEIGHTS_URL)
    urllib.request.urlretrieve(HYBRIDGNET_WEIGHTS_URL, str(weights_path))
    return weights_path


def create_hybridgnet_model(device: torch.device | str = "cpu") -> HybridGNet:
    """Inisialisasi arsitektur HybridGNet (tanpa memuat bobot)."""
    return HybridGNet(device=device)


def load_frozen_hybridgnet(
    weights_path: Path | str,
    device: torch.device | str | None = None,
) -> HybridGNet:
    """Muat bobot HybridGNet (CheXmask), kunci 100% parameter (`requires_grad=False`), dan set `eval()`."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Berkas bobot HybridGNet tidak ditemukan di: {weights_path}. "
            "Jalankan `download_hybridgnet_weights(...)` terlebih dahulu."
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = HybridGNet(device=device)
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)

    model.load_state_dict(state_dict, strict=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == 0, "[FATAL] Terdeteksi parameter HybridGNet yang tidak beku!"
    return model


def predict_lung_mask_hybridgnet(
    model: HybridGNet,
    image: Image.Image | np.ndarray,
    device: torch.device | str | None = None,
    return_float_prob: bool = True,
) -> np.ndarray:
    """Prediksi lung mask biner (paru kanan + paru kiri) pada ukuran citra asli menggunakan HybridGNet."""
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    else:
        arr = image.mean(axis=-1).astype(np.float32) if image.ndim == 3 else image.astype(np.float32)

    if arr.max() > 1.0:
        arr = arr / 255.0

    h0, w0 = arr.shape[:2]
    if h0 > w0:
        padw = h0 - w0
        img_sq = np.pad(arr, ((0, 0), (padw // 2, padw // 2 + padw % 2)), mode="constant")
        padh = 0
    else:
        padh = w0 - h0
        img_sq = np.pad(arr, ((padh // 2, padh // 2 + padh % 2), (0, 0)), mode="constant")
        padw = 0

    hs = img_sq.shape[0]
    img_1024 = (
        cv2.resize(img_sq, (HYBRIDGNET_INPUT_SIZE, HYBRIDGNET_INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        if hs != HYBRIDGNET_INPUT_SIZE
        else img_sq
    )
    tensor = torch.from_numpy(img_1024).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = model(tensor)[0].cpu().numpy().reshape(-1, 2)

    out = out * float(hs)
    out[:, 0] -= padw // 2
    out[:, 1] -= padh // 2
    lm = out.astype(np.int32)
    right_lung = lm[0:44].reshape(-1, 1, 2)
    left_lung = lm[44:94].reshape(-1, 1, 2)

    mask = np.zeros((h0, w0), dtype=np.uint8)
    cv2.drawContours(mask, [right_lung], -1, 1, -1)
    cv2.drawContours(mask, [left_lung], -1, 1, -1)

    if return_float_prob:
        return mask.astype(np.float32)
    return mask
