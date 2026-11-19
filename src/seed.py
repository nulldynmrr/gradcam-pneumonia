"""Modul pengaturan determinisme (reproducibility) dan snapshot lingkungan komputasi.

Mengacu pada:
- Proposal Bab III (§3.4 L133 & Tabel 3.2): pengaturan deterministik lintas 5 random seed.
- AMBIG-9: `torch.use_deterministic_algorithms(True, warn_only=True)` + pencatatan versi paket.
- `docs/1.md` baris 108: sel pertama tiap notebook mencetak versi library + GPU dan set seed.
"""

from __future__ import annotations

import importlib
import os
import platform
import random
import sys
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """Tetapkan seluruh RNG (Python, NumPy, PyTorch CPU/CUDA) secara deterministik.

    Args:
        seed: Bilangan bulat seed acak.
        deterministic: Jika True, aktifkan mode deterministik cuDNN & PyTorch (warn_only=True).

    Returns:
        Dict informasi seed dan status determinisme yang terpasang.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }


def _safe_pkg_version(pkg_name: str) -> str:
    """Ambil versi paket secara aman tanpa crash jika paket opsional belum terpasang."""
    try:
        mod = importlib.import_module(pkg_name)
        return str(getattr(mod, "__version__", "installed"))
    except ImportError:
        return "not-installed"


def get_env_snapshot(seed: int | None = None) -> Dict[str, Any]:
    """Kumpulkan informasi versi Python, pustaka utama, dan perangkat keras (CPU/GPU)."""
    packages = [
        "torch",
        "torchvision",
        "timm",
        "segmentation_models_pytorch",
        "cv2",
        "PIL",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "yaml",
        "tqdm",
    ]
    versions = {pkg: _safe_pkg_version(pkg) for pkg in packages}

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device_name": device_name,
        "seed": seed,
        "packages": versions,
    }


def print_env_banner(snapshot: Dict[str, Any]) -> None:
    """Cetak ringkasan lingkungan komputasi secara rapi pada sel pertama notebook."""
    print("=== RINGKASAN LINGKUNGAN EKSEKUSI ===")
    print(f"Timestamp UTC : {snapshot['timestamp_utc']}")
    print(f"Python        : {snapshot['python']} ({snapshot['platform']})")
    print(f"Device        : {snapshot['device_name']} (CUDA={snapshot['cuda_available']})")
    print(f"Seed Aktif    : {snapshot['seed']}")
    pkgs = snapshot["packages"]
    print(
        f"Versi Utama   : torch={pkgs['torch']} | torchvision={pkgs['torchvision']} | "
        f"timm={pkgs['timm']} | smp={pkgs['segmentation_models_pytorch']}"
    )
    print(
        f"Versi Data    : numpy={pkgs['numpy']} | pandas={pkgs['pandas']} | "
        f"cv2={pkgs['cv2']} | scipy={pkgs['scipy']} | matplotlib={pkgs['matplotlib']}"
    )
