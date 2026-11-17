"""Skrip pengunduhan dan verifikasi bobot U-Net pralatih publik (Zenodo 8393556).

Bobot dilatih pada gabungan dataset segmentasi paru Darwin, Montgomery County, dan Shenzhen:
- Arsitektur: smp.Unet (encoder: se_resnext101_32x4d)
- Input: 384x384 (3 channel, normalisasi ImageNet)
- Output: 1 channel (aktivasi sigmoid)
- DOI: 10.5281/zenodo.8393556
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

from src.paths import ensure_output_dirs, get_project_paths, load_config

ZENODO_UNET_ZIP_URL = "https://zenodo.org/api/records/8393556/files/Unet.zip/content"
EXPECTED_ZIP_MD5 = "9ddfcf1d5cf1927dd095b7ca333c6350"
EXPECTED_ZIP_SIZE = 208426482


def download_and_extract_unet_weights(
    target_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Unduh Unet.zip dari Zenodo dan ekstrak weights.pth ke target_dir."""
    if target_dir is None:
        cfg = load_config()
        paths = get_project_paths(cfg=cfg)
        target_dir = paths["unet_frozen"]

    target_dir.mkdir(parents=True, exist_ok=True)
    weights_path = target_dir / "weights.pth"
    config_path = target_dir / "config.json"

    if weights_path.exists() and config_path.exists() and not force_download:
        print(f"[OK] Bobot U-Net sudah tersedia di: {weights_path}")
        return weights_path

    zip_path = target_dir / "Unet.zip"
    print(f"Mengunduh bobot U-Net dari Zenodo ({ZENODO_UNET_ZIP_URL})...")
    print(f"Ukuran berkas: ~{EXPECTED_ZIP_SIZE / (1024**2):.1f} MB")

    # Unduh dengan pelacak progres
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        mb_downloaded = (count * block_size) / (1024 * 1024)
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        print(f"\rProgres: {percent}% ({mb_downloaded:.1f}/{total_mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(ZENODO_UNET_ZIP_URL, zip_path, reporthook=progress_hook)
    print("\nUnduhan selesai! Memeriksa integritas MD5...")

    # Cek MD5
    md5_hash = hashlib.md5()
    with open(zip_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5_hash.update(chunk)
    actual_md5 = md5_hash.hexdigest()

    if actual_md5 != EXPECTED_ZIP_MD5:
        print(f"[PERINGATAN] MD5 tidak cocok! Diharapkan: {EXPECTED_ZIP_MD5}, Ditemukan: {actual_md5}")
    else:
        print(f"[OK] Integritas MD5 terverifikasi: {actual_md5}")

    print("Mengekstrak berkas ke direktori model...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Ekstrak file di dalam folder Unet/ ke target_dir langsung
            filename = Path(member.filename).name
            if filename:
                target_file = target_dir / filename
                with zf.open(member) as source, open(target_file, "wb") as dest:
                    dest.write(source.read())

    # Hapus zip sementara untuk menghemat ruang
    if zip_path.exists():
        zip_path.unlink()

    assert weights_path.exists(), f"weights.pth tidak ditemukan di {target_dir}!"
    print(f"[OK] Bobot U-Net berhasil dipasang: {weights_path}")
    return weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unduh bobot U-Net Zenodo 8393556")
    parser.add_argument("--force", action="store_true", help="Paksa unduh ulang meskipun berkas sudah ada")
    args = parser.parse_args()

    download_and_extract_unet_weights(force_download=args.force)
