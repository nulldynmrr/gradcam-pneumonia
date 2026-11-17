# gradcam-pneumonia

Implementasi proposal TA: *Evaluasi Lokalisasi Grad-CAM pada Klasifikasi Pneumonia Menggunakan EfficientNet-B0 dengan Fusi Multi-Channel Berbasis Segmentasi Paru* (Dinar Muhammad Akbar, 103012300333).

**Status: Fase 0 selesai (dokumen spesifikasi + skeleton). Belum ada kode pipeline — dikerjakan per gerbang (G1–G5), tiap gerbang menunggu persetujuan.**

- Sumber kebenaran: proposal (`.tex`). Lihat `docs/SPEC_FROM_PROPOSAL.md`, `docs/AMBIGUITIES.md`, `docs/DEVIATIONS.md`.
- Nilai terkunci: `configs/config.yaml` (`locked:`). Alur Gambar 3.1: `configs/pipeline_stages.yaml`.
- Pengujian lokal memakai data **sintetis** (bukan bukti hasil pada data asli).
- Data asli + run penuh: Google Colab Pro + Drive (`LAB_TA_GradCAM_Pneumonia`).

Setup lokal (setelah Gerbang 1 dimulai):
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-local.txt -r requirements-extra.txt
```
