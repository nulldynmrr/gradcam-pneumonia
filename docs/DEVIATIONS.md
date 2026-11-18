# DEVIATIONS — Status Rekonsiliasi Deviasi dari Proposal

Aturan baku: **Tidak ada deviasi diam-diam**. Seluruh ketidaksesuaian awal antara draf naskah awal dan data/bobot riil telah diaudit, disetujui, dan **SELESAI DIREKONSILIASI** langsung di dalam naskah proposal resmi (`Metodologi.tex` dan `docs/main.pdf`).

---

## Tabel Pelacakan Deviasi

| ID | Status Akhir | Bagian Proposal | Isi Draf Awal | Realitas Empiris & Kode | Resolusi Akademik di Naskah Proposal |
|---|:---:|---|---|---|---|
| **DEV-1** | ✅ **GUGUR / SELESAI DIREKONSILIASI** | Metodologi L26 (hlm 26) | Menyebut pola penamaan Kaggle `person<nomor>` (Pneumonia) dan `IM-<nomor>` (Normal). | Dataset riil di Google Drive adalah rilis resmi Mendeley Data `rscbjbr9sj-3` dengan format: `(NORMAL\|BACTERIA\|VIRUS)-<pid>-<seq>.jpeg`. Pola Kaggle 0% cocok. | Naskah `Metodologi.tex` L26 telah diperbarui secara resmi memasukkan pola regex Mendeley Data. Seluruh 5.856 citra berhasil dipetakan ke 3.081 pasien unik tanpa tumpang tindih. |
| **DEV-2** | ✅ **GUGUR / SELESAI DIREKONSILIASI** | Metodologi L68 (hlm 27) | Menyebut bobot U-Net pralatih publik dilatih murni pada *Montgomery County + Shenzhen*. | Bobot resmi publik terverifikasi (Zenodo Record 8393556) dilatih pada gabungan dataset *Darwin + Montgomery County + Shenzhen* menggunakan arsitektur U-Net dengan encoder `se_resnext101_32x4d`. | Naskah `Metodologi.tex` L68 telah disempurnakan secara resmi menegaskan gabungan dataset publik independen *Darwin, Montgomery County, dan Shenzhen*. |

---

## Catatan Integritas Penelitian
* **NOL DEVIASI AKTIF:** Saat ini tidak ada satupun deviasi yang belum terselesaikan.
* **Kepatuhan Metodologi:** Seluruh batasan masalah tetap dipegang teguh (tidak ada LR scheduler, tidak ada class weight, tidak ada cropping, tidak ada penalti CAM pada fungsi loss, dan 5 random seed independen pada partisi data terkunci).
