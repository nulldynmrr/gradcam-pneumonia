# DEVIATIONS — Status Rekonsiliasi Deviasi dari Proposal

Aturan baku: **Tidak ada deviasi diam-diam**. Seluruh ketidaksesuaian awal antara draf naskah awal dan data/bobot riil telah diaudit, disetujui, dan **SELESAI DIREKONSILIASI** langsung di dalam naskah proposal resmi (`Metodologi.tex` dan `docs/main.pdf`).

---

## Tabel Pelacakan Deviasi

| ID | Status Akhir | Bagian Proposal | Isi Draf Awal | Realitas Empiris & Kode | Resolusi Akademik di Naskah Proposal |
|---|:---:|---|---|---|---|
| **DEV-1** | ✅ **GUGUR / SELESAI DIREKONSILIASI** | Metodologi L26 (hlm 26) | Menyebut pola penamaan Kaggle `person<nomor>` (Pneumonia) dan `IM-<nomor>` (Normal). | Dataset riil di Google Drive adalah rilis resmi Mendeley Data `rscbjbr9sj-3` dengan format: `(NORMAL\|BACTERIA\|VIRUS)-<pid>-<seq>.jpeg`. Pola Kaggle 0% cocok. | Naskah `Metodologi.tex` L26 telah diperbarui secara resmi memasukkan pola regex Mendeley Data. Seluruh 5.856 citra berhasil dipetakan ke 3.081 pasien unik tanpa tumpang tindih. |
| **DEV-2** | ✅ **GUGUR / SELESAI DIREKONSILIASI** | Metodologi L68 (hlm 27) | Menyebut bobot U-Net pralatih publik dilatih murni pada *Montgomery County + Shenzhen*. | Bobot resmi publik terverifikasi (Zenodo Record 8393556) dilatih pada gabungan dataset *Darwin + Montgomery County + Shenzhen* menggunakan arsitektur U-Net dengan encoder `se_resnext101_32x4d`. | Naskah `Metodologi.tex` L68 telah disempurnakan secara resmi menegaskan gabungan dataset publik independen *Darwin, Montgomery County, dan Shenzhen*. |
| **DEV-3** | ✅ **SELESAI DIREKONSILIASI** | Pendahuluan, Kajian Pustaka (Tabel 2.2), Metodologi (§3.2.1, Tabel 3.1, §3.2.2) | Dataset awal hanya menggunakan Kermany Pediatric CXR (5.827 citra pascadeduplikasi, 1 RS). | Perluasan scope atas masukan pembimbing: menggabungkan Kermany (anak, 1:1 = 3.164 citra) + RSNA Pneumonia Detection Challenge (dewasa, 1:1 = 12.024 citra; 11.821 kelas ambigu dikeluarkan) → total 15.188 citra dengan harmonisasi PNG 8-bit lossless (`MONOCHROME1` dikoreksi), dedup 2-tahap (SHA-256 + `pHash`), QC keseimbangan paru kiri-kanan ($r_{\text{LR}}$), serta pelaporan per sumber. | Seluruh naskah `.tex`, `References.bib`, `main.pdf`, `configs/config.yaml`, modul `src/data/`, `src/segmentation/`, dan notebook `00`, `01`, `02`, `02b` telah disinkronkan 100%. Arsitektur, loss BCE murni, optimizer, dan 3 skema fusi tetap tidak berubah. |

---

## Catatan Integritas Penelitian
* **NOL DEVIASI AKTIF:** Saat ini tidak ada satupun deviasi yang belum terselesaikan.
* **Kepatuhan Metodologi:** Seluruh batasan masalah tetap dipegang teguh (tidak ada LR scheduler, tidak ada class weight, tidak ada cropping, tidak ada penalti CAM pada fungsi loss, dan 5 random seed independen pada partisi data terkunci).
