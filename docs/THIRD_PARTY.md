# THIRD_PARTY — Sumber eksternal, lisensi, dan status verifikasi

Aturan: dilarang mengarang link/bobot/angka/sitasi; setiap sumber diverifikasi lewat sumber primer; menyalin kode memerlukan pencatatan sumber + lisensi di sini. Repo referensi = pembanding read-only.

| Sumber | Peran | Digunakan sebagai | Lisensi | Status verifikasi |
|---|---|---|---|---|
| `timm` (huggingface/pytorch-image-models) | arsitektur EfficientNet-B0 + bobot ImageNet | dependensi (di-pin) | Apache-2.0 *(diverifikasi saat instal: cek metadata paket)* | Belum diperiksa — G4 |
| `torch`, `torchvision` | framework | dependensi (bawaan Colab; lokal CPU) | BSD-style *(verifikasi saat instal)* | Belum diperiksa — G4 |
| scikit-learn, SciPy, pandas, pyarrow, OpenCV (headless), Matplotlib, PyYAML, pytest, tqdm | analisis/uji | dependensi (di-pin) | — | Dicatat saat `pip freeze` — G1 |
| Kermany pediatric CXR, rilis **Mendeley** `rscbjbr9sj` (ID di nama folder lokal `rscbjbr9sj-3`) | data utama | data (read-only) | **Belum diverifikasi** (cek lisensi pada halaman Mendeley) | Struktur nyata dicatat di `DATA_CONTRACT.md` (G1) |
| RSUA Chest X-Ray dataset (Wahyuningrum dkk., Data in Brief 2023; Mendeley `10.17632/2jg8vfdmpm.1` per docs/3.md) | QC lung mask | data (read-only) | **Belum diverifikasi** | G1 |
| Bobot U-Net / model segmentasi paru (kandidat: Zenodo, repo `ViacheslavDanilov/covid_scoring`) | lung mask beku | bobot (read-only) | **Belum diverifikasi** | G3 — **wajib** diverifikasi: DOI/URL, versi, lisensi, arsitektur nyata, data latih, format masukan, ukuran + checksum |
| Dataset gabungan Mendeley `8gf9vpkhgy` (kandidat docs/3.md) | alternatif sumber segmentasi | – | **Belum diverifikasi** | G3 |
| `jacobgil/pytorch-grad-cam` (pip `grad-cam`) | **hanya** cross-check Grad-CAM (G4); bukan hasil akhir | verifikasi | MIT *(verifikasi saat instal)* | G4 |
| Selvaraju dkk. 2017 (Grad-CAM) | rumus Pers. 2.5–2.6 | implementasi sendiri dari makalah | – | tidak menyalin kode |

Catatan: tidak ada kode pihak ketiga yang disalin ke repo hingga saat ini.
