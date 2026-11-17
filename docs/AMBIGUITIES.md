# AMBIGUITIES — Bagian proposal yang ambigu/kurang spesifik

Format: **ID** · bagian proposal · opsi · **default yang diimplementasikan** · status.
Default dipilih paling masuk akal dan **tidak diam-diam**; Anda dapat membantahnya kapan saja. Perubahan default → ubah `config.yaml::assumed` dan baris ini.

| ID | Bagian proposal | Ambiguitas / opsi | Default | Status |
|---|---|---|---|---|
| AMBIG-1 | Metodologi L77 + L79 (hlm 32) | Urutan resize vs threshold: (a) threshold pada resolusi keluaran U-Net lalu resize; (b) resize probabilitas ke ukuran citra asli lalu threshold | (b): probabilitas di-resize (bilinear) ke ukuran asli → thr 0,5 → 2 komponen terbesar → hole filling → `M`; `M224` = nearest dari `M` | Default terpasang |
| AMBIG-2 | L24 (hlm 30) | "Citra identik": hash apa? | sha256 byte berkas **dan** sha256 array piksel grayscale (+ shape); satu per kelompok dipertahankan (urut path); duplikat lintas kelas → berhenti & lapor | Default terpasang |
| AMBIG-3 | L28–36 | Algoritma stratified group split & seed tidak ditentukan; proporsi 70/15/15 tidak bisa persis pada level pasien | greedy deterministik: grup (pasien) diurut menurun menurut ukuran (tie: shuffle bersemi), tiap grup ke partisi dengan defisit relatif terbesar untuk kelasnya; `SPLIT_SEED=42` (terpisah dari seed training); hasil dikunci + hash | Default terpasang |
| AMBIG-4 | L79 | Kalimat resize citra tidak menyebut "bilinear" (hanya L83) | bilinear | Default terpasang |
| AMBIG-5 | L130; Tabel 3.2 | "conv head": urutan timm `conv_head → bn2 → act2`; di mana hook dipasang? | hook di **keluaran `conv_head`** (1280×7×7, pra-BN, pra-SiLU); layer configurable (`gradcam_target_layer`); perbandingan layer lain hanya di `exploratory/` berlabel "DI LUAR PROPOSAL" | Default terpasang |
| AMBIG-6 | Pers. 2.6 | Heatmap ReLU seluruhnya nol → min-max membagi nol | `H=0` (seragam nol), ACR=0 lewat ε, jumlah kasus dilaporkan | Default terpasang |
| AMBIG-7 | Pers. 2.8 | "30% piksel aktivasi tertinggi" ambigu saat banyak nilai sama (plato ReLU) | ambang = kuantil ke-70 dari H; hanya piksel dengan aktivasi > 0 yang bisa terpilih | Default terpasang |
| AMBIG-8 | L253 + docs/1 S8a | Wilcoxon tidak bisa dipakai pada AUC/F1 agregat | CI bootstrap persentil + p-value bootstrap berpasangan dua-sisi untuk AUC/F1; Wilcoxon signed-rank dua-sisi untuk ACR/IoU-Paru | Default terpasang |
| AMBIG-9 | L133 | "Deterministik" vs op CUDA non-deterministik (mis. backward `adaptive_avg_pool2d`) | `torch.use_deterministic_algorithms(True, warn_only=True)`; peringatan dicatat di `env_snapshot.json`; 5 seed menangkap sisa variasi | Default terpasang |
| AMBIG-10 | docs/1 S8b,c | F1 dari prob rata-rata; pasien banyak citra | prob citra → rata-rata per pasien → rata-rata 5 seed; threshold 0,5; label pasien tunggal (assert) | Default terpasang |
| AMBIG-11 | L253, L257–258 | Anggota keluarga Holm per hipotesis | H1: p-value (AUC, F1) vs hard masking; H2: p-value (ACR, IoU-Paru) vs baseline; non-inferioritas vs baseline hanya lewat CI | Default terpasang |
| AMBIG-12 | L77, L251 | ACR_norm tidak masuk keluarga H2 | dilaporkan deskriptif + CI bootstrap, tanpa Holm, tidak memengaruhi verdict | Default terpasang |
| AMBIG-13 | L133, Pers. 3.3 | BCE pada σ(z) vs implementasi stabil | `BCEWithLogitsLoss` (setara matematis, numerik stabil) | Default terpasang |
| AMBIG-14 | L73, L77, L251 | `r_paru` dihitung di `M` asli atau `M224` | `M224` (definisi L251); versi `M` asli dicatat sebagai kolom tambahan | Default terpasang |
| AMBIG-15 | docs/2 aturan 3 vs docs/3 daftar notebook | Notebook `00`, `00b`, `01a`, `01b`, `08` tidak ada di Gambar 3.1 | `infrastructure_notebooks` di `pipeline_stages.yaml` (tidak dihitung tahap) | Default terpasang |
| AMBIG-16 | L253 | Resample bootstrap yang tidak memuat salah satu kelas (AUC tak terdefinisi) | resample dilewati & jumlahnya dicatat | Default terpasang |
| AMBIG-17 | L79, L83 | Cache `I224` hasil resize: kuantisasi uint8 akan menambah galat di luar proposal | simpan float32 | Default terpasang |
| AMBIG-18 | Tabel 3.2 | Early stopping `min_delta` tidak disebut; tie-break epoch terbaik | `min_delta=0` (perbaikan strict `<`); epoch terbaik = pertama yang mencapai val loss minimum | Default terpasang |
| AMBIG-19 | L81 | Nilai mean/std ImageNet tidak ditulis angka | mean (0,485; 0,456; 0,406), std (0,229; 0,224; 0,225) (statistik standar torchvision/timm) | Default terpasang |
| AMBIG-20 | L130, Tabel 3.2 | Inisialisasi head Linear/Dropout dan inisialisasi layer baru tak disebut | default PyTorch `nn.Linear` dengan seed run; stem 4-ch hanya Pers. 3.2 | Default terpasang |
| AMBIG-21 | L83 | Translasi "hingga 10%": per sumbu (±10% lebar/tinggi) atau total; zoom seragam per sumbu | per sumbu ±10% × ukuran; zoom satu skala isotropik ∈ [0,9; 1,1]; rotasi seragam ∈ [−15°, 15°]; rotasi → zoom → translasi dalam satu affine; pusat gambar | Default terpasang |
| AMBIG-22 | L71 | Pemetaan kelas RSUA ke label proposal | `Non_Covid_Pneumonia` = Pneumonia (53), `Non_Covid` = Normal (32), `Covid` (207) dikeluarkan; hanya folder `Validated` (292) | Default terpasang; diverifikasi nyata di G1 |
| AMBIG-23 | L73 | Perbandingan QC pada resolusi apa | RSUA: prediksi di-resize nearest ke 256×256 (ukuran GT); Kermany: pada ukuran citra asli (kontur manual dibuat pada citra asli) | Default terpasang |
| AMBIG-24 | L14 vs L73 | Teks Bab III L14 menyatakan QC "telah" lulus (DSC ≥ 0,85, IoU ≥ 0,75) sebelum QC nyata dijalankan | Kode tidak mengasumsikan lulus; hasil nyata menentukan. Jika FAIL, kalimat proposal perlu revisi oleh penulis | **Perlu tindakan penulis jika FAIL** |
| AMBIG-25 | docs/1 vs docs/3 | `docs/1.md` mengatur Fase 0 dan struktur notebook; `docs/3.md` menambah `00b`, `01a`, `01b` dan "hanya per gerbang" | Satu garis waktu: Fase 0 → G1 → G2 → G3 → G4 → G5; modul murni-matematika (metrik, stats, Grad-CAM, stem 4-ch) dibangun di G3–G4 (bukan lebih awal), sesuai "HANYA per gerbang" | Default terpasang |
| AMBIG-26 | docs/1 | `register_hook` pada keluaran `conv_head`: gradien yang diambil = `∂y^c/∂A` pada tensor keluaran konv (pra-BN) | sesuai AMBIG-5 | Default terpasang |
| AMBIG-27 | Gambar 3.1 vs proposal | Gambar menampilkan tahap "Prapemrosesan" sebelum "Split"; teks Bab III menyatakan dedup → split → U-Net; docs/2 menyatakan urutan eksekusi nyata sengaja berbeda | ikuti urutan eksekusi docs/2 (dedup → split → U-Net → QC → prapemrosesan); tidak "diperbaiki" | Default terpasang |
| AMBIG-28 | `docs/main.pdf` vs `.tex` | Salinan `docs/main.pdf` dan `WORKSPACE/.../main.pdf` lebih lama dari `.tex` (beda kalimat §3.1) | `.tex` terbaru dipakai sebagai sumber kebenaran | Default terpasang |
| AMBIG-29 | `master_prd_audit_tugas_akhir_dinar (1).md` | Dokumen audit lama; redaksi RM1/H2 dan "n=879 citra" berbeda dari `.tex` | diabaikan (dikonfirmasi lewat persetujuan rencana, default Q3) | Default terpasang |

## Ketidaksesuaian data nyata ↔ proposal (lihat juga DEVIATIONS.md)

- **Rilis dataset**: Mendeley (nama `NORMAL-<id>-<n>`, `BACTERIA-…`, `VIRUS-…`) vs pola Kaggle yang tertulis di proposal → **DEV-1**.
- **`train/NORMAL` di Drive tidak lengkap** (1.090 vs 1.349 pada salinan lokal) → tindakan penulis, Gerbang 1 FAIL sampai diperbaiki; bukan deviasi desain.
- **RSUA** ada dua versi (`Annotated` 371, `Validated` 292) → hanya `Validated` (AMBIG-22).
