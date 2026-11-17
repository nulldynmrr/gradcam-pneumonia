# DATA_CONTRACT — Spesifikasi Struktur, Format, dan Karakteristik Nyata Dataset

Dokumen ini disusun pada **Gerbang 1 (Data Contract)** setelah verifikasi langsung pada struktur Google Drive (`LAB_TA_GradCAM_Pneumonia`) dan salinan dataset lokal lengkap. Dokumen ini menjadi **acuan tunggal** bagi implementasi loader, parser manifest, dan deduplikasi di `src/data/`.

---

## 1. Status Verifikasi Proposal (PASS / FAIL)

| Komponen Uji | Target Proposal (Tabel 3.1) | Nilai Nyata Terverifikasi | Status | Catatan |
|---|---|---|---|---|
| **Kermany Normal** | 1.583 citra | 1.583 citra | **PASS** | `train/NORMAL`: 1.349, `test/NORMAL`: 234 |
| **Kermany Pneumonia** | 4.273 citra | 4.273 citra | **PASS** | `train/PNEUMONIA`: 3.883, `test/PNEUMONIA`: 390 |
| **Total Kermany** | 5.856 citra | 5.856 citra | **PASS** | 100% konsisten dengan Tabel 3.1 proposal |
| **Kermany Test Bawaan** | 624 citra | 624 citra | **PASS** | 234 Normal + 390 Pneumonia |
| **RSUA Validated Normal** | 32 citra | 32 citra | **PASS** | Folder `Non_Covid`, 32 pasang citra & mask |
| **RSUA Validated Pneumonia**| 53 citra | 53 citra | **PASS** | Folder `Non_Covid_Pneumonia`, 53 pasang |
| **RSUA Validated Covid** | 207 citra (dikeluarkan) | 207 citra | **PASS** | Dikeluarkan dari evaluasi QC |
| **Total RSUA Evaluasi QC** | 85 citra | 85 citra | **PASS** | 32 Normal + 53 Pneumonia |
| **Pola Regex Proposal** | Kaggle (`person*`, `IM-*`) | 0 berkas cocok | **DEV-1** | Pola Mendeley terverifikasi 100% |

---

## 2. Dataset Utama: Kermany Pediatric CXR

### 2.1 Lokasi dan Struktur Direktori
- **Path relatif di `PROJECT_ROOT`:** `data/raw/kermany/chest_xray/`
- **Subdirektori:**
  ```
  data/raw/kermany/chest_xray/
  ├── test/
  │   ├── NORMAL/      (234 berkas)
  │   └── PNEUMONIA/   (390 berkas)
  └── train/
      ├── NORMAL/      (1.349 berkas)
      └── PNEUMONIA/   (3.883 berkas)
  ```
- **Catatan Partisi Bawaan:** Tidak terdapat folder `val/` bawaan pada rilis Mendeley. Sesuai proposal (§3.2.1, Bab III L24), seluruh 5.856 citra dari `train/` dan `test/` (termasuk 624 citra uji bawaan) digabungkan terlebih dahulu sebelum dilakukan pembagian ulang (*split* ulang).

### 2.2 Pola Penamaan Berkas dan Ekstraksi ID Pasien (DEV-1)
Seluruh 5.856 berkas mengikuti konvensi penamaan rilis Mendeley `rscbjbr9sj`:
$$\textbf{(disease)-(randomized patient ID)-(image number)}$$

- **Regular Expression Standar:**
  ```regex
  ^(?P<cls>NORMAL|BACTERIA|VIRUS)-(?P<pid>\d+)-(?P<seq>\d+)\.jpeg$
  ```
- **Aturan Pemetaan:**
  - `cls == 'NORMAL'` $\rightarrow$ Label Kelas `Normal` (nilai target numerik: `0`)
  - `cls in ('BACTERIA', 'VIRUS')` $\rightarrow$ Label Kelas `Pneumonia` (nilai target numerik: `1`)
  - `pid` $\rightarrow$ Identitas Pasien (*patient ID*)
  - `seq` $\rightarrow$ Nomor urut citra untuk pasien tersebut
- **Verifikasi Ketercakupan (*Coverage*):**
  - Cocok pola Mendeley: **5.856 / 5.856 (100,00%)**
  - Berkas tak cocok (*unmatched*): **0 berkas**
- **Verifikasi Irisan Pasien Lintas Kelas (*Cross-class Patient Leakage*):**
  - Jumlah pasien unik kelas Normal: **1.237 pasien**
  - Jumlah pasien unik kelas Pneumonia: **1.844 pasien**
  - **Irisan $\text{Pasien}_{\text{Normal}} \cap \text{Pasien}_{\text{Pneumonia}}$ = 0 (KOSONG)**.
  - Setiap ID pasien terbukti secara ketat hanya memiliki citra pada satu kelas.

### 2.3 Properti Citra Kermany (Hasil Sampel 200 per Kelas)
- **Ekstensi:** Seluruhnya `.jpeg` (mengabaikan artefak sistem `.DS_Store`).
- **Mode Warna:**
  - Kelas `Normal`: 100% bermode `'L'` (grayscale 8-bit).
  - Kelas `Pneumonia`: Mayoritas bermode `'L'` (~94%), sebagian kecil bermode `'RGB'` (~6%).
  - **Ketentuan Loader:** Semua citra **wajib** dikonversi secara eksplisit ke grayscale 1-channel (`Image.convert('L')`) sebelum dinormalisasi ke rentang $[0, 1]$ (`float32`), memenuhi aturan `D5`.
- **Rentang Resolusi Citra Asli ($W \times H$):**
  - `Normal`: Lebar [1.040, 2.694] (median 1.652), Tinggi [747, 2.713] (median 1.316).
  - `Pneumonia`: Lebar [481, 2.008] (median 1.160), Tinggi [234, 2.096] (median 768).
  - Terlihat perbedaan resolusi median antar-kelas (konfirmasi potensi artefak/confounder akuisisi yang dibahas di Bab 2.3).

### 2.4 Duplikasi Berkas Byte-Identik (SHA-256)
- **Kelompok duplikat byte:** Ditemukan **27 kelompok** berkas dengan SHA-256 identik.
- **Jumlah berkas redundan:** **29 berkas ekstra** yang harus disaring pada tahap deduplikasi (`D2`).
- **Aturan:** Deduplikasi menyaring duplikat sebelum dilakukan partisi data *patient-wise*.

---

## 3. Dataset Verifikasi: RSUA Surabaya

### 3.1 Lokasi dan Pemilihan Folder
- **Path relatif di `PROJECT_ROOT`:**
  `data/raw/rsua/Data Chest X-Ray RSUA (Validated)-20230618T025557Z-001/Data Chest X-Ray RSUA (Validated)/`
- **Aturan Pemilihan:**
  - Folder `Data Chest X-Ray RSUA (Annotated)` (berisi 371 citra pra-validasi) **TIDAK DIGUNAKAN**.
  - Folder `Data Chest X-Ray RSUA (Validated)` (berisi 292 citra tervalidasi dokter radiolog) adalah **satu-satunya sumber yang digunakan** untuk evaluasi QC.

### 3.2 Pemetaan Subfolder dan Jumlah Sampel QC
| Subfolder RSUA | Peran Evaluasi | Jumlah Citra | Jumlah Mask | Keterangan |
|---|---|---|---|---|
| `Non_Covid/` | Kelas Normal | 32 | 32 | Indeks `CXR_Images_208.bmp` s.d. `239.bmp` |
| `Non_Covid_Pneumonia/` | Kelas Pneumonia | 53 | 53 | Indeks `CXR_Images_240.bmp` s.d. `292.bmp` |
| `Covid/` | Dikeluarkan | 207 | 207 | **DIKELUARKAN** dari evaluasi QC (sesuai batasan proposal) |
| **Total Evaluasi QC** | **Normal + Pneumonia** | **85** | **85** | Pasangan citra-mask 100% lengkap |

### 3.3 Format Citra dan Mask RSUA
- **Format Berkas:** `.bmp` (tersedia pula representasi `.npy`).
- **Resolusi:** Tepat $256 \times 256$ piksel untuk seluruh citra dan mask.
- **Mode:** Grayscale `'L'`.
- **Nilai Mask Ground Truth:** Biner ketat $\{0, 255\}$ (latar belakang = 0, wilayah paru = 255).
- **Protokol QC:** Mask keluaran U-Net diuji terhadap mask ground truth $256 \times 256$ ini dengan metrik Dice (DSC) dan IoU per kelas.

---

## 4. Folder Tambahan di `data/raw/` dan `models/`

| Folder | Isi Aktual | Kebutuhan Implementasi |
|---|---|---|
| `data/raw/kermany_manual_masks/` | Kosong | Akan diisi 50 kontur manual Kermany ($25\text{ Normal} + 25\text{ Pneumonia}$) dari $D_{\text{test}}$ via skrip pembantu. |
| `data/raw/seg_public/` | Kosong | Opsional / arsip dataset segmentasi publik. |
| `models/unet_frozen/` | Kosong | Tempat meletakkan bobot U-Net terverifikasi pada Gerbang 3. |

---

## 5. Ringkasan Ketentuan Loader Data (`src/data/`)

1. **Parser Manifest (`manifest.py`):**
   - Wajib menggunakan regex Mendeley `^(NORMAL|BACTERIA|VIRUS)-(?P<pid>\d+)-(?P<seq>\d+)\.jpeg$` sesuai `config.assumed.patient_id_regex`.
   - Wajib memvalidasi bahwa tidak ada berkas citra yang gagal diparsing ID-nya (`assert unmatched == 0`).
2. **Deduplikasi (`dedup.py`):**
   - Menghitung hash SHA-256 byte berkas dan SHA-256 array piksel uint8.
   - Menyimpan seluruh riwayat penyaringan ke `data/interim/dedup_log.csv`.
3. **Pemisahan Data (*Split*) (`split.py`):**
   - Pengelompokan berdasarkan `patient_id` dengan rasio 70% train, 15% validation, 15% test.
   - Wajib menjamin $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset$, $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$, $\mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$ baik pada tingkat pasien maupun citra.
   - Hasil split dikunci ke `data/interim/split_patientwise.csv` dan dilindungi dengan *hash check*.
4. **Prapemrosesan (`transforms.py` & `dataset.py`):**
   - Citra diubah ke mode `'L'` $[0, 1]$ $\rightarrow$ resize $224 \times 224$ (bilinear) $\rightarrow$ replikasi 3-channel $\rightarrow$ normalisasi ImageNet.
   - Mask diubah ke ukuran $224 \times 224$ (nearest-neighbor) $\rightarrow$ biner $\{0, 1\}$.
