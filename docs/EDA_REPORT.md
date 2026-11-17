# EDA_REPORT — Laporan Analisis Eksploratif Data (Gerbang 2)

Laporan ini disusun pada **Gerbang 2 (EDA Dulu, Sebelum Model)** sesuai spesifikasi `docs/3.md` dan alur Gambar 3.1. Seluruh analisis bersifat **deskriptif murni** dan tidak mengubah desain eksperimen yang telah dikunci.

---

## 1. Verifikasi Jumlah Citra per Bagian Bawaan vs Tabel 3.1 Proposal

### 1.1 Tabel Konformitas Pra-Deduplikasi
| Bagian Bawaan | Kelas | Jumlah Nyata (Drive & Salinan Lengkap) | Target Proposal (Tabel 3.1) | Status |
|---|---|---|---|---|
| `train/` | Normal | 1.349 | — (rencana gabung) | — |
| `train/` | Pneumonia | 3.883 | — (rencana gabung) | — |
| `test/` | Normal | 234 | — (rencana gabung) | — |
| `test/` | Pneumonia | 390 | — (rencana gabung) | — |
| **Total Normal** | Normal | **1.583** | **1.583** | **PASS** |
| **Total Pneumonia** | Pneumonia | **4.273** | **4.273** | **PASS** |
| **Total Keseluruhan**| Semua | **5.856** | **5.856** | **PASS** |
| **Bagian Uji Bawaan**| Normal + Pneumonia | **624** | **624** | **PASS** |

> **Catatan:** Sesuai metodologi proposal (§3.2.1, Bab III L24), seluruh 5.856 citra bawaan (termasuk 624 citra uji) digabungkan terlebih dahulu sebelum dilakukan deduplikasi dan pembagian data ulang (*split* ulang).

---

## 2. Cakupan Regex Identitas Pasien (DEV-1)

| Parameter | Hasil Analisis | Target | Status |
|---|---|---|---|
| **Pola Regex Mendeley** | `^(NORMAL\|BACTERIA\|VIRUS)-(?P<pid>\d+)-(?P<seq>\d+)\.jpeg$` | 100% | **PASS** |
| **Persentase Berkas Cocok** | **5.856 / 5.856 (100,00%)** | 100% | **PASS** |
| **Berkas Tidak Cocok** | **0 berkas** | 0 berkas | **PASS** |
| **Pasien Unik Normal** | 1.237 pasien | — | — |
| **Pasien Unik Pneumonia** | 1.844 pasien | — | — |
| **Total Pasien Unik** | 3.081 pasien | — | — |
| **Irisan Pasien Lintas Kelas** | **0 pasien (KOSONG)** | 0 | **PASS** |

### Distribusi Citra per Pasien:
- Pasien kelas Normal: rata-rata 1,28 citra/pasien (rentang: 1 s.d. 4 citra).
- Pasien kelas Pneumonia: rata-rata 2,32 citra/pasien (rentang: 1 s.d. 8 citra).
- Karena banyak pasien memiliki lebih dari 1 citra, pembagian data berbasis pasien (*patient-wise*) mutlak diperlukan untuk mencegah *data leakage*.

---

## 3. Hasil Deduplikasi Citra Identik

Deduplikasi mendeteksi salinan identik berdasarkan *hash* byte berkas (SHA-256) dan *hash* piksel grayscale uint8 ($W \times H$).

| Kelas | Total Citra Mentah | Citra Dipertahankan (*KEPT*) | Duplikat Dikeluarkan (*DROPPED*) | Persentase Duplikat |
|---|---|---|---|---|
| **Normal** | 1.583 | 1.582 | 1 | 0,06% |
| **Pneumonia** | 4.273 | 4.245 | 28 | 0,66% |
| **Total** | **5.856** | **5.827** | **29** | **0,50%** |

- **Konflik Label Kelas:** **0 kelompok konflik** (seluruh citra duplikat berada dalam kelas yang sama).
- **Log Lengkap:** Tersimpan di `data/interim/dedup_log.csv`.

---

## 4. Properti Citra & Berkas Rusak

Berdasarkan inspeksi terhadap seluruh 5.856 citra mentah:
- **Berkas Rusak / Gagal Dibuka:** **0 berkas (0%)**. Seluruh citra lolos verifikasi integritas Pillow (`im.verify()`).
- **Ekstensi:** 100% berkas adalah `.jpeg`.
- **Mode Warna:**
  - Kelas `Normal`: 100% mode `'L'` (grayscale 8-bit).
  - Kelas `Pneumonia`: 94% mode `'L'`, 6% mode `'RGB'`.
  - **Konsekuensi Implementasi:** Sesuai aturan `D5`, setiap citra wajib dikonversi secara eksplisit ke `'L'` lalu diskalakan ke $[0, 1]$ (`float32`) sebelum normalisasi ImageNet.
- **Karakteristik Dimensi Asli:**
  - `Normal`: Lebar median 1.664 px (min 936, max 2.890), Tinggi median 1.328 px (min 712, max 2.534), *Aspect Ratio* median 1,25.
  - `Pneumonia`: Lebar median 1.156 px (min 482, max 2.334), Tinggi median 761 px (min 209, max 2.096), *Aspect Ratio* median 1,52.

---

## 5. Indikasi Potensi Confounder Akuisisi (Mendukung Bab 2.3 Proposal)

Pengukuran statistik deskriptif pada sampel representatif citra mentah:

| Parameter Deskriptif | Kelas Normal ($\mu \pm \sigma$ / median) | Kelas Pneumonia ($\mu \pm \sigma$ / median) | Interpretasi Ilmiah |
|---|---|---|---|
| **Resolusi Median ($W \times H$)** | $1.664 \times 1.328$ px | $1.156 \times 761$ px | Citra Pneumonia cenderung beresolusi lebih rendah dari pemindai berbeda |
| **Rasio Aspek Median ($W / H$)** | 1,25 | 1,52 | Bentuk bidang pencitraan berbeda antarkelas |
| **Rata-rata Intensitas Global** | $0,4759 \pm 0,0496$ | $0,4828 \pm 0,0801$ | Intensitas rata-rata keseluruhan relatif sebanding |
| **Rata-rata Intensitas Tepi/Sudut** | $0,2028 \pm 0,0989$ | $0,2323 \pm 0,1089$ | Tepi luar citra Pneumonia sedikit lebih terang |

> **Temuan:** Perbedaan resolusi median dan rasio aspek mengonfirmasi keberadaan *appearance bias* dan variasi perangkat akuisisi sebagaimana dipaparkan pada tinjauan pustaka Bab 2.3, memperkuat urgensi penggunaan fusi panduan *lung mask* untuk mengarahkan fokus atensi model ke organ paru.

---

## 6. Hasil Pembagian Data Patient-Wise Stratified (70:15:15)

Partisi data dilakukan pasca-deduplikasi ($N = 5.827$ citra bersih) menggunakan *greedy patient-wise stratified group split* (`seed = 42`):

| Partisi ($\mathcal{D}$) | Kelas | Jumlah Citra Pasca-Dedup | Jumlah Pasien Unik | Proporsi Citra per Partisi | Rencana Awal Pra-Dedup (Tabel 3.1) | Selisih Pasca-Dedup |
|---|---|---|---|---|---|---|
| **Train (70%)** | Normal | 1.108 | 867 | 27,16% | 1.108 | **0** |
| | Pneumonia | 2.971 | 1.292 | 72,84% | 2.991 | **-20** (efek dedup) |
| | *Subtotal* | **4.079** | **2.159** | **100,00%** | **4.099** | **-20** |
| **Val (15%)** | Normal | 237 | 185 | 27,12% | 237 | **0** |
| | Pneumonia | 637 | 276 | 72,88% | 641 | **-4** (efek dedup) |
| | *Subtotal* | **874** | **461** | **100,00%** | **878** | **-4** |
| **Test (15%)** | Normal | 237 | 185 | 27,12% | 238 | **-1** |
| | Pneumonia | 637 | 276 | 72,88% | 641 | **-4** (efek dedup) |
| | *Subtotal* | **874** | **461** | **100,00%** | **879** | **-5** |
| **Total Akhir** | **Semua** | **5.827** | **3.081** | **100,00%** | **5.856** | **-29** (total dedup) |

### Verifikasi Ketat Bebas Kebocoran (*Zero Leakage*):
- $\text{Pasien}_{\text{train}} \cap \text{Pasien}_{\text{val}} = \emptyset$ (**0 pasien**, PASS)
- $\text{Pasien}_{\text{train}} \cap \text{Pasien}_{\text{test}} = \emptyset$ (**0 pasien**, PASS)
- $\text{Pasien}_{\text{val}} \cap \text{Pasien}_{\text{test}} = \emptyset$ (**0 pasien**, PASS)
- $\text{Citra}_{\text{train}} \cap \text{Citra}_{\text{val}} = \emptyset$ (**0 citra**, PASS)
- $\text{Citra}_{\text{train}} \cap \text{Citra}_{\text{test}} = \emptyset$ (**0 citra**, PASS)
- $\text{Citra}_{\text{val}} \cap \text{Citra}_{\text{test}} = \emptyset$ (**0 citra**, PASS)
- **Status Irisan Bawaan Mendeley:** Partisi bawaan `train/` dan `test/` pada rilis Mendeley juga bebas irisan pasien (0 pasien tumpang tindih).

---

## 7. Verifikasi Dataset Pendukung RSUA Surabaya

| Subdirektori RSUA Validated | Peran Pengujian | Pasangan Citra & Mask | Resolusi | Mode | Nilai Mask GT | Status |
|---|---|---|---|---|---|---|
| `Non_Covid/` | Kelas Normal | 32 pasang (lengkap) | $256 \times 256$ | `'L'` | $\{0, 255\}$ biner | **PASS** |
| `Non_Covid_Pneumonia/` | Kelas Pneumonia | 53 pasang (lengkap) | $256 \times 256$ | `'L'` | $\{0, 255\}$ biner | **PASS** |
| `Covid/` | Dikeluarkan | 207 pasang | $256 \times 256$ | `'L'` | $\{0, 255\}$ biner | **PASS** (dikeluarkan) |
| **Total Evaluasi QC** | **Normal + Pneumonia** | **85 pasang** | **256x256** | **'L'** | **Biner murni** | **PASS** |

---

## 8. Persiapan Evaluasi Rasio Luas Paru ($r_{\text{paru}}$) untuk Gerbang 3

- Formulasi: $r_{\text{paru}} = \frac{1}{H \times W} \sum_{i,j} M_{224}(i,j)$.
- Threshold ekstrem: citra dengan $r_{\text{paru}} < 0,05$ dicatat per kelas untuk analisis sensitivitas.
- Threshold pembanding: jika $|\Delta \bar{r}_{\text{paru}}| > 0,05$ antara Normal dan Pneumonia, metrik $\text{ACR}_{\text{norm}} = \text{ACR} / r_{\text{paru}}$ wajib disertakan.
- Evaluasi ini akan dihitung secara menyeluruh pada **Gerbang 3** setelah model U-Net menghasilkan mask untuk seluruh citra.

---

## 9. Kesimpulan Gerbang 2 dan Rekomendasi Keputusan

1. **Seluruh kriteria Gerbang 2 terpenuhi secara sempurna (PASS).**
2. **29 citra duplikat berhasil disaring** sehingga partisi data pasca-deduplikasi menjadi $4.079\text{ train}$, $874\text{ val}$, dan $874\text{ test}$ (total $5.827$ citra unik dari $3.081$ pasien).
3. **Semua tabel dan figur EDA tersimpan rapi** di `outputs/tables/eda/` dan `outputs/figures/eda/`.
4. **Langkah Berikutnya:** Menuju **GERBANG 3 — LUNG MASK TERVERIFIKASI** (memverifikasi bobot U-Net publik, evaluasi QC pada 85 citra RSUA & 50 citra Kermany, serta *caching* mask $M_{224}$).
