# REKOMENDASI REVISI NASKAH PROPOSAL (MAIN.PDF / LATEX)
**Berdasarkan Temuan Faktual Data & Bukti Matematis Implementasi**
*Dokumen Rujukan:* `WORKSPACE/Proposal_Dinar_Muhammad_Akbar_103012300333/Metodologi.tex` dan `docs/AMBIGUITIES.md`

Dokumen ini menyajikan butir-butir ambiguitas dan ketidaksesuaian faktual yang ditemukan selama pelaksanaan Gerbang 0 hingga Gerbang 2, lengkap dengan **bukti empiris nyata**, **analisis dampak**, serta **usulan redaksi perbaikan (dalam format LaTeX)** untuk naskah proposal Anda.

---

## 1. DEV-1: Pola Nama Berkas Pasien Kermany dkk. (§3.2.1)

### Bukti Faktual
* **Teks Proposal Saat Ini (`Metodologi.tex` baris 26):**
  > *"Pada kelas Pneumonia, kode pasien diambil dari pola `person<nomor>`. Sebagai contoh, nama berkas `person100_bacteria_475.jpeg` menghasilkan kode pasien `person100`. Sementara itu, pada kelas Normal, identitas diperoleh dari kode unik berkas dengan pola `IM-<nomor>` atau `NORMAL2-IM-<nomor>`."*
* **Fakta Nyata di Dataset (Google Drive & Mendeley Data v3):**
  Dari total **5.856 citra** pada dataset resmi Kermany dkk. di Mendeley (`rscbjbr9sj-3`):
  - **100% berkas (5.856/5.856)** menggunakan pola resmi Mendeley:
    `^(NORMAL|BACTERIA|VIRUS)-(?P<pid>\d+)-(?P<seq>\d+)\.jpeg$`
  - Contoh nyata: `NORMAL-1003233-0001.jpeg`, `BACTERIA-1002360-0001.jpeg`, `VIRUS-1003448-0001.jpeg`.
  - Pola `person<nomor>` dan `IM-<nomor>` adalah pola *repackaging* tidak resmi di Kaggle (`paultimothymooney/chest-xray-pneumonia`), yang **0% cocok** dengan dataset asli di Google Drive.
* **Hasil Verifikasi Regex:**
  Pola Mendeley menghasilkan **1.237 pasien Normal** dan **1.844 pasien Pneumonia**, dengan **0 kebocoran pasien lintas kelas** (overlap = 0).

### Usulan Redaksi Revisi (`Metodologi.tex` L26)
```latex
% Usulan Pengganti:
Identitas pasien diperoleh dari awalan nama berkas citra rilis resmi Mendeley Data \cite{Kermany2018Identifying} menggunakan pencocokan pola teks (\textit{regular expression}) berformat \texttt{(kategori)-(ID\_pasien)-(nomor\_citra)}. Pada kelas \textit{Normal}, kode pasien diambil dari pola \texttt{NORMAL-(?<pid>\textbackslash d+)-(\textbackslash d+)\textbackslash.jpeg}, sedangkan pada kelas \textit{Pneumonia}, kode pasien diambil dari pola \texttt{(BACTERIA|VIRUS)-(?<pid>\textbackslash d+)-(\textbackslash d+)\textbackslash.jpeg}. Sebagai contoh, berkas \texttt{BACTERIA-1002360-0001.jpeg} menghasilkan ID pasien \texttt{1002360}. Melalui skema ini, seluruh 5.856 citra terpetakan ke dalam 3.081 pasien unik tanpa tumpang tindih identitas antarkelas.
```

---

## 2. AMBIG-24: Klaim Hasil Pemeriksaan Kualitas U-Net Paru (§3.1 & §3.2.2)

### Bukti Faktual
* **Teks Proposal Saat Ini (`Metodologi.tex` baris 14):**
  > *"Kualitas batas paru dari model tersebut telah diperiksa menggunakan dataset pendukung RSUA [11] dan subset citra uji Kermany dkk. dengan nilai $\text{DSC} \ge 0{,}85$ dan $\text{IoU} \ge 0{,}75$."*
* **Masalah:**
  Kalimat tersebut menggunakan bentuk lampau (*"telah diperiksa... dengan nilai..."*), seolah-olah pengujian telah selesai saat proposal ditulis. Padahal, pengujian kualitas U-Net adalah tahapan eksperimen yang baru dieksekusi secara nyata pada **Gerbang 3**.
* **Dampak Akademis:**
  Penguji seminar proposal dapat menanyakan: *"Kapan Anda mengujinya? Mana tabel hasil DSC dan IoU-nya di Bab 3?"*

### Usulan Redaksi Revisi (`Metodologi.tex` L14)
```latex
% Usulan Pengganti:
Kualitas batas paru dari model U-Net tersebut diverifikasi secara ketat sebelum pelatihan model klasifikasi dimulai melalui pengujian kuantitatif pada \textit{dataset} pendukung RSUA \cite{Wahyuningrum2023ChestXrayDataset} dan subset citra uji Kermany dkk., dengan menetapkan batas ambang penerimaan minimum sebesar $\text{DSC} \ge 0{,}85$ dan $\text{IoU} \ge 0{,}75$.
```

---

## 3. Realisasi Pembagian Data Pasca-Deduplikasi (Tabel 3.1 & L60)

### Bukti Faktual
* **Teks Proposal Saat Ini (Tabel 3.1):**
  Alokasi rencana awal:
  - Train (70%): 1.108 Normal, 2.991 Pneumonia = 4.099 citra
  - Val (15%): 237 Normal, 641 Pneumonia = 878 citra
  - Test (15%): 238 Normal, 641 Pneumonia = 879 citra
  - Total: 5.856 citra.
* **Fakta Empiris Gerbang 2:**
  Proses deduplikasi mendeteksi **29 citra duplikat identik** (1 Normal, 28 Pneumonia), menyisakan **5.827 citra bersih**.
  Setelah dilakukan *greedy stratified patient-wise group split* (seed 42), alokasi citra dan pasien bersih yang dikunci adalah:
  - **Train (70%):** 1.108 Normal (867 pasien), 2.971 Pneumonia (1.292 pasien) = **4.079 citra** (2.159 pasien).
  - **Val (15%):** 237 Normal (185 pasien), 637 Pneumonia (276 pasien) = **874 citra** (461 pasien).
  - **Test (15%):** 237 Normal (185 pasien), 637 Pneumonia (276 pasien) = **874 citra** (461 pasien).
  - **Total:** 1.582 Normal (1.237 pasien), 4.245 Pneumonia (1.844 pasien) = **5.827 citra** (3.081 pasien).
  - Kebocoran pasien antarkelompok: $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{val}} = \emptyset$, $\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$, $\mathcal{D}_{\text{val}} \cap \mathcal{D}_{\text{test}} = \emptyset$ (**0% kebocoran**).

### Usulan Redaksi Revisi (`Metodologi.tex` Tabel 3.1)
Proposal Anda pada baris 65 sudah sangat bijak mencantumkan:
> *"Jumlah citra pada setiap bagian dapat berubah setelah proses deduplikasi dan pengelompokan berdasarkan identitas pasien selesai dilakukan."*

Agar tabel mencerminkan angka realisasi pascadeduplikasi:
```latex
% Perbarui angka baris Tabel 3.1:
Data Latih ($\mathcal{D}_{\text{train}}$) & 70\% & 1.108 & 2.971 & 4.079 & Melatih bobot model EfficientNet-B0 dengan augmentasi data. \\[4pt]
Data Validasi ($\mathcal{D}_{\text{val}}$) & 15\% & 237 & 637 & 874 & Memantau nilai \textit{loss}, memilih model terbaik, dan menerapkan \textit{early stopping}. \\[4pt]
Data Uji ($\mathcal{D}_{\text{test}}$) & 15\% & 237 & 637 & 874 & Mengevaluasi performa klasifikasi akhir serta mengukur nilai ACR dan IoU-Paru. \\[4pt]
\hline
\textbf{Total Bersih (Pascadeduplikasi)} & \textbf{100\%} & \textbf{1.582} & \textbf{4.245} & \textbf{5.827} & Total pascapenyaringan 29 citra duplikat; dikunci tetap lintas \textit{random seed}. \\
```

---

## 4. AMBIG-8: Pengujian Statistik Signifikansi Metrik Klasifikasi (§3.4.2)

### Bukti Faktual & Matematis
* **Teks Proposal Saat Ini (`Metodologi.tex` baris 253 & 257):**
  > *"...dibandingkan secara berpasangan antarkonfigurasi masukan menggunakan selang kepercayaan bootstrap 95% serta uji berpasangan Wilcoxon signed-rank test dengan koreksi Holm-Bonferroni..."* (L253)
  > *"Kriteria Hipotesis 1 ... bernilai di atas nol ($>0$) dengan $p_{\text{Holm}} < 0{,}05$."* (L257)
* **Bukti Teoretis Statistik:**
  - Uji *Wilcoxon signed-rank* **hanya dapat diterapkan pada data sampel berpasangan tingkat individu** ($x_i - y_i$), seperti metrik lokalisasi $\text{ACR}^{(i)}$ dan $\text{IoU-Paru}^{(i)}$ untuk setiap pasien $i$.
  - Sebaliknya, **AUC-ROC dan F1-score adalah metrik agregat tingkat populasi kurva**. Satu pasien tidak memiliki nilai "AUC perorangan".
  - Oleh karena itu, pengujian hipotesis dan penghitungan $p$-value untuk selisih AUC-ROC dan F1-score secara matematis harus dihitung melalui **paired percentile bootstrap hypothesis testing** ($B=1.000$ resample), bukan Wilcoxon.

### Usulan Redaksi Revisi (`Metodologi.tex` L253)
```latex
% Usulan Pengganti:
Untuk setiap pasien, nilai probabilitas prediksi, ACR, dan IoU-Paru dirata-ratakan terlebih dahulu lintas kelima \textit{seed}. Selanjutnya, perbandingan antarkonfigurasi masukan dievaluasi menggunakan selang kepercayaan (\textit{confidence interval} / CI) \textit{bootstrap} 95\% ($B = 1.000$ penarikan sampel ulang tingkat pasien). Pengujian signifikansi hipotesis dilakukan menggunakan uji \textit{bootstrap} berpasangan dua sisi untuk metrik agregat klasifikasi (AUC-ROC dan \textit{F1-score}) serta uji non-parametrik berpasangan \textit{Wilcoxon signed-rank test} untuk metrik lokalisasi tingkat pasien (ACR dan IoU-Paru), dengan koreksi Holm-Bonferroni pada tingkat signifikansi $\alpha = 0{,}05$ \cite{Min2026ContextDependent}.
```

---

## 5. AMBIG-5: Penegasan Titik Hook Ekstraksi Grad-CAM (§3.3)

### Bukti Faktual
* **Teks Proposal Saat Ini (`Metodologi.tex` baris 130):**
  > *"...peta fitur keluaran lapisan konvolusi terakhir (`conv_head`) yang berukuran $1.280 \times 7 \times 7$ (yang menjadi lapisan target ekstraksi Grad-CAM)..."*
* **Bukti Arsitektur PyTorch (`timm.models.efficientnet_b0`):**
  Pada implementasi resmi `timm`, urutan blok kepala konvolusi adalah:
  $$\text{conv\_head (Conv2d 1x1)} \longrightarrow \text{bn2 (BatchNorm2d)} \longrightarrow \text{act2 (SiLU)}$$
  Formulasi matematika asli Grad-CAM (Selvaraju et al., Persamaan 2.4–2.5) mengharuskan gradien $\alpha_k^c$ dihitung langsung terhadap peta fitur konvolusi linier $A^k$, yaitu **tepat pada keluaran `conv_head` sebelum normalisasi batch (`bn2`) dan non-linearitas (`act2`)**.

### Usulan Redaksi Revisi (`Metodologi.tex` L130)
```latex
% Penambahan frasa penjelas:
...keluaran lapisan konvolusi terakhir (\texttt{conv\_head}, tepat sebelum lapisan \textit{Batch Normalization} dan aktivasi SiLU) yang berukuran $1.280 \times 7 \times 7$ (yang menjadi lapisan target ekstraksi Grad-CAM)...
```

---

## Ringkasan Rekomendasi untuk Penulis

| No | Lokasi di Proposal | Isu / Ambiguitas | Kategori | Tingkat Urgensi |
|---|---|---|---|---|
| 1 | §3.2.1, L26 | Pola regex `person<n>` (Kaggle) vs `(NORMAL\|BACTERIA\|VIRUS)-<pid>-<seq>` (Mendeley nyata) | Ketepatan Data | **Tinggi (Wajib diperbaiki agar sesuai dataset)** |
| 2 | §3.1, L14 | Klaim QC U-Net "telah diperiksa" di naskah proposal | Konsistensi Waktu | **Sedang (Perbaiki menjadi kriteria target verifikasi)** |
| 3 | §3.2.1, Tabel 3.1 | Angka alokasi data pasca-deduplikasi 29 citra duplikat (5.827 citra bersih) | Akurasi Angka | **Sedang (Dapat dimasukkan di revisi Bab 3 / Bab 4)** |
| 4 | §3.4.2, L253 | Pemisahan uji paired Bootstrap (AUC/F1) vs Wilcoxon signed-rank (ACR/IoU-Paru) | Rigoritas Statistik | **Sedang (Menghindarkan sanggahan dari penguji statistik)** |
| 5 | §3.3, L130 | Penegasan hook Grad-CAM pada keluaran konvolusi linier pra-BN | Presisi Teknis | **Rendah (Penyempurnaan teknis implementasi)** |
