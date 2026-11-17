# DEVIATIONS — Penyimpangan dari proposal

Aturan: tidak ada deviasi diam-diam. Semua butir `[PERLU PERSETUJUAN]` **belum final** sampai Anda menyetujui. Butir tanpa persetujuan eksplisit tetap tertandai dan muncul di "Laporan Kepatuhan" akhir.

| ID | Status | Bagian proposal | Isi proposal | Yang akan dilakukan kode | Alasan | Dampak |
|---|---|---|---|---|---|---|
| **DEV-1** | **[PERLU PERSETUJUAN]** (default A1 disetujui lewat rencana; verifikasi nyata di Gerbang 1) | Metodologi L26 (hlm 30): kode pasien dari `person<nomor>` (Pneumonia) dan `IM-<nomor>` / `NORMAL2-IM-<nomor>` (Normal) | Regex pola Kaggle | Regex `^(?P<cls>NORMAL\|BACTERIA\|VIRUS)-(?P<pid>\d+)-(?P<seq>\d+)\.jpeg$`; kunci grup = `pid`; assert satu `pid` ↔ satu kelas | Dataset di Drive = rilis Mendeley; 100% nama berkas tidak cocok regex proposal. Deskripsi Mendeley: penamaan `(disease)-(randomized patient ID)-(image number)` → tujuan *patient-wise split* tetap tercapai | Hanya cara memperoleh ID pasien; tujuan, jumlah, dan prosedur split tidak berubah. Alternatif **A2**: pakai rilis Kaggle (diunduh/diunggah penulis) → regex proposal berlaku, DEV-1 gugur |
| **DEV-2** | **[PERLU PERSETUJUAN]** — **BELUM diputuskan**; menunggu verifikasi nyata di Gerbang 3 | Metodologi L68 (hlm 31): U-Net **standar** pralatih publik, dilatih pada **Montgomery County + Shenzhen** (di luar Kermany dan RSUA) | Bobot yang persis demikian | Default A (docs/3.md): model segmentasi paru Zenodo (kandidat dilaporkan penulis: record 8393556; pencarian awal agen mengaitkan repo `ViacheslavDanilov/covid_scoring` dengan DOI 8393555 — **belum terverifikasi**) dilatih pada Darwin + Montgomery + Shenzhen; arsitektur nyata bobot belum dipastikan (paper terkait menyebut DeepLabV3+) | Bobot murni Montgomery+Shenzhen berformat U-Net tidak ditemukan penulis | Potensi **dua** deviasi: (i) data latih ≠ proposal (ada Darwin), (ii) arsitektur ≠ "U-Net standar". Jika hasil verifikasi buruk → **berhenti dan tanya**, **tidak** melatih sendiri |

## Aturan penanganan

- Tidak ada deviasi lain yang diizinkan. Temuan baru → tambah baris di sini + tandai `[PERLU PERSETUJUAN]` + tanya penulis.
- Ambiguitas (proposal diam) bukan deviasi → `AMBIGUITIES.md`.
- Hal yang ditolak tegas (di luar proposal): LR scheduler, class weight, balanced sampler, label smoothing, gradient clipping, AMP (default off), TTA, ensemble, cropping, arsitektur lain.
