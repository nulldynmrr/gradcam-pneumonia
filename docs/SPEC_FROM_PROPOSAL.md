# SPEC_FROM_PROPOSAL — Traceability Matrix

Sumber kebenaran: `Pendahuluan.tex`, `Kajian-Pustaka.tex`, `Metodologi.tex` (versi `.tex` terbaru; `main.pdf` di `docs/` dan di `WORKSPACE` sedikit lebih lama).
Acuan: `L<n>` = nomor baris `.tex`; `hlm` = halaman **fisik** `main.pdf` (44 hlm; Bab I mulai hlm 5, Bab II hlm 12, Bab III hlm 28), diperoleh dari pencarian kata kunci `pdftotext` per halaman — **bisa ±1**; verifikasi ulang saat penulisan skripsi.
Status kolom **Cek**: `[ ]` belum diimplementasi/diverifikasi · `[x]` selesai + test hijau · `[~]` selesai tetapi menunggu data nyata/GPU.
Kolom **Mengacu pada proposal**: ✔ = sama persis dengan teks · ⚠ = ada ambiguitas (lihat `AMBIGUITIES.md`) · ✖ = bertentangan dengan data/teks (lihat `DEVIATIONS.md`).

> Gerbang tempat butir dikerjakan: **G1** Data contract · **G2** EDA/dedup/split · **G3** lung mask · **G4** model/XAI/statistik + smoke · **G5** run penuh.

## A. Data & split

| ID | Spesifikasi | Proposal | Mengacu | Implementasi (rencana) | Test | Gerbang | Cek |
|---|---|---|---|---|---|---|---|
| D1 | Kermany 5.856 citra (1.583 N / 4.273 P); gabung semua bagian bawaan (termasuk 624 test) lalu split ULANG | Metodologi L22, L24; Tabel 3.1 (hlm 29–31) | ✔ | `src/data/manifest.py::build_manifest`, nb `00b`, `01` | `test_manifest_counts` (sintetis), cek nyata di `EDA_REPORT` | G1/G2 | [x] |
| D2 | Dedup citra identik (hash byte DAN hash piksel) SEBELUM split; simpan log | Metodologi L24 (hlm 30); Gambar 3.1 | ⚠ AMBIG-2 | `src/data/dedup.py::find_duplicates`; `data/interim/dedup_log.csv` | `test_dedup` | G2 | [x] |
| D3 | ID pasien via regex dari nama berkas | Metodologi L26 (hlm 30) | ✖ DEV-1 (nama berkas Mendeley ≠ pola proposal) | `src/data/manifest.py::extract_patient_id` (regex di `config.assumed.patient_id_regex`) | `test_patient_regex` | G1/G2 | [x] |
| D4 | Stratified group split 70/15/15 (kelas × pasien), tanpa irisan, dikunci untuk semua konfigurasi & seed; laporkan jumlah citra & pasien unik vs Tabel 3.1 | Metodologi L28–36; Tabel 3.1 (hlm 30–31) | ⚠ AMBIG-3 | `src/data/split.py::patientwise_split`; `data/interim/split_patientwise.csv` (+hash) | `test_split_no_leakage`, `test_split_locked_hash` | G2 | [x] |
| D5 | Grayscale 1-ch, skala [0,1] sebelum normalisasi | Metodologi L81 (hlm 33) | ✔ | `src/data/transforms.py::load_gray01` | `test_load_gray01_and_resize_pair_from_config` | G3 | [x] |

## B. Lung mask

| ID | Spesifikasi | Proposal | Mengacu | Implementasi | Test | G | Cek |
|---|---|---|---|---|---|---|---|
| L1 | U-Net standar pralatih PUBLIK (Darwin+Montgomery+Shenzhen, bukan Kermany/RSUA), bobot beku; bobot harus terverifikasi | Metodologi L68 (hlm 27); Pendahuluan L56 | ✔ *(DEV-2 direkonsiliasi di `.tex`)* | `src/segmentation/unet.py::load_frozen_unet`, `get_unet_spec` | `test_unet_spec_verified_fields`, `test_create_unet_architecture_parameters` | G3 | [~] |
| L2 | prob → biner thr 0,5 → 2 komponen terbesar kiri-kanan → hole filling; (default) prob di-resize ke ukuran asli sebelum thr | Metodologi L77 (hlm 28) | ⚠ AMBIG-1 | `src/segmentation/lung_mask.py::postprocess_lung_mask` | `test_lung_mask.py` | G3 | [x] |
| L3 | QC sebelum training tanpa melihat keluaran klasifikator: (1) RSUA 85 (53 P + 32 N; 207 COVID dikeluarkan) vs GT 256×256; (2) 50 citra D_test Kermany (25+25, seed tetap, daftar disimpan) vs kontur manual `data/raw/kermany_manual_masks/*.png` | Metodologi L71–73 (hlm 28) | ✔ | `src/segmentation/qc.py`; `scripts/export_overlay_for_manual_masks.py`, `import_manual_masks.py` | `test_qc_metrics.py`, `test_gerbang3_pipeline.py` | G3 | [~] |
| L4 | Kriteria rata-rata DSC ≥ 0,85 dan IoU ≥ 0,75 PER KELAS; PASS/FAIL; tidak memblokir, tidak melatih ulang; FAIL → `ACKNOWLEDGE_QC_FAIL=True` wajib sebelum nb 04 | Metodologi L14, L73 (hlm 25, 28) | ✔ | `qc.py::evaluate_rsua_dataset`, `evaluate_kermany_manual_masks`; nb `02` GATE CHECK | `test_gerbang3_pipeline.py` | G3 | [~] |
| L5 | `r_paru=(1/HW)ΣM`; bandingkan rata-rata N vs P; \|Δ\|>0,05 → ACR_norm jadi metrik pembanding; hitung `r_paru<0,05` per kelas + sensitivitas (dengan/tanpa) | Metodologi L73, L77, L251 (hlm 28, 35) | ⚠ AMBIG-14 | `lung_mask.py::compute_r_paru`, `qc.py::summarize_full_r_paru`, nb `02` | `test_lung_mask.py`, `test_gerbang3_pipeline.py` | G3 | [~] |

## C. Prapemrosesan & skema masukan

| ID | Spesifikasi | Proposal | Mengacu | Implementasi | Test | G | Cek |
|---|---|---|---|---|---|---|---|
| P1 | citra → 224×224 bilinear; mask → 224×224 NEAREST | Metodologi L79 (hlm 28), L83 | ✔ *(AMBIG-4 eksplisit di L83 & Alg. 1)* | `transforms.py::resize_pair`, `cache.py` | `test_load_gray01_and_resize_pair_from_config`, `test_synthetic_end_to_end_cache_qc_and_plots` | G3 | [x] |
| P2 | Baseline 3-ch: gray×3 → normalisasi ImageNet | Metodologi L81 (hlm 29) | ✔ | `input_schemes.py::build_baseline_3ch` | `test_input_schemes.py` | G3 | [x] |
| P3 | Hard masking: I224 ⊙ M224 DULU (skala [0,1]) → gray×3 → normalisasi ImageNet | Metodologi L81, L111 | ✔ | `input_schemes.py::build_hardmask_3ch` | `test_input_schemes.py` | G3 | [x] |
| P4 | Fusi 4-ch: [X_RGB(ternormalisasi) ‖ M224 biner {0,1}], ch-4 TIDAK dinormalisasi | Metodologi L81, L113 (hlm 29, 30) | ✔ | `input_schemes.py::build_fusion_4ch` | `test_input_schemes.py` | G3 | [x] |
| P5 | Augmentasi HANYA train: rot ±15°, hflip 0,5, translasi ≤10%, zoom 0,9–1,1; parameter acak identik untuk (I224,M224); bilinear/nearest; fill 0; urutan: aug pasangan → ⊙ → normalisasi → concat | Metodologi L83 (hlm 29) | ✔ | `transforms.py::PairwiseAffineTransform` | `test_input_schemes.py`, `test_gerbang3_pipeline.py` | G3 | [x] |

## D. Model & pelatihan

| ID | Spesifikasi | Proposal | Mengacu | Implementasi | Test | G | Cek |
|---|---|---|---|---|---|---|---|
| M1 | EfficientNet-B0 (timm) ImageNet, full fine-tuning; GAP → Dropout(0,2) → Linear → 1 logit z; p̂=σ(z); ŷ=1 jika p̂ ≥ 0,5 | Metodologi L86, L130 (hlm 34, 37) | ✔ | `models/factory.py::create_model` | `test_head_shape_single_logit` | G4 | [ ] |
| M2 | Fusi 4-ch: stem 32×4×3×3 MANUAL sesuai Pers. 3.2 (ch 1–3 = ImageNet; ch-4 = rata-rata 3 ch); JANGAN `in_chans=4` timm | Metodologi L117–128 (hlm 36–37) | ✔ | `models/fusion4ch.py::adapt_stem_to_4ch` | `test_stem_4ch_matches_eq32`, `test_differs_from_timm_in_chans4` | G4 | [ ] |
| M3 | BCE unweighted (tanpa penalti Grad-CAM); AdamW lr=1e-4 tetap, wd=1e-4; batch 32; ≤50 epoch; early stopping patience 10 pada val loss; simpan bobot val-loss terbaik; deterministik; fp32 | Metodologi L133–140; Tabel 3.2 (hlm 35–36) | ✔ (⚠ AMBIG-9, AMBIG-13) | `train/trainer.py`, `early_stopping.py`, `checkpoint.py`, `seed.py` | `test_trainer_no_scheduler`, `test_early_stopping`, `test_resume_training`, `test_determinism_cpu` | G4 | [ ] |
| M4 | 3 konfigurasi × 5 seed {42,100,2024,2025,2026} = 15 run; seed mengatur init head, dropout, urutan batch, augmentasi; split tidak berubah | Metodologi L140 (hlm 36) | ✔ | nb `04_train_all` | `test_seed_controls_head_dropout_order_aug` | G4/G5 | [ ] |

## E. Grad-CAM & metrik

| ID | Spesifikasi | Proposal | Mengacu | Implementasi | Test | G | Cek |
|---|---|---|---|---|---|---|---|
| G1 | Post-hoc di D_test tanpa augmentasi, `model.eval()`; target = ŷ: y^c=+z (ŷ=1), −z (ŷ=0); implementasi sendiri (Pers. 2.5–2.6): forward hook di `conv_head` (1280×7×7) + `tensor.register_hook`; layer configurable; upsample bilinear 224 → min-max [0,1] | Kajian L320–328 (hlm 26); Metodologi L130, L217–220 (hlm 37–38) | ⚠ AMBIG-5, AMBIG-6 | `xai/gradcam.py` | `test_gradcam_target_sign`, `test_gradcam_batch_equals_manual`, `test_gradcam_layer_configurable` | G4 | [ ] |
| G2 | ACR (Pers. 2.7, ε=1e-6); ACR_norm = ACR/r_paru; IoU-Paru (Pers. 2.8) top-30%; M224 yang SAMA untuk semua konfigurasi | Kajian L334–345 (hlm 27–28) | ⚠ AMBIG-7 | `xai/localization_metrics.py` | `test_acr_bounds`, `test_uniform_heatmap_acr_eq_rparu`, `test_iou_top30` | G4 | [ ] |
| G3 | AUC-ROC & F1 (utama); akurasi, presisi, recall, spesifisitas, balanced accuracy (pendukung); positif = Pneumonia, thr 0,5 | Metodologi L251 (hlm 40) | ✔ | `evaluation/classification_metrics.py` | `test_classification_metrics_known_case` | G4 | [ ] |

## F. Pengujian & statistik (Bab III.5)

| ID | Spesifikasi | Proposal | Mengacu | Implementasi | Test | G | Cek |
|---|---|---|---|---|---|---|---|
| S1 | Skenario 1: kurva val-loss 5 seed; STABIL jika semua 5 seed konvergen, tanpa NaN, σ AUC antar-seed ≤ 0,010; evaluasi utama fusi 4-ch (juga baseline & hard masking); µ±σ semua metrik | Metodologi L243 (hlm 38) | ✔ | `evaluation/aggregate.py`, `hypothesis.py::stability` | `test_stability_verdict` | G4/G5 | [ ] |
| S2 | Skenario 2: µ±σ ACR, ACR_norm, IoU-Paru lintas 5 seed × 3 konfigurasi | Metodologi L245 (hlm 38) | ✔ | nb 06, `aggregate.py` | `test_aggregate_mu_sigma` | G5 | [ ] |
| S3 | Skenario 3: TP/TN/FP/FN per seed; deskriptif jika n ≥ 5 per seed; overlay heatmap + kontur paru | Metodologi L247 (hlm 38) | ✔ | nb 06, `viz/overlays.py` | `test_group_min_n` | G5 | [ ] |
| S4 | Unit analisis = pasien; per pasien rata-rata prob/ACR/IoU lintas 5 seed (dan lintas citra); bootstrap 95% (B=1000, level pasien, persentil); Wilcoxon signed-rank dua sisi berpasangan; Holm dalam tiap kelompok (H1={AUC,F1}, H2={ACR,IoU-Paru}) | Metodologi L253 (hlm 39) | ⚠ AMBIG-8, 10, 11, 16 | `evaluation/stats.py` | `test_bootstrap_patient_level`, `test_holm`, `test_wilcoxon_paired` | G4 | [ ] |
| S5 | H1 DITERIMA jika (a) batas bawah CI Δ(4-ch − baseline) > −0,010 untuk AUC DAN F1; (b) batas bawah CI Δ(4-ch − hard masking) > 0 dengan p_Holm < 0,05 untuk AUC DAN F1; positif tapi tidak signifikan → dilaporkan sebagai temuan empiris | Metodologi L257 (hlm 39); Pendahuluan L63 | ✔ | `evaluation/hypothesis.py::evaluate_h1` | `test_h1_pass_and_fail` | G4 | [ ] |
| S6 | H2 DITERIMA jika (4-ch − baseline): ΔACR ≥ +0,020 dan ΔIoU-Paru ≥ +0,010, batas bawah CI > 0, p_Holm < 0,05; σ antar-seed & trade-off vs hard masking = deskriptif | Metodologi L258 (hlm 40); Pendahuluan L65 | ✔ | `hypothesis.py::evaluate_h2` | `test_h2_pass_and_fail` | G4 | [ ] |
| S7 | `hypothesis.py` → verdict DITERIMA/DITOLAK otomatis + alasan per kriteria → CSV + `.tex` | docs/1.md S7 | ✔ | `hypothesis.py`, `viz/tables.py` | `test_verdict_tables_written` | G4 | [ ] |
| S8 | Ambiguitas sadar: (a) Wilcoxon tidak bisa pada AUC/F1 agregat → CI bootstrap + p bootstrap berpasangan; (b) F1 dari prob rata-rata lintas seed thr 0,5; (c) pasien multi-citra → rata-rata per pasien | docs/1.md S8 | ⚠ AMBIG-8, 10 | `stats.py`, `aggregate.py` | `test_patient_aggregation` | G4 | [ ] |

## G. Peta Rumusan Masalah → implementasi

| RM / Tujuan | Skenario | Hipotesis | Notebook | Output utama | Cek |
|---|---|---|---|---|---|
| RM1 / T1 (stabilitas 4-ch) | 1 | – | 04, 05 | `outputs/metrics/stability.csv` | [ ] |
| RM2 / T2 (performa klasifikasi) | 1 | H1 | 05, 07 | `outputs/stats/h1_verdict.csv` | [ ] |
| RM3 / T3 (lokalisasi Grad-CAM) | 2 + 3 | H2 | 06, 07 | `outputs/stats/h2_verdict.csv`, `outputs/metrics/localization_*.csv` | [ ] |

(RM: Pendahuluan L23–27, hlm 8; Tujuan L41–45; Hipotesis L63–65, hlm 9–10.)

## H. Konformitas Gambar 3.1 (patch v2.1, `docs/2.md`)

Gambar 3.1 = `figure/Desain Arsitektur.pdf` (alur wajib; PNG di `PROPOSAL\gambar\` punya typo, tidak dipakai).

| Tahap | Kotak di gambar | Implementasi | Test | Cek |
|---|---|---|---|---|
| 1 | Dataset CXR | nb `01`; `manifest.py`, `dedup.py` | `test_diagram_conformity` | [ ] |
| 1 | U-Net Beku (pralatih, bobot beku, Lung mask M) | nb `02`; `unet.py`, `lung_mask.py` | idem | [ ] |
| 1 | Uji Kualitas mask (RSUA dan subset Kermany) | nb `02`; `qc.py` | idem | [ ] |
| 2 | Prapemrosesan 224×224 | nb `03`; `transforms.py` | idem | [ ] |
| 2 | Training 70% + augmentasi / Validation 15% / Testing 15% | nb `01` (split) + `dataset.py` | idem | [ ] |
| 3 | Baseline 3-channel / Hard Masking 3-channel / Fusi 4-Channel (Usulan) | `input_schemes.py`, `fusion4ch.py`; nb `03` | idem | [ ] |
| 4 | EfficientNet-B0 + AdamW, BCE, early stopping (patience 10) + 5 seed | nb `04`; `train/` | idem | [ ] |
| 4 | Prediksi Normal / Pneumonia | nb `05` | idem | [ ] |
| 5 | Klasifikasi (AUC-ROC, F1, akurasi, presisi, sensitivitas, spesifisitas, balanced accuracy, prediksi vs label asli) | nb `05` | idem | [ ] |
| 5 | Grad-CAM (ACR, IoU-Paru) | nb `06` | idem | [ ] |
| 5 | Kelompok prediksi (TP, TN, FP, FN) | nb `06–07` | idem | [ ] |

Aturan konformitas (docs/2.md) → butir implementasi:

| # | Aturan | Implementasi/Test | Cek |
|---|---|---|---|
| 1 | Judul markdown tiap notebook: `Tahap <n> Gambar 3.1 — <label persis>` | semua notebook; dicek di `test_diagram_conformity` | [ ] |
| 2 | Label persis di registry/kolom/legenda/tabel; warna konsisten (abu-abu / merah muda / hijau mint / kuning muda) | `pipeline_stages.yaml::labels,colors`; `viz/` | [ ] |
| 3 | `pipeline_stages.yaml` 5 tahap + kotak; test gagal bila tahap tanpa notebook/modul atau notebook tanpa tahap (kecuali `infrastructure_notebooks`, AMBIG-15) | `test_diagram_conformity` | [ ] |
| 4 | Confusion matrix 2×2 per (konfigurasi, seed) → `outputs/metrics/confusion_{config}_seed{s}.csv` + figur + agregat µ±σ | nb 05; `test_confusion_written` | [ ] |
| 5 | Panah merah: SEMUA keluaran tahap 5 dari D_test dengan `augment=False`; assert di dataset/evaluator | `dataset.py`, `aggregate.py`; `test_stage5_only_test_noaug` | [ ] |
| 6 | Validation 15% hanya untuk early stopping & checkpoint terbaik; DILARANG untuk metrik tahap 5; assert | `evaluator`; `test_val_not_in_stage5_metrics` | [ ] |
| 7 | U-Net dijalankan pada SEMUA citra (train/val/test) SATU kali; hasil di-cache di `data/processed` | nb `02`; `test_mask_cache_complete` | [ ] |
| 8 | Ketidaksesuaian lain gambar vs proposal dicatat di `AMBIGUITIES.md`, tidak dipilih diam-diam | `AMBIGUITIES.md` | [x] (awal) |

## I. Patch v2.2 (gerbang) — status

| Gerbang | Artefak bukti | Cek |
|---|---|---|
| G1 Data contract | `data/interim/inventory.json`, `raw_manifest.csv`, `docs/DATA_CONTRACT.md` | [x] |
| G2 EDA | `docs/EDA_REPORT.md`, `outputs/figures/eda/`, `outputs/tables/eda/` | [x] |
| G3 Lung mask | bobot termuat, ≥10 overlay Kermany, laporan QC L3/L4, format masukan model | [ ] |
| G4 Smoke | `docs/SMOKE_REPORT.md` (overfit-32, ekuivalensi 4-ch, heatmap seragam, cross-check Grad-CAM, waktu/epoch) | [ ] |
| G5 Run penuh | 15 run + analisis + `RUNBOOK_COLAB.md` + laporan kepatuhan | [ ] |
