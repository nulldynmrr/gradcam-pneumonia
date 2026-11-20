"""Test konformitas nilai `locked:` pada `configs/config.yaml` terhadap konstanta proposal.

Jika ada satu saja nilai di `locked:` yang diubah dan menyimpang dari angka proposal,
test ini WAJIB gagal.
"""

from __future__ import annotations

from pathlib import Path
import yaml

from src.paths import get_repo_root, load_config


EXPECTED_LOCKED = {
    "data": {
        "sources": ["kermany", "rsna"],
        "kermany_raw_total": 5856,
        "kermany_clean_total": 5827,
        "kermany_balanced_total": 3164,
        "rsna_raw_total": 26684,
        "rsna_excluded_not_normal": 11821,
        "rsna_clean_binary_total": 14863,
        "rsna_balanced_total": 12024,
        "total_images": 15188,
        "n_normal": 7594,
        "n_pneumonia": 7594,
        "builtin_test_images": 624,
        "source_class_ratio_target": 1.0,
        "phash_max_hamming": 5,
        "harmonized_format": "png_8bit_lossless",
        "split_ratio_train": 0.70,
        "split_ratio_val": 0.15,
        "split_ratio_test": 0.15,
        "table31_train": {"normal": 5316, "pneumonia": 5316, "total": 10632},
        "table31_val": {"normal": 1139, "pneumonia": 1139, "total": 2278},
        "table31_test": {"normal": 1139, "pneumonia": 1139, "total": 2278},
        "grayscale_channels": 1,
        "intensity_range": [0.0, 1.0],
    },
    "lung_mask": {
        "segmentation_model": "hybridgnet_chexmask",
        "unet_frozen": True,
        "binarize_threshold": 0.5,
        "keep_largest_components": 2,
        "hole_filling": True,
        "qc_dsc_min": 0.85,
        "qc_iou_min": 0.75,
        "qc_per_class": True,
        "qc_rsua_total": 85,
        "qc_rsua_pneumonia": 53,
        "qc_rsua_normal": 32,
        "rsua_covid_excluded": 207,
        "rsua_mask_size": [256, 256],
        "qc_kermany_total": 50,
        "qc_kermany_normal": 25,
        "qc_kermany_pneumonia": 25,
        "r_paru_extreme_threshold": 0.05,
        "r_paru_class_diff_threshold": 0.05,
        "lr_balance_min_ratio": 0.30,
    },
    "preprocess": {
        "image_size": 224,
        "mask_interp": "nearest",
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "mask_channel_normalized": False,
        "hardmask_before_normalization": True,
    },
    "augmentation": {
        "rotation_deg": 15,
        "hflip_p": 0.5,
        "translate_frac": 0.10,
        "zoom_range": [0.9, 1.1],
        "fill_value": 0,
        "image_interp": "bilinear",
        "mask_interp": "nearest",
        "identical_params_for_pair": True,
        "val_test_augmentation": False,
    },
    "model": {
        "architecture": "efficientnet_b0",
        "pretrained": "imagenet",
        "full_finetuning": True,
        "stem_conv_3ch": [32, 3, 3, 3],
        "stem_conv_4ch": [32, 4, 3, 3],
        "stem_ch4_init": "mean_of_rgb_pretrained",
        "head_feature_map": [1280, 7, 7],
        "dropout": 0.2,
        "n_logits": 1,
        "decision_threshold": 0.5,
        "positive_class": "pneumonia",
    },
    "train": {
        "loss": "bce_unweighted",
        "cam_penalty": False,
        "optimizer": "adamw",
        "lr": 1.0e-4,
        "weight_decay": 1.0e-4,
        "lr_scheduler": "none",
        "batch_size": 32,
        "max_epochs": 50,
        "early_stopping_patience": 10,
        "early_stopping_monitor": "val_loss",
        "keep_best_val_loss_weights": True,
        "seeds": [42, 100, 2024, 2025, 2026],
        "configs": ["baseline_3ch", "hardmask_3ch", "fusion_4ch"],
        "n_runs": 15,
        "deterministic": True,
        "precision": "fp32",
        "split_fixed_across_seeds": True,
    },
    "xai": {
        "method": "gradcam",
        "target_class": "predicted_class",
        "target_score_pos": "+z",
        "target_score_neg": "-z",
        "on_split": "test_no_augmentation",
        "upsample": "bilinear",
        "normalize": "minmax_0_1",
        "acr_epsilon": 1.0e-6,
        "iou_top_fraction": 0.30,
        "heatmap_size": [224, 224],
    },
    "metrics": {
        "primary_classification": ["auc_roc", "f1"],
        "supporting_classification": [
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
        ],
        "localization_primary": "acr",
        "localization_complement": "iou_paru",
        "localization_normalized": "acr_norm",
    },
    "stats": {
        "unit_of_analysis": "patient",
        "bootstrap_B": 1000,
        "bootstrap_ci_level": 0.95,
        "bootstrap_ci_method": "percentile",
        "bootstrap_resample_level": "patient",
        "wilcoxon_alternative": "two-sided",
        "multiple_testing": "holm_bonferroni",
        "alpha": 0.05,
        "holm_family_h1": ["auc_roc", "f1"],
        "holm_family_h2": ["acr", "iou_paru"],
        "stability_sigma_auc_max": 0.010,
        "stability_no_nan_loss": True,
        "min_group_n_per_seed": 5,
        "h1_noninferiority_margin": -0.010,
        "h1_vs_hardmask_lower_ci": 0.0,
        "h2_delta_acr_min": 0.020,
        "h2_delta_iou_min": 0.010,
        "h2_lower_ci": 0.0,
    },
}


def test_locked_values_match_proposal_constants() -> None:
    cfg = load_config()
    assert "locked" in cfg, "Bagian `locked:` wajib ada di configs/config.yaml"
    locked = cfg["locked"]

    assert set(locked.keys()) == set(EXPECTED_LOCKED.keys()), (
        f"Kategori di `locked:` berubah: {set(locked.keys()) ^ set(EXPECTED_LOCKED.keys())}"
    )

    for section, expected_items in EXPECTED_LOCKED.items():
        actual_section = locked[section]
        assert set(actual_section.keys()) == set(expected_items.keys()), (
            f"Kunci pada locked.{section} berubah: "
            f"{set(actual_section.keys()) ^ set(expected_items.keys())}"
        )
        for key, expected_val in expected_items.items():
            entry = actual_section[key]
            assert isinstance(entry, dict) and "value" in entry and "ref" in entry, (
                f"locked.{section}.{key} wajib berupa dict {{value, ref}}"
            )
            assert entry["value"] == expected_val, (
                f"Nilai locked.{section}.{key} = {entry['value']!r} != konstanta proposal {expected_val!r}"
            )
            assert isinstance(entry["ref"], str) and len(entry["ref"].strip()) > 0, (
                f"locked.{section}.{key} wajib memiliki rujukan `ref` ke proposal"
            )


def test_pipeline_stages_labels_match_figure_3_1() -> None:
    stages_path = get_repo_root() / "configs" / "pipeline_stages.yaml"
    with open(stages_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data["labels"] == {
        "baseline_3ch": "Baseline 3-channel",
        "hardmask_3ch": "Hard Masking 3-channel",
        "fusion_4ch": "Fusi 4-Channel (Usulan)",
    }
    assert len(data["stages"]) == 5
    assert [s["id"] for s in data["stages"]] == [1, 2, 3, 4, 5]
    assert data["red_arrow"] == {"stage5_split": "test", "stage5_augment": False}
