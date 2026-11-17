"""Unit test untuk parser nama berkas dan manifest Kermany CXR."""

from __future__ import annotations

import re
import pytest
from src.data.manifest import parse_kermany_filename, DEFAULT_MENDELEY_REGEX


def test_parse_mendeley_valid_filenames() -> None:
    cases = [
        ("NORMAL-1003233-0001.jpeg", "NORMAL", "Normal", 0, "1003233", 1),
        ("BACTERIA-1008087-0001.jpeg", "BACTERIA", "Pneumonia", 1, "1008087", 1),
        ("VIRUS-1025587-0004.jpeg", "VIRUS", "Pneumonia", 1, "1025587", 4),
        ("normal-999-0002.jpg", "NORMAL", "Normal", 0, "999", 2),
    ]
    for fn, raw_cls, label, label_id, pid, seq in cases:
        res = parse_kermany_filename(fn, pattern=DEFAULT_MENDELEY_REGEX)
        assert res is not None, f"Gagal parsing {fn}"
        assert res["raw_cls"] == raw_cls
        assert res["label"] == label
        assert res["label_id"] == label_id
        assert res["patient_id"] == pid
        assert res["seq_num"] == seq


def test_parse_invalid_filenames_returns_none() -> None:
    bad_cases = [
        "random_image.jpeg",
        "NORMAL-abc-0001.jpeg",
        "COVID-100-0001.jpeg",
        ".DS_Store",
        "Thumbs.db",
    ]
    for fn in bad_cases:
        assert parse_kermany_filename(fn, pattern=DEFAULT_MENDELEY_REGEX) is None


def test_kaggle_pattern_variant_for_comparison() -> None:
    # Menguji varian pola proposal (Kaggle) untuk memastikan parser fleksibel
    kaggle_pneu = re.compile(
        r"^person(?P<pid>\d+)_(?P<subcls>bacteria|virus)_\d+\.(jpeg|jpg|png)$",
        re.IGNORECASE,
    )
    kaggle_norm = re.compile(
        r"^(?:NORMAL2-)?IM-(?P<pid>\d+)(?:-\d+)?\.(jpeg|jpg|png)$",
        re.IGNORECASE,
    )
    m = kaggle_pneu.match("person100_bacteria_475.jpeg")
    assert m is not None
    assert m.group("pid") == "100"

    m_norm = kaggle_norm.match("NORMAL2-IM-0105-0001.jpeg")
    assert m_norm is not None
    assert m_norm.group("pid") == "0105"
