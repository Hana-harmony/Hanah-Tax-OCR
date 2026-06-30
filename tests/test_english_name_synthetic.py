from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.training.english_name_synthetic import (
    SYNTHETIC_AUGMENTATION_PREFIX,
    generate_synthetic_english_name_hard_cases,
)
from PIL import Image


def test_generate_synthetic_english_name_hard_cases_emits_entries(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir(parents=True)
    crop_dir = field_crops_root / "train" / "english_name_org" / "first_name"
    crop_dir.mkdir(parents=True)
    crop_path = crop_dir / "sample.png"
    Image.new("RGB", (180, 64), "white").save(crop_path)

    manifest_entry = {
        "case_id": "case_001",
        "document_type": "withholding_tax_form",
        "field_name": "first_name",
        "field_group": "english_name_org",
        "text": "MARIA",
        "split": "train",
        "crop_path": str(crop_path),
        "quality": {"accepted": True},
    }
    (field_crops_root / "manifest.jsonl").write_text(
        json.dumps(manifest_entry, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "hard_cases"
    summary = generate_synthetic_english_name_hard_cases(
        field_crops_root,
        output_root,
        variants_per_entry=2,
        seed=7,
    )

    assert summary["base_entry_count"] == 1
    assert summary["synthetic_entry_count"] == 2
    manifest_entries = [
        json.loads(line)
        for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(manifest_entries) == 2
    assert all(
        entry["augmentation_type"] == SYNTHETIC_AUGMENTATION_PREFIX
        for entry in manifest_entries
    )
    assert all(Path(entry["crop_path"]).exists() for entry in manifest_entries)


def test_generate_synthetic_english_name_hard_cases_keeps_label_text_for_multiline_rendering(
    tmp_path: Path,
) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir(parents=True)
    crop_dir = field_crops_root / "train" / "english_name_org" / "applicant_name"
    crop_dir.mkdir(parents=True)
    crop_path = crop_dir / "sample.png"
    Image.new("RGB", (220, 96), "white").save(crop_path)

    manifest_entry = {
        "case_id": "case_001",
        "document_type": "withholding_tax_form",
        "field_name": "applicant_name",
        "field_group": "english_name_org",
        "text": "MARIA L. CHEN",
        "split": "train",
        "crop_path": str(crop_path),
        "quality": {"accepted": True},
    }
    (field_crops_root / "manifest.jsonl").write_text(
        json.dumps(manifest_entry, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "hard_cases"
    generate_synthetic_english_name_hard_cases(
        field_crops_root,
        output_root,
        variants_per_entry=1,
        seed=7,
    )

    manifest_entries = [
        json.loads(line)
        for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(manifest_entries) == 1
    entry = manifest_entries[0]
    assert entry["recognizer_text"] == entry["text"]
    assert "\n" not in entry["recognizer_text"]
    assert "render_text" in entry
