from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.training.hard_cases import augment_hard_cases
from hanah_tax_ocr.training.recognizer import prepare_recognizer_datasets
from PIL import Image


def test_augment_hard_cases_writes_variant_manifest(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()

    base_entries = []
    for name, group, split in (
        ("name_001.png", "english_name_org", "train"),
        ("mixed_001.png", "korean_mixed_form", "train"),
    ):
        image_path = image_root / name
        Image.new("RGB", (120, 40), "white").save(image_path)
        base_entries.append(
            {
                "case_id": name,
                "field_group": group,
                "field_name": "taxpayer_name",
                "text": "SAMPLE",
                "split": split,
                "crop_path": str(image_path),
            }
        )

    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in base_entries) + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "hard_cases"
    summary = augment_hard_cases(field_crops_root, output_root, seed=7)

    assert summary["total_augmented_crops"] >= 2
    manifest_lines = (
        (output_root / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    first_entry = json.loads(manifest_lines[0])
    assert first_entry["augmentation_type"] in {"left_clip", "rotate", "low_res", "overlay_patch"}
    assert Path(first_entry["crop_path"]).exists()


def test_prepare_recognizer_datasets_can_include_hard_cases(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_path = tmp_path / "base.png"
    Image.new("RGB", (120, 40), "white").save(image_path)
    field_entry = {
        "case_id": "case_001",
        "document_type": "withholding_tax_form",
        "field_group": "numeric_tin_code",
        "field_name": "tin",
        "text": "987-65-4321",
        "split": "train",
        "crop_path": str(image_path),
    }
    (field_crops_root / "manifest.jsonl").write_text(
        json.dumps(field_entry) + "\n",
        encoding="utf-8",
    )

    hard_cases_root = tmp_path / "hard_cases"
    hard_cases_root.mkdir()
    augmented_path = tmp_path / "base__low_res.png"
    Image.new("RGB", (120, 40), "white").save(augmented_path)
    hard_case_entry = {
        **field_entry,
        "augmentation_type": "low_res",
        "base_crop_path": str(image_path),
        "crop_path": str(augmented_path),
    }
    (hard_cases_root / "manifest.jsonl").write_text(
        json.dumps(hard_case_entry) + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "recognizer"
    summary = prepare_recognizer_datasets(
        field_crops_root,
        output_root,
        hard_cases_root=hard_cases_root,
        include_hard_cases=True,
    )

    assert summary["include_hard_cases"] is True
    profile = summary["groups"]["numeric_tin_code"]["data_profile"]
    assert profile["counts_by_document_type"]["train"] == {"withholding_tax_form": 2}
    assert profile["counts_by_source_type"]["train"] == {"base": 1, "hard_case": 1}
    assert profile["hard_case_train_ratio"] == 0.5
    train_lines = (
        (output_root / "numeric_tin_code" / "train.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(train_lines) == 2
