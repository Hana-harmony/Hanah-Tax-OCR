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
        ("tin_001.png", "numeric_tin_code", "train"),
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
    assert first_entry["augmentation_type"] in {
        "left_clip",
        "rotate",
        "low_res",
        "overlay_patch",
        "edge_overlap",
    }
    assert Path(first_entry["crop_path"]).exists()


def test_augment_hard_cases_adds_edge_overlap_variant_for_numeric_groups(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()

    entries = []
    for name, field_group in (
        ("numeric_001.png", "numeric_tin_code"),
        ("mixed_001.png", "korean_mixed_form"),
    ):
        image_path = image_root / name
        Image.new("RGB", (120, 40), "white").save(image_path)
        entries.append(
            {
                "case_id": name,
                "document_type": "withholding_tax_form",
                "field_group": field_group,
                "field_name": "tin",
                "text": "123-45-6789",
                "split": "train",
                "crop_path": str(image_path),
            }
        )
    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    summary = augment_hard_cases(field_crops_root, tmp_path / "hard_cases", seed=7)

    assert summary["counts_by_variant"]["edge_overlap"] == 1
    manifest_entries = [
        json.loads(line)
        for line in (tmp_path / "hard_cases" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    edge_overlap_entry = next(
        entry for entry in manifest_entries if entry["augmentation_type"] == "edge_overlap"
    )
    assert edge_overlap_entry["field_group"] == "numeric_tin_code"
    assert Path(edge_overlap_entry["crop_path"]).exists()


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
    assert summary["max_hard_case_ratio"] == 0.5
    profile = summary["groups"]["numeric_tin_code"]["data_profile"]
    assert profile["counts_by_document_type"]["train"] == {"withholding_tax_form": 2}
    assert profile["counts_by_document_type_and_source"]["train"] == {
        "base": {"withholding_tax_form": 1},
        "hard_case": {"withholding_tax_form": 1},
    }
    assert profile["counts_by_source_type"]["train"] == {"base": 1, "hard_case": 1}
    assert profile["hard_case_train_ratio"] == 0.5
    assert profile["filtered_hard_case_train_count"] == 0
    train_lines = (
        (output_root / "numeric_tin_code" / "train.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(train_lines) == 2


def test_prepare_recognizer_datasets_caps_hard_case_share(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    base_path = tmp_path / "base.png"
    Image.new("RGB", (120, 40), "white").save(base_path)
    field_entry = {
        "case_id": "case_001",
        "document_type": "withholding_tax_form",
        "field_group": "numeric_tin_code",
        "field_name": "tin",
        "text": "987-65-4321",
        "split": "train",
        "crop_path": str(base_path),
    }
    (field_crops_root / "manifest.jsonl").write_text(
        json.dumps(field_entry) + "\n",
        encoding="utf-8",
    )

    hard_cases_root = tmp_path / "hard_cases"
    hard_cases_root.mkdir()
    hard_case_entries = []
    for variant in ("left_clip", "rotate", "low_res"):
        augmented_path = tmp_path / f"base__{variant}.png"
        Image.new("RGB", (120, 40), "white").save(augmented_path)
        hard_case_entries.append(
            {
                **field_entry,
                "augmentation_type": variant,
                "base_crop_path": str(base_path),
                "crop_path": str(augmented_path),
            }
        )
    (hard_cases_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in hard_case_entries) + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "recognizer"
    summary = prepare_recognizer_datasets(
        field_crops_root,
        output_root,
        hard_cases_root=hard_cases_root,
        include_hard_cases=True,
    )

    profile = summary["groups"]["numeric_tin_code"]["data_profile"]
    assert profile["counts_by_source_type"]["train"] == {"base": 1, "hard_case": 2}
    assert profile["counts_by_document_type_and_source"]["train"]["hard_case"] == {
        "withholding_tax_form": 2
    }
    assert profile["filtered_hard_case_train_count"] == 1
    assert profile["hard_case_train_ratio"] == 0.6667
    assert "hard_case_train_capped" in profile["warnings"]
    assert "hard_case_variant_floor_applied" in profile["warnings"]
    assert "hard_case_dominant_train_split" in profile["warnings"]
    train_lines = (
        (output_root / "numeric_tin_code" / "train.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(train_lines) == 3


def test_prepare_recognizer_datasets_balances_capped_hard_cases_by_document_type(
    tmp_path: Path,
) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()

    base_entries = []
    for case_id, document_type in (
        ("apostille_case", "apostille"),
        ("residency_case", "residency_certificate"),
        ("withholding_case_001", "withholding_tax_form"),
        ("withholding_case_002", "withholding_tax_form"),
    ):
        image_path = image_root / f"{case_id}.png"
        Image.new("RGB", (120, 40), "white").save(image_path)
        base_entries.append(
            {
                "case_id": case_id,
                "document_type": document_type,
                "field_group": "english_name_org",
                "field_name": "taxpayer_name",
                "text": case_id.upper(),
                "split": "train",
                "crop_path": str(image_path),
            }
        )
    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in base_entries) + "\n",
        encoding="utf-8",
    )

    hard_cases_root = tmp_path / "hard_cases"
    hard_cases_root.mkdir()
    hard_case_entries = []
    for entry in base_entries:
        source_path = Path(entry["crop_path"])
        for variant in ("left_clip", "rotate", "low_res"):
            augmented_path = tmp_path / f"{source_path.stem}__{variant}.png"
            Image.new("RGB", (120, 40), "white").save(augmented_path)
            hard_case_entries.append(
                {
                    **entry,
                    "augmentation_type": variant,
                    "base_crop_path": str(source_path),
                    "crop_path": str(augmented_path),
                }
            )
    (hard_cases_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in hard_case_entries) + "\n",
        encoding="utf-8",
    )

    summary = prepare_recognizer_datasets(
        field_crops_root,
        tmp_path / "recognizer",
        hard_cases_root=hard_cases_root,
        include_hard_cases=True,
    )

    profile = summary["groups"]["english_name_org"]["data_profile"]
    assert profile["counts_by_document_type_and_source"]["train"]["base"] == {
        "apostille": 1,
        "residency_certificate": 1,
        "withholding_tax_form": 2,
    }
    assert profile["counts_by_document_type_and_source"]["train"]["hard_case"] == {
        "apostille": 1,
        "residency_certificate": 1,
        "withholding_tax_form": 2,
    }
    assert profile["counts_by_document_type"]["train"] == {
        "apostille": 2,
        "residency_certificate": 2,
        "withholding_tax_form": 4,
    }
    assert profile["filtered_hard_case_train_count"] == 8
