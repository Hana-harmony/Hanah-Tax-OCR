from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.training.field_crops import assign_case_splits, export_field_crops
from PIL import Image


def test_export_field_crops_writes_manifest_and_grouped_outputs(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_data" / "거주자증명서"
    sample_dir.mkdir(parents=True)
    image_path = sample_dir / "미국 TREASURY주.png"
    Image.new("RGB", (400, 300), "white").save(image_path)

    label_root = tmp_path / "data" / "labeled" / "residency_certificate" / "case_001"
    label_root.mkdir(parents=True)
    label_payload = {
        "case_id": "residency_case_001",
        "document_type": "residency_certificate",
        "source_path": str(image_path),
        "expected_fields": {
            "taxpayer_name": "MARIA L. CHEN",
            "tin": "987-65-4321",
            "tax_year": "2026",
            "issue_date": "January 12, 2026",
        },
    }
    (label_root / "label.json").write_text(
        json.dumps(label_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_root = tmp_path / "data" / "training" / "field_crops"
    summary = export_field_crops(tmp_path / "data" / "labeled", output_root, val_ratio=0.0)

    assert summary["total_crops"] == 4
    manifest_lines = (
        (output_root / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(manifest_lines) == 4
    first_entry = json.loads(manifest_lines[0])
    assert first_entry["field_group"] in {"english_name_org", "numeric_tin_code", "date"}
    assert "quality" in first_entry
    assert "accepted" in first_entry["quality"]
    assert Path(first_entry["crop_path"]).exists()
    assert (output_root / "manifests" / "english_name_org" / "train.jsonl").exists()
    assert summary["accepted_crops"] + summary["rejected_crops"] == 4


def test_export_field_crops_skips_non_file_sources(tmp_path: Path) -> None:
    label_root = tmp_path / "data" / "labeled" / "withholding_tax_form" / "case_002"
    label_root.mkdir(parents=True)
    payload = {
        "case_id": "withholding_case_002",
        "document_type": "withholding_tax_form",
        "source_path": "synthetic://withholding_tax_form/withholding_case_002",
        "expected_fields": {
            "first_name": "SAMPLE",
        },
    }
    (label_root / "label.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = export_field_crops(tmp_path / "data" / "labeled", tmp_path / "output")

    assert summary["total_crops"] == 0
    assert summary["skipped_reasons"]["non_file_source"] == 1


def test_export_field_crops_marks_low_quality_blank_crop_as_rejected(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_data" / "거주자증명서"
    sample_dir.mkdir(parents=True)
    image_path = sample_dir / "미국 TREASURY주.png"
    Image.new("RGB", (400, 300), "white").save(image_path)

    label_root = tmp_path / "data" / "labeled" / "residency_certificate" / "case_003"
    label_root.mkdir(parents=True)
    (label_root / "label.json").write_text(
        json.dumps(
            {
                "case_id": "residency_case_003",
                "document_type": "residency_certificate",
                "source_path": str(image_path),
                "expected_fields": {"taxpayer_name": "MARIA L. CHEN"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = export_field_crops(
        tmp_path / "data" / "labeled",
        tmp_path / "field_crops",
        min_dark_ratio=0.5,
    )

    assert summary["rejected_crops"] == 1
    assert summary["quality_flag_counts"]["low_dark_ratio"] == 1


def test_export_field_crops_rejects_dense_edge_content_crop(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample_data" / "거주자증명서"
    sample_dir.mkdir(parents=True)
    image_path = sample_dir / "dense_crop.png"
    Image.new("RGB", (400, 300), "black").save(image_path)

    label_root = tmp_path / "data" / "labeled" / "residency_certificate" / "case_dense"
    label_root.mkdir(parents=True)
    (label_root / "label.json").write_text(
        json.dumps(
            {
                "case_id": "residency_case_dense",
                "document_type": "residency_certificate",
                "source_path": str(image_path),
                "expected_fields": {"taxpayer_name": "MARIA L. CHEN"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "field_crops"
    summary = export_field_crops(tmp_path / "data" / "labeled", output_root)
    manifest_entry = json.loads(
        (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert summary["rejected_crops"] == 1
    assert summary["quality_flag_counts"]["foreground_fills_crop"] == 1
    assert summary["quality_flag_counts"]["dense_edge_content"] == 1
    assert manifest_entry["quality"]["accepted"] is False
    assert manifest_entry["quality"]["foreground_bbox_ratio"] == 1.0
    assert sorted(manifest_entry["quality"]["touched_edges"]) == [
        "bottom",
        "left",
        "right",
        "top",
    ]


def test_assign_case_splits_keeps_document_types_in_both_splits_when_possible() -> None:
    assignments = assign_case_splits(
        [
            {"case_id": "apostille_case_001", "document_type": "apostille"},
            {"case_id": "apostille_case_002", "document_type": "apostille"},
            {"case_id": "residency_case_001", "document_type": "residency_certificate"},
            {"case_id": "residency_case_002", "document_type": "residency_certificate"},
        ],
        val_ratio=0.2,
    )

    apostille_splits = {
        assignments["apostille_case_001"],
        assignments["apostille_case_002"],
    }
    residency_splits = {
        assignments["residency_case_001"],
        assignments["residency_case_002"],
    }
    assert apostille_splits == {"train", "val"}
    assert residency_splits == {"train", "val"}


def test_export_field_crops_assigns_val_case_per_document_type_when_possible(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample_data"
    sample_dir.mkdir(parents=True)
    residency_image = sample_dir / "residency_treasury.png"
    apostille_image = sample_dir / "california_apostille.png"
    Image.new("RGB", (400, 300), "white").save(residency_image)
    Image.new("RGB", (400, 300), "white").save(apostille_image)

    label_specs = [
        (
            "residency_certificate",
            "residency_case_001",
            residency_image,
            {"taxpayer_name": "RESIDENCY ONE"},
        ),
        (
            "residency_certificate",
            "residency_case_002",
            residency_image,
            {"taxpayer_name": "RESIDENCY TWO"},
        ),
        (
            "apostille",
            "apostille_case_001",
            apostille_image,
            {"signed_by": "APOSTILLE ONE"},
        ),
        (
            "apostille",
            "apostille_case_002",
            apostille_image,
            {"signed_by": "APOSTILLE TWO"},
        ),
    ]
    for document_type, case_id, image_path, expected_fields in label_specs:
        label_root = tmp_path / "data" / "labeled" / document_type / case_id
        label_root.mkdir(parents=True)
        (label_root / "label.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "document_type": document_type,
                    "source_path": str(image_path),
                    "expected_fields": expected_fields,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    output_root = tmp_path / "field_crops"
    export_field_crops(tmp_path / "data" / "labeled", output_root, val_ratio=0.2)

    manifest_entries = [
        json.loads(line)
        for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    splits_by_document_type: dict[str, set[str]] = {}
    for entry in manifest_entries:
        splits_by_document_type.setdefault(entry["document_type"], set()).add(entry["split"])

    assert splits_by_document_type["apostille"] == {"train", "val"}
    assert splits_by_document_type["residency_certificate"] == {"train", "val"}
