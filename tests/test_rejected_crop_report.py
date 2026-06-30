from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.training.rejected_crops import build_rejected_field_crop_report


def test_build_rejected_field_crop_report_sorts_by_data_gap_priority(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case_001",
                        "document_type": "apostille",
                        "field_group": "numeric_tin_code",
                        "field_name": "certificate_number",
                        "split": "val",
                        "crop_path": "data/training/field_crops/val/numeric_tin_code/case_001.png",
                        "source_path": "sample_data/apostille.png",
                        "quality": {
                            "accepted": False,
                            "quality_flags": [
                                "foreground_fills_crop",
                                "dense_edge_content",
                            ],
                        },
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case_002",
                        "document_type": "residency_certificate",
                        "field_group": "english_name_org",
                        "field_name": "taxpayer_name",
                        "split": "train",
                        "crop_path": (
                            "data/training/field_crops/train/english_name_org/case_002.png"
                        ),
                        "source_path": "sample_data/residency.png",
                        "quality": {
                            "accepted": False,
                            "quality_flags": ["low_contrast"],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    data_gap_report_path = tmp_path / "data_gap_report.json"
    data_gap_report_path.write_text(
        json.dumps(
            {
                "priorities": [
                    {
                        "field_group": "numeric_tin_code",
                        "priority_score": 22.75,
                        "recommendations": ["review_rejected_field_crops"],
                    },
                    {
                        "field_group": "english_name_org",
                        "priority_score": 6.0,
                        "recommendations": [],
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = build_rejected_field_crop_report(
        field_crops_root,
        data_gap_report_path=data_gap_report_path,
    )

    assert report["rejected_crop_count"] == 2
    assert report["counts_by_field_group"] == {
        "numeric_tin_code": 1,
        "english_name_org": 1,
    }
    assert report["groups"][0]["field_group"] == "numeric_tin_code"
    assert report["groups"][0]["priority_score"] == 22.75
    assert report["groups"][0]["entries"][0]["review_actions"] == [
        "inspect_seal_or_noise_overlap",
        "tighten_region_box_or_verify_template",
    ]
    assert report["groups"][1]["entries"][0]["review_actions"] == [
        "check_blank_or_faint_text_crop"
    ]


def test_build_rejected_field_crop_report_handles_missing_priority_report(
    tmp_path: Path,
) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    (field_crops_root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case_003",
                "document_type": "withholding_tax_form",
                "field_group": "date",
                "field_name": "signature_date",
                "split": "train",
                "crop_path": "date.png",
                "source_path": "sample_data/withholding.png",
                "quality": {
                    "accepted": False,
                    "quality_flags": ["too_short"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_rejected_field_crop_report(field_crops_root)

    assert report["data_gap_report_path"] is None
    assert report["groups"][0]["field_group"] == "date"
    assert report["groups"][0]["priority_score"] is None
    assert report["groups"][0]["entries"][0]["review_actions"] == [
        "review_region_box_size"
    ]
