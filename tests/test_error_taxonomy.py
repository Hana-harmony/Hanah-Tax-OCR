from __future__ import annotations

import json
from pathlib import Path

from scripts.evals.sync_error_taxonomy import build_hard_case_manifest


def test_build_hard_case_manifest_classifies_historical_root_causes(tmp_path: Path) -> None:
    review_queue_dir = tmp_path / "review_queue"
    review_queue_dir.mkdir()
    (review_queue_dir / "case_001.json").write_text(
        json.dumps(
            {
                "case_id": "case_001",
                "review_result": {
                    "status": "reject",
                    "findings": [
                        {"field_name": "issued_on", "code": "required_apostille_date_invalid"},
                        {
                            "field_name": "residency_country",
                            "code": "required_residency_country_invalid",
                        },
                    ],
                },
                "documents": [
                    {
                        "document_type": "apostille",
                        "source_path": "sample_data/apostille.jpg",
                        "fields": {"issued_on": "10TH DAY OF APRIL2014"},
                        "quality_checks": {"blur_score": 1200},
                    },
                    {
                        "document_type": "withholding_tax_form",
                        "source_path": "sample_data/withholding.png",
                        "fields": {"residency_country": "nited States 이f America 전화"},
                        "quality_checks": {"blur_score": 300},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manual_annotations_path = tmp_path / "manual_case_annotations.json"
    manual_annotations_path.write_text(
        json.dumps(
            {
                "cases": {
                    "case_001": {
                        "root_causes": ["label_bleed_name_header"],
                        "notes": "manual note",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = build_hard_case_manifest(
        review_queue_dir,
        manual_annotations_path=manual_annotations_path,
    )

    assert manifest["summary"]["case_count"] == 1
    case = manifest["cases"][0]
    assert case["manual_notes"] == "manual note"
    assert sorted(case["root_causes"]) == [
        "date_spacing_loss",
        "label_bleed_name_header",
        "low_quality_input",
        "mixed_korean_english_interference",
    ]


def test_build_hard_case_manifest_detects_middle_name_and_address_patterns(
    tmp_path: Path,
) -> None:
    review_queue_dir = tmp_path / "review_queue"
    review_queue_dir.mkdir()
    (review_queue_dir / "case_002.json").write_text(
        json.dumps(
            {
                "case_id": "case_002",
                "review_result": {
                    "status": "reject",
                    "findings": [
                        {"field_name": "middle_name", "code": "required_middle_name_invalid"},
                        {"field_name": "address", "code": "required_address_invalid"},
                    ],
                },
                "documents": [
                    {
                        "document_type": "withholding_tax_form",
                        "source_path": "sample_data/withholding.png",
                        "fields": {
                            "middle_name": "CHEN MARIA",
                            "address": "1234Main Street Suite 1",
                        },
                        "quality_checks": {"blur_score": 820},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_hard_case_manifest(review_queue_dir)

    case = manifest["cases"][0]
    assert "middle_name_segmentation_ambiguity" in case["root_causes"]
    assert "address_spacing_merge" in case["root_causes"]


def test_build_hard_case_manifest_detects_address_label_bleed(tmp_path: Path) -> None:
    review_queue_dir = tmp_path / "review_queue"
    review_queue_dir.mkdir()
    (review_queue_dir / "case_003.json").write_text(
        json.dumps(
            {
                "case_id": "case_003",
                "review_result": {
                    "status": "reject",
                    "findings": [
                        {"field_name": "address", "code": "required_address_invalid"},
                    ],
                },
                "documents": [
                    {
                        "document_type": "withholding_tax_form",
                        "source_path": "sample_data/withholding.png",
                        "fields": {
                            "address": (
                                "12 Last Name First Name Middle Name CHEN MARIA "
                                "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 "
                                "United States of America"
                            )
                        },
                        "quality_checks": {"blur_score": 820},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = build_hard_case_manifest(review_queue_dir)

    case = manifest["cases"][0]
    assert "address_label_bleed" in case["root_causes"]
