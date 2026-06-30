from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.training.data_gaps import build_data_gap_report


def test_build_data_gap_report_prioritizes_low_coverage_and_low_accuracy_groups(
    tmp_path: Path,
) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    recognizer_root = tmp_path / "recognizer"
    recognizer_root.mkdir()

    field_crop_entries = [
        {
            "field_group": "english_name_org",
            "field_name": "taxpayer_name",
            "document_type": "apostille",
            "split": "train",
            "source_path": "sample_data/apostille_train.png",
            "quality": {"accepted": True},
        },
        {
            "field_group": "english_name_org",
            "field_name": "taxpayer_name",
            "document_type": "apostille",
            "split": "val",
            "source_path": "sample_data/apostille_val.png",
            "quality": {"accepted": True},
        },
        {
            "field_group": "english_name_org",
            "field_name": "taxpayer_name",
            "document_type": "withholding_tax_form",
            "split": "train",
            "source_path": "sample_data/withholding_train.png",
            "quality": {"accepted": False},
        },
        {
            "field_group": "numeric_tin_code",
            "field_name": "tin",
            "document_type": "residency_certificate",
            "split": "train",
            "source_path": "sample_data/residency_train.png",
            "quality": {"accepted": True},
        },
        {
            "field_group": "numeric_tin_code",
            "field_name": "tin",
            "document_type": "residency_certificate",
            "split": "train",
            "source_path": "sample_data/residency_train.png",
            "quality": {"accepted": True},
        },
        {
            "field_group": "numeric_tin_code",
            "field_name": "tin",
            "document_type": "residency_certificate",
            "split": "val",
            "source_path": "sample_data/residency_val.png",
            "quality": {"accepted": True},
        },
    ]
    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in field_crop_entries) + "\n",
        encoding="utf-8",
    )

    recognizer_summary = {
        "groups": {
            "english_name_org": {
                "train_count": 4,
                "val_count": 1,
                "data_profile": {
                    "hard_case_train_ratio": 0.5,
                    "filtered_hard_case_train_count": 3,
                    "unique_source_counts": {"train": 1, "val": 1},
                    "counts_by_source_type": {"train": {"hard_case": 3}, "val": {}},
                    "hard_case_variant_counts": {
                        "train": {"left_clip": 3},
                        "val": {},
                    },
                    "hard_case_variant_counts_by_document_type": {
                        "train": {"apostille": {"left_clip": 3}},
                        "val": {},
                    },
                    "unique_hard_case_variant_counts": {"train": 1, "val": 0},
                    "hard_case_selection_strategy": "base_document_balance",
                    "hard_case_variant_floor_applied": False,
                    "warnings": ["low_train_sample_count", "hard_case_train_capped"],
                },
                "training_readiness": {
                    "status": "review_required",
                    "ready_for_execution": True,
                    "blocking_warnings": [],
                    "advisory_warnings": [
                        "low_train_sample_count",
                        "hard_case_train_capped",
                    ],
                },
            },
            "numeric_tin_code": {
                "train_count": 3,
                "val_count": 1,
                "data_profile": {
                    "hard_case_train_ratio": 0.0,
                    "filtered_hard_case_train_count": 0,
                    "unique_source_counts": {"train": 1, "val": 1},
                    "warnings": [],
                },
                "training_readiness": {
                    "status": "ready",
                    "ready_for_execution": True,
                    "blocking_warnings": [],
                    "advisory_warnings": [],
                },
            },
        }
    }
    (recognizer_root / "summary.json").write_text(
        json.dumps(recognizer_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    eval_report_path = tmp_path / "eval_report.json"
    eval_report_path.write_text(
        json.dumps(
            {
                "compared_cases": 2,
                "missing_cases": [],
                "comparisons": [],
                "field_metrics": {
                    "apostille.taxpayer_name": {
                        "comparisons": 2,
                        "exact_matches": 1,
                        "exact_match_rate": 0.5,
                        "average_character_error_rate": 0.2,
                        "average_word_error_rate": 0.4,
                    },
                    "residency_certificate.tin": {
                        "comparisons": 2,
                        "exact_matches": 2,
                        "exact_match_rate": 1.0,
                        "average_character_error_rate": 0.0,
                        "average_word_error_rate": 0.0,
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = build_data_gap_report(
        field_crops_root,
        recognizer_root,
        eval_report_path=eval_report_path,
        min_base_train_count=3,
        min_base_val_count=2,
    )

    assert report["priority_order"][0] == "english_name_org"
    english_group = report["priorities"][0]
    assert english_group["base_train_count"] == 1
    assert english_group["base_val_count"] == 1
    assert english_group["base_train_source_count"] == 1
    assert english_group["base_val_source_count"] == 1
    assert english_group["rejected_count"] == 1
    assert english_group["missing_document_types"]["train"] == ["residency_certificate"]
    assert english_group["source_counts_by_document_type"]["train"] == {"apostille": 1}
    assert english_group["recognizer_profile"]["filtered_hard_case_train_count"] == 3
    assert english_group["recognizer_profile"]["train_source_count"] == 1
    assert english_group["recognizer_profile"]["val_source_count"] == 1
    assert english_group["recognizer_profile"]["hard_case_variant_counts"] == {
        "train": {"left_clip": 3},
        "val": {},
    }
    assert english_group["recognizer_profile"]["unique_hard_case_variant_counts"] == {
        "train": 1,
        "val": 0,
    }
    assert (
        english_group["recognizer_profile"]["hard_case_selection_strategy"]
        == "base_document_balance"
    )
    assert english_group["recognizer_profile"]["hard_case_variant_floor_applied"] is False
    assert english_group["recognizer_profile"]["training_readiness"]["status"] == "review_required"
    assert english_group["score_breakdown"]["train_source_gap"] == 5.0
    assert english_group["score_breakdown"]["val_source_gap"] == 2.0
    assert "prioritize_low_accuracy_group" in english_group["recommendations"]
    assert "collect_distinct_train_sources" in english_group["recommendations"]
    assert "collect_distinct_val_sources" in english_group["recommendations"]
    assert "add_base_samples_before_more_hard_cases" in english_group["recommendations"]
    assert "expand_hard_case_variant_coverage" in english_group["recommendations"]

    numeric_group = next(
        item for item in report["priorities"] if item["field_group"] == "numeric_tin_code"
    )
    assert numeric_group["eval_metrics"]["exact_match_rate"] == 1.0
    assert numeric_group["score_breakdown"]["accuracy_gap"] == 0.0


def test_data_gap_report_writes_priority_report_with_missing_eval(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    recognizer_root = tmp_path / "recognizer"
    recognizer_root.mkdir()

    (field_crops_root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "field_group": "date",
                "field_name": "issue_date",
                "document_type": "residency_certificate",
                "split": "train",
                "source_path": "sample_data/residency_train.png",
                "quality": {"accepted": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (recognizer_root / "summary.json").write_text(
        json.dumps({"groups": {"date": {"train_count": 1, "val_count": 0, "data_profile": {}}}}),
        encoding="utf-8",
    )

    report = build_data_gap_report(field_crops_root, recognizer_root)

    assert report["eval_report_path"] is None
    assert report["priority_order"] == ["date"]
    assert report["priorities"][0]["recommendations"] == [
        "collect_base_train_samples",
        "collect_base_val_samples",
        "expand_val_document_coverage",
        "collect_distinct_train_sources",
        "collect_distinct_val_sources",
    ]
    assert report["priorities"][0]["recognizer_profile"]["training_readiness"] == {}
