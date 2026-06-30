from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.training.sample_label_priorities import (
    build_sample_label_priority_report,
)


def test_build_sample_label_priority_report_prioritizes_blocked_val_samples(
    tmp_path: Path,
) -> None:
    coverage_report_path = tmp_path / "sample_data_coverage.json"
    coverage_report_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_path": "sample_data/거주자증명서/6.jpg",
                        "covered_by_labeled": False,
                        "pending_review_case_ids": ["residency_university_hawaii_001"],
                        "sample_dataset_document_type": "residency_certificate",
                        "sample_dataset_case_id": "residency_university_hawaii_001",
                        "sample_dataset_split": "val",
                    },
                    {
                        "sample_path": "sample_data/아포스티유 샘플/미국 california 주.png",
                        "covered_by_labeled": False,
                        "pending_review_case_ids": ["apostille_california_001"],
                        "sample_dataset_document_type": "apostille",
                        "sample_dataset_case_id": "apostille_california_001",
                        "sample_dataset_split": "train",
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    data_gap_report_path = tmp_path / "data_gap_report.json"
    data_gap_report_path.write_text(
        json.dumps(
            {
                "priorities": [
                    {
                        "field_group": "date",
                        "priority_score": 32.5,
                        "score_breakdown": {
                            "train_gap": 21.0,
                            "train_source_gap": 0.0,
                            "val_gap": 6.0,
                            "val_source_gap": 4.0,
                        },
                        "recognizer_profile": {
                            "training_readiness": {
                                "status": "blocked",
                                "blocking_warnings": ["no_val_samples"],
                            },
                        },
                        "recommendations": ["collect_base_val_samples"],
                    },
                    {
                        "field_group": "numeric_tin_code",
                        "priority_score": 21.25,
                        "score_breakdown": {
                            "train_gap": 6.0,
                            "train_source_gap": 2.5,
                            "val_gap": 6.0,
                            "val_source_gap": 4.0,
                        },
                        "recognizer_profile": {
                            "training_readiness": {
                                "status": "blocked",
                                "blocking_warnings": ["no_val_samples"],
                            },
                        },
                        "recommendations": ["collect_distinct_val_sources"],
                    },
                    {
                        "field_group": "english_name_org",
                        "priority_score": 3.0,
                        "score_breakdown": {
                            "train_gap": 0.0,
                            "train_source_gap": 0.0,
                            "val_gap": 0.0,
                            "val_source_gap": 2.0,
                        },
                        "recognizer_profile": {
                            "training_readiness": {"status": "review_required"},
                        },
                        "recommendations": ["expand_val_document_coverage"],
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    labeled_root = tmp_path / "labeled"
    residency_label = labeled_root / "residency_certificate" / "reviewed_001" / "label.json"
    residency_label.parent.mkdir(parents=True)
    residency_label.write_text(
        json.dumps(
            {
                "document_type": "residency_certificate",
                "expected_fields": {
                    "tin": "123-45-6789",
                    "issue_date": "January 12, 2026",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    apostille_label = labeled_root / "apostille" / "reviewed_002" / "label.json"
    apostille_label.parent.mkdir(parents=True)
    apostille_label.write_text(
        json.dumps(
            {
                "document_type": "apostille",
                "expected_fields": {
                    "signed_by": "CHOI",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = build_sample_label_priority_report(
        coverage_report_path,
        data_gap_report_path,
        labeled_root=labeled_root,
    )

    assert report["priority_order"] == [
        "sample_data/거주자증명서/6.jpg",
        "sample_data/아포스티유 샘플/미국 california 주.png",
    ]
    first_sample = report["samples"][0]
    assert first_sample["matched_field_groups"] == ["date", "numeric_tin_code"]
    assert first_sample["blocked_field_groups"] == ["date", "numeric_tin_code"]
    assert first_sample["status"] == "pending_review"
    assert first_sample["score_breakdown"] == {
        "gap_score": 53.75,
        "blocked_boost": 10.0,
        "split_gap_boost": 20.0,
        "blocking_split_boost": 10.0,
        "queue_ready_boost": 1.0,
    }
    assert first_sample["priority_score"] == 94.75
    assert first_sample["recommendations"] == [
        "review_pending_label",
        "unblock_recognizer_training",
        "collect_base_val_samples",
        "collect_distinct_val_sources",
    ]
    assert (
        first_sample["label_path"]
        == str(
            labeled_root
            / "pending_review"
            / "residency_certificate"
            / "residency_university_hawaii_001"
            / "label.json"
        )
    )
