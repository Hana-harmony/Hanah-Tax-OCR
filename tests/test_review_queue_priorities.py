from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.training.review_queue_priorities import (
    build_review_queue_priority_report,
)


def test_build_review_queue_priority_report_orders_cases_by_gap_score(tmp_path: Path) -> None:
    review_queue_dir = tmp_path / "review_queue"
    review_queue_dir.mkdir()
    data_gap_report_path = tmp_path / "data_gap_report.json"

    data_gap_report_path.write_text(
        json.dumps(
            {
                "priorities": [
                    {
                        "field_group": "date",
                        "priority_score": 37.5,
                        "recommendations": ["collect_base_train_samples"],
                    },
                    {
                        "field_group": "english_name_org",
                        "priority_score": 25.5,
                        "recommendations": ["expand_train_document_coverage"],
                    },
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (review_queue_dir / "queue_001.json").write_text(
        json.dumps(
            {
                "case_id": "queue_001",
                "review_result": {
                    "status": "reject",
                    "findings": [{"code": "bad_date", "message": "bad date"}],
                },
                "documents": [
                    {
                        "document_type": "withholding_tax_form",
                        "source_path": "withholding.png",
                        "fields": {
                            "signature_date": "2026/01/01",
                            "first_name": "MARIA",
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (review_queue_dir / "queue_002.json").write_text(
        json.dumps(
            {
                "case_id": "queue_002",
                "review_result": {
                    "status": "needs_review",
                    "findings": [],
                },
                "documents": [
                    {
                        "document_type": "apostille",
                        "source_path": "apostille.png",
                        "fields": {"signed_by": "CHOI"},
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (review_queue_dir / "run_result_like.json").write_text(
        json.dumps(
            {
                "case_id": "run_result_like",
                "review_result": {"status": "pass", "findings": []},
                "extracted_documents": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = build_review_queue_priority_report(review_queue_dir, data_gap_report_path)

    assert report["priority_order"] == ["queue_001", "queue_002"]
    first_case = report["cases"][0]
    assert first_case["matched_field_groups"] == ["date", "english_name_org"]
    assert first_case["priority_score"] == 65.5
    assert first_case["score_breakdown"] == {
        "gap_score": 63.0,
        "status_boost": 2.0,
        "findings_boost": 0.5,
    }
    assert first_case["recommendations"] == [
        "collect_base_train_samples",
        "expand_train_document_coverage",
    ]


def test_build_review_queue_priority_report_adds_label_target_paths(tmp_path: Path) -> None:
    review_queue_dir = tmp_path / "review_queue"
    review_queue_dir.mkdir()
    data_gap_report_path = tmp_path / "data_gap_report.json"

    data_gap_report_path.write_text(
        json.dumps(
            {
                "priorities": [
                    {
                        "field_group": "numeric_tin_code",
                        "priority_score": 32.5,
                        "recommendations": ["collect_base_train_samples"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (review_queue_dir / "queue_003.json").write_text(
        json.dumps(
            {
                "case_id": "queue_003",
                "review_result": {"status": "reject", "findings": []},
                "documents": [
                    {
                        "document_type": "residency_certificate",
                        "source_path": "residency.png",
                        "fields": {"tin": "123-45-6789"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_review_queue_priority_report(review_queue_dir, data_gap_report_path)

    assert report["prioritized_case_count"] == 1
    assert report["cases"][0]["label_targets"] == [
        {
            "document_type": "residency_certificate",
            "label_path": "data/labeled/pending_review/residency_certificate/queue_003/label.json",
        }
    ]


def test_build_review_queue_priority_report_skips_already_reviewed_cases(
    tmp_path: Path,
) -> None:
    review_queue_dir = tmp_path / "review_queue"
    review_queue_dir.mkdir()
    labeled_root = tmp_path / "labeled"
    reviewed_label = labeled_root / "apostille" / "queue_004" / "label.json"
    reviewed_label.parent.mkdir(parents=True)
    reviewed_label.write_text("{}", encoding="utf-8")
    data_gap_report_path = tmp_path / "data_gap_report.json"
    data_gap_report_path.write_text(
        json.dumps(
            {
                "priorities": [
                    {
                        "field_group": "english_name_org",
                        "priority_score": 25.5,
                        "recommendations": ["expand_train_document_coverage"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (review_queue_dir / "queue_004.json").write_text(
        json.dumps(
            {
                "case_id": "queue_004",
                "review_result": {"status": "reject", "findings": []},
                "documents": [
                    {
                        "document_type": "apostille",
                        "source_path": "apostille.png",
                        "fields": {"signed_by": "CHOI"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_review_queue_priority_report(
        review_queue_dir,
        data_gap_report_path,
        labeled_root=labeled_root,
    )

    assert report["prioritized_case_count"] == 0
    assert report["priority_order"] == []
