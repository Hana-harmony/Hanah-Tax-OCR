from __future__ import annotations

import json
from pathlib import Path

from scripts.evals.summarize_eval_report import summarize_eval_report


def test_summarize_eval_report_includes_low_quality_subset(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "compared_cases": 2,
                "missing_cases": [],
                "comparisons": [
                    {
                        "case_id": "case_low_001",
                        "document_type": "residency_certificate",
                        "field_name": "issue_date",
                        "expected": "January 1, 2026",
                        "actual": "January 1, 2026",
                        "exact_match": True,
                        "character_error_rate": 0.0,
                        "word_error_rate": 0.0,
                    },
                    {
                        "case_id": "case_std_001",
                        "document_type": "residency_certificate",
                        "field_name": "issue_date",
                        "expected": "January 1, 2026",
                        "actual": "January 4, 2026",
                        "exact_match": False,
                        "character_error_rate": 0.1,
                        "word_error_rate": 0.2,
                    },
                ],
                "field_metrics": {
                    "residency_certificate.issue_date": {
                        "comparisons": 2,
                        "exact_matches": 1,
                        "exact_match_rate": 0.5,
                        "average_character_error_rate": 0.05,
                        "average_word_error_rate": 0.1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    external_manifest_path = tmp_path / "external_manifest.json"
    external_manifest_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "case_low_001", "subset_tags": ["low_quality"]},
                    {"case_id": "case_std_001", "subset_tags": ["format_variation"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_eval_report(
        report_path,
        external_manifest_path=external_manifest_path,
    )

    assert summary["comparison_count"] == 2
    assert summary["exact_match_rate"] == 0.5
    assert summary["document_pass_rate"] == 0.5
    assert summary["low_quality_subset"]["comparison_count"] == 1
    assert summary["low_quality_subset"]["exact_match_rate"] == 1.0
