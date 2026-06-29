from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.evaluation import build_field_error_report, character_error_rate, word_error_rate


def test_error_rates_handle_exact_and_partial_matches() -> None:
    assert character_error_rate("ABC", "ABC") == 0.0
    assert character_error_rate("ABC", "ADC") > 0.0
    assert word_error_rate("UNITED STATES", "UNITED STATES") == 0.0
    assert word_error_rate("UNITED STATES", "UNITED STATE") > 0.0


def test_build_field_error_report_aggregates_field_metrics(tmp_path: Path) -> None:
    expected_root = tmp_path / "evals" / "cases" / "case_001"
    expected_root.mkdir(parents=True)
    expected_payload = {
        "case_id": "case_001",
        "document_type": "withholding_tax_form",
        "expected_fields": {
            "first_name": "MARIA",
            "residency_country_code": "US",
        },
    }
    (expected_root / "expected.json").write_text(
        json.dumps(expected_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    actual_dir = tmp_path / "actual"
    actual_dir.mkdir()
    actual_payload = {
        "case_id": "case_001",
        "extracted_documents": [
            {
                "document_type": "withholding_tax_form",
                "source_path": "withholding.png",
                "fields": {
                    "first_name": "MARlA",
                    "residency_country_code": "US",
                },
                "quality_checks": {},
                "parser_warnings": [],
            }
        ],
        "review_result": {
            "status": "pass",
            "findings": [],
            "cross_check": {},
        },
        "queued_review_path": None,
    }
    (actual_dir / "case_001.json").write_text(
        json.dumps(actual_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = build_field_error_report(tmp_path / "evals", actual_dir)

    assert report.compared_cases == 1
    assert report.missing_cases == []
    assert len(report.comparisons) == 2
    first_name_metric = report.field_metrics["withholding_tax_form.first_name"]
    assert first_name_metric.exact_match_rate == 0.0
    assert first_name_metric.average_character_error_rate > 0.0
    country_code_metric = report.field_metrics["withholding_tax_form.residency_country_code"]
    assert country_code_metric.exact_match_rate == 1.0


def test_build_field_error_report_can_skip_missing_cases(tmp_path: Path) -> None:
    expected_root = tmp_path / "evals" / "cases"
    for case_id in ("case_001", "case_002"):
        case_dir = expected_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "expected.json").write_text(
            json.dumps(
                {
                    "case_id": case_id,
                    "document_type": "residency_certificate",
                    "expected_fields": {"taxpayer_name": "JANE DOE"},
                }
            ),
            encoding="utf-8",
        )

    actual_dir = tmp_path / "actual"
    actual_dir.mkdir()
    (actual_dir / "case_001.json").write_text(
        json.dumps(
            {
                "case_id": "case_001",
                "extracted_documents": [
                    {
                        "document_type": "residency_certificate",
                        "source_path": "residency.png",
                        "fields": {"taxpayer_name": "JANE DOE"},
                        "quality_checks": {},
                        "parser_warnings": [],
                    }
                ],
                "review_result": {
                    "status": "pass",
                    "findings": [],
                    "cross_check": {},
                },
                "queued_review_path": None,
            }
        ),
        encoding="utf-8",
    )

    report = build_field_error_report(
        expected_root,
        actual_dir,
        include_missing_cases=False,
    )

    assert report.compared_cases == 1
    assert report.missing_cases == []
