from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from hanah_tax_ocr.cli import main
from hanah_tax_ocr.evaluation import (
    FieldErrorReport,
    FieldMetricSummary,
    compare_field_error_report_files,
    compare_field_error_reports,
)


def build_metric(
    *,
    comparisons: int,
    exact_matches: int,
    exact_match_rate: float,
    average_character_error_rate: float,
    average_word_error_rate: float,
) -> FieldMetricSummary:
    return FieldMetricSummary(
        comparisons=comparisons,
        exact_matches=exact_matches,
        exact_match_rate=exact_match_rate,
        average_character_error_rate=average_character_error_rate,
        average_word_error_rate=average_word_error_rate,
    )


def test_compare_field_error_reports_classifies_metric_changes() -> None:
    baseline_report = FieldErrorReport(
        compared_cases=4,
        missing_cases=["case_001", "case_002"],
        field_metrics={
            "withholding_tax_form.first_name": build_metric(
                comparisons=4,
                exact_matches=2,
                exact_match_rate=0.5,
                average_character_error_rate=0.25,
                average_word_error_rate=0.5,
            ),
            "withholding_tax_form.last_name": build_metric(
                comparisons=4,
                exact_matches=4,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
            "withholding_tax_form.address": build_metric(
                comparisons=4,
                exact_matches=2,
                exact_match_rate=0.5,
                average_character_error_rate=0.2,
                average_word_error_rate=0.2,
            ),
            "withholding_tax_form.country_code": build_metric(
                comparisons=4,
                exact_matches=4,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
            "withholding_tax_form.tin": build_metric(
                comparisons=3,
                exact_matches=3,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
        },
    )
    candidate_report = FieldErrorReport(
        compared_cases=5,
        missing_cases=["case_002", "case_003"],
        field_metrics={
            "withholding_tax_form.first_name": build_metric(
                comparisons=5,
                exact_matches=4,
                exact_match_rate=0.8,
                average_character_error_rate=0.1,
                average_word_error_rate=0.2,
            ),
            "withholding_tax_form.last_name": build_metric(
                comparisons=5,
                exact_matches=3,
                exact_match_rate=0.6,
                average_character_error_rate=0.3,
                average_word_error_rate=0.4,
            ),
            "withholding_tax_form.address": build_metric(
                comparisons=5,
                exact_matches=2,
                exact_match_rate=0.5,
                average_character_error_rate=0.1,
                average_word_error_rate=0.3,
            ),
            "withholding_tax_form.country_code": build_metric(
                comparisons=5,
                exact_matches=5,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
            "withholding_tax_form.beneficial_owner": build_metric(
                comparisons=5,
                exact_matches=5,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
        },
    )

    comparison = compare_field_error_reports(baseline_report, candidate_report)

    assert comparison.baseline_compared_cases == 4
    assert comparison.candidate_compared_cases == 5
    assert comparison.resolved_missing_cases == ["case_001"]
    assert comparison.new_missing_cases == ["case_003"]
    assert comparison.improved_fields == ["withholding_tax_form.first_name"]
    assert comparison.regressed_fields == ["withholding_tax_form.last_name"]
    assert comparison.mixed_fields == ["withholding_tax_form.address"]
    assert comparison.unchanged_fields == ["withholding_tax_form.country_code"]
    assert comparison.added_fields == ["withholding_tax_form.beneficial_owner"]
    assert comparison.removed_fields == ["withholding_tax_form.tin"]

    first_name_delta = comparison.field_deltas["withholding_tax_form.first_name"]
    assert first_name_delta.status == "improved"
    assert first_name_delta.comparison_delta == 1
    assert first_name_delta.exact_match_rate_delta == pytest.approx(0.3)
    assert first_name_delta.average_character_error_rate_delta == pytest.approx(-0.15)
    assert first_name_delta.average_word_error_rate_delta == pytest.approx(-0.3)

    address_delta = comparison.field_deltas["withholding_tax_form.address"]
    assert address_delta.status == "mixed"


def test_compare_field_error_report_files_and_cli(tmp_path: Path, capsys) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(
        json.dumps(
            {
                "compared_cases": 1,
                "missing_cases": [],
                "comparisons": [],
                "field_metrics": {
                    "residency_certificate.taxpayer_name": {
                        "comparisons": 1,
                        "exact_matches": 0,
                        "exact_match_rate": 0.0,
                        "average_character_error_rate": 0.5,
                        "average_word_error_rate": 1.0,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "compared_cases": 1,
                "missing_cases": [],
                "comparisons": [],
                "field_metrics": {
                    "residency_certificate.taxpayer_name": {
                        "comparisons": 1,
                        "exact_matches": 1,
                        "exact_match_rate": 1.0,
                        "average_character_error_rate": 0.0,
                        "average_word_error_rate": 0.0,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison = compare_field_error_report_files(baseline_path, candidate_path)
    assert comparison.improved_fields == ["residency_certificate.taxpayer_name"]

    original_argv = sys.argv
    sys.argv = [
        "hanah-tax-ocr",
        "compare-eval-reports",
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
    ]
    try:
        exit_code = main()
    finally:
        sys.argv = original_argv

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["improved_fields"] == ["residency_certificate.taxpayer_name"]
