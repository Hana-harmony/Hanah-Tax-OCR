import json
from pathlib import Path

from hanah_tax_ocr.evaluation import evaluate_run_result, load_harness_run_result


def test_evaluate_run_result_matches_expected_fields(tmp_path: Path) -> None:
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(
        json.dumps(
            {
                "document_type": "residency_certificate",
                "expected_status": "needs_review",
                "expected_fields": {
                    "residency_country": "United States of America",
                    "residency_country_code": "US",
                },
                "expected_quality_checks": {
                    "seal_present": True,
                },
                "expected_cross_check": {
                    "matched": True,
                },
            }
        ),
        encoding="utf-8",
    )

    actual_path = tmp_path / "actual.json"
    actual_path.write_text(
        json.dumps(
            {
                "case_id": "case_001",
                "extracted_documents": [
                    {
                        "document_type": "residency_certificate",
                        "source_path": "residency.png",
                        "fields": {
                            "residency_country": "United States of America",
                            "residency_country_code": "US",
                        },
                        "quality_checks": {
                            "seal_present": True,
                        },
                        "parser_warnings": [],
                    }
                ],
                "review_result": {
                    "status": "needs_review",
                    "findings": [],
                    "cross_check": {
                        "matched": True,
                    },
                },
                "queued_review_path": None,
            }
        ),
        encoding="utf-8",
    )

    run_result = load_harness_run_result(actual_path)
    evaluation = evaluate_run_result(expected_path, run_result)

    assert evaluation.passed is True
    assert evaluation.mismatches == []
