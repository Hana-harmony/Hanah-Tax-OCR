from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.evals.run_azure_document_intelligence_protocol_eval import (
    build_azure_document_intelligence_protocol_eval,
)


def test_build_azure_document_intelligence_protocol_eval_normalizes_and_scores_same_protocol(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    expected_root = tmp_path / "cases"
    case_dir = expected_root / "holdout_case_001"
    case_dir.mkdir(parents=True)
    output_dir = tmp_path / "actual"
    report_output = tmp_path / "report.json"
    summary_output = tmp_path / "summary.json"
    metadata_output = tmp_path / "metadata.json"
    comparison_output = tmp_path / "comparison.json"
    baseline_report = tmp_path / "baseline_report.json"
    external_manifest = tmp_path / "external_manifest.json"
    contract_path = tmp_path / "contract.json"

    case_id = "holdout_case_001"
    (case_dir / "expected.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "document_type": "withholding_tax_form",
                "expected_fields": {
                    "first_name": "MARIA",
                    "last_name": "CHEN",
                    "middle_name": "L",
                    "address": (
                        "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 "
                        "United States of America"
                    ),
                    "residency_country_code": "US",
                    "signature_date": "2026-01-12",
                    "applicant_name": "MARIA L. CHEN"
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    external_manifest.write_text(
        json.dumps(
            {"cases": [{"case_id": case_id, "subset_tags": ["low_quality"]}]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    contract_path.write_text(
        json.dumps(
            {
                "document_type_aliases": {
                    "withholding_tax_form": {
                        "first_name": ["FirstName"],
                        "last_name": ["LastName"],
                        "middle_name": ["MiddleName"],
                        "address": ["ResidenceAddress"],
                        "residency_country_code": ["ResidencyCountryCode"],
                        "signature_date": ["SignatureDate"],
                        "applicant_name": ["ApplicantName"]
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (raw_dir / f"{case_id}.json").write_text(
        json.dumps(
            {
                "source_path": "external://azure/holdout_case_001",
                "documents": [
                    {
                        "fields": {
                            "FirstName": {"type": "string", "valueString": "MARIA"},
                            "LastName": {"type": "string", "valueString": "CHEN"},
                            "MiddleName": {"type": "string", "content": "L"},
                            "ResidenceAddress": {
                                "type": "address",
                                "content": (
                                    "1234 Sunset Blvd, Apt 5B, Los Angeles, CA 90026, "
                                    "United States of America"
                                )
                            },
                            "ResidencyCountryCode": {
                                "type": "countryRegion",
                                "valueCountryRegion": "US"
                            },
                            "SignatureDate": {"type": "date", "valueDate": "2026-01-12"},
                            "ApplicantName": {"type": "string", "valueString": "MARIA L. CHEN"}
                        }
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    baseline_report.write_text(
        json.dumps(
            {"compared_cases": 1, "missing_cases": [], "comparisons": [], "field_metrics": {}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = build_azure_document_intelligence_protocol_eval(
        Namespace(
            raw_dir=raw_dir,
            expected_root=expected_root,
            output_dir=output_dir,
            report_output=report_output,
            summary_output=summary_output,
            metadata_output=metadata_output,
            comparison_output=comparison_output,
            baseline_report=baseline_report,
            external_manifest=external_manifest,
            contract=contract_path,
        )
    )

    run_result = json.loads((output_dir / f"{case_id}.json").read_text(encoding="utf-8"))
    fields = run_result["extracted_documents"][0]["fields"]
    assert fields["address"] == (
        "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 United States of America"
    )
    assert fields["signature_date"] == "2026-01-12"
    assert fields["residency_country_code"] == "US"

    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["exact_match_rate"] == 1.0
    assert summary["low_quality_subset"]["exact_match_rate"] == 1.0

    metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
    assert metadata["written_case_ids"] == [case_id]
    assert result["comparison"]["comparison_output"] == str(comparison_output)
