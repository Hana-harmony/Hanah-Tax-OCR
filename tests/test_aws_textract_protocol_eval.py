from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.evals.run_aws_textract_protocol_eval import build_aws_textract_protocol_eval


def test_build_aws_textract_protocol_eval_normalizes_queries_and_form_keys(
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
                    "address": (
                        "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 "
                        "United States of America"
                    ),
                    "signature_date": "2026-01-12"
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    external_manifest.write_text(
        json.dumps(
            {"cases": [{"case_id": case_id, "subset_tags": ["format_variation"]}]},
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
                        "first_name": ["FIRST_NAME", "First Name"],
                        "last_name": ["LAST_NAME", "Last Name"],
                        "address": ["ADDRESS", "Residence Address"],
                        "signature_date": ["SIGNATURE_DATE", "Signature Date"]
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
                "source_path": "external://aws_textract/holdout_case_001",
                "Blocks": [
                    {
                        "BlockType": "QUERY",
                        "Id": "q1",
                        "Query": {"Text": "What is the first name?", "Alias": "FIRST_NAME"},
                        "Relationships": [{"Type": "ANSWER", "Ids": ["a1"]}]
                    },
                    {
                        "BlockType": "QUERY_RESULT",
                        "Id": "a1",
                        "Text": "MARIA"
                    },
                    {
                        "BlockType": "KEY_VALUE_SET",
                        "Id": "k1",
                        "EntityTypes": ["KEY"],
                        "Relationships": [
                            {"Type": "CHILD", "Ids": ["w1", "w2"]},
                            {"Type": "VALUE", "Ids": ["v1"]}
                        ]
                    },
                    {"BlockType": "WORD", "Id": "w1", "Text": "Last"},
                    {"BlockType": "WORD", "Id": "w2", "Text": "Name"},
                    {
                        "BlockType": "KEY_VALUE_SET",
                        "Id": "v1",
                        "EntityTypes": ["VALUE"],
                        "Relationships": [{"Type": "CHILD", "Ids": ["w3"]}]
                    },
                    {"BlockType": "WORD", "Id": "w3", "Text": "CHEN"},
                    {
                        "BlockType": "QUERY",
                        "Id": "q2",
                        "Query": {"Text": "What is the address?", "Alias": "ADDRESS"},
                        "Relationships": [{"Type": "ANSWER", "Ids": ["a2"]}]
                    },
                    {
                        "BlockType": "QUERY_RESULT",
                        "Id": "a2",
                        "Text": (
                            "1234 Sunset Blvd, Apt 5B, Los Angeles, CA 90026, "
                            "United States of America"
                        )
                    },
                    {
                        "BlockType": "QUERY",
                        "Id": "q3",
                        "Query": {"Text": "What is the signature date?", "Alias": "SIGNATURE_DATE"},
                        "Relationships": [{"Type": "ANSWER", "Ids": ["a3"]}]
                    },
                    {
                        "BlockType": "QUERY_RESULT",
                        "Id": "a3",
                        "Text": "2026-01-12"
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

    result = build_aws_textract_protocol_eval(
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
    assert fields["first_name"] == "MARIA"
    assert fields["last_name"] == "CHEN"
    assert fields["address"] == (
        "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 United States of America"
    )
    assert fields["signature_date"] == "2026-01-12"

    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["exact_match_rate"] == 1.0

    metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
    assert metadata["written_case_ids"] == [case_id]
    assert result["comparison"]["comparison_output"] == str(comparison_output)
