from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.evals.run_external_holdout_eval import run_external_holdout_eval


def _write_run_result(
    path: Path,
    *,
    case_id: str,
    document_type: str,
    source_path: str,
    field_name: str,
    field_value: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "extracted_documents": [
                    {
                        "document_type": document_type,
                        "source_path": source_path,
                        "template_id": None,
                        "fields": {
                            field_name: field_value,
                        },
                        "quality_checks": {},
                        "parser_warnings": [],
                    }
                ],
                "review_result": {
                    "status": "pass",
                    "findings": [],
                    "cross_check": {
                        "matched": False,
                        "reason": "cross-check skipped because one of the documents is missing",
                    },
                },
                "queued_review_path": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_run_external_holdout_eval_writes_report_and_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected_root = tmp_path / "evals" / "external_holdout" / "cases"
    case_dir = expected_root / "holdout_case_001"
    case_dir.mkdir(parents=True)
    (case_dir / "expected.json").write_text(
        json.dumps(
            {
                "case_id": "holdout_case_001",
                "document_type": "residency_certificate",
                "source_path": "sample_data/residency.png",
                "expected_fields": {
                    "issue_date": "January 2, 2026",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    external_manifest = tmp_path / "evals" / "external_holdout" / "manifest.json"
    external_manifest.parent.mkdir(parents=True, exist_ok=True)
    external_manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "holdout_case_001",
                        "subset_tags": ["low_quality", "format_variation"],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "actual"
    report_output = tmp_path / "report.json"
    summary_output = tmp_path / "summary.json"
    metadata_output = tmp_path / "metadata.json"

    def fake_run_eval_suite(
        expected_root_arg: Path,
        labeled_root: Path,
        output_dir_arg: Path,
        *,
        recognizer_root: Path | None = None,
        inference_subdir: str = "inference",
        paddleocr_home: Path = Path("PaddleOCR"),
        case_ids: set[str] | None = None,
        lang: str = "en",
    ) -> dict[str, object]:
        assert expected_root_arg == expected_root
        output_dir_arg.mkdir(parents=True, exist_ok=True)
        _write_run_result(
            output_dir_arg / "holdout_case_001.json",
            case_id="holdout_case_001",
            document_type="residency_certificate",
            source_path="sample_data/residency.png",
            field_name="issue_date",
            field_value="January 2, 2026",
        )
        return {
            "expected_root": str(expected_root_arg),
            "labeled_root": str(labeled_root),
            "output_dir": str(output_dir_arg),
            "written_case_ids": ["holdout_case_001"],
            "missing_case_ids": [],
            "region_override_count": 0,
            "recognizer_root": None if recognizer_root is None else str(recognizer_root),
            "inference_subdir": inference_subdir,
            "lang": lang,
            "case_ids": sorted(case_ids) if case_ids else [],
            "paddleocr_home": str(paddleocr_home),
        }

    monkeypatch.setattr(
        "scripts.evals.run_external_holdout_eval.run_eval_suite",
        fake_run_eval_suite,
    )

    result = run_external_holdout_eval(
        Namespace(
            expected_root=expected_root,
            labeled_root=tmp_path / "data" / "labeled",
            external_manifest=external_manifest,
            output_dir=output_dir,
            report_output=report_output,
            summary_output=summary_output,
            metadata_output=metadata_output,
            baseline_report=None,
            comparison_output=None,
            recognizer_root=None,
            inference_subdir="inference",
            paddleocr_home=tmp_path / "PaddleOCR",
            case_id=[],
            lang="en",
        )
    )

    report = json.loads(report_output.read_text(encoding="utf-8"))
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_output.read_text(encoding="utf-8"))

    assert report["compared_cases"] == 1
    assert summary["exact_match_rate"] == 1.0
    assert summary["low_quality_subset"]["comparison_count"] == 1
    assert metadata["metadata"]["written_case_ids"] == ["holdout_case_001"]
    assert result["report_output"] == str(report_output)
