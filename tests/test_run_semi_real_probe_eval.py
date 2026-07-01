from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from scripts.evals.run_semi_real_probe_eval import run_semi_real_probe_eval


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


def test_run_semi_real_probe_eval_materializes_and_scores_selected_case(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_output_root = tmp_path / "suite"
    actual_output_dir = tmp_path / "actual"
    report_output = tmp_path / "report.json"
    summary_output = tmp_path / "summary.json"
    metadata_output = tmp_path / "metadata.json"
    manifest_path = tmp_path / "manifest.json"
    external_manifest = tmp_path / "external_manifest.json"
    external_manifest.write_text(json.dumps({"cases": []}), encoding="utf-8")

    def fake_materialize_probe_suite(
        manifest: Path,
        output_root: Path,
        *,
        seed: int,
    ) -> dict[str, object]:
        assert manifest == manifest_path
        (output_root / "cases" / "probe_001").mkdir(parents=True, exist_ok=True)
        (output_root / "labeled" / "withholding_tax_form" / "probe_001").mkdir(
            parents=True,
            exist_ok=True,
        )
        (output_root / "cases" / "probe_001" / "expected.json").write_text(
            json.dumps(
                {
                    "case_id": "probe_001",
                    "document_type": "withholding_tax_form",
                    "expected_fields": {"tin": "987-65-4321"},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "manifest_path": str(manifest),
            "output_root": str(output_root),
            "probe_count": 1,
        }

    def fake_run_eval_suite(
        expected_root: Path,
        labeled_root: Path,
        output_dir: Path,
        *,
        recognizer_root: Path | None = None,
        inference_subdir: str = "inference",
        paddleocr_home: Path = Path("PaddleOCR"),
        case_ids: set[str] | None = None,
        lang: str = "en",
    ) -> dict[str, object]:
        assert expected_root == suite_output_root / "cases"
        assert labeled_root == suite_output_root / "labeled"
        assert case_ids == {"probe_001"}
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_run_result(
            output_dir / "probe_001.json",
            case_id="probe_001",
            document_type="withholding_tax_form",
            source_path="probe.png",
            field_name="tin",
            field_value="987-65-4321",
        )
        return {
            "expected_root": str(expected_root),
            "labeled_root": str(labeled_root),
            "output_dir": str(output_dir),
            "written_case_ids": ["probe_001"],
            "missing_case_ids": [],
            "region_override_count": 0,
            "recognizer_root": None if recognizer_root is None else str(recognizer_root),
            "inference_subdir": inference_subdir,
            "lang": lang,
            "paddleocr_home": str(paddleocr_home),
        }

    monkeypatch.setattr(
        "scripts.evals.run_semi_real_probe_eval.materialize_probe_suite",
        fake_materialize_probe_suite,
    )
    monkeypatch.setattr(
        "scripts.evals.run_semi_real_probe_eval.run_eval_suite",
        fake_run_eval_suite,
    )

    result = run_semi_real_probe_eval(
        Namespace(
            manifest=manifest_path,
            suite_output_root=suite_output_root,
            actual_output_dir=actual_output_dir,
            report_output=report_output,
            summary_output=summary_output,
            metadata_output=metadata_output,
            external_manifest=external_manifest,
            recognizer_root=None,
            inference_subdir="inference",
            paddleocr_home=tmp_path / "PaddleOCR",
            lang="en",
            seed=20260701,
            case_id=["probe_001"],
        )
    )

    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_output.read_text(encoding="utf-8"))

    assert summary["comparison_count"] == 1
    assert summary["exact_match_rate"] == 1.0
    assert metadata["eval_metadata"]["written_case_ids"] == ["probe_001"]
    assert result["selected_case_ids"] == ["probe_001"]
