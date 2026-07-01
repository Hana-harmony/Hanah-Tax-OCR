from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.harness import HarnessRunResult

from scripts.evals.replay_hybrid_eval import (
    build_hybrid_eval_dir,
    should_rerun_case,
)


def _write_run_result(path: Path, *, case_id: str, document_type: str, source_path: str) -> None:
    path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "extracted_documents": [
                    {
                        "document_type": document_type,
                        "source_path": source_path,
                        "template_id": None,
                        "fields": {"dummy": case_id},
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


def test_should_rerun_case_matches_case_ids_and_prefixes() -> None:
    assert should_rerun_case(
        "withholding_regression_001",
        rerun_case_ids={"residency_maria_chen_001"},
        rerun_prefixes=("withholding_",),
    )
    assert should_rerun_case(
        "residency_maria_chen_001",
        rerun_case_ids={"residency_maria_chen_001"},
        rerun_prefixes=(),
    )
    assert not should_rerun_case(
        "apostille_north_carolina_001",
        rerun_case_ids={"residency_maria_chen_001"},
        rerun_prefixes=("withholding_",),
    )


def test_build_hybrid_eval_dir_copies_untouched_and_reruns_selected(tmp_path: Path) -> None:
    baseline_actual_dir = tmp_path / "baseline"
    baseline_actual_dir.mkdir()
    _write_run_result(
        baseline_actual_dir / "withholding_regression_001.json",
        case_id="withholding_regression_001",
        document_type="withholding_tax_form",
        source_path="withholding.png",
    )
    _write_run_result(
        baseline_actual_dir / "residency_maria_chen_001.json",
        case_id="residency_maria_chen_001",
        document_type="residency_certificate",
        source_path="residency.png",
    )

    output_dir = tmp_path / "replay"

    def rerun_case_fn(run_result: HarnessRunResult) -> HarnessRunResult:
        return HarnessRunResult.model_validate(
            {
                "case_id": run_result.case_id,
                "extracted_documents": [
                    {
                        "document_type": run_result.extracted_documents[0].document_type,
                        "source_path": run_result.extracted_documents[0].source_path,
                        "template_id": "rerun.template",
                        "fields": {"dummy": f"rerun:{run_result.case_id}"},
                        "quality_checks": {},
                        "parser_warnings": [],
                    }
                ],
                "review_result": run_result.review_result.model_dump(mode="json"),
                "queued_review_path": None,
            }
        )

    metadata = build_hybrid_eval_dir(
        baseline_actual_dir,
        output_dir,
        rerun_case_ids=set(),
        rerun_prefixes=("withholding_",),
        rerun_case_fn=rerun_case_fn,
    )

    assert metadata["copied_case_count"] == 1
    assert metadata["rerun_case_count"] == 1
    assert metadata["copied_case_ids"] == ["residency_maria_chen_001"]
    assert metadata["rerun_case_ids"] == ["withholding_regression_001"]

    copied_payload = json.loads((output_dir / "residency_maria_chen_001.json").read_text())
    rerun_payload = json.loads((output_dir / "withholding_regression_001.json").read_text())
    assert copied_payload["extracted_documents"][0]["fields"]["dummy"] == "residency_maria_chen_001"
    assert (
        rerun_payload["extracted_documents"][0]["fields"]["dummy"]
        == "rerun:withholding_regression_001"
    )
