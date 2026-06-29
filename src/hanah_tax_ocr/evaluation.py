from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from hanah_tax_ocr.harness import HarnessRunResult
from hanah_tax_ocr.schemas import DocumentType


class EvaluationResult(BaseModel):
    passed: bool
    mismatches: list[str] = Field(default_factory=list)


def load_harness_run_result(path: str | Path) -> HarnessRunResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HarnessRunResult.model_validate(payload)


def evaluate_run_result(
    expected_path: str | Path,
    run_result: HarnessRunResult,
) -> EvaluationResult:
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    mismatches: list[str] = []

    expected_status = expected.get("expected_status")
    if expected_status and run_result.review_result.status.value != expected_status:
        mismatches.append(
            f"expected status {expected_status}, got {run_result.review_result.status.value}"
        )

    expected_document_type = expected.get("document_type")
    target_document = None
    if expected_document_type:
        document_type = DocumentType(expected_document_type)
        target_document = next(
            (
                document
                for document in run_result.extracted_documents
                if document.document_type == document_type
            ),
            None,
        )
        if target_document is None:
            mismatches.append(f"missing document type {expected_document_type} in run result")

    for field_name, expected_value in expected.get("expected_fields", {}).items():
        actual_value = None if target_document is None else target_document.fields.get(field_name)
        if actual_value != expected_value:
            mismatches.append(
                f"expected field {field_name}={expected_value!r}, got {actual_value!r}"
            )

    expected_codes = set(expected.get("expected_finding_codes", []))
    if expected_codes:
        actual_codes = {finding.code for finding in run_result.review_result.findings}
        missing_codes = expected_codes - actual_codes
        if missing_codes:
            mismatches.append(
                "missing finding codes: " + ", ".join(sorted(missing_codes))
            )

    return EvaluationResult(passed=not mismatches, mismatches=mismatches)
