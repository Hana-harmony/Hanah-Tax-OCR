from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from hanah_tax_ocr.harness import HarnessRunResult
from hanah_tax_ocr.schemas import DocumentType


class EvaluationResult(BaseModel):
    passed: bool
    mismatches: list[str] = Field(default_factory=list)


class FieldComparison(BaseModel):
    case_id: str
    document_type: str
    field_name: str
    expected: str | None = None
    actual: str | None = None
    exact_match: bool
    character_error_rate: float
    word_error_rate: float


class FieldMetricSummary(BaseModel):
    comparisons: int
    exact_matches: int
    exact_match_rate: float
    average_character_error_rate: float
    average_word_error_rate: float


class FieldErrorReport(BaseModel):
    compared_cases: int
    missing_cases: list[str] = Field(default_factory=list)
    comparisons: list[FieldComparison] = Field(default_factory=list)
    field_metrics: dict[str, FieldMetricSummary] = Field(default_factory=dict)


def load_harness_run_result(path: str | Path) -> HarnessRunResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HarnessRunResult.model_validate(payload)


def load_harness_run_results_from_dir(path: str | Path) -> dict[str, HarnessRunResult]:
    indexed: dict[str, HarnessRunResult] = {}
    for json_path in sorted(Path(path).glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if "case_id" not in payload or "extracted_documents" not in payload:
            continue
        run_result = HarnessRunResult.model_validate(payload)
        indexed[run_result.case_id] = run_result
    return indexed


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(expected: str | None, actual: str | None) -> float:
    expected_text = "" if expected is None else str(expected)
    actual_text = "" if actual is None else str(actual)
    if not expected_text:
        return 0.0 if not actual_text else 1.0
    return _levenshtein_distance(list(expected_text), list(actual_text)) / len(expected_text)


def word_error_rate(expected: str | None, actual: str | None) -> float:
    expected_tokens = [] if expected is None else str(expected).split()
    actual_tokens = [] if actual is None else str(actual).split()
    if not expected_tokens:
        return 0.0 if not actual_tokens else 1.0
    return _levenshtein_distance(expected_tokens, actual_tokens) / len(expected_tokens)


def build_field_error_report(
    expected_root: str | Path,
    actual_dir: str | Path,
    *,
    case_ids: set[str] | None = None,
    include_missing_cases: bool = True,
) -> FieldErrorReport:
    comparisons: list[FieldComparison] = []
    missing_cases: list[str] = []
    actual_results = load_harness_run_results_from_dir(actual_dir)

    for expected_path in sorted(Path(expected_root).rglob("expected.json")):
        expected_payload = json.loads(expected_path.read_text(encoding="utf-8"))
        case_id = expected_payload.get("case_id")
        expected_document_type = expected_payload.get("document_type")
        if not case_id or not expected_document_type:
            continue
        if case_ids is not None and case_id not in case_ids:
            continue

        run_result = actual_results.get(case_id)
        if run_result is None:
            if include_missing_cases:
                missing_cases.append(case_id)
            continue

        document_type = DocumentType(expected_document_type)
        target_document = next(
            (
                document
                for document in run_result.extracted_documents
                if document.document_type == document_type
            ),
            None,
        )
        for field_name, expected_value in expected_payload.get("expected_fields", {}).items():
            actual_value = (
                None if target_document is None else target_document.fields.get(field_name)
            )
            comparisons.append(
                FieldComparison(
                    case_id=case_id,
                    document_type=document_type.value,
                    field_name=field_name,
                    expected=None if expected_value is None else str(expected_value),
                    actual=None if actual_value is None else str(actual_value),
                    exact_match=actual_value == expected_value,
                    character_error_rate=character_error_rate(expected_value, actual_value),
                    word_error_rate=word_error_rate(expected_value, actual_value),
                )
            )

    grouped: dict[str, list[FieldComparison]] = {}
    for comparison in comparisons:
        key = f"{comparison.document_type}.{comparison.field_name}"
        grouped.setdefault(key, []).append(comparison)

    metrics: dict[str, FieldMetricSummary] = {}
    for key, values in grouped.items():
        exact_matches = sum(1 for value in values if value.exact_match)
        metrics[key] = FieldMetricSummary(
            comparisons=len(values),
            exact_matches=exact_matches,
            exact_match_rate=exact_matches / len(values),
            average_character_error_rate=sum(value.character_error_rate for value in values)
            / len(values),
            average_word_error_rate=sum(value.word_error_rate for value in values) / len(values),
        )

    return FieldErrorReport(
        compared_cases=len({comparison.case_id for comparison in comparisons}),
        missing_cases=sorted(missing_cases),
        comparisons=comparisons,
        field_metrics=dict(sorted(metrics.items())),
    )


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

    for field_name, expected_value in expected.get("expected_quality_checks", {}).items():
        actual_value = None if target_document is None else target_document.quality_checks.get(
            field_name
        )
        if actual_value != expected_value:
            mismatches.append(
                f"expected quality check {field_name}={expected_value!r}, got {actual_value!r}"
            )

    expected_codes = set(expected.get("expected_finding_codes", []))
    if expected_codes:
        actual_codes = {finding.code for finding in run_result.review_result.findings}
        missing_codes = expected_codes - actual_codes
        if missing_codes:
            mismatches.append(
                "missing finding codes: " + ", ".join(sorted(missing_codes))
            )

    for field_name, expected_value in expected.get("expected_cross_check", {}).items():
        actual_value = run_result.review_result.cross_check.get(field_name)
        if actual_value != expected_value:
            mismatches.append(
                f"expected cross_check {field_name}={expected_value!r}, got {actual_value!r}"
            )

    return EvaluationResult(passed=not mismatches, mismatches=mismatches)
