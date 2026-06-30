from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

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


class FieldMetricDelta(BaseModel):
    field_key: str
    status: Literal["added", "removed", "improved", "regressed", "mixed", "unchanged"]
    baseline: FieldMetricSummary | None = None
    candidate: FieldMetricSummary | None = None
    comparison_delta: int | None = None
    exact_match_rate_delta: float | None = None
    average_character_error_rate_delta: float | None = None
    average_word_error_rate_delta: float | None = None


class FieldErrorReportComparison(BaseModel):
    baseline_compared_cases: int
    candidate_compared_cases: int
    baseline_missing_cases: list[str] = Field(default_factory=list)
    candidate_missing_cases: list[str] = Field(default_factory=list)
    resolved_missing_cases: list[str] = Field(default_factory=list)
    new_missing_cases: list[str] = Field(default_factory=list)
    improved_fields: list[str] = Field(default_factory=list)
    regressed_fields: list[str] = Field(default_factory=list)
    mixed_fields: list[str] = Field(default_factory=list)
    unchanged_fields: list[str] = Field(default_factory=list)
    added_fields: list[str] = Field(default_factory=list)
    removed_fields: list[str] = Field(default_factory=list)
    field_deltas: dict[str, FieldMetricDelta] = Field(default_factory=dict)


def load_harness_run_result(path: str | Path) -> HarnessRunResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HarnessRunResult.model_validate(payload)


def load_field_error_report(path: str | Path) -> FieldErrorReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FieldErrorReport.model_validate(payload)


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


def _classify_metric_delta(
    baseline: FieldMetricSummary,
    candidate: FieldMetricSummary,
    *,
    tolerance: float = 1e-9,
) -> str:
    improvement_signals: list[str] = []
    regression_signals: list[str] = []

    exact_match_rate_delta = candidate.exact_match_rate - baseline.exact_match_rate
    if exact_match_rate_delta > tolerance:
        improvement_signals.append("exact_match_rate")
    elif exact_match_rate_delta < -tolerance:
        regression_signals.append("exact_match_rate")

    character_error_rate_delta = (
        candidate.average_character_error_rate - baseline.average_character_error_rate
    )
    if character_error_rate_delta < -tolerance:
        improvement_signals.append("character_error_rate")
    elif character_error_rate_delta > tolerance:
        regression_signals.append("character_error_rate")

    word_error_rate_delta = candidate.average_word_error_rate - baseline.average_word_error_rate
    if word_error_rate_delta < -tolerance:
        improvement_signals.append("word_error_rate")
    elif word_error_rate_delta > tolerance:
        regression_signals.append("word_error_rate")

    if improvement_signals and regression_signals:
        return "mixed"
    if improvement_signals:
        return "improved"
    if regression_signals:
        return "regressed"
    return "unchanged"


def compare_field_error_reports(
    baseline_report: FieldErrorReport,
    candidate_report: FieldErrorReport,
) -> FieldErrorReportComparison:
    field_deltas: dict[str, FieldMetricDelta] = {}
    improved_fields: list[str] = []
    regressed_fields: list[str] = []
    mixed_fields: list[str] = []
    unchanged_fields: list[str] = []
    added_fields: list[str] = []
    removed_fields: list[str] = []

    all_field_keys = sorted(
        set(baseline_report.field_metrics) | set(candidate_report.field_metrics)
    )
    for field_key in all_field_keys:
        baseline_metric = baseline_report.field_metrics.get(field_key)
        candidate_metric = candidate_report.field_metrics.get(field_key)

        if baseline_metric is None:
            added_fields.append(field_key)
            field_deltas[field_key] = FieldMetricDelta(
                field_key=field_key,
                status="added",
                candidate=candidate_metric,
            )
            continue

        if candidate_metric is None:
            removed_fields.append(field_key)
            field_deltas[field_key] = FieldMetricDelta(
                field_key=field_key,
                status="removed",
                baseline=baseline_metric,
            )
            continue

        status = _classify_metric_delta(baseline_metric, candidate_metric)
        if status == "improved":
            improved_fields.append(field_key)
        elif status == "regressed":
            regressed_fields.append(field_key)
        elif status == "mixed":
            mixed_fields.append(field_key)
        else:
            unchanged_fields.append(field_key)

        field_deltas[field_key] = FieldMetricDelta(
            field_key=field_key,
            status=status,
            baseline=baseline_metric,
            candidate=candidate_metric,
            comparison_delta=candidate_metric.comparisons - baseline_metric.comparisons,
            exact_match_rate_delta=candidate_metric.exact_match_rate
            - baseline_metric.exact_match_rate,
            average_character_error_rate_delta=candidate_metric.average_character_error_rate
            - baseline_metric.average_character_error_rate,
            average_word_error_rate_delta=candidate_metric.average_word_error_rate
            - baseline_metric.average_word_error_rate,
        )

    baseline_missing_cases = sorted(baseline_report.missing_cases)
    candidate_missing_cases = sorted(candidate_report.missing_cases)

    return FieldErrorReportComparison(
        baseline_compared_cases=baseline_report.compared_cases,
        candidate_compared_cases=candidate_report.compared_cases,
        baseline_missing_cases=baseline_missing_cases,
        candidate_missing_cases=candidate_missing_cases,
        resolved_missing_cases=sorted(
            set(baseline_missing_cases) - set(candidate_missing_cases)
        ),
        new_missing_cases=sorted(set(candidate_missing_cases) - set(baseline_missing_cases)),
        improved_fields=improved_fields,
        regressed_fields=regressed_fields,
        mixed_fields=mixed_fields,
        unchanged_fields=unchanged_fields,
        added_fields=added_fields,
        removed_fields=removed_fields,
        field_deltas=field_deltas,
    )


def compare_field_error_report_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
) -> FieldErrorReportComparison:
    baseline_report = load_field_error_report(baseline_path)
    candidate_report = load_field_error_report(candidate_path)
    return compare_field_error_reports(baseline_report, candidate_report)


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
