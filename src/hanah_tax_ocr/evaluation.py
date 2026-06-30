from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from hanah_tax_ocr.harness import HarnessRunResult
from hanah_tax_ocr.schemas import DocumentType
from hanah_tax_ocr.training.field_crops import field_group_for


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


class EvalPromotionAssessment(BaseModel):
    status: Literal["promote", "review", "reject"]
    blocking_reasons: list[str] = Field(default_factory=list)
    warning_reasons: list[str] = Field(default_factory=list)
    severe_regressed_fields: list[str] = Field(default_factory=list)
    coverage_regressed_fields: list[str] = Field(default_factory=list)
    notable_improved_fields: list[str] = Field(default_factory=list)


class RecognizerTrainingProfile(BaseModel):
    field_group: str
    train_count: int = 0
    val_count: int = 0
    train_source_count: int = 0
    val_source_count: int = 0
    hard_case_train_ratio: float = 0.0
    unique_hard_case_variant_count: int = 0
    training_readiness_status: str = ""
    blocking_warnings: list[str] = Field(default_factory=list)
    advisory_warnings: list[str] = Field(default_factory=list)
    hard_case_selection_strategy: str = ""
    hard_case_variant_floor_applied: bool = False


class RecognizerTrainingProfileDelta(BaseModel):
    field_group: str
    status: Literal["added", "removed", "improved", "regressed", "mixed", "unchanged"]
    baseline: RecognizerTrainingProfile | None = None
    candidate: RecognizerTrainingProfile | None = None
    train_count_delta: int | None = None
    val_count_delta: int | None = None
    train_source_count_delta: int | None = None
    val_source_count_delta: int | None = None
    hard_case_train_ratio_delta: float | None = None
    unique_hard_case_variant_count_delta: int | None = None
    readiness_rank_delta: int | None = None
    readiness_transition: str | None = None


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
    improved_documents: list[str] = Field(default_factory=list)
    regressed_documents: list[str] = Field(default_factory=list)
    mixed_documents: list[str] = Field(default_factory=list)
    unchanged_documents: list[str] = Field(default_factory=list)
    added_documents: list[str] = Field(default_factory=list)
    removed_documents: list[str] = Field(default_factory=list)
    document_deltas: dict[str, FieldMetricDelta] = Field(default_factory=dict)
    improved_field_groups: list[str] = Field(default_factory=list)
    regressed_field_groups: list[str] = Field(default_factory=list)
    mixed_field_groups: list[str] = Field(default_factory=list)
    unchanged_field_groups: list[str] = Field(default_factory=list)
    added_field_groups: list[str] = Field(default_factory=list)
    removed_field_groups: list[str] = Field(default_factory=list)
    field_group_deltas: dict[str, FieldMetricDelta] = Field(default_factory=dict)
    improved_training_field_groups: list[str] = Field(default_factory=list)
    regressed_training_field_groups: list[str] = Field(default_factory=list)
    mixed_training_field_groups: list[str] = Field(default_factory=list)
    unchanged_training_field_groups: list[str] = Field(default_factory=list)
    added_training_field_groups: list[str] = Field(default_factory=list)
    removed_training_field_groups: list[str] = Field(default_factory=list)
    training_profile_deltas: dict[str, RecognizerTrainingProfileDelta] = Field(
        default_factory=dict
    )
    overall_delta: FieldMetricDelta | None = None
    promotion_assessment: EvalPromotionAssessment | None = None


def load_harness_run_result(path: str | Path) -> HarnessRunResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return HarnessRunResult.model_validate(payload)


def load_field_error_report(path: str | Path) -> FieldErrorReport:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FieldErrorReport.model_validate(payload)


def load_recognizer_training_profiles(
    path: str | Path,
) -> dict[str, RecognizerTrainingProfile]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = payload.get("groups", {})
    profiles: dict[str, RecognizerTrainingProfile] = {}
    for field_group, group_payload in sorted(groups.items()):
        if not isinstance(group_payload, dict):
            continue
        data_profile = group_payload.get("data_profile", {})
        training_readiness = group_payload.get("training_readiness", {})
        unique_variant_counts = data_profile.get("unique_hard_case_variant_counts", {})
        unique_source_counts = data_profile.get("unique_source_counts", {})
        profiles[field_group] = RecognizerTrainingProfile(
            field_group=field_group,
            train_count=int(group_payload.get("train_count", 0) or 0),
            val_count=int(group_payload.get("val_count", 0) or 0),
            train_source_count=int(unique_source_counts.get("train", 0) or 0),
            val_source_count=int(unique_source_counts.get("val", 0) or 0),
            hard_case_train_ratio=float(data_profile.get("hard_case_train_ratio", 0.0) or 0.0),
            unique_hard_case_variant_count=int(unique_variant_counts.get("train", 0) or 0),
            training_readiness_status=str(training_readiness.get("status", "") or ""),
            blocking_warnings=list(training_readiness.get("blocking_warnings", []) or []),
            advisory_warnings=list(training_readiness.get("advisory_warnings", []) or []),
            hard_case_selection_strategy=str(
                data_profile.get("hard_case_selection_strategy", "") or ""
            ),
            hard_case_variant_floor_applied=bool(
                data_profile.get("hard_case_variant_floor_applied", False)
            ),
        )
    return profiles


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


def _compare_metric_summary_maps(
    baseline_metrics: dict[str, FieldMetricSummary],
    candidate_metrics: dict[str, FieldMetricSummary],
) -> dict[str, Any]:
    metric_deltas: dict[str, FieldMetricDelta] = {}
    improved_keys: list[str] = []
    regressed_keys: list[str] = []
    mixed_keys: list[str] = []
    unchanged_keys: list[str] = []
    added_keys: list[str] = []
    removed_keys: list[str] = []

    all_metric_keys = sorted(set(baseline_metrics) | set(candidate_metrics))
    for metric_key in all_metric_keys:
        baseline_metric = baseline_metrics.get(metric_key)
        candidate_metric = candidate_metrics.get(metric_key)

        if baseline_metric is None:
            added_keys.append(metric_key)
            metric_deltas[metric_key] = FieldMetricDelta(
                field_key=metric_key,
                status="added",
                candidate=candidate_metric,
            )
            continue

        if candidate_metric is None:
            removed_keys.append(metric_key)
            metric_deltas[metric_key] = FieldMetricDelta(
                field_key=metric_key,
                status="removed",
                baseline=baseline_metric,
            )
            continue

        status = _classify_metric_delta(baseline_metric, candidate_metric)
        if status == "improved":
            improved_keys.append(metric_key)
        elif status == "regressed":
            regressed_keys.append(metric_key)
        elif status == "mixed":
            mixed_keys.append(metric_key)
        else:
            unchanged_keys.append(metric_key)

        metric_deltas[metric_key] = FieldMetricDelta(
            field_key=metric_key,
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

    return {
        "improved_keys": improved_keys,
        "regressed_keys": regressed_keys,
        "mixed_keys": mixed_keys,
        "unchanged_keys": unchanged_keys,
        "added_keys": added_keys,
        "removed_keys": removed_keys,
        "metric_deltas": metric_deltas,
    }


def _is_severe_regression(delta: FieldMetricDelta) -> bool:
    if delta.status != "regressed":
        return False
    exact_match_rate_delta = delta.exact_match_rate_delta or 0.0
    character_error_rate_delta = delta.average_character_error_rate_delta or 0.0
    word_error_rate_delta = delta.average_word_error_rate_delta or 0.0
    return (
        exact_match_rate_delta <= -0.05
        or character_error_rate_delta >= 0.05
        or word_error_rate_delta >= 0.1
    )


def _is_notable_improvement(delta: FieldMetricDelta) -> bool:
    if delta.status != "improved":
        return False
    exact_match_rate_delta = delta.exact_match_rate_delta or 0.0
    character_error_rate_delta = delta.average_character_error_rate_delta or 0.0
    word_error_rate_delta = delta.average_word_error_rate_delta or 0.0
    return (
        exact_match_rate_delta >= 0.05
        or character_error_rate_delta <= -0.05
        or word_error_rate_delta <= -0.1
    )


def _build_promotion_assessment(
    *,
    baseline_compared_cases: int,
    candidate_compared_cases: int,
    new_missing_cases: list[str],
    regressed_fields: list[str],
    mixed_fields: list[str],
    removed_fields: list[str],
    overall_delta: FieldMetricDelta | None,
    field_deltas: dict[str, FieldMetricDelta],
) -> EvalPromotionAssessment:
    severe_regressed_fields = sorted(
        field_key
        for field_key in regressed_fields
        if _is_severe_regression(field_deltas[field_key])
    )
    coverage_regressed_fields = sorted(
        field_key
        for field_key, delta in field_deltas.items()
        if delta.baseline is not None
        and delta.candidate is not None
        and (delta.comparison_delta or 0) < 0
    )
    notable_improved_fields = sorted(
        field_key
        for field_key, delta in field_deltas.items()
        if _is_notable_improvement(delta)
    )

    blocking_reasons: list[str] = []
    if new_missing_cases:
        blocking_reasons.append("new_missing_cases")
    if candidate_compared_cases < baseline_compared_cases:
        blocking_reasons.append("candidate_compared_cases_dropped")
    if overall_delta is not None and overall_delta.status == "regressed":
        blocking_reasons.append("overall_delta_regressed")

    warning_reasons: list[str] = []
    if severe_regressed_fields:
        warning_reasons.append("severe_field_regressions")
    if regressed_fields and "severe_field_regressions" not in blocking_reasons:
        warning_reasons.append("field_regressions")
    if mixed_fields:
        warning_reasons.append("mixed_field_deltas")
    if removed_fields:
        warning_reasons.append("removed_fields")
    if coverage_regressed_fields and "candidate_compared_cases_dropped" not in blocking_reasons:
        warning_reasons.append("reduced_field_coverage")
    if overall_delta is not None and overall_delta.status == "mixed":
        warning_reasons.append("overall_delta_mixed")

    if blocking_reasons:
        status: Literal["promote", "review", "reject"] = "reject"
    elif warning_reasons:
        status = "review"
    else:
        status = "promote"

    return EvalPromotionAssessment(
        status=status,
        blocking_reasons=blocking_reasons,
        warning_reasons=warning_reasons,
        severe_regressed_fields=severe_regressed_fields,
        coverage_regressed_fields=coverage_regressed_fields,
        notable_improved_fields=notable_improved_fields,
    )


def _combine_metric_summaries(
    summaries: list[FieldMetricSummary],
) -> FieldMetricSummary | None:
    if not summaries:
        return None

    comparisons = sum(summary.comparisons for summary in summaries)
    if comparisons <= 0:
        return FieldMetricSummary(
            comparisons=0,
            exact_matches=0,
            exact_match_rate=0.0,
            average_character_error_rate=0.0,
            average_word_error_rate=0.0,
        )

    exact_matches = sum(summary.exact_matches for summary in summaries)
    return FieldMetricSummary(
        comparisons=comparisons,
        exact_matches=exact_matches,
        exact_match_rate=exact_matches / comparisons,
        average_character_error_rate=sum(
            summary.average_character_error_rate * summary.comparisons
            for summary in summaries
        )
        / comparisons,
        average_word_error_rate=sum(
            summary.average_word_error_rate * summary.comparisons
            for summary in summaries
        )
        / comparisons,
    )


def _aggregate_metric_summaries(
    metrics: dict[str, FieldMetricSummary],
    key_builder,
) -> dict[str, FieldMetricSummary]:
    grouped: dict[str, list[FieldMetricSummary]] = {}
    for metric_key, summary in metrics.items():
        grouped.setdefault(str(key_builder(metric_key)), []).append(summary)

    aggregated: dict[str, FieldMetricSummary] = {}
    for aggregate_key, summaries in grouped.items():
        combined = _combine_metric_summaries(summaries)
        if combined is not None:
            aggregated[aggregate_key] = combined
    return dict(sorted(aggregated.items()))


def _document_type_for_field_key(field_key: str) -> str:
    document_type, _, _ = field_key.partition(".")
    return document_type or field_key


def _field_name_for_field_key(field_key: str) -> str:
    _, _, field_name = field_key.partition(".")
    return field_name or field_key


def _field_group_for_field_key(field_key: str) -> str:
    return field_group_for(_field_name_for_field_key(field_key))


def _training_readiness_rank(status: str) -> int:
    return {
        "blocked": 0,
        "review_required": 1,
        "ready": 2,
    }.get(status, -1)


def _classify_training_profile_delta(
    baseline: RecognizerTrainingProfile,
    candidate: RecognizerTrainingProfile,
) -> str:
    positive_signals = 0
    negative_signals = 0

    readiness_delta = _training_readiness_rank(candidate.training_readiness_status) - (
        _training_readiness_rank(baseline.training_readiness_status)
    )
    if readiness_delta > 0:
        positive_signals += 1
    elif readiness_delta < 0:
        negative_signals += 1

    for delta in (
        candidate.train_count - baseline.train_count,
        candidate.val_count - baseline.val_count,
        candidate.train_source_count - baseline.train_source_count,
        candidate.val_source_count - baseline.val_source_count,
        candidate.unique_hard_case_variant_count - baseline.unique_hard_case_variant_count,
    ):
        if delta > 0:
            positive_signals += 1
        elif delta < 0:
            negative_signals += 1

    if positive_signals and negative_signals:
        return "mixed"
    if positive_signals:
        return "improved"
    if negative_signals:
        return "regressed"
    return "unchanged"


def _compare_training_profiles(
    baseline_profiles: dict[str, RecognizerTrainingProfile],
    candidate_profiles: dict[str, RecognizerTrainingProfile],
) -> dict[str, Any]:
    deltas: dict[str, RecognizerTrainingProfileDelta] = {}
    improved_keys: list[str] = []
    regressed_keys: list[str] = []
    mixed_keys: list[str] = []
    unchanged_keys: list[str] = []
    added_keys: list[str] = []
    removed_keys: list[str] = []

    all_keys = sorted(set(baseline_profiles) | set(candidate_profiles))
    for field_group in all_keys:
        baseline_profile = baseline_profiles.get(field_group)
        candidate_profile = candidate_profiles.get(field_group)
        if baseline_profile is None:
            added_keys.append(field_group)
            deltas[field_group] = RecognizerTrainingProfileDelta(
                field_group=field_group,
                status="added",
                candidate=candidate_profile,
            )
            continue
        if candidate_profile is None:
            removed_keys.append(field_group)
            deltas[field_group] = RecognizerTrainingProfileDelta(
                field_group=field_group,
                status="removed",
                baseline=baseline_profile,
            )
            continue

        status = _classify_training_profile_delta(baseline_profile, candidate_profile)
        if status == "improved":
            improved_keys.append(field_group)
        elif status == "regressed":
            regressed_keys.append(field_group)
        elif status == "mixed":
            mixed_keys.append(field_group)
        else:
            unchanged_keys.append(field_group)

        deltas[field_group] = RecognizerTrainingProfileDelta(
            field_group=field_group,
            status=status,
            baseline=baseline_profile,
            candidate=candidate_profile,
            train_count_delta=candidate_profile.train_count - baseline_profile.train_count,
            val_count_delta=candidate_profile.val_count - baseline_profile.val_count,
            train_source_count_delta=(
                candidate_profile.train_source_count - baseline_profile.train_source_count
            ),
            val_source_count_delta=(
                candidate_profile.val_source_count - baseline_profile.val_source_count
            ),
            hard_case_train_ratio_delta=round(
                candidate_profile.hard_case_train_ratio - baseline_profile.hard_case_train_ratio,
                4,
            ),
            unique_hard_case_variant_count_delta=(
                candidate_profile.unique_hard_case_variant_count
                - baseline_profile.unique_hard_case_variant_count
            ),
            readiness_rank_delta=(
                _training_readiness_rank(candidate_profile.training_readiness_status)
                - _training_readiness_rank(baseline_profile.training_readiness_status)
            ),
            readiness_transition=(
                f"{baseline_profile.training_readiness_status}"
                f"->{candidate_profile.training_readiness_status}"
            ),
        )

    return {
        "improved_keys": improved_keys,
        "regressed_keys": regressed_keys,
        "mixed_keys": mixed_keys,
        "unchanged_keys": unchanged_keys,
        "added_keys": added_keys,
        "removed_keys": removed_keys,
        "deltas": deltas,
    }


def _normalize_training_profiles(
    profiles: dict[str, RecognizerTrainingProfile | dict[str, Any]] | None,
) -> dict[str, RecognizerTrainingProfile]:
    normalized: dict[str, RecognizerTrainingProfile] = {}
    for field_group, payload in (profiles or {}).items():
        if isinstance(payload, RecognizerTrainingProfile):
            normalized[field_group] = payload
            continue
        normalized[field_group] = RecognizerTrainingProfile.model_validate(
            {
                "field_group": field_group,
                **payload,
            }
        )
    return normalized


def compare_field_error_reports(
    baseline_report: FieldErrorReport,
    candidate_report: FieldErrorReport,
    *,
    baseline_training_profiles: dict[str, RecognizerTrainingProfile | dict[str, Any]] | None = None,
    candidate_training_profiles: (
        dict[str, RecognizerTrainingProfile | dict[str, Any]] | None
    ) = None,
) -> FieldErrorReportComparison:
    field_comparison = _compare_metric_summary_maps(
        baseline_report.field_metrics,
        candidate_report.field_metrics,
    )
    document_comparison = _compare_metric_summary_maps(
        _aggregate_metric_summaries(
            baseline_report.field_metrics,
            _document_type_for_field_key,
        ),
        _aggregate_metric_summaries(
            candidate_report.field_metrics,
            _document_type_for_field_key,
        ),
    )
    field_group_comparison = _compare_metric_summary_maps(
        _aggregate_metric_summaries(
            baseline_report.field_metrics,
            _field_group_for_field_key,
        ),
        _aggregate_metric_summaries(
            candidate_report.field_metrics,
            _field_group_for_field_key,
        ),
    )
    overall_comparison = _compare_metric_summary_maps(
        (
            {"overall": summary}
            if (
                summary := _combine_metric_summaries(
                    list(baseline_report.field_metrics.values())
                )
            )
            is not None
            else {}
        ),
        (
            {"overall": summary}
            if (
                summary := _combine_metric_summaries(
                    list(candidate_report.field_metrics.values())
                )
            )
            is not None
            else {}
        ),
    )

    baseline_missing_cases = sorted(baseline_report.missing_cases)
    candidate_missing_cases = sorted(candidate_report.missing_cases)
    training_profile_comparison = _compare_training_profiles(
        _normalize_training_profiles(baseline_training_profiles),
        _normalize_training_profiles(candidate_training_profiles),
    )

    promotion_assessment = _build_promotion_assessment(
        baseline_compared_cases=baseline_report.compared_cases,
        candidate_compared_cases=candidate_report.compared_cases,
        new_missing_cases=sorted(set(candidate_missing_cases) - set(baseline_missing_cases)),
        regressed_fields=field_comparison["regressed_keys"],
        mixed_fields=field_comparison["mixed_keys"],
        removed_fields=field_comparison["removed_keys"],
        overall_delta=overall_comparison["metric_deltas"].get("overall"),
        field_deltas=field_comparison["metric_deltas"],
    )

    return FieldErrorReportComparison(
        baseline_compared_cases=baseline_report.compared_cases,
        candidate_compared_cases=candidate_report.compared_cases,
        baseline_missing_cases=baseline_missing_cases,
        candidate_missing_cases=candidate_missing_cases,
        resolved_missing_cases=sorted(
            set(baseline_missing_cases) - set(candidate_missing_cases)
        ),
        new_missing_cases=sorted(set(candidate_missing_cases) - set(baseline_missing_cases)),
        improved_fields=field_comparison["improved_keys"],
        regressed_fields=field_comparison["regressed_keys"],
        mixed_fields=field_comparison["mixed_keys"],
        unchanged_fields=field_comparison["unchanged_keys"],
        added_fields=field_comparison["added_keys"],
        removed_fields=field_comparison["removed_keys"],
        field_deltas=field_comparison["metric_deltas"],
        improved_documents=document_comparison["improved_keys"],
        regressed_documents=document_comparison["regressed_keys"],
        mixed_documents=document_comparison["mixed_keys"],
        unchanged_documents=document_comparison["unchanged_keys"],
        added_documents=document_comparison["added_keys"],
        removed_documents=document_comparison["removed_keys"],
        document_deltas=document_comparison["metric_deltas"],
        improved_field_groups=field_group_comparison["improved_keys"],
        regressed_field_groups=field_group_comparison["regressed_keys"],
        mixed_field_groups=field_group_comparison["mixed_keys"],
        unchanged_field_groups=field_group_comparison["unchanged_keys"],
        added_field_groups=field_group_comparison["added_keys"],
        removed_field_groups=field_group_comparison["removed_keys"],
        field_group_deltas=field_group_comparison["metric_deltas"],
        improved_training_field_groups=training_profile_comparison["improved_keys"],
        regressed_training_field_groups=training_profile_comparison["regressed_keys"],
        mixed_training_field_groups=training_profile_comparison["mixed_keys"],
        unchanged_training_field_groups=training_profile_comparison["unchanged_keys"],
        added_training_field_groups=training_profile_comparison["added_keys"],
        removed_training_field_groups=training_profile_comparison["removed_keys"],
        training_profile_deltas=training_profile_comparison["deltas"],
        overall_delta=overall_comparison["metric_deltas"].get("overall"),
        promotion_assessment=promotion_assessment,
    )


def compare_field_error_report_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    baseline_training_summary_path: str | Path | None = None,
    candidate_training_summary_path: str | Path | None = None,
) -> FieldErrorReportComparison:
    baseline_report = load_field_error_report(baseline_path)
    candidate_report = load_field_error_report(candidate_path)
    baseline_training_profiles = (
        load_recognizer_training_profiles(baseline_training_summary_path)
        if baseline_training_summary_path is not None
        else None
    )
    candidate_training_profiles = (
        load_recognizer_training_profiles(candidate_training_summary_path)
        if candidate_training_summary_path is not None
        else None
    )
    return compare_field_error_reports(
        baseline_report,
        candidate_report,
        baseline_training_profiles=baseline_training_profiles,
        candidate_training_profiles=candidate_training_profiles,
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
