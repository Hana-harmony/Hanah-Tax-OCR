from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from hanah_tax_ocr.cli import main
from hanah_tax_ocr.evaluation import (
    FieldErrorReport,
    FieldMetricSummary,
    compare_field_error_report_files,
    compare_field_error_reports,
)


def build_metric(
    *,
    comparisons: int,
    exact_matches: int,
    exact_match_rate: float,
    average_character_error_rate: float,
    average_word_error_rate: float,
) -> FieldMetricSummary:
    return FieldMetricSummary(
        comparisons=comparisons,
        exact_matches=exact_matches,
        exact_match_rate=exact_match_rate,
        average_character_error_rate=average_character_error_rate,
        average_word_error_rate=average_word_error_rate,
    )


def test_compare_field_error_reports_classifies_metric_changes() -> None:
    baseline_report = FieldErrorReport(
        compared_cases=4,
        missing_cases=["case_001", "case_002"],
        field_metrics={
            "withholding_tax_form.first_name": build_metric(
                comparisons=4,
                exact_matches=2,
                exact_match_rate=0.5,
                average_character_error_rate=0.25,
                average_word_error_rate=0.5,
            ),
            "withholding_tax_form.last_name": build_metric(
                comparisons=4,
                exact_matches=4,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
            "withholding_tax_form.address": build_metric(
                comparisons=4,
                exact_matches=2,
                exact_match_rate=0.5,
                average_character_error_rate=0.2,
                average_word_error_rate=0.2,
            ),
            "withholding_tax_form.country_code": build_metric(
                comparisons=4,
                exact_matches=4,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
            "withholding_tax_form.tin": build_metric(
                comparisons=3,
                exact_matches=3,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
        },
    )
    candidate_report = FieldErrorReport(
        compared_cases=5,
        missing_cases=["case_002", "case_003"],
        field_metrics={
            "withholding_tax_form.first_name": build_metric(
                comparisons=5,
                exact_matches=4,
                exact_match_rate=0.8,
                average_character_error_rate=0.1,
                average_word_error_rate=0.2,
            ),
            "withholding_tax_form.last_name": build_metric(
                comparisons=5,
                exact_matches=3,
                exact_match_rate=0.6,
                average_character_error_rate=0.3,
                average_word_error_rate=0.4,
            ),
            "withholding_tax_form.address": build_metric(
                comparisons=5,
                exact_matches=2,
                exact_match_rate=0.5,
                average_character_error_rate=0.1,
                average_word_error_rate=0.3,
            ),
            "withholding_tax_form.country_code": build_metric(
                comparisons=5,
                exact_matches=5,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
            "withholding_tax_form.beneficial_owner": build_metric(
                comparisons=5,
                exact_matches=5,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
        },
    )

    comparison = compare_field_error_reports(baseline_report, candidate_report)

    assert comparison.baseline_compared_cases == 4
    assert comparison.candidate_compared_cases == 5
    assert comparison.resolved_missing_cases == ["case_001"]
    assert comparison.new_missing_cases == ["case_003"]
    assert comparison.improved_fields == ["withholding_tax_form.first_name"]
    assert comparison.regressed_fields == ["withholding_tax_form.last_name"]
    assert comparison.mixed_fields == ["withholding_tax_form.address"]
    assert comparison.unchanged_fields == ["withholding_tax_form.country_code"]
    assert comparison.added_fields == ["withholding_tax_form.beneficial_owner"]
    assert comparison.removed_fields == ["withholding_tax_form.tin"]

    first_name_delta = comparison.field_deltas["withholding_tax_form.first_name"]
    assert first_name_delta.status == "improved"
    assert first_name_delta.comparison_delta == 1
    assert first_name_delta.exact_match_rate_delta == pytest.approx(0.3)
    assert first_name_delta.average_character_error_rate_delta == pytest.approx(-0.15)
    assert first_name_delta.average_word_error_rate_delta == pytest.approx(-0.3)

    address_delta = comparison.field_deltas["withholding_tax_form.address"]
    assert address_delta.status == "mixed"
    assert comparison.regressed_documents == ["withholding_tax_form"]
    assert comparison.document_deltas["withholding_tax_form"].status == "regressed"
    assert comparison.regressed_field_groups == ["english_name_org"]
    assert comparison.field_group_deltas["english_name_org"].status == "regressed"
    assert comparison.overall_delta is not None
    assert comparison.overall_delta.status == "regressed"
    assert comparison.promotion_assessment is not None
    assert comparison.promotion_assessment.status == "reject"
    assert comparison.promotion_assessment.blocking_reasons == [
        "new_missing_cases",
        "overall_delta_regressed",
    ]
    assert comparison.promotion_assessment.warning_reasons == [
        "severe_field_regressions",
        "field_regressions",
        "mixed_field_deltas",
        "removed_fields",
    ]
    assert comparison.promotion_assessment.severe_regressed_fields == [
        "withholding_tax_form.last_name"
    ]
    assert comparison.promotion_assessment.notable_improved_fields == [
        "withholding_tax_form.first_name"
    ]


def test_compare_field_error_report_files_and_cli(tmp_path: Path, capsys) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_training_summary_path = tmp_path / "baseline_summary.json"
    candidate_training_summary_path = tmp_path / "candidate_summary.json"
    baseline_path.write_text(
        json.dumps(
            {
                "compared_cases": 1,
                "missing_cases": [],
                "comparisons": [],
                "field_metrics": {
                    "residency_certificate.taxpayer_name": {
                        "comparisons": 1,
                        "exact_matches": 0,
                        "exact_match_rate": 0.0,
                        "average_character_error_rate": 0.5,
                        "average_word_error_rate": 1.0,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "compared_cases": 1,
                "missing_cases": [],
                "comparisons": [],
                "field_metrics": {
                    "residency_certificate.taxpayer_name": {
                        "comparisons": 1,
                        "exact_matches": 1,
                        "exact_match_rate": 1.0,
                        "average_character_error_rate": 0.0,
                        "average_word_error_rate": 0.0,
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    baseline_training_summary_path.write_text(
        json.dumps(
            {
                "groups": {
                    "english_name_org": {
                        "train_count": 12,
                        "val_count": 1,
                        "data_profile": {
                            "unique_source_counts": {"train": 2, "val": 1},
                            "hard_case_train_ratio": 0.5,
                            "unique_hard_case_variant_counts": {"train": 1, "val": 0},
                            "hard_case_selection_strategy": "base_document_balance",
                            "hard_case_variant_floor_applied": False,
                        },
                        "training_readiness": {
                            "status": "blocked",
                            "blocking_warnings": ["no_val_samples"],
                            "advisory_warnings": ["low_train_sample_count"],
                        },
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    candidate_training_summary_path.write_text(
        json.dumps(
            {
                "groups": {
                    "english_name_org": {
                        "train_count": 18,
                        "val_count": 3,
                        "data_profile": {
                            "unique_source_counts": {"train": 3, "val": 2},
                            "hard_case_train_ratio": 0.5,
                            "unique_hard_case_variant_counts": {"train": 3, "val": 0},
                            "hard_case_selection_strategy": "base_document_balance",
                            "hard_case_variant_floor_applied": False,
                        },
                        "training_readiness": {
                            "status": "ready",
                            "blocking_warnings": [],
                            "advisory_warnings": [],
                        },
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison = compare_field_error_report_files(
        baseline_path,
        candidate_path,
        baseline_training_summary_path=baseline_training_summary_path,
        candidate_training_summary_path=candidate_training_summary_path,
    )
    assert comparison.improved_fields == ["residency_certificate.taxpayer_name"]
    assert comparison.improved_documents == ["residency_certificate"]
    assert comparison.improved_field_groups == ["english_name_org"]
    assert comparison.improved_training_field_groups == ["english_name_org"]
    training_delta = comparison.training_profile_deltas["english_name_org"]
    assert training_delta.status == "improved"
    assert training_delta.train_count_delta == 6
    assert training_delta.val_count_delta == 2
    assert training_delta.train_source_count_delta == 1
    assert training_delta.val_source_count_delta == 1
    assert training_delta.unique_hard_case_variant_count_delta == 2
    assert training_delta.readiness_transition == "blocked->ready"
    assert training_delta.readiness_rank_delta == 2
    assert comparison.overall_delta is not None
    assert comparison.overall_delta.status == "improved"

    original_argv = sys.argv
    sys.argv = [
        "hanah-tax-ocr",
        "compare-eval-reports",
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--baseline-training-summary",
        str(baseline_training_summary_path),
        "--candidate-training-summary",
        str(candidate_training_summary_path),
    ]
    try:
        exit_code = main()
    finally:
        sys.argv = original_argv

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["improved_fields"] == ["residency_certificate.taxpayer_name"]
    assert payload["improved_documents"] == ["residency_certificate"]
    assert payload["improved_field_groups"] == ["english_name_org"]
    assert payload["improved_training_field_groups"] == ["english_name_org"]
    assert payload["training_profile_deltas"]["english_name_org"]["status"] == "improved"
    assert (
        payload["training_profile_deltas"]["english_name_org"]["readiness_transition"]
        == "blocked->ready"
    )
    assert payload["overall_delta"]["status"] == "improved"
    assert payload["promotion_assessment"]["status"] == "promote"
    assert payload["promotion_assessment"]["blocking_reasons"] == []
    assert payload["promotion_assessment"]["warning_reasons"] == []


def test_compare_field_error_reports_classifies_training_profile_regressions_and_additions(
) -> None:
    baseline_report = FieldErrorReport(
        compared_cases=1,
        missing_cases=[],
        field_metrics={
            "apostille.signed_by": build_metric(
                comparisons=1,
                exact_matches=1,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            )
        },
    )
    candidate_report = FieldErrorReport(
        compared_cases=1,
        missing_cases=[],
        field_metrics={
            "apostille.signed_by": build_metric(
                comparisons=1,
                exact_matches=1,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            )
        },
    )

    comparison = compare_field_error_reports(
        baseline_report,
        candidate_report,
        baseline_training_profiles={
            "english_name_org": {
                "field_group": "english_name_org",
                "train_count": 20,
                "val_count": 3,
                "train_source_count": 3,
                "val_source_count": 2,
                "hard_case_train_ratio": 0.5,
                "unique_hard_case_variant_count": 4,
                "training_readiness_status": "ready",
            },
            "numeric_tin_code": {
                "field_group": "numeric_tin_code",
                "train_count": 22,
                "val_count": 0,
                "train_source_count": 3,
                "val_source_count": 0,
                "hard_case_train_ratio": 0.5,
                "unique_hard_case_variant_count": 4,
                "training_readiness_status": "blocked",
            },
        },
        candidate_training_profiles={
            "english_name_org": {
                "field_group": "english_name_org",
                "train_count": 18,
                "val_count": 2,
                "train_source_count": 3,
                "val_source_count": 1,
                "hard_case_train_ratio": 0.5,
                "unique_hard_case_variant_count": 3,
                "training_readiness_status": "review_required",
            },
            "korean_mixed_form": {
                "field_group": "korean_mixed_form",
                "train_count": 3,
                "val_count": 1,
                "train_source_count": 1,
                "val_source_count": 1,
                "hard_case_train_ratio": 0.6667,
                "unique_hard_case_variant_count": 2,
                "training_readiness_status": "review_required",
            },
        },
    )

    assert comparison.regressed_training_field_groups == ["english_name_org"]
    assert comparison.added_training_field_groups == ["korean_mixed_form"]
    assert comparison.removed_training_field_groups == ["numeric_tin_code"]
    assert comparison.training_profile_deltas["english_name_org"].status == "regressed"
    assert comparison.training_profile_deltas["english_name_org"].readiness_transition == (
        "ready->review_required"
    )
    assert comparison.training_profile_deltas["korean_mixed_form"].status == "added"
    assert comparison.training_profile_deltas["numeric_tin_code"].status == "removed"


def test_compare_field_error_reports_rolls_up_document_level_metrics() -> None:
    baseline_report = FieldErrorReport(
        compared_cases=2,
        missing_cases=[],
        field_metrics={
            "residency_certificate.taxpayer_name": build_metric(
                comparisons=2,
                exact_matches=0,
                exact_match_rate=0.0,
                average_character_error_rate=0.4,
                average_word_error_rate=0.5,
            ),
            "apostille.signed_by": build_metric(
                comparisons=2,
                exact_matches=2,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
        },
    )
    candidate_report = FieldErrorReport(
        compared_cases=3,
        missing_cases=[],
        field_metrics={
            "residency_certificate.taxpayer_name": build_metric(
                comparisons=2,
                exact_matches=2,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
            "apostille.signed_by": build_metric(
                comparisons=2,
                exact_matches=1,
                exact_match_rate=0.5,
                average_character_error_rate=0.2,
                average_word_error_rate=0.5,
            ),
            "withholding_tax_form.tin": build_metric(
                comparisons=1,
                exact_matches=1,
                exact_match_rate=1.0,
                average_character_error_rate=0.0,
                average_word_error_rate=0.0,
            ),
        },
    )

    comparison = compare_field_error_reports(baseline_report, candidate_report)

    assert comparison.improved_documents == ["residency_certificate"]
    assert comparison.regressed_documents == ["apostille"]
    assert comparison.added_documents == ["withholding_tax_form"]
    assert comparison.improved_field_groups == ["english_name_org"]
    assert comparison.regressed_field_groups == []
    assert comparison.added_field_groups == ["numeric_tin_code"]
    assert comparison.field_group_deltas["english_name_org"].status == "improved"
    assert comparison.field_group_deltas["numeric_tin_code"].status == "added"
    assert comparison.document_deltas["residency_certificate"].status == "improved"
    assert comparison.document_deltas["apostille"].status == "regressed"
    assert comparison.document_deltas["withholding_tax_form"].status == "added"
    assert comparison.overall_delta is not None
    assert comparison.overall_delta.status == "improved"
    assert comparison.promotion_assessment is not None
    assert comparison.promotion_assessment.status == "review"
    assert comparison.promotion_assessment.blocking_reasons == []
    assert comparison.promotion_assessment.warning_reasons == [
        "severe_field_regressions",
        "field_regressions",
    ]
