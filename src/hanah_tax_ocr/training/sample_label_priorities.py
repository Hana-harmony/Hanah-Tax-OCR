from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from hanah_tax_ocr.training.data_gaps import DEFAULT_OUTPUT_PATH as DEFAULT_DATA_GAP_REPORT_PATH
from hanah_tax_ocr.training.data_gaps import load_json
from hanah_tax_ocr.training.field_crops import field_group_for
from hanah_tax_ocr.training.sample_coverage import (
    DEFAULT_LABELED_ROOT,
    PENDING_REVIEW_SPLIT,
)

DEFAULT_COVERAGE_REPORT_PATH = Path("data/training/reports/sample_data_coverage.json")
DEFAULT_OUTPUT_PATH = Path("data/training/reports/sample_label_priority.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prioritize sample_data labels using current recognizer data-gap pressure."
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=DEFAULT_COVERAGE_REPORT_PATH,
    )
    parser.add_argument(
        "--data-gap-report",
        type=Path,
        default=DEFAULT_DATA_GAP_REPORT_PATH,
    )
    parser.add_argument(
        "--labeled-root",
        type=Path,
        default=DEFAULT_LABELED_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def _reviewed_document_field_groups(labeled_root: Path) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for label_path in sorted(labeled_root.rglob("label.json")):
        try:
            relative_path = label_path.relative_to(labeled_root)
        except ValueError:
            continue
        if relative_path.parts and relative_path.parts[0] == PENDING_REVIEW_SPLIT:
            continue

        payload = load_json(label_path)
        document_type = payload.get("document_type")
        expected_fields = payload.get("expected_fields") or {}
        if not document_type or not expected_fields:
            continue

        grouped[str(document_type)].update(
            field_group_for(field_name) for field_name in expected_fields
        )
    return {
        document_type: sorted(field_groups)
        for document_type, field_groups in sorted(grouped.items())
    }


def _group_recommendations(
    matched_field_groups: list[str],
    priority_by_group: dict[str, dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []
    for field_group in matched_field_groups:
        for recommendation in priority_by_group[field_group].get("recommendations", []):
            if recommendation not in recommendations:
                recommendations.append(recommendation)
    return recommendations


def _readiness_blocking_warnings(
    priority_by_group: dict[str, dict[str, Any]],
    field_group: str,
) -> list[str]:
    return (
        priority_by_group[field_group]
        .get("recognizer_profile", {})
        .get("training_readiness", {})
        .get("blocking_warnings", [])
    )


def build_sample_label_priority_report(
    coverage_report_path: Path,
    data_gap_report_path: Path,
    *,
    labeled_root: Path = DEFAULT_LABELED_ROOT,
) -> dict[str, Any]:
    coverage_report = load_json(coverage_report_path)
    data_gap_report = load_json(data_gap_report_path)
    priority_by_group = {
        item["field_group"]: item for item in data_gap_report.get("priorities", [])
    }
    document_field_groups = _reviewed_document_field_groups(labeled_root)

    samples: list[dict[str, Any]] = []
    for sample in coverage_report.get("samples", []):
        if sample.get("covered_by_labeled"):
            continue

        document_type = sample.get("sample_dataset_document_type")
        case_id = sample.get("sample_dataset_case_id")
        sample_split = sample.get("sample_dataset_split")
        pending_review_case_ids = list(sample.get("pending_review_case_ids", []))
        matched_field_groups = sorted(
            [
                field_group
                for field_group in document_field_groups.get(str(document_type), [])
                if field_group in priority_by_group
            ],
            key=lambda field_group: (
                -priority_by_group[field_group]["priority_score"],
                field_group,
            ),
        )
        blocked_field_groups = [
            field_group
            for field_group in matched_field_groups
            if priority_by_group[field_group]
            .get("recognizer_profile", {})
            .get("training_readiness", {})
            .get("status")
            == "blocked"
        ]

        gap_score = round(
            sum(
                priority_by_group[field_group]["priority_score"]
                for field_group in matched_field_groups
            ),
            4,
        )
        blocked_boost = round(len(blocked_field_groups) * 5.0, 4)
        split_gap_boost = 0.0
        if sample_split == "train":
            split_gap_boost = round(
                sum(
                    priority_by_group[field_group]
                    .get("score_breakdown", {})
                    .get("train_gap", 0.0)
                    + priority_by_group[field_group]
                    .get("score_breakdown", {})
                    .get("train_source_gap", 0.0)
                    for field_group in matched_field_groups
                ),
                4,
            )
        elif sample_split == "val":
            split_gap_boost = round(
                sum(
                    priority_by_group[field_group]
                    .get("score_breakdown", {})
                    .get("val_gap", 0.0)
                    + priority_by_group[field_group]
                    .get("score_breakdown", {})
                    .get("val_source_gap", 0.0)
                    for field_group in matched_field_groups
                ),
                4,
            )
        blocking_split_boost = 0.0
        if sample_split == "train":
            blocking_split_boost = round(
                sum(
                    5.0
                    for field_group in matched_field_groups
                    if "no_train_samples"
                    in _readiness_blocking_warnings(priority_by_group, field_group)
                ),
                4,
            )
        elif sample_split == "val":
            blocking_split_boost = round(
                sum(
                    5.0
                    for field_group in matched_field_groups
                    if "no_val_samples"
                    in _readiness_blocking_warnings(priority_by_group, field_group)
                ),
                4,
            )
        queue_ready_boost = 1.0 if pending_review_case_ids else 0.0
        priority_score = round(
            gap_score
            + blocked_boost
            + split_gap_boost
            + blocking_split_boost
            + queue_ready_boost,
            4,
        )

        status = "pending_review" if pending_review_case_ids else "unbootstrapped"
        recommendations = [
            (
                "review_pending_label"
                if pending_review_case_ids
                else "bootstrap_label_scaffold"
            )
        ]
        if blocked_field_groups:
            recommendations.append("unblock_recognizer_training")
        for recommendation in _group_recommendations(matched_field_groups, priority_by_group):
            if recommendation not in recommendations:
                recommendations.append(recommendation)

        samples.append(
            {
                "sample_path": sample["sample_path"],
                "case_id": case_id,
                "document_type": document_type,
                "sample_dataset_split": sample_split,
                "status": status,
                "pending_review_case_ids": pending_review_case_ids,
                "matched_field_groups": matched_field_groups,
                "blocked_field_groups": blocked_field_groups,
                "priority_score": priority_score,
                "score_breakdown": {
                    "gap_score": gap_score,
                    "blocked_boost": blocked_boost,
                    "split_gap_boost": split_gap_boost,
                    "blocking_split_boost": blocking_split_boost,
                    "queue_ready_boost": queue_ready_boost,
                },
                "recommendations": recommendations,
                "label_path": None
                if not document_type or not case_id
                else str(
                    labeled_root
                    / PENDING_REVIEW_SPLIT
                    / document_type
                    / case_id
                    / "label.json"
                ),
            }
        )

    samples.sort(
        key=lambda item: (
            -item["priority_score"],
            0 if item["status"] == "pending_review" else 1,
            item["sample_path"],
        )
    )
    return {
        "coverage_report_path": str(coverage_report_path),
        "data_gap_report_path": str(data_gap_report_path),
        "labeled_root": str(labeled_root),
        "prioritized_sample_count": len(samples),
        "priority_order": [item["sample_path"] for item in samples],
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    report = build_sample_label_priority_report(
        args.coverage_report,
        args.data_gap_report,
        labeled_root=args.labeled_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
