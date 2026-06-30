from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hanah_tax_ocr.evaluation import load_field_error_report
from hanah_tax_ocr.training.field_crops import field_group_for

DEFAULT_FIELD_CROPS_ROOT = Path("data/training/field_crops")
DEFAULT_RECOGNIZER_ROOT = Path("data/training/recognizer")
DEFAULT_OUTPUT_PATH = Path("data/training/reports/data_gap_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report labeling and coverage gaps for PaddleOCR recognizer training."
    )
    parser.add_argument(
        "--field-crops-root",
        type=Path,
        default=DEFAULT_FIELD_CROPS_ROOT,
    )
    parser.add_argument(
        "--recognizer-root",
        type=Path,
        default=DEFAULT_RECOGNIZER_ROOT,
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=None,
        help="Optional eval-report JSON to fold accuracy signals into prioritization.",
    )
    parser.add_argument(
        "--min-base-train-count",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--min-base-val-count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-train-source-count",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-val-source-count",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _counts_by(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter()
    for entry in entries:
        value = entry.get(key) or "unknown"
        counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _sum_counts(values: dict[str, int]) -> int:
    return sum(values.values())


def _unique_source_counts_by(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        group_key = str(entry.get(key) or "unknown")
        source_path = str(entry.get("source_path") or "unknown")
        grouped[group_key].add(source_path)
    return dict(
        sorted((group_key, len(source_paths)) for group_key, source_paths in grouped.items())
    )


def _aggregate_eval_metrics(eval_report_path: Path | None) -> dict[str, dict[str, float]]:
    if eval_report_path is None or not eval_report_path.exists():
        return {}

    report = load_field_error_report(eval_report_path)
    grouped: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    for field_key, metric in report.field_metrics.items():
        _, _, field_name = field_key.partition(".")
        group = field_group_for(field_name)
        grouped[group].append(
            (
                metric.comparisons,
                metric.exact_match_rate,
                metric.average_character_error_rate,
                metric.average_word_error_rate,
            )
        )

    aggregated: dict[str, dict[str, float]] = {}
    for field_group, values in grouped.items():
        total_comparisons = sum(item[0] for item in values)
        if total_comparisons <= 0:
            continue
        aggregated[field_group] = {
            "comparisons": total_comparisons,
            "exact_match_rate": round(
                sum(item[0] * item[1] for item in values) / total_comparisons,
                4,
            ),
            "average_character_error_rate": round(
                sum(item[0] * item[2] for item in values) / total_comparisons,
                4,
            ),
            "average_word_error_rate": round(
                sum(item[0] * item[3] for item in values) / total_comparisons,
                4,
            ),
        }
    return aggregated


def build_data_gap_report(
    field_crops_root: Path,
    recognizer_root: Path,
    *,
    eval_report_path: Path | None = None,
    min_base_train_count: int = 12,
    min_base_val_count: int = 3,
    min_train_source_count: int = 3,
    min_val_source_count: int = 2,
) -> dict[str, Any]:
    field_crop_entries = load_jsonl(field_crops_root / "manifest.jsonl")
    recognizer_summary_path = recognizer_root / "summary.json"
    recognizer_summary = (
        load_json(recognizer_summary_path) if recognizer_summary_path.exists() else {"groups": {}}
    )
    eval_metrics_by_group = _aggregate_eval_metrics(eval_report_path)

    accepted_entries = [
        entry for entry in field_crop_entries if entry.get("quality", {}).get("accepted", True)
    ]
    rejected_entries = [
        entry for entry in field_crop_entries if not entry.get("quality", {}).get("accepted", True)
    ]

    document_types = sorted(
        {
            str(entry.get("document_type"))
            for entry in accepted_entries
            if entry.get("document_type")
        }
    )
    known_groups = sorted(
        {
            str(entry.get("field_group"))
            for entry in field_crop_entries
            if entry.get("field_group")
        }
        | set(recognizer_summary.get("groups", {}))
        | set(eval_metrics_by_group)
    )

    priorities: list[dict[str, Any]] = []
    for field_group in known_groups:
        group_entries = [
            entry for entry in accepted_entries if entry.get("field_group") == field_group
        ]
        train_entries = [entry for entry in group_entries if entry.get("split") == "train"]
        val_entries = [entry for entry in group_entries if entry.get("split") == "val"]
        rejected_group_entries = [
            entry for entry in rejected_entries if entry.get("field_group") == field_group
        ]

        train_counts_by_document = _counts_by(train_entries, "document_type")
        val_counts_by_document = _counts_by(val_entries, "document_type")
        rejected_counts_by_document = _counts_by(rejected_group_entries, "document_type")
        train_source_counts_by_document = _unique_source_counts_by(train_entries, "document_type")
        val_source_counts_by_document = _unique_source_counts_by(val_entries, "document_type")
        base_train_count = _sum_counts(train_counts_by_document)
        base_val_count = _sum_counts(val_counts_by_document)
        base_train_source_count = _sum_counts(train_source_counts_by_document)
        base_val_source_count = _sum_counts(val_source_counts_by_document)
        rejected_count = len(rejected_group_entries)

        missing_train_document_types = [
            document_type
            for document_type in document_types
            if train_counts_by_document.get(document_type, 0) == 0
        ]
        missing_val_document_types = [
            document_type
            for document_type in document_types
            if val_counts_by_document.get(document_type, 0) == 0
        ]

        recognizer_group = recognizer_summary.get("groups", {}).get(field_group, {})
        data_profile = recognizer_group.get("data_profile", {})
        eval_metrics = eval_metrics_by_group.get(field_group)

        train_gap = max(0, min_base_train_count - base_train_count)
        val_gap = max(0, min_base_val_count - base_val_count)
        train_source_gap = max(0, min_train_source_count - base_train_source_count)
        val_source_gap = max(0, min_val_source_count - base_val_source_count)
        document_gap = len(missing_train_document_types) + (0.5 * len(missing_val_document_types))
        rejected_gap = rejected_count * 0.25
        accuracy_gap = 0.0
        if eval_metrics is not None:
            accuracy_gap = (
                (1.0 - eval_metrics["exact_match_rate"]) * 10
                + eval_metrics["average_character_error_rate"] * 5
                + eval_metrics["average_word_error_rate"] * 2
            )

        priority_score = round(
            (train_gap * 3.0)
            + (val_gap * 2.0)
            + (train_source_gap * 2.5)
            + (val_source_gap * 2.0)
            + document_gap
            + rejected_gap
            + accuracy_gap,
            4,
        )

        recommendations: list[str] = []
        if train_gap > 0:
            recommendations.append("collect_base_train_samples")
        if val_gap > 0:
            recommendations.append("collect_base_val_samples")
        if missing_train_document_types:
            recommendations.append("expand_train_document_coverage")
        if missing_val_document_types:
            recommendations.append("expand_val_document_coverage")
        if train_source_gap > 0:
            recommendations.append("collect_distinct_train_sources")
        if val_source_gap > 0:
            recommendations.append("collect_distinct_val_sources")
        if rejected_count > 0:
            recommendations.append("review_rejected_field_crops")
        if eval_metrics is not None and (
            eval_metrics["exact_match_rate"] < 0.9
            or eval_metrics["average_character_error_rate"] > 0.05
        ):
            recommendations.append("prioritize_low_accuracy_group")
        if data_profile.get("filtered_hard_case_train_count", 0) > 0:
            recommendations.append("add_base_samples_before_more_hard_cases")
        if (
            data_profile.get("counts_by_source_type", {}).get("train", {}).get("hard_case", 0) > 0
            and data_profile.get("unique_hard_case_variant_counts", {}).get("train", 0) < 2
        ):
            recommendations.append("expand_hard_case_variant_coverage")

        priorities.append(
            {
                "field_group": field_group,
                "priority_score": priority_score,
                "base_train_count": base_train_count,
                "base_val_count": base_val_count,
                "base_train_source_count": base_train_source_count,
                "base_val_source_count": base_val_source_count,
                "rejected_count": rejected_count,
                "counts_by_document_type": {
                    "train": train_counts_by_document,
                    "val": val_counts_by_document,
                    "rejected": rejected_counts_by_document,
                },
                "source_counts_by_document_type": {
                    "train": train_source_counts_by_document,
                    "val": val_source_counts_by_document,
                },
                "missing_document_types": {
                    "train": missing_train_document_types,
                    "val": missing_val_document_types,
                },
                "recognizer_profile": {
                    "train_count": recognizer_group.get("train_count", 0),
                    "val_count": recognizer_group.get("val_count", 0),
                    "train_source_count": data_profile.get("unique_source_counts", {}).get(
                        "train",
                        0,
                    ),
                    "val_source_count": data_profile.get("unique_source_counts", {}).get("val", 0),
                    "hard_case_train_ratio": data_profile.get("hard_case_train_ratio", 0.0),
                    "filtered_hard_case_train_count": data_profile.get(
                        "filtered_hard_case_train_count",
                        0,
                    ),
                    "filtered_stale_hard_case_count": data_profile.get(
                        "filtered_stale_hard_case_count",
                        0,
                    ),
                    "hard_case_variant_counts": data_profile.get(
                        "hard_case_variant_counts",
                        {},
                    ),
                    "hard_case_variant_counts_by_document_type": data_profile.get(
                        "hard_case_variant_counts_by_document_type",
                        {},
                    ),
                    "unique_hard_case_variant_counts": data_profile.get(
                        "unique_hard_case_variant_counts",
                        {},
                    ),
                    "hard_case_selection_strategy": data_profile.get(
                        "hard_case_selection_strategy",
                        "",
                    ),
                    "hard_case_variant_floor_applied": data_profile.get(
                        "hard_case_variant_floor_applied",
                        False,
                    ),
                    "warnings": data_profile.get("warnings", []),
                    "training_readiness": recognizer_group.get("training_readiness", {}),
                },
                "eval_metrics": eval_metrics,
                "score_breakdown": {
                    "train_gap": round(train_gap * 3.0, 4),
                    "val_gap": round(val_gap * 2.0, 4),
                    "train_source_gap": round(train_source_gap * 2.5, 4),
                    "val_source_gap": round(val_source_gap * 2.0, 4),
                    "document_gap": round(document_gap, 4),
                    "rejected_gap": round(rejected_gap, 4),
                    "accuracy_gap": round(accuracy_gap, 4),
                },
                "recommendations": recommendations,
            }
        )

    priorities.sort(
        key=lambda item: (
            -item["priority_score"],
            item["base_train_count"],
            item["field_group"],
        )
    )

    return {
        "field_crops_root": str(field_crops_root),
        "recognizer_root": str(recognizer_root),
        "eval_report_path": None if eval_report_path is None else str(eval_report_path),
        "min_base_train_count": min_base_train_count,
        "min_base_val_count": min_base_val_count,
        "min_train_source_count": min_train_source_count,
        "min_val_source_count": min_val_source_count,
        "document_types": document_types,
        "priority_order": [item["field_group"] for item in priorities],
        "priorities": priorities,
    }


def main() -> None:
    args = parse_args()
    report = build_data_gap_report(
        args.field_crops_root,
        args.recognizer_root,
        eval_report_path=args.eval_report,
        min_base_train_count=args.min_base_train_count,
        min_base_val_count=args.min_base_val_count,
        min_train_source_count=args.min_train_source_count,
        min_val_source_count=args.min_val_source_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
