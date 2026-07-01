from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from hanah_tax_ocr.evaluation import load_field_error_report
from hanah_tax_ocr.training.field_crops import field_group_for

DEFAULT_OUTPUT_PATH = Path("evals/augmentation_effects/ledger.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append field-group deltas for a hard-case augmentation experiment."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--variants", action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _aggregate_by_field_group(report_path: Path) -> dict[str, dict[str, float]]:
    report = load_field_error_report(report_path)
    grouped: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "comparisons": 0.0,
            "exact_weighted": 0.0,
            "cer_weighted": 0.0,
            "wer_weighted": 0.0,
        }
    )
    for field_key, metric in report.field_metrics.items():
        _, _, field_name = field_key.partition(".")
        field_group = field_group_for(field_name)
        grouped[field_group]["comparisons"] += metric.comparisons
        grouped[field_group]["exact_weighted"] += metric.comparisons * metric.exact_match_rate
        grouped[field_group]["cer_weighted"] += (
            metric.comparisons * metric.average_character_error_rate
        )
        grouped[field_group]["wer_weighted"] += metric.comparisons * metric.average_word_error_rate

    summary: dict[str, dict[str, float]] = {}
    for field_group, values in grouped.items():
        comparisons = values["comparisons"]
        if comparisons <= 0:
            continue
        summary[field_group] = {
            "comparisons": comparisons,
            "exact_match_rate": round(values["exact_weighted"] / comparisons, 6),
            "average_character_error_rate": round(values["cer_weighted"] / comparisons, 6),
            "average_word_error_rate": round(values["wer_weighted"] / comparisons, 6),
        }
    return summary


def record_augmentation_effects(
    experiment_id: str,
    baseline_report: Path,
    candidate_report: Path,
    *,
    variants: list[str],
    output_path: Path,
) -> dict[str, Any]:
    baseline = _aggregate_by_field_group(baseline_report)
    candidate = _aggregate_by_field_group(candidate_report)
    field_groups = sorted(set(baseline) | set(candidate))
    deltas: list[dict[str, Any]] = []
    for field_group in field_groups:
        baseline_metrics = baseline.get(field_group, {})
        candidate_metrics = candidate.get(field_group, {})
        deltas.append(
            {
                "field_group": field_group,
                "baseline": baseline_metrics,
                "candidate": candidate_metrics,
                "exact_match_rate_delta": round(
                    candidate_metrics.get("exact_match_rate", 0.0)
                    - baseline_metrics.get("exact_match_rate", 0.0),
                    6,
                ),
                "average_character_error_rate_delta": round(
                    candidate_metrics.get("average_character_error_rate", 0.0)
                    - baseline_metrics.get("average_character_error_rate", 0.0),
                    6,
                ),
                "average_word_error_rate_delta": round(
                    candidate_metrics.get("average_word_error_rate", 0.0)
                    - baseline_metrics.get("average_word_error_rate", 0.0),
                    6,
                ),
            }
        )

    payload = {
        "version": "2026-07-01",
        "experiments": [],
    }
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload.setdefault("experiments", []).append(
        {
            "experiment_id": experiment_id,
            "baseline_report": str(baseline_report),
            "candidate_report": str(candidate_report),
            "variants": variants,
            "field_group_deltas": deltas,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    payload = record_augmentation_effects(
        args.experiment_id,
        args.baseline_report,
        args.candidate_report,
        variants=args.variants,
        output_path=args.output,
    )
    print(json.dumps(payload["experiments"][-1], ensure_ascii=False))


if __name__ == "__main__":
    main()
