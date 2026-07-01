from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from hanah_tax_ocr.evaluation import load_field_error_report

DEFAULT_OUTPUT_PATH = Path("evals/reports/latest_eval_summary.json")
DEFAULT_EXTERNAL_MANIFEST_PATH = Path("evals/external_holdout/manifest.json")
DEFAULT_SUBSET_TAGS: dict[str, set[str]] = {
    "low_quality_subset": {"low_quality", "blur"},
    "format_variation_subset": {"format_variation", "pdf_render"},
    "mixed_language_subset": {"mixed_language"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize exact/CER/WER/document-pass metrics from an eval-report JSON."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, default=DEFAULT_EXTERNAL_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _load_external_manifest_case_tags(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(case.get("case_id")): list(case.get("subset_tags", []))
        for case in payload.get("cases", [])
        if case.get("case_id")
    }


def _metrics_for_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    if not comparisons:
        return {
            "comparison_count": 0,
            "exact_match_rate": 0.0,
            "average_character_error_rate": 0.0,
            "average_word_error_rate": 0.0,
            "document_pass_rate": 0.0,
            "passed_case_ids": [],
            "failed_case_ids": [],
        }

    case_exact: dict[str, bool] = defaultdict(lambda: True)
    exact_matches = 0
    total_cer = 0.0
    total_wer = 0.0
    for comparison in comparisons:
        case_id = str(comparison["case_id"])
        exact_match = bool(comparison["exact_match"])
        case_exact[case_id] = case_exact[case_id] and exact_match
        exact_matches += int(exact_match)
        total_cer += float(comparison["character_error_rate"])
        total_wer += float(comparison["word_error_rate"])

    passed_case_ids = sorted(case_id for case_id, passed in case_exact.items() if passed)
    failed_case_ids = sorted(case_id for case_id, passed in case_exact.items() if not passed)
    return {
        "comparison_count": len(comparisons),
        "exact_match_rate": round(exact_matches / len(comparisons), 6),
        "average_character_error_rate": round(total_cer / len(comparisons), 6),
        "average_word_error_rate": round(total_wer / len(comparisons), 6),
        "document_pass_rate": round(len(passed_case_ids) / len(case_exact), 6),
        "passed_case_ids": passed_case_ids,
        "failed_case_ids": failed_case_ids,
    }


def summarize_eval_report(
    report_path: Path,
    *,
    external_manifest_path: Path | None = None,
) -> dict[str, Any]:
    report = load_field_error_report(report_path)
    comparisons = [comparison.model_dump(mode="json") for comparison in report.comparisons]
    summary = _metrics_for_comparisons(comparisons)
    case_tags = _load_external_manifest_case_tags(external_manifest_path)
    subset_summaries = {}
    for subset_name, subset_tags in DEFAULT_SUBSET_TAGS.items():
        subset_case_ids = {
            case_id
            for case_id, tags in case_tags.items()
            if subset_tags & set(tags)
        }
        subset_comparisons = [
            comparison
            for comparison in comparisons
            if str(comparison["case_id"]) in subset_case_ids
        ]
        subset_summaries[subset_name] = {
            "case_ids": sorted(subset_case_ids),
            **_metrics_for_comparisons(subset_comparisons),
        }

    return {
        "report_path": str(report_path),
        "compared_cases": report.compared_cases,
        "missing_cases": report.missing_cases,
        "field_metric_count": len(report.field_metrics),
        **summary,
        **subset_summaries,
    }


def main() -> None:
    args = parse_args()
    summary = summarize_eval_report(
        args.report,
        external_manifest_path=args.external_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
