from __future__ import annotations

import argparse
import json
from pathlib import Path

from hanah_tax_ocr.evaluation import (
    build_field_error_report,
    compare_field_error_report_files,
)

from scripts.evals.run_eval_suite import run_eval_suite
from scripts.evals.summarize_eval_report import summarize_eval_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the OCR harness on evals/external_holdout/cases and emit report/summary artifacts "
            "under the benchmark protocol."
        )
    )
    parser.add_argument(
        "--expected-root",
        type=Path,
        default=Path("evals/external_holdout/cases"),
    )
    parser.add_argument(
        "--labeled-root",
        type=Path,
        default=Path("data/labeled"),
    )
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=Path("evals/external_holdout/manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--recognizer-root", type=Path)
    parser.add_argument("--inference-subdir", default="inference")
    parser.add_argument("--paddleocr-home", type=Path, default=Path("PaddleOCR"))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--lang", default="en")
    return parser.parse_args()


def run_external_holdout_eval(args: argparse.Namespace) -> dict[str, object]:
    metadata = run_eval_suite(
        args.expected_root,
        args.labeled_root,
        args.output_dir,
        recognizer_root=args.recognizer_root,
        inference_subdir=args.inference_subdir,
        paddleocr_home=args.paddleocr_home,
        case_ids=set(args.case_id) or None,
        lang=args.lang,
    )

    report = build_field_error_report(
        args.expected_root,
        args.output_dir,
        include_missing_cases=False,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = summarize_eval_report(
        args.report_output,
        external_manifest_path=args.external_manifest,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    comparison_payload: dict[str, object] | None = None
    if args.baseline_report and args.comparison_output:
        comparison = compare_field_error_report_files(
            args.baseline_report,
            args.report_output,
        )
        args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_output.write_text(
            json.dumps(comparison.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        comparison_payload = {
            "comparison_output": str(args.comparison_output),
            "overall_status": comparison.overall_delta.status if comparison.overall_delta else None,
            "overall_exact_match_rate_delta": (
                comparison.overall_delta.exact_match_rate_delta
                if comparison.overall_delta
                else None
            ),
        }

    result = {
        "metadata": metadata,
        "report_output": str(args.report_output),
        "summary_output": str(args.summary_output),
        "external_manifest": str(args.external_manifest),
        "baseline_report": str(args.baseline_report) if args.baseline_report else None,
        "comparison": comparison_payload,
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def main() -> None:
    args = parse_args()
    result = run_external_holdout_eval(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
