from __future__ import annotations

import argparse
import json
from pathlib import Path

from hanah_tax_ocr.evaluation import build_field_error_report

from scripts.evals.materialize_semi_real_probe_suite import materialize_probe_suite
from scripts.evals.run_eval_suite import run_eval_suite
from scripts.evals.summarize_eval_report import summarize_eval_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize semi-real document probes, run the OCR harness on selected "
            "probe case_ids, and emit report/summary artifacts."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evals/semi_real_probes/manifest.json"),
    )
    parser.add_argument("--suite-output-root", type=Path, required=True)
    parser.add_argument("--actual-output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=Path("evals/external_holdout/manifest.json"),
    )
    parser.add_argument("--recognizer-root", type=Path)
    parser.add_argument("--inference-subdir", default="inference")
    parser.add_argument("--paddleocr-home", type=Path, default=Path("PaddleOCR"))
    parser.add_argument("--lang", default="en")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--case-id", action="append", default=[])
    return parser.parse_args()


def run_semi_real_probe_eval(args: argparse.Namespace) -> dict[str, object]:
    materialization = materialize_probe_suite(
        args.manifest,
        args.suite_output_root,
        seed=args.seed,
    )
    case_ids = set(args.case_id) or None
    eval_metadata = run_eval_suite(
        args.suite_output_root / "cases",
        args.suite_output_root / "labeled",
        args.actual_output_dir,
        recognizer_root=args.recognizer_root,
        inference_subdir=args.inference_subdir,
        paddleocr_home=args.paddleocr_home,
        case_ids=case_ids,
        lang=args.lang,
    )

    report = build_field_error_report(
        args.suite_output_root / "cases",
        args.actual_output_dir,
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

    result = {
        "materialization": materialization,
        "eval_metadata": eval_metadata,
        "report_output": str(args.report_output),
        "summary_output": str(args.summary_output),
        "selected_case_ids": sorted(case_ids) if case_ids else [],
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
    result = run_semi_real_probe_eval(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
