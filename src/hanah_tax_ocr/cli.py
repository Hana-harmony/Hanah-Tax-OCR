from __future__ import annotations

import argparse
import json
from pathlib import Path

from hanah_tax_ocr.evaluation import (
    build_field_error_report,
    compare_field_error_report_files,
    evaluate_run_result,
    load_harness_run_result,
)
from hanah_tax_ocr.harness import CaseDocument, HarnessRunner
from hanah_tax_ocr.ocr import PaddleOCREngine
from hanah_tax_ocr.schemas import DocumentType
from hanah_tax_ocr.template_profiles import classify_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hanah tax OCR harness CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_review = subparsers.add_parser(
        "run-review",
        help="Run OCR parsing and review for a case.",
    )
    run_review.add_argument("--case-id", required=True)
    run_review.add_argument(
        "--document",
        action="append",
        required=True,
        help="Document spec in the form document_type=/absolute/or/relative/path",
    )
    run_review.add_argument(
        "--output",
        type=Path,
        default=Path("evals/fixtures/last_run_result.json"),
    )
    run_review.add_argument(
        "--review-queue-dir",
        type=Path,
        default=Path("data/review_queue/index"),
    )
    run_review.add_argument("--lang", default="en")

    eval_case = subparsers.add_parser(
        "eval-case",
        help="Evaluate a run result against expected.json.",
    )
    eval_case.add_argument("--expected", type=Path, required=True)
    eval_case.add_argument("--actual", type=Path, required=True)

    eval_report = subparsers.add_parser(
        "eval-report",
        help="Build a field-level OCR error report from expected cases and run results.",
    )
    eval_report.add_argument("--expected-root", type=Path, required=True)
    eval_report.add_argument("--actual-dir", type=Path, required=True)
    eval_report.add_argument("--case-id", action="append", default=[])
    eval_report.add_argument(
        "--only-available",
        action="store_true",
        help="Skip missing expected cases and report only cases that exist in the actual dir.",
    )

    compare_eval_reports = subparsers.add_parser(
        "compare-eval-reports",
        help="Compare baseline and candidate eval-report JSON outputs.",
    )
    compare_eval_reports.add_argument("--baseline", type=Path, required=True)
    compare_eval_reports.add_argument("--candidate", type=Path, required=True)
    compare_eval_reports.add_argument(
        "--baseline-training-summary",
        type=Path,
        help="Optional recognizer summary.json used to explain baseline training-data context.",
    )
    compare_eval_reports.add_argument(
        "--candidate-training-summary",
        type=Path,
        help="Optional recognizer summary.json used to explain candidate training-data context.",
    )

    return parser


def parse_document_specs(specs: list[str]) -> list[CaseDocument]:
    documents: list[CaseDocument] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --document value: {spec!r}")
        document_spec, source_path = spec.split("=", 1)
        document_type_raw, _, ocr_lang = document_spec.partition("@")
        documents.append(
            CaseDocument(
                document_type=DocumentType(document_type_raw),
                source_path=source_path,
                ocr_lang=ocr_lang or None,
            )
        )
    return documents


def run_review_command(args: argparse.Namespace) -> int:
    documents = parse_document_specs(args.document)
    engines: dict[str, PaddleOCREngine] = {}
    hydrated_documents: list[CaseDocument] = []
    for document in documents:
        lang = document.ocr_lang or args.lang
        engine = engines.get(lang)
        if engine is None:
            engine = PaddleOCREngine(lang=lang)
            engines[lang] = engine
        ocr_result = engine.run(document.source_path)
        profile = classify_template(
            document.document_type,
            document.source_path,
            ocr_result.combined_text(),
        )
        if profile is not None:
            ocr_result.template_id = profile.template_id
            ocr_result.regions = engine.run_regions(document.source_path, profile.ocr_regions)
        hydrated_documents.append(
            CaseDocument(
                document_type=document.document_type,
                source_path=document.source_path,
                ocr_lang=lang,
                ocr_result=ocr_result,
            )
        )

    runner = HarnessRunner(review_queue_dir=args.review_queue_dir)
    result = runner.run_case(args.case_id, hydrated_documents)
    runner.write_run_result(result, args.output)
    print(
        json.dumps(
            {
                "case_id": result.case_id,
                "status": result.review_result.status.value,
                "output": str(args.output),
                "queued_review_path": result.queued_review_path,
            },
            ensure_ascii=False,
        )
    )
    return 0


def eval_case_command(args: argparse.Namespace) -> int:
    run_result = load_harness_run_result(args.actual)
    evaluation = evaluate_run_result(args.expected, run_result)
    print(json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False))
    return 0 if evaluation.passed else 1


def eval_report_command(args: argparse.Namespace) -> int:
    report = build_field_error_report(
        args.expected_root,
        args.actual_dir,
        case_ids=set(args.case_id) or None,
        include_missing_cases=not args.only_available,
    )
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False))
    return 0


def compare_eval_reports_command(args: argparse.Namespace) -> int:
    comparison = compare_field_error_report_files(
        args.baseline,
        args.candidate,
        baseline_training_summary_path=args.baseline_training_summary,
        candidate_training_summary_path=args.candidate_training_summary,
    )
    print(json.dumps(comparison.model_dump(mode="json"), ensure_ascii=False))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-review":
        return run_review_command(args)
    if args.command == "eval-case":
        return eval_case_command(args)
    if args.command == "eval-report":
        return eval_report_command(args)
    if args.command == "compare-eval-reports":
        return compare_eval_reports_command(args)

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
