from __future__ import annotations

import argparse
import json
from pathlib import Path

from hanah_tax_ocr.evaluation import evaluate_run_result, load_harness_run_result
from hanah_tax_ocr.harness import CaseDocument, HarnessRunner
from hanah_tax_ocr.ocr import PaddleOCREngine
from hanah_tax_ocr.schemas import DocumentType


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

    return parser


def parse_document_specs(specs: list[str]) -> list[CaseDocument]:
    documents: list[CaseDocument] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --document value: {spec!r}")
        document_type_raw, source_path = spec.split("=", 1)
        documents.append(
            CaseDocument(
                document_type=DocumentType(document_type_raw),
                source_path=source_path,
            )
        )
    return documents


def run_review_command(args: argparse.Namespace) -> int:
    documents = parse_document_specs(args.document)
    runner = HarnessRunner(
        ocr_engine=PaddleOCREngine(lang=args.lang),
        review_queue_dir=args.review_queue_dir,
    )
    result = runner.run_case(args.case_id, documents)
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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-review":
        return run_review_command(args)
    if args.command == "eval-case":
        return eval_case_command(args)

    raise ValueError(f"Unsupported command: {args.command}")
