from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path

from hanah_tax_ocr.evaluation import (
    build_field_error_report,
    compare_field_error_report_files,
    load_harness_run_results_from_dir,
)
from hanah_tax_ocr.harness import CaseDocument, HarnessRunner, HarnessRunResult
from hanah_tax_ocr.ocr import PaddleOCREngine
from hanah_tax_ocr.schemas import DocumentType

from scripts.evals.summarize_eval_report import summarize_eval_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hybrid eval directory by copying baseline run results for untouched cases "
            "and rerunning only selected cases on the current code."
        )
    )
    parser.add_argument("--baseline-actual-dir", type=Path, required=True)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=Path("evals/external_holdout/manifest.json"),
    )
    parser.add_argument("--review-queue-dir", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--rerun-case-id", action="append", default=[])
    parser.add_argument("--rerun-prefix", action="append", default=[])
    return parser.parse_args()


def should_rerun_case(
    case_id: str,
    *,
    rerun_case_ids: set[str],
    rerun_prefixes: tuple[str, ...],
) -> bool:
    if case_id in rerun_case_ids:
        return True
    return any(case_id.startswith(prefix) for prefix in rerun_prefixes)


def rerun_case_from_baseline(
    run_result: HarnessRunResult,
    *,
    lang: str,
    review_queue_dir: Path,
) -> HarnessRunResult:
    engine = PaddleOCREngine(lang=lang)
    runner = HarnessRunner(ocr_engine=engine, review_queue_dir=review_queue_dir)
    documents = [
        CaseDocument(
            document_type=DocumentType(document.document_type),
            source_path=document.source_path,
            ocr_lang=lang,
        )
        for document in run_result.extracted_documents
    ]
    return runner.run_case(run_result.case_id, documents)


def build_hybrid_eval_dir(
    baseline_actual_dir: Path,
    output_dir: Path,
    *,
    rerun_case_ids: set[str],
    rerun_prefixes: tuple[str, ...],
    rerun_case_fn: Callable[[HarnessRunResult], HarnessRunResult],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_results = load_harness_run_results_from_dir(baseline_actual_dir)

    copied_case_ids: list[str] = []
    rerun_case_id_list: list[str] = []

    for case_id, run_result in sorted(baseline_results.items()):
        destination = output_dir / f"{case_id}.json"
        if should_rerun_case(
            case_id,
            rerun_case_ids=rerun_case_ids,
            rerun_prefixes=rerun_prefixes,
        ):
            rerun_result = rerun_case_fn(run_result)
            destination.write_text(
                json.dumps(rerun_result.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            rerun_case_id_list.append(case_id)
            continue

        shutil.copy2(baseline_actual_dir / f"{case_id}.json", destination)
        copied_case_ids.append(case_id)

    return {
        "baseline_actual_dir": str(baseline_actual_dir),
        "output_dir": str(output_dir),
        "copied_case_ids": copied_case_ids,
        "rerun_case_ids": rerun_case_id_list,
        "copied_case_count": len(copied_case_ids),
        "rerun_case_count": len(rerun_case_id_list),
    }


def run_hybrid_eval(args: argparse.Namespace) -> dict[str, object]:
    replay_review_queue_dir = args.review_queue_dir or (args.output_dir / "review_queue")
    replay_review_queue_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_hybrid_eval_dir(
        args.baseline_actual_dir,
        args.output_dir,
        rerun_case_ids=set(args.rerun_case_id),
        rerun_prefixes=tuple(args.rerun_prefix),
        rerun_case_fn=lambda run_result: rerun_case_from_baseline(
            run_result,
            lang=args.lang,
            review_queue_dir=replay_review_queue_dir,
        ),
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
    result = run_hybrid_eval(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
