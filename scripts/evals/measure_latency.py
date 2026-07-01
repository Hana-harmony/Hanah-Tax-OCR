from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hanah_tax_ocr.harness import CaseDocument, HarnessRunner
from hanah_tax_ocr.ocr import PaddleOCREngine
from hanah_tax_ocr.schemas import DocumentType
from hanah_tax_ocr.template_profiles import classify_template

from scripts.evals.run_eval_suite import (
    _default_ocr_lang,
    _region_overrides_from_recognizer_root,
)

DEFAULT_LABELED_ROOT = Path("data/labeled")
DEFAULT_OUTPUT_PATH = Path("evals/benchmark_latency_observations.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure local CPU end-to-end latency for selected OCR cases."
    )
    parser.add_argument("--labeled-root", type=Path, default=DEFAULT_LABELED_ROOT)
    parser.add_argument("--case-id", action="append", required=True, default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--recognizer-root", type=Path, default=None)
    parser.add_argument("--inference-subdir", default="inference")
    parser.add_argument("--paddleocr-home", type=Path, default=Path("PaddleOCR"))
    parser.add_argument("--lang", default="en")
    return parser.parse_args()


def _load_case_documents(labeled_root: Path, case_ids: set[str]) -> dict[str, list[CaseDocument]]:
    cases: dict[str, list[CaseDocument]] = {}
    for label_path in sorted(labeled_root.rglob("label.json")):
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        case_id = str(payload.get("case_id") or "")
        if case_id not in case_ids:
            continue
        source_path = payload.get("source_path")
        document_type = payload.get("document_type")
        if not source_path or not document_type:
            continue
        cases.setdefault(case_id, []).append(
            CaseDocument(
                document_type=DocumentType(str(document_type)),
                source_path=str(source_path),
                ocr_lang=_default_ocr_lang(DocumentType(str(document_type))),
            )
        )
    for documents in cases.values():
        documents.sort(key=lambda item: item.document_type.value)
    return cases


def measure_latency(
    labeled_root: Path,
    case_ids: list[str],
    *,
    recognizer_root: Path | None = None,
    inference_subdir: str = "inference",
    paddleocr_home: Path = Path("PaddleOCR"),
    lang: str = "en",
) -> dict[str, Any]:
    case_id_set = set(case_ids)
    cases = _load_case_documents(labeled_root, case_id_set)
    region_overrides = _region_overrides_from_recognizer_root(
        recognizer_root,
        inference_subdir=inference_subdir,
        paddleocr_home=paddleocr_home,
    )
    runner = HarnessRunner(review_queue_dir=Path("tmp/latency_review_queue"))
    engines: dict[str, PaddleOCREngine] = {}
    observed_cases: list[dict[str, Any]] = []

    for case_id in case_ids:
        documents = cases.get(case_id, [])
        if not documents:
            continue
        start = time.perf_counter()
        hydrated_documents: list[CaseDocument] = []
        for document in documents:
            document_lang = document.ocr_lang or lang
            engine = engines.get(document_lang)
            if engine is None:
                engine = PaddleOCREngine(lang=document_lang, region_overrides=region_overrides)
                engines[document_lang] = engine
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
                    ocr_lang=document_lang,
                    ocr_result=ocr_result,
                )
            )
        run_result = runner.run_case(case_id, hydrated_documents)
        elapsed_seconds = time.perf_counter() - start
        observed_cases.append(
            {
                "case_id": case_id,
                "document_types": [
                    document.document_type.value for document in hydrated_documents
                ],
                "source_paths": [document.source_path for document in hydrated_documents],
                "elapsed_seconds": round(elapsed_seconds, 4),
                "document_count": len(hydrated_documents),
                "review_status": run_result.review_result.status.value,
            }
        )

    case_count = len(observed_cases)
    total_elapsed = sum(case["elapsed_seconds"] for case in observed_cases)
    total_documents = sum(case["document_count"] for case in observed_cases)
    return {
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_mode": "local_cpu_observation",
        },
        "recognizer_root": None if recognizer_root is None else str(recognizer_root),
        "cases": observed_cases,
        "summary": {
            "case_count": case_count,
            "document_count": total_documents,
            "average_case_seconds": round(total_elapsed / case_count, 4) if case_count else 0.0,
            "average_document_seconds": (
                round(total_elapsed / total_documents, 4) if total_documents else 0.0
            ),
        },
    }


def main() -> None:
    args = parse_args()
    report = measure_latency(
        args.labeled_root,
        args.case_id,
        recognizer_root=args.recognizer_root,
        inference_subdir=args.inference_subdir,
        paddleocr_home=args.paddleocr_home,
        lang=args.lang,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
