from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hanah_tax_ocr.document_checks import compute_document_checks
from hanah_tax_ocr.ocr import PaddleOCREngine
from hanah_tax_ocr.parsers import build_parser_registry
from hanah_tax_ocr.quality import average_ocr_confidence, compute_quality_metrics
from hanah_tax_ocr.review import TaxDocumentReviewer
from hanah_tax_ocr.schemas import (
    DocumentType,
    ExtractedDocument,
    OCRResult,
    ReviewResult,
    ReviewStatus,
)
from hanah_tax_ocr.template_profiles import classify_template


class CaseDocument(BaseModel):
    document_type: DocumentType
    source_path: str
    ocr_lang: str | None = None
    ocr_result: OCRResult | None = None


class HarnessRunResult(BaseModel):
    case_id: str
    extracted_documents: list[ExtractedDocument] = Field(default_factory=list)
    review_result: ReviewResult
    queued_review_path: str | None = None


class HarnessRunner:
    def __init__(
        self,
        *,
        reviewer: TaxDocumentReviewer | None = None,
        ocr_engine: PaddleOCREngine | None = None,
        review_queue_dir: str | Path = "data/review_queue/index",
        blur_threshold: float = 100.0,
        min_ocr_confidence: float = 0.75,
    ) -> None:
        self._reviewer = reviewer or TaxDocumentReviewer()
        self._ocr_engine = ocr_engine
        self._review_queue_dir = Path(review_queue_dir)
        self._blur_threshold = blur_threshold
        self._min_ocr_confidence = min_ocr_confidence
        self._parsers = build_parser_registry()

    def run_case(
        self,
        case_id: str,
        documents: list[CaseDocument],
    ) -> HarnessRunResult:
        extracted_documents: list[ExtractedDocument] = []

        for document in documents:
            ocr_result = document.ocr_result or self._run_ocr(document.source_path)
            profile = classify_template(
                document.document_type,
                document.source_path,
                ocr_result.combined_text(),
            )
            if profile and not ocr_result.template_id:
                ocr_result.template_id = profile.template_id
            if self._ocr_engine is not None and profile and not ocr_result.regions:
                ocr_result.regions = self._ocr_engine.run_regions(
                    document.source_path,
                    profile.ocr_regions,
                )
            extracted = self._parsers[document.document_type].parse(
                ocr_result,
                document.source_path,
            )
            extracted.template_id = ocr_result.template_id

            extracted.quality_checks.update(
                compute_document_checks(
                    document.document_type,
                    document.source_path,
                    template_id=ocr_result.template_id,
                    ocr_text=ocr_result.combined_text(),
                )
            )
            extracted.quality_checks["detected_template_id"] = ocr_result.template_id
            quality_metrics = compute_quality_metrics(
                document.source_path,
                blur_threshold=self._blur_threshold,
            )
            avg_confidence = average_ocr_confidence(ocr_result.pages)
            if avg_confidence is not None:
                quality_metrics["average_ocr_confidence"] = avg_confidence
                quality_metrics["low_ocr_confidence"] = avg_confidence < self._min_ocr_confidence

            extracted.quality_checks.update(quality_metrics)
            extracted_documents.append(extracted)

        review_result = self._reviewer.review(extracted_documents)
        queued_review_path = None
        if self._should_queue_review(review_result, extracted_documents):
            queued_review_path = self._write_review_queue(
                case_id,
                extracted_documents,
                review_result,
            )

        return HarnessRunResult(
            case_id=case_id,
            extracted_documents=extracted_documents,
            review_result=review_result,
            queued_review_path=queued_review_path,
        )

    def write_run_result(self, run_result: HarnessRunResult, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(run_result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _run_ocr(self, source_path: str) -> OCRResult:
        if self._ocr_engine is None:
            raise RuntimeError(
                "No OCR result was provided and no PaddleOCR engine is configured."
            )
        return self._ocr_engine.run(source_path)

    def _should_queue_review(
        self,
        review_result: ReviewResult,
        documents: list[ExtractedDocument],
    ) -> bool:
        if review_result.status != ReviewStatus.PASS:
            return True
        for document in documents:
            if document.quality_checks.get("blurry") is True:
                return True
            if document.quality_checks.get("low_ocr_confidence") is True:
                return True
        return False

    def _write_review_queue(
        self,
        case_id: str,
        documents: list[ExtractedDocument],
        review_result: ReviewResult,
    ) -> str:
        payload: dict[str, Any] = {
            "case_id": case_id,
            "review_result": review_result.model_dump(mode="json"),
            "documents": [document.model_dump(mode="json") for document in documents],
        }
        self._review_queue_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._review_queue_dir / f"{case_id}.json"
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(output_path)
