from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    RESIDENCY_CERTIFICATE = "residency_certificate"
    APOSTILLE = "apostille"
    WITHHOLDING_TAX_FORM = "withholding_tax_form"


class ReviewStatus(StrEnum):
    PASS = "pass"
    REJECT = "reject"
    NEEDS_REVIEW = "needs_review"


class OCRWordBox(BaseModel):
    text: str
    confidence: float | None = None
    points: list[list[float]] = Field(default_factory=list)


class OCRPage(BaseModel):
    page_number: int
    words: list[OCRWordBox] = Field(default_factory=list)
    raw_text: str = ""


class OCRResult(BaseModel):
    pages: list[OCRPage] = Field(default_factory=list)

    def combined_text(self) -> str:
        return "\n".join(page.raw_text for page in self.pages if page.raw_text).strip()


class ExtractedDocument(BaseModel):
    document_type: DocumentType
    source_path: str
    fields: dict[str, Any] = Field(default_factory=dict)
    quality_checks: dict[str, Any] = Field(default_factory=dict)
    parser_warnings: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    code: str
    message: str
    field_name: str | None = None


class ReviewResult(BaseModel):
    status: ReviewStatus
    findings: list[ReviewFinding] = Field(default_factory=list)
    cross_check: dict[str, Any] = Field(default_factory=dict)
