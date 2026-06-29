from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, OCRResult


class BaseDocumentParser(ABC):
    document_type: DocumentType

    @abstractmethod
    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        raise NotImplementedError

    @staticmethod
    def _find_first(pattern: str, text: str, *, flags: int = re.IGNORECASE) -> str | None:
        match = re.search(pattern, text, flags)
        if not match:
            return None
        return next((group.strip() for group in match.groups() if group), match.group(0).strip())

    @staticmethod
    def _contains_any(text: str, needles: list[str]) -> bool:
        lowered = text.lower()
        return any(needle.lower() in lowered for needle in needles)
