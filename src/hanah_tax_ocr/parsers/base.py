from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path

from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, OCRPage, OCRResult, OCRWordBox


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

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_name(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = re.sub(r"\s+", " ", value).strip(" ,;:")
        normalized = re.sub(r"(?<=\b[A-Z])\.(?=[A-Z])", ". ", normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized)
        return normalized or None

    @staticmethod
    def _extract_first_pattern(
        patterns: list[str],
        text: str,
        *,
        flags: int = re.IGNORECASE,
    ) -> str | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags)
            if match:
                return next(
                    (group.strip() for group in match.groups() if group),
                    match.group(0).strip(),
                )
        return None

    @staticmethod
    def _normalize_english_date(value: str | None) -> str | None:
        if not value:
            return None
        normalized = re.sub(r"\s+", " ", value).strip(" ,.;:")
        match = re.search(
            r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,.]?\s*(\d{4})",
            normalized,
            re.IGNORECASE,
        )
        if not match:
            return normalized or None
        month, day, year = match.groups()
        return f"{month} {int(day)}, {year}"

    @staticmethod
    def _normalize_iso_date(value: str | None) -> str | None:
        if not value:
            return None
        normalized = re.sub(r"\s+", "", value)
        match = re.search(r"(\d{4})[년./-]?(\d{1,2})[월./-]?(\d{1,2})", normalized)
        if not match:
            return value.strip()
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"

    @staticmethod
    def _region_text(
        ocr_result: OCRResult,
        box: tuple[float, float, float, float],
    ) -> str | None:
        if not ocr_result.pages:
            return None
        page = ocr_result.pages[0]
        page_size = BaseDocumentParser._page_size(page)
        if page_size is None:
            return None

        width, height = page_size
        positioned_words: list[tuple[float, float, str]] = []
        left, top, right, bottom = box
        for word in page.words:
            center = BaseDocumentParser._word_center(word)
            if center is None:
                continue
            center_x, center_y = center
            normalized_x = center_x / width
            normalized_y = center_y / height
            if left <= normalized_x <= right and top <= normalized_y <= bottom:
                positioned_words.append((normalized_y, normalized_x, word.text.strip()))

        if not positioned_words:
            return None

        positioned_words.sort(key=lambda item: (round(item[0], 3), item[1]))
        text = " ".join(text for _, _, text in positioned_words if text)
        return BaseDocumentParser._normalize_whitespace(text) or None

    @staticmethod
    def _page_size(page: OCRPage) -> tuple[float, float] | None:
        xs: list[float] = []
        ys: list[float] = []
        for word in page.words:
            for point in word.points:
                if len(point) >= 2:
                    xs.append(float(point[0]))
                    ys.append(float(point[1]))
        if not xs or not ys:
            return None
        return max(xs), max(ys)

    @staticmethod
    def _word_center(word: OCRWordBox) -> tuple[float, float] | None:
        if not word.points:
            return None
        x_values = [float(point[0]) for point in word.points if len(point) >= 2]
        y_values = [float(point[1]) for point in word.points if len(point) >= 2]
        if not x_values or not y_values:
            return None
        return sum(x_values) / len(x_values), sum(y_values) / len(y_values)
