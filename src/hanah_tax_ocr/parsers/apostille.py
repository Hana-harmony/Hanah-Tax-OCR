from __future__ import annotations

import re
from pathlib import Path

from hanah_tax_ocr.parsers.base import BaseDocumentParser
from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, OCRResult


class ApostilleParser(BaseDocumentParser):
    document_type = DocumentType.APOSTILLE

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        single_line = self._normalize_whitespace(text)

        item_1 = self._extract_item_block(single_line, 1, 2)
        item_2 = self._extract_item_block(single_line, 2, 3)
        item_3 = self._extract_item_block(single_line, 3, 4)
        item_4 = self._extract_item_block(single_line, 4, 5)
        item_5 = self._extract_item_block(single_line, 5, 6)
        item_6 = self._extract_item_block(single_line, 6, 7)
        item_7 = self._extract_item_block(single_line, 7, 8)
        item_8 = self._extract_item_block(single_line, 8, 9)
        certificate_number_source = self._normalize_whitespace(
            " ".join(
                part
                for part in [
                    self._region_text(ocr_result, (0.24, 0.73, 0.48, 0.81)),
                    item_8,
                ]
                if part
            )
        )

        issuing_country = self._clean_item_value(
            self._region_text(ocr_result, (0.34, 0.31, 0.68, 0.36))
            or self._find_first(r"Country[:\s]+(.+)", item_1),
            stop_phrases=["This public document"],
        )
        signer_name = self._clean_item_value(
            self._region_text(ocr_result, (0.44, 0.38, 0.66, 0.43))
            or self._find_first(r"has been signed by[:\s]*(.+)", item_2),
            stop_phrases=["acting in the capacity"],
        )
        signer_capacity = self._clean_item_value(
            self._find_first(
                r"acting in the capacity of\s+(.+?)(?=\s+4\.|\s+bears the seal|$)",
                single_line,
            )
            or self._region_text(ocr_result, (0.43, 0.44, 0.72, 0.50))
            or self._find_first(r"acting in the capacity of[:\s]*(.+)", item_3),
            stop_phrases=["bears the seal"],
        )
        seal_owner = self._clean_item_value(
            self._find_first(
                r"seal(?:/stamp)?\s+of[_\s]*([A-Z ,]+?)(?=\s+CERTIFIED|\s+5\.|$)",
                single_line,
            )
            or self._region_text(ocr_result, (0.42, 0.49, 0.79, 0.55))
            or self._find_first(r"bears the seal(?:/stamp)? of[:\s]*(.+)", item_4),
            stop_phrases=["CERTIFIED"],
        )
        issued_at = self._clean_item_value(
            self._find_first(r"5\.\s*at\s+([A-Za-z ,]+?)(?=\s+6\.|$)", single_line)
            or self._region_text(ocr_result, (0.30, 0.57, 0.56, 0.62))
            or self._find_first(r"at[:\s]*(.+)", item_5),
            stop_phrases=["the", "by"],
        )
        issued_on = self._clean_item_value(
            self._find_first(
                r"((?:\d{1,2}(?:ST|ND|RD|TH)?\s+DAY\s+OF\s+[A-Z]+[,]?\s*\d{4}))",
                single_line,
            )
            or self._region_text(ocr_result, (0.40, 0.61, 0.66, 0.67))
            or self._find_first(r"(?:the\s+)?(.+)", item_6),
            stop_phrases=["by"],
        )
        issuing_authority = self._clean_item_value(
            self._find_first(
                r"by\s+(Secretary of State.*?North Carolina)",
                single_line,
            )
            or self._region_text(ocr_result, (0.26, 0.68, 0.86, 0.74))
            or self._find_first(r"by[:\s]*(.+)", item_7),
            stop_phrases=["8.", "9.", "10."],
        )
        certificate_number = self._clean_item_value(
            self._extract_first_pattern(
                [
                    r"(?:NO\.?|No\.?)\s*([A-Z0-9-]+)",
                    r"(\d+)",
                ],
                certificate_number_source or item_8,
            )
        )

        fields = {
            "issuing_country": issuing_country,
            "signed_by": signer_name,
            "signer_capacity": self._normalize_capacity(signer_capacity),
            "seal_owner": seal_owner,
            "issued_at": issued_at,
            "issued_on": self._normalize_apostille_date(issued_on),
            "issuing_authority": self._normalize_authority(issuing_authority),
            "certificate_number": certificate_number,
        }
        quality_checks = {
            "has_apostille_heading": self._contains_any(
                single_line,
                ["apostille", "hague convention", "convention de la haye"],
            ),
            "filled_item_count": sum(1 for value in fields.values() if value),
        }
        return ExtractedDocument(
            document_type=self.document_type,
            source_path=str(source_path),
            fields=fields,
            quality_checks=quality_checks,
            parser_warnings=[],
        )

    def _extract_item_block(self, text: str, current: int, next_item: int) -> str:
        pattern = rf"{current}\.?\s*(.+?)(?=\s+{next_item}\.?\s)"
        match = self._find_first(pattern, text)
        if match:
            return match
        tail_pattern = rf"{current}\.?\s*(.+)$"
        return self._find_first(tail_pattern, text) or ""

    def _clean_item_value(
        self,
        value: str | None,
        *,
        stop_phrases: list[str] | None = None,
    ) -> str | None:
        cleaned = self._normalize_whitespace(value or "")
        if not cleaned:
            return None
        cleaned = re.sub(r"[_|]+", " ", cleaned)
        cleaned = re.sub(r"^(Country:|COUNTRY:)\s*", "", cleaned)
        cleaned = re.sub(r"^10\.\s*Signature\s*$", "", cleaned, flags=re.IGNORECASE)
        for phrase in stop_phrases or []:
            match = re.search(re.escape(phrase), cleaned, re.IGNORECASE)
            if match:
                cleaned = cleaned[: match.start()].strip()
        cleaned = cleaned.strip(" :;|\\/")
        if not any(character.isalnum() for character in cleaned):
            return None
        return cleaned or None

    def _normalize_capacity(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"\s+\d+$", "", value).strip()
        return cleaned or None

    def _normalize_apostille_date(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"([A-Za-z])(\d{4})", r"\1, \2", value)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    def _normalize_authority(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = value.replace("StateState", "State State")
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned
