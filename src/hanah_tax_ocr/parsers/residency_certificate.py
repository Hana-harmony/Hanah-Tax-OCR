from __future__ import annotations

import re
from pathlib import Path

from hanah_tax_ocr.normalization import normalize_country
from hanah_tax_ocr.parsers.base import BaseDocumentParser
from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, OCRResult


class ResidencyCertificateParser(BaseDocumentParser):
    document_type = DocumentType.RESIDENCY_CERTIFICATE

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        single_line = self._normalize_whitespace(text)
        tin_source = self._normalize_whitespace(
            " ".join(
                part
                for part in [
                    self._region_value(ocr_result, "tin"),
                    single_line,
                ]
                if part
            )
        )

        taxpayer_name = self._normalize_name(
            self._clean_taxpayer_value(self._region_value(ocr_result, "taxpayer_name"))
            or self._clean_taxpayer_value(self._find_first(r"Taxpayer\s*[:;]?\s*([^\n]+)", text))
            or self._clean_taxpayer_value(
                self._find_first(
                    r"Taxpayer\s*[:;]?\s*(.+?)\s+(?:TIN|Tax Year|Date)\b",
                    single_line,
                )
            )
        )
        tin = self._extract_first_pattern(
            [
                r"\bTIN\s*[:;]?\s*(\d{3}-\d{2}-\d{4})",
                r"\bTIN\s*[:;]?\s*(\d{2}-\d{7})",
                r"\b(\d{3}-\d{2}-\d{4})\b",
                r"\b(\d{2}-\d{7})\b",
            ],
            tin_source,
        )
        tax_year = (
            self._find_first(r"(\d{4})", self._region_value(ocr_result, "tax_year") or "")
            or self._find_first(r"Tax\s*Year\s*[:;]?\s*(\d{4})", single_line)
        )
        region_issue_date = self._normalize_english_date(
            self._region_value(ocr_result, "issue_date")
        )
        fallback_issue_date = self._normalize_english_date(
            self._find_first(
                r"Date\s*[:;]?\s*([A-Za-z]+\s+\d{1,2}[,.]?\s*\d{4})",
                single_line,
            )
        )
        issue_date = (
            region_issue_date
            if self._is_valid_english_date(region_issue_date)
            else fallback_issue_date
        )
        residency_country = normalize_country(
            "United States of America"
            if self._contains_any(
                single_line,
                [
                    "resident of the united states of america",
                    "united states of america for purposes of u.s. taxation",
                ],
            )
            else None
        )

        fields = {
            "taxpayer_name": taxpayer_name,
            "tin": tin,
            "tax_year": tax_year,
            "issue_date": issue_date,
            "residency_country": residency_country,
            "residency_country_code": "US" if residency_country else None,
        }
        quality_checks = {
            "has_certification_text": self._contains_any(
                single_line,
                [
                    "resident of the united states of america for purposes of u.s. taxation",
                    "certification",
                    "i certify that",
                ],
            ),
            "has_irs_heading": self._contains_any(
                single_line,
                [
                    "department of the treasury",
                    "internal revenue service",
                ],
            ),
        }
        return ExtractedDocument(
            document_type=self.document_type,
            source_path=str(source_path),
            template_id=ocr_result.template_id,
            fields=fields,
            quality_checks=quality_checks,
            parser_warnings=[],
        )

    def _clean_taxpayer_value(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"Taxpayer\s*:?", " ", value, flags=re.IGNORECASE)
        cleaned = re.sub(r"TIN\s*:?.*$", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z .'-]", " ", cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        return cleaned or None
