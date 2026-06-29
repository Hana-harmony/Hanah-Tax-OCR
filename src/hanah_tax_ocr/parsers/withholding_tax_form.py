from __future__ import annotations

import re
from pathlib import Path

from hanah_tax_ocr.parsers.base import BaseDocumentParser
from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, OCRResult


class WithholdingTaxFormParser(BaseDocumentParser):
    document_type = DocumentType.WITHHOLDING_TAX_FORM

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        single_line = self._normalize_whitespace(text)
        tin_source = self._normalize_whitespace(
            " ".join(
                part
                for part in [
                    self._region_text(ocr_result, (0.14, 0.27, 0.31, 0.33)),
                    single_line,
                ]
                if part
            )
        )
        dividend_rate_source = self._normalize_whitespace(
            " ".join(
                part
                for part in [
                    self._region_text(ocr_result, (0.72, 0.39, 0.90, 0.45)),
                    single_line,
                ]
                if part
            )
        )

        last_name = self._normalize_name(
            self._compact_name_value(self._region_text(ocr_result, (0.14, 0.16, 0.30, 0.20)))
            or self._compact_name_value(self._region_text(ocr_result, (0.14, 0.15, 0.31, 0.21)))
            or self._find_first(r"Last Name\)?\s*([A-Z]{2,})\b", single_line)
            or self._find_first(
                r"Last Name\)?\s*([A-Za-z][A-Za-z .'-]+?)\s+\(?First Name\)?",
                single_line,
            )
            or self._find_first(r"Last Name\)?\s*([A-Za-z][A-Za-z .'-]+)", single_line)
        )
        first_name = self._normalize_name(
            self._compact_name_value(self._region_text(ocr_result, (0.40, 0.16, 0.54, 0.20)))
            or self._compact_name_value(self._region_text(ocr_result, (0.40, 0.15, 0.55, 0.21)))
            or self._find_first(r"First Name\)?\s*([A-Z]{2,})\b", single_line)
            or
            self._find_first(
                r"First Name\)?\s*([A-Za-z][A-Za-z .'-]+?)\s+\(?Middle Name\)?",
                single_line,
            )
            or self._find_first(r"First Name\)?\s*([A-Za-z][A-Za-z .'-]+)", single_line)
        )
        middle_name = self._normalize_name(
            self._compact_name_value(self._region_text(ocr_result, (0.68, 0.16, 0.76, 0.20)))
            or self._compact_name_value(self._region_text(ocr_result, (0.68, 0.15, 0.77, 0.21)))
            or self._find_first(r"Middle Name\)?\s*([A-Z])\b", single_line)
            or
            self._find_first(
                r"Middle Name\)?\s*([A-Za-z][A-Za-z .'-]+?)\s+"
                r"(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|Address|주소|납세자번호|Taxpayer)",
                single_line,
            )
            or self._find_first(r"Middle Name\)?\s*([A-Za-z][A-Za-z .'-]+)", single_line)
        )
        tin = self._extract_first_pattern(
            [
                r"납세자번호\s*(\d{3}-\d{2}-\d{4})",
                r"(?:Taxpayer Identification Number|TIN)\s*[:;]?\s*(\d{3}-\d{2}-\d{4})",
                r"(?:Taxpayer Identification Number|TIN)\s*[:;]?\s*(\d{2}-\d{7})",
                r"\b(\d{3}-\d{2}-\d{4})\b",
                r"\b(\d{2}-\d{7})\b",
            ],
            tin_source,
        )
        address = self._normalize_whitespace(
            self._find_first(
                r"(1\d{3,4}\s+[A-Za-z0-9 ,.'#-]+?(?:United States of America|USA))",
                single_line,
            )
            or self._region_text(ocr_result, (0.16, 0.21, 0.86, 0.25))
            or
            self._find_first(
                r"주소(?:\s*\([^)]+\))?\s*([0-9A-Za-z ,.'#-]+?(?:United States of America|USA))",
                single_line,
            )
            or self._find_first(
                r"Address\s*[:;]?\s*([0-9A-Za-z ,.'#-]+?(?:United States of America|USA))",
                single_line,
            )
            or ""
        ) or None
        country = self._clean_country_value(
            self._region_text(ocr_result, (0.56, 0.27, 0.83, 0.33))
            or self._find_first(
                r"거주지국\s*(United States of America|USA)",
                single_line,
            )
            or self._find_first(
                r"Country\s*[:;]?\s*(United States of America|USA)",
                single_line,
            )
        )
        country_code = self._clean_country_code_value(
            self._region_text(ocr_result, (0.86, 0.27, 0.98, 0.33))
            or self._find_first(
                r"거주지국코드\s*([A-Z]{2})",
                single_line,
            )
            or self._find_first(r"Country Code\s*[:;]?\s*([A-Z]{2})", single_line)
        )
        dividend_tax_rate = self._extract_first_pattern(
            [
                r"배당소득\s*세율\s*(\d{1,2}\s*%)",
                r"배당소득\s*세율\s*(\d{1,2})(?=\s*(?:양도소득|%))",
                r"Dividend(?:\s+Income)?\s+Tax\s+Rate\s*(\d{1,2}\s*%)",
                r"(15\s*%)",
                r"\b(15)\b",
            ],
            dividend_rate_source,
        )
        signature_date = self._normalize_iso_date(
            self._find_first(
                r"(\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일)",
                single_line,
            )
            or self._find_first(
                r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})",
                single_line,
            )
            or self._region_text(ocr_result, (0.70, 0.74, 0.96, 0.81))
        )
        applicant_name = self._normalize_name(
            self._clean_name_value(self._region_text(ocr_result, (0.40, 0.79, 0.72, 0.85)))
            or
            self._find_first(
                r"신청인\s+([A-Za-z][A-Za-z .'-]+?)\s+\(?서명",
                single_line,
            )
            or self._find_first(r"신청인\s+([A-Za-z][A-Za-z .'-]+)", single_line)
        )
        middle_name = self._derive_middle_name(
            middle_name,
            applicant_name=applicant_name,
            first_name=first_name,
            last_name=last_name,
        )

        fields = {
            "first_name": first_name,
            "last_name": last_name,
            "middle_name": middle_name,
            "tin": tin,
            "address": address,
            "residency_country": country,
            "residency_country_code": country_code,
            "dividend_tax_rate": self._normalize_tax_rate(dividend_tax_rate),
            "signature_date": signature_date,
            "applicant_name": applicant_name,
        }
        quality_checks = {
            "contains_non_resident_heading": self._contains_any(
                single_line,
                ["국내원천소득 제한세율 적용신청서", "비거주자용"],
            ),
        }
        return ExtractedDocument(
            document_type=self.document_type,
            source_path=str(source_path),
            fields=fields,
            quality_checks=quality_checks,
            parser_warnings=[],
        )

    def _clean_name_value(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(
            r"\b(Last|First|Middle|Name)\b",
            " ",
            value,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"[^A-Za-z .'-]", " ", cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        return cleaned or None

    def _compact_name_value(self, value: str | None) -> str | None:
        cleaned = self._clean_name_value(value)
        if not cleaned:
            return None
        tokens = cleaned.split()
        if not tokens:
            return None
        if len(tokens) == 1:
            return tokens[0]
        if len(tokens) == 2 and all(len(token) <= 10 for token in tokens):
            return tokens[0]
        return None

    def _clean_country_value(self, value: str | None) -> str | None:
        if not value:
            return None
        if "United States" in value:
            return "United States of America"
        if re.search(r"\bUSA\b", value, re.IGNORECASE):
            return "USA"
        return None

    def _clean_country_code_value(self, value: str | None) -> str | None:
        if not value:
            return None
        match = re.search(r"\b([A-Z]{2})\b", value.upper())
        return match.group(1) if match else None

    def _derive_middle_name(
        self,
        middle_name: str | None,
        *,
        applicant_name: str | None,
        first_name: str | None,
        last_name: str | None,
    ) -> str | None:
        if middle_name and len(middle_name.split()) == 1 and len(middle_name) <= 4:
            return middle_name
        if not applicant_name or not first_name or not last_name:
            return middle_name
        tokens = [token.strip(".") for token in applicant_name.split()]
        if len(tokens) < 3:
            return middle_name
        if tokens[0].lower() == first_name.lower() and tokens[-1].lower() == last_name.lower():
            return tokens[1][:1].upper()
        return middle_name

    def _normalize_tax_rate(self, value: str | None) -> str | None:
        if not value:
            return None
        digits = re.search(r"(\d{1,2})", value)
        if not digits:
            return None
        return f"{digits.group(1)}%"
