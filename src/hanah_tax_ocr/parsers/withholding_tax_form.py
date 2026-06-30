from __future__ import annotations

import re
from pathlib import Path

from hanah_tax_ocr.normalization import (
    canonicalize_name,
    normalize_address,
    normalize_country,
    normalize_country_code,
    normalize_percentage,
)
from hanah_tax_ocr.parsers.base import BaseDocumentParser
from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, OCRResult


class WithholdingTaxFormParser(BaseDocumentParser):
    document_type = DocumentType.WITHHOLDING_TAX_FORM
    _LABEL_FRAGMENTS = {"last", "ast", "first", "irst", "middle", "iddle", "name"}

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        single_line = self._normalize_whitespace(text)
        tin_source = self._normalize_whitespace(
            " ".join(
                part
                for part in [self._region_value(ocr_result, "tin"), single_line]
                if part
            )
        )
        dividend_rate_source = self._normalize_whitespace(
            " ".join(
                part
                for part in [self._region_value(ocr_result, "dividend_tax_rate"), single_line]
                if part
            )
        )
        applicant_name = self._normalize_name(
            self._clean_name_value(self._region_value(ocr_result, "applicant_name"))
            or self._find_first(
                r"신청인\s+([A-Za-z][A-Za-z .'-]+?)\s+\(?서명",
                single_line,
            )
            or self._find_first(r"신청인\s+([A-Za-z][A-Za-z .'-]+)", single_line)
        )
        derived_first_name, derived_middle_name, derived_last_name = self._derive_name_parts(
            applicant_name
        )

        last_name = self._normalize_name(
            self._compact_name_value(self._region_value(ocr_result, "last_name"))
            or derived_last_name
            or self._find_first(r"Last Name\)?\s*([A-Z]{2,})\b", single_line)
            or self._find_first(
                r"Last Name\)?\s*([A-Za-z][A-Za-z .'-]+?)\s+\(?First Name\)?",
                single_line,
            )
        )
        first_name = self._normalize_name(
            self._compact_name_value(self._region_value(ocr_result, "first_name"))
            or derived_first_name
            or self._find_first(r"First Name\)?\s*([A-Z]{2,})\b", single_line)
            or self._find_first(
                r"First Name\)?\s*([A-Za-z][A-Za-z .'-]+?)\s+\(?Middle Name\)?",
                single_line,
            )
        )
        middle_name = self._normalize_name(
            self._compact_name_value(self._region_value(ocr_result, "middle_name"))
            or derived_middle_name
            or self._find_first(r"Middle Name\)?\s*([A-Z])\b", single_line)
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
        address = normalize_address(
            self._find_first(
                r"(\d{1,5}\s+[A-Za-z0-9 ,.'#-]+?(?:United States of America|USA))",
                single_line,
            )
            or self._region_value(ocr_result, "address")
            or self._find_first(
                r"주소(?:\s*\([^)]+\))?\s*([0-9A-Za-z ,.'#-]+?(?:United States of America|USA))",
                single_line,
            )
        )
        country = normalize_country(
            self._region_value(ocr_result, "residency_country")
            or self._find_first(r"거주지국\s*(United States of America|USA)", single_line)
            or self._find_first(r"Country\s*[:;]?\s*(United States of America|USA)", single_line)
            or address
        )
        country_code = normalize_country_code(
            self._region_value(ocr_result, "residency_country_code")
        ) or normalize_country_code(
            self._find_first(r"거주지국코드\s*([A-Z]{2})", single_line)
            or self._find_first(r"Country Code\s*[:;]?\s*([A-Z]{2})", single_line)
            or country
            or address
        )
        dividend_tax_rate = normalize_percentage(
            self._extract_first_pattern(
                [
                    r"배당소득\s*세율\s*(\d{1,2}\s*%)",
                    r"배당소득\s*세율\s*(\d{1,2})(?=\s*(?:양도소득|%))",
                    r"Dividend(?:\s+Income)?\s+Tax\s+Rate\s*(\d{1,2}\s*%)",
                    r"(15\s*%)",
                    r"\b(15)\b",
                ],
                dividend_rate_source,
            )
        )
        region_signature_date = self._normalize_iso_date(
            self._region_value(ocr_result, "signature_date")
        )
        fallback_signature_date = self._normalize_iso_date(
            self._find_first(r"(\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일)", single_line)
            or self._find_first(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})", single_line)
        )
        signature_date = (
            region_signature_date
            if self._is_valid_iso_date(region_signature_date)
            else fallback_signature_date
        )
        middle_name = self._derive_middle_name(
            middle_name,
            applicant_name=applicant_name,
            first_name=first_name,
            last_name=last_name,
        )
        applicant_name = self._rebuild_applicant_name(
            applicant_name,
            first_name=first_name,
            middle_name=middle_name,
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
            "dividend_tax_rate": dividend_tax_rate,
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
            template_id=ocr_result.template_id,
            fields=fields,
            quality_checks=quality_checks,
            parser_warnings=[],
        )

    def _clean_name_value(self, value: str | None) -> str | None:
        if not value:
            return None
        cleaned = re.sub(r"\b(Last|First|Middle|Name)\b", " ", value, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^A-Za-z0-9 .'-]", " ", cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        return cleaned or None

    def _compact_name_value(self, value: str | None) -> str | None:
        cleaned = self._clean_name_value(value)
        if not cleaned:
            return None
        tokens = cleaned.split()
        if not tokens:
            return None
        lowered_tokens = [token.strip(".").lower() for token in tokens]
        if any(token in self._LABEL_FRAGMENTS for token in lowered_tokens):
            return None
        if len(tokens) == 1:
            return tokens[0]
        if len(tokens) == 2 and all(len(token) <= 10 for token in tokens):
            return tokens[0]
        return None

    def _derive_name_parts(
        self,
        applicant_name: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        if not applicant_name:
            return None, None, None
        tokens = [token.strip(".") for token in applicant_name.split() if token.strip(".")]
        if len(tokens) < 2:
            return None, None, None
        first_name = tokens[0]
        last_name = tokens[-1]
        middle_name = tokens[1][:1].upper() if len(tokens) >= 3 else None
        return first_name, middle_name, last_name

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

    def _rebuild_applicant_name(
        self,
        applicant_name: str | None,
        *,
        first_name: str | None,
        middle_name: str | None,
        last_name: str | None,
    ) -> str | None:
        rebuilt = self._format_applicant_name(first_name, middle_name, last_name)
        if not rebuilt:
            return applicant_name
        if not applicant_name:
            return rebuilt
        if canonicalize_name(applicant_name) != canonicalize_name(rebuilt):
            return rebuilt
        if applicant_name != rebuilt and re.fullmatch(r"[A-Za-z '.-]+", rebuilt):
            return rebuilt
        return applicant_name

    def _format_applicant_name(
        self,
        first_name: str | None,
        middle_name: str | None,
        last_name: str | None,
    ) -> str | None:
        if not first_name and not middle_name and not last_name:
            return None
        formatted_middle = middle_name
        if (
            middle_name
            and len(middle_name) == 1
            and middle_name.isalpha()
            and (first_name or "").isalpha()
            and (last_name or "").isalpha()
        ):
            formatted_middle = f"{middle_name}."
        return " ".join(part for part in [first_name, formatted_middle, last_name] if part).strip()
