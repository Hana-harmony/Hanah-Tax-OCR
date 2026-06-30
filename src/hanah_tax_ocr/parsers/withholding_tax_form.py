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
    _AMBIGUOUS_ALNUM_DIGIT_MAP = {
        "O": "0",
        "I": "1",
        "L": "1",
        "T": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        single_line = self._normalize_whitespace(text)
        region_address = normalize_address(self._region_value(ocr_result, "address"))
        full_text_address = normalize_address(
            self._find_first(
                r"(\d{1,5}\s+[A-Za-z0-9 ,.'#-]+?(?:United States of America|USA))",
                single_line,
            )
            or self._find_first(
                r"주소(?:\s*\([^)]+\))?\s*([0-9A-Za-z ,.'#-]+?(?:United States of America|USA))",
                single_line,
            )
        )
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
            self._merge_address_candidates(
                region_address=region_address,
                full_text_address=full_text_address,
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
        first_name = self._prefer_applicant_first_name(
            first_name,
            applicant_name=applicant_name,
            middle_name=middle_name,
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

    def _merge_address_candidates(
        self,
        *,
        region_address: str | None,
        full_text_address: str | None,
    ) -> str | None:
        if region_address and full_text_address:
            sanitized_full_text_address = self._sanitize_full_text_address(full_text_address)
            full_text_street_number = self._leading_street_number(full_text_address)
            repaired_region_address = region_address
            if full_text_street_number and not self._leading_street_number(region_address):
                repaired_region_address = f"{full_text_street_number} {region_address}"
            if self._contains_leading_name_noise(full_text_address):
                region_has_suite_number = re.search(
                    r"\bSuite\s+[0-9][A-Za-z0-9-]*",
                    region_address,
                    re.IGNORECASE,
                )
                sanitized_has_suite_number = sanitized_full_text_address and re.search(
                    r"\bSuite\s+[0-9][A-Za-z0-9-]*",
                    sanitized_full_text_address,
                    re.IGNORECASE,
                )
                if sanitized_full_text_address and (
                    not region_has_suite_number or sanitized_has_suite_number
                ):
                    return sanitized_full_text_address
                return repaired_region_address
            return sanitized_full_text_address or full_text_address
        return self._sanitize_full_text_address(full_text_address) or region_address

    def _leading_street_number(self, value: str | None) -> str | None:
        if not value:
            return None
        match = re.match(r"^(\d{1,5})\b", value)
        if not match:
            return None
        return match.group(1)

    def _contains_leading_name_noise(self, value: str | None) -> bool:
        if not value:
            return False
        return re.search(r"^\d{1,5}\s+USER\b", value, re.IGNORECASE) is not None

    def _sanitize_full_text_address(self, value: str | None) -> str | None:
        if not value:
            return None
        sanitized = re.sub(r"^\d{1,5}\s+USER\b\s+", "", value, flags=re.IGNORECASE)
        sanitized = normalize_address(sanitized)
        return sanitized or None

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
        first_name = self._normalize_alphanumeric_name_token(tokens[0])
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
        normalized_middle_name = self._normalize_middle_initial(middle_name)
        if normalized_middle_name and len(normalized_middle_name.split()) == 1 and len(normalized_middle_name) <= 4:
            return normalized_middle_name
        if not applicant_name or not first_name or not last_name:
            return normalized_middle_name
        tokens = [token.strip(".") for token in applicant_name.split()]
        if len(tokens) < 3:
            return normalized_middle_name
        if tokens[0].lower() == first_name.lower() and tokens[-1].lower() == last_name.lower():
            return self._normalize_middle_initial(tokens[1][:1].upper())
        return normalized_middle_name

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

    def _prefer_applicant_first_name(
        self,
        first_name: str | None,
        *,
        applicant_name: str | None,
        middle_name: str | None,
        last_name: str | None,
    ) -> str | None:
        derived_first_name, derived_middle_name, derived_last_name = self._derive_name_parts(
            applicant_name
        )
        normalized_first_name = self._normalize_alphanumeric_name_token(first_name)
        normalized_derived_first_name = self._normalize_alphanumeric_name_token(derived_first_name)
        if not derived_first_name or not first_name:
            return normalized_first_name or normalized_derived_first_name
        if not last_name or derived_last_name != last_name:
            return normalized_first_name
        if middle_name and derived_middle_name and middle_name != derived_middle_name:
            return normalized_first_name
        if canonicalize_name(normalized_derived_first_name) == canonicalize_name(normalized_first_name):
            return normalized_first_name
        if self._digit_count(normalized_derived_first_name) > self._digit_count(normalized_first_name):
            return normalized_derived_first_name
        return normalized_first_name

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

    def _normalize_middle_initial(self, value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.strip().strip(".")
        if len(normalized) != 1:
            return value
        if normalized == "0":
            return "O"
        return normalized.upper()

    def _normalize_alphanumeric_name_token(self, value: str | None) -> str | None:
        if not value:
            return None
        token = value.strip().strip(".").upper()
        if not re.fullmatch(r"[A-Z0-9]+", token):
            return value
        first_digit_index = next((index for index, char in enumerate(token) if char.isdigit()), -1)
        if first_digit_index <= 0:
            return value
        prefix = token[:first_digit_index]
        suffix = token[first_digit_index:]
        normalized_suffix = "".join(
            self._AMBIGUOUS_ALNUM_DIGIT_MAP.get(char, char) for char in suffix
        )
        if normalized_suffix == suffix:
            return value
        if not normalized_suffix.isdigit():
            return value
        normalized = prefix + normalized_suffix
        if self._digit_count(normalized) <= self._digit_count(token):
            return value
        return normalized

    def _digit_count(self, value: str | None) -> int:
        if not value:
            return 0
        return sum(character.isdigit() for character in value)
