from __future__ import annotations

from pathlib import Path

from hanah_tax_ocr.parsers.base import BaseDocumentParser
from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, OCRResult


class WithholdingTaxFormParser(BaseDocumentParser):
    document_type = DocumentType.WITHHOLDING_TAX_FORM

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        first_name = self._find_first(r"First Name[\s:]+([A-Za-z ,.'-]+)", text)
        last_name = self._find_first(r"Last Name[\s:]+([A-Za-z ,.'-]+)", text)
        tin = self._find_first(r"(?:Taxpayer Identification Number|TIN)[\s:]+([A-Z0-9-]+)", text)
        address = self._find_first(r"Address[\s:]+(.+)", text)
        country = self._find_first(r"Country[\s:]+([A-Za-z ]+)", text)
        country_code = self._find_first(r"Country Code[\s:]+([A-Z]{2})", text)

        fields = {
            "first_name": first_name,
            "last_name": last_name,
            "tin": tin,
            "address": address,
            "residency_country": country,
            "residency_country_code": country_code,
            "dividend_tax_rate": self._find_first(r"(15\s*%)", text),
        }
        quality_checks = {
            "all_no_boxes_checked": self._contains_any(text, ["no", "아니오"]),
            "signature_present": None,
            "date_present": self._find_first(r"(\d{4}[./-]\d{1,2}[./-]\d{1,2})", text)
            or self._find_first(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", text),
        }
        warnings = [
            "Checkbox and signature image detection are not implemented yet.",
        ]
        return ExtractedDocument(
            document_type=self.document_type,
            source_path=str(source_path),
            fields=fields,
            quality_checks=quality_checks,
            parser_warnings=warnings,
        )
