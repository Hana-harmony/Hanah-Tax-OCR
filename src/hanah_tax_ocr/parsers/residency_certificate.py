from __future__ import annotations

from pathlib import Path

from hanah_tax_ocr.parsers.base import BaseDocumentParser
from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, OCRResult


class ResidencyCertificateParser(BaseDocumentParser):
    document_type = DocumentType.RESIDENCY_CERTIFICATE

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        taxpayer_name = self._find_first(r"Taxpayer:\s*(.+)", text)
        tin = self._find_first(r"TIN:\s*([A-Z0-9-]+)", text)
        tax_year = self._find_first(r"Tax Year:\s*(\d{4})", text)
        issue_date = self._find_first(r"Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)

        fields = {
            "taxpayer_name": taxpayer_name,
            "tin": tin,
            "tax_year": tax_year,
            "issue_date": issue_date,
            "residency_country": "United States of America",
            "residency_country_code": "US",
        }
        quality_checks = {
            "has_certification_text": self._contains_any(
                text,
                [
                    "resident of the united states of america for purposes of u.s. taxation",
                    "certification",
                ],
            ),
            "signature_present": self._contains_any(text, ["director", "accounts management"]),
        }
        warnings = [
            "Manual signature/stamp detection is not implemented yet.",
        ]
        return ExtractedDocument(
            document_type=self.document_type,
            source_path=str(source_path),
            fields=fields,
            quality_checks=quality_checks,
            parser_warnings=warnings,
        )
