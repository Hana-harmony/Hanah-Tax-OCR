from __future__ import annotations

from pathlib import Path

from hanah_tax_ocr.parsers.base import BaseDocumentParser
from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, OCRResult


class ApostilleParser(BaseDocumentParser):
    document_type = DocumentType.APOSTILLE

    def parse(self, ocr_result: OCRResult, source_path: str | Path) -> ExtractedDocument:
        text = ocr_result.combined_text()
        issuing_country = self._find_first(r"United States of America", text)
        signer_name = self._find_first(r"(?:signed by|name of signer)[\s:]+(.+)", text)
        signer_capacity = self._find_first(
            r"(?:acting in the capacity of|capacity)[\s:]+(.+)",
            text,
        )

        fields = {
            "issuing_country": issuing_country or "United States of America",
            "signer_name": signer_name,
            "signer_capacity": signer_capacity,
        }
        quality_checks = {
            "has_apostille_heading": self._contains_any(text, ["apostille", "hague convention"]),
            "seal_present": None,
            "signature_present": None,
        }
        warnings = [
            "Apostille 10-field schema is not finalized in specification.",
            "Seal/signature image detection is pending.",
        ]
        return ExtractedDocument(
            document_type=self.document_type,
            source_path=str(source_path),
            fields=fields,
            quality_checks=quality_checks,
            parser_warnings=warnings,
        )
