from __future__ import annotations

from hanah_tax_ocr.parsers.apostille import ApostilleParser
from hanah_tax_ocr.parsers.base import BaseDocumentParser
from hanah_tax_ocr.parsers.residency_certificate import ResidencyCertificateParser
from hanah_tax_ocr.parsers.withholding_tax_form import WithholdingTaxFormParser
from hanah_tax_ocr.schemas import DocumentType


def build_parser_registry() -> dict[DocumentType, BaseDocumentParser]:
    return {
        DocumentType.RESIDENCY_CERTIFICATE: ResidencyCertificateParser(),
        DocumentType.APOSTILLE: ApostilleParser(),
        DocumentType.WITHHOLDING_TAX_FORM: WithholdingTaxFormParser(),
    }
