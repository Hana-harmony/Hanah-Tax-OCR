from hanah_tax_ocr.review import TaxDocumentReviewer
from hanah_tax_ocr.schemas import DocumentType, ExtractedDocument, ReviewStatus


def test_reviewer_rejects_missing_required_fields() -> None:
    reviewer = TaxDocumentReviewer()
    residency = ExtractedDocument(
        document_type=DocumentType.RESIDENCY_CERTIFICATE,
        source_path="residency.png",
        fields={
            "taxpayer_name": "Jane Doe",
            "tin": "123-45-6789",
            "tax_year": "2026",
            "residency_country": "United States of America",
        },
    )
    withholding = ExtractedDocument(
        document_type=DocumentType.WITHHOLDING_TAX_FORM,
        source_path="withholding.png",
        fields={
            "first_name": "Jane",
            "last_name": "Doe",
            "tin": "123-45-6789",
            "address": None,
            "residency_country": "United States of America",
            "dividend_tax_rate": "10%",
        },
        quality_checks={"signature_present": None},
    )

    result = reviewer.review([residency, withholding])

    assert result.status == ReviewStatus.REJECT
    assert {finding.code for finding in result.findings} >= {
        "required_address_missing",
        "required_dividend_rate_invalid",
        "required_signature_review",
    }


def test_reviewer_passes_clean_cross_check_pair() -> None:
    reviewer = TaxDocumentReviewer()
    residency = ExtractedDocument(
        document_type=DocumentType.RESIDENCY_CERTIFICATE,
        source_path="residency.png",
        fields={
            "taxpayer_name": "Jane Doe",
            "tin": "123-45-6789",
            "tax_year": "2026",
            "residency_country": "United States of America",
        },
        quality_checks={"has_certification_text": True},
    )
    withholding = ExtractedDocument(
        document_type=DocumentType.WITHHOLDING_TAX_FORM,
        source_path="withholding.png",
        fields={
            "first_name": "Jane",
            "last_name": "Doe",
            "tin": "123-45-6789",
            "address": "1 Main St",
            "residency_country": "United States of America",
            "dividend_tax_rate": "15%",
        },
        quality_checks={"signature_present": True},
    )

    result = reviewer.review([residency, withholding])

    assert result.status == ReviewStatus.PASS
    assert result.cross_check["matched"] is True
