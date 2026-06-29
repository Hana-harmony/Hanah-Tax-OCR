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
            "issue_date": "January 12, 2026",
            "residency_country": "United States of America",
            "residency_country_code": "US",
        },
        quality_checks={
            "has_certification_text": True,
            "seal_present": True,
            "signature_present": True,
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
            "residency_country_code": "US",
            "dividend_tax_rate": "10%",
            "signature_date": None,
        },
        quality_checks={"signature_present": False, "all_no_boxes_checked": False},
    )

    result = reviewer.review([residency, withholding])

    assert result.status == ReviewStatus.REJECT
    assert {finding.code for finding in result.findings} >= {
        "required_address_missing",
        "required_dividend_rate_invalid",
        "required_signature_missing",
        "required_no_checkbox_missing",
    }


def test_reviewer_passes_clean_cross_check_pair() -> None:
    reviewer = TaxDocumentReviewer()
    residency = ExtractedDocument(
        document_type=DocumentType.RESIDENCY_CERTIFICATE,
        source_path="residency.png",
        fields={
            "taxpayer_name": "Jane Q. Doe",
            "tin": "123-45-6789",
            "tax_year": "2026",
            "issue_date": "January 12, 2026",
            "residency_country": "United States of America",
            "residency_country_code": "US",
        },
        quality_checks={
            "has_certification_text": True,
            "seal_present": True,
            "signature_present": True,
        },
    )
    withholding = ExtractedDocument(
        document_type=DocumentType.WITHHOLDING_TAX_FORM,
        source_path="withholding.png",
        fields={
            "first_name": "Jane",
            "middle_name": "Q.",
            "last_name": "Doe",
            "tin": "123-45-6789",
            "address": "1 Main St",
            "residency_country": "United States of America",
            "residency_country_code": "US",
            "dividend_tax_rate": "15%",
            "signature_date": "2026-01-12",
        },
        quality_checks={"signature_present": True, "all_no_boxes_checked": True},
    )

    result = reviewer.review([residency, withholding])

    assert result.status == ReviewStatus.PASS
    assert result.cross_check["matched"] is True
