from __future__ import annotations

import re

from hanah_tax_ocr.schemas import (
    DocumentType,
    ExtractedDocument,
    ReviewFinding,
    ReviewResult,
    ReviewStatus,
)


class TaxDocumentReviewer:
    """Business-rule reviewer for extracted OCR fields and cross-document consistency."""

    def review(self, documents: list[ExtractedDocument]) -> ReviewResult:
        findings: list[ReviewFinding] = []
        indexed = {document.document_type: document for document in documents}

        residency_doc = indexed.get(DocumentType.RESIDENCY_CERTIFICATE)
        if residency_doc:
            findings.extend(self._review_residency_certificate(residency_doc))

        apostille_doc = indexed.get(DocumentType.APOSTILLE)
        if apostille_doc:
            findings.extend(self._review_apostille(apostille_doc))

        withholding_doc = indexed.get(DocumentType.WITHHOLDING_TAX_FORM)
        if withholding_doc:
            findings.extend(self._review_withholding_tax_form(withholding_doc))

        cross_check = self._cross_check(residency_doc, withholding_doc)
        findings.extend(cross_check["findings"])

        severe_findings = [item for item in findings if item.code.startswith("required_")]
        status = (
            ReviewStatus.PASS
            if not severe_findings and not findings
            else ReviewStatus.NEEDS_REVIEW
        )
        if severe_findings:
            status = ReviewStatus.REJECT

        return ReviewResult(
            status=status,
            findings=findings,
            cross_check={key: value for key, value in cross_check.items() if key != "findings"},
        )

    def _review_residency_certificate(self, document: ExtractedDocument) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        if not document.fields.get("taxpayer_name"):
            findings.append(
                self._finding(
                    "required_name_missing",
                    "Taxpayer name is missing.",
                    "taxpayer_name",
                )
            )
        if not self._is_ssn_or_ein(document.fields.get("tin")):
            findings.append(self._finding("required_tin_invalid", "TIN format is invalid.", "tin"))
        if not document.fields.get("tax_year"):
            findings.append(
                self._finding(
                    "required_tax_year_missing",
                    "Tax year is missing.",
                    "tax_year",
                )
            )
        return findings

    def _review_apostille(self, document: ExtractedDocument) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        if not document.quality_checks.get("has_apostille_heading"):
            findings.append(
                self._finding(
                    "required_apostille_heading_missing",
                    "Apostille heading not found.",
                )
            )
        if document.quality_checks.get("seal_present") is None:
            findings.append(
                self._finding(
                    "manual_seal_review_needed",
                    "Seal detection requires manual review.",
                )
            )
        return findings

    def _review_withholding_tax_form(self, document: ExtractedDocument) -> list[ReviewFinding]:
        findings: list[ReviewFinding] = []
        if not document.fields.get("first_name") or not document.fields.get("last_name"):
            findings.append(self._finding("required_name_missing", "First/last name is missing."))
        if not document.fields.get("address"):
            findings.append(
                self._finding(
                    "required_address_missing",
                    "Address is missing.",
                    "address",
                )
            )
        if not self._is_ssn_or_ein(document.fields.get("tin")):
            findings.append(self._finding("required_tin_invalid", "TIN format is invalid.", "tin"))
        if document.fields.get("dividend_tax_rate") != "15%":
            findings.append(
                self._finding(
                    "required_dividend_rate_invalid",
                    "Dividend tax rate must be 15%.",
                    "dividend_tax_rate",
                )
            )
        if document.quality_checks.get("signature_present") is None:
            findings.append(
                self._finding(
                    "required_signature_review",
                    "Signature detection requires manual review.",
                    "signature",
                )
            )
        return findings

    def _cross_check(
        self,
        residency_doc: ExtractedDocument | None,
        withholding_doc: ExtractedDocument | None,
    ) -> dict[str, object]:
        findings: list[ReviewFinding] = []
        matched = True

        if not residency_doc or not withholding_doc:
            return {
                "matched": False,
                "reason": "cross-check skipped because one of the documents is missing",
                "findings": findings,
            }

        pairs = [
            ("tin", residency_doc.fields.get("tin"), withholding_doc.fields.get("tin")),
            (
                "residency_country",
                residency_doc.fields.get("residency_country"),
                withholding_doc.fields.get("residency_country"),
            ),
        ]
        for field_name, left_value, right_value in pairs:
            if left_value and right_value and str(left_value).strip() != str(right_value).strip():
                matched = False
                findings.append(
                    self._finding(
                        "required_cross_check_mismatch",
                        f"Cross-check mismatch for {field_name}.",
                        field_name,
                    )
                )

        residency_name = residency_doc.fields.get("taxpayer_name")
        withholding_name_parts = [
            withholding_doc.fields.get("first_name"),
            withholding_doc.fields.get("last_name"),
        ]
        withholding_name = " ".join(part for part in withholding_name_parts if part).strip()
        if (
            residency_name
            and withholding_name
            and residency_name.lower() != withholding_name.lower()
        ):
            matched = False
            findings.append(
                self._finding(
                    "required_cross_check_mismatch",
                    "Cross-check mismatch for taxpayer name.",
                    "taxpayer_name",
                )
            )

        return {
            "matched": matched,
            "reason": None if matched else "one or more cross-check fields do not match",
            "findings": findings,
        }

    @staticmethod
    def _is_ssn_or_ein(value: str | None) -> bool:
        if not value:
            return False
        normalized = value.strip()
        return bool(
            re.fullmatch(r"\d{3}-\d{2}-\d{4}", normalized)
            or re.fullmatch(r"\d{2}-\d{7}", normalized)
        )

    @staticmethod
    def _finding(code: str, message: str, field_name: str | None = None) -> ReviewFinding:
        return ReviewFinding(code=code, message=message, field_name=field_name)
