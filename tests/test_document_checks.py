from pathlib import Path

from hanah_tax_ocr.document_checks import compute_document_checks
from hanah_tax_ocr.schemas import DocumentType

ROOT = Path(__file__).resolve().parents[1]


def test_document_checks_detect_residency_seal_and_signature() -> None:
    checks = compute_document_checks(
        DocumentType.RESIDENCY_CERTIFICATE,
        ROOT / "sample_data" / "거주자증명서" / "미국 TREASURY주.png",
    )

    assert checks["seal_present"] is True
    assert checks["signature_present"] is True


def test_document_checks_detect_withholding_signature_and_no_checkboxes() -> None:
    checks = compute_document_checks(
        DocumentType.WITHHOLDING_TAX_FORM,
        ROOT
        / "sample_data"
        / "국내원천소득 제한세율"
        / "국내원천소득 제한세율 적용신청서-1.png",
    )

    assert checks["signature_present"] is True
    assert checks["all_no_boxes_checked"] is True
    assert checks["checked_no_box_count"] == 7


def test_document_checks_detect_apostille_seal_and_signature() -> None:
    checks = compute_document_checks(
        DocumentType.APOSTILLE,
        ROOT / "sample_data" / "아포스티유 샘플" / "미국 california 주2.jpg",
    )

    assert checks["seal_present"] is True
    assert checks["signature_present"] is True
