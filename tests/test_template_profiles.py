from hanah_tax_ocr.schemas import DocumentType
from hanah_tax_ocr.template_profiles import classify_template


def test_classify_template_detects_north_carolina_apostille_from_text() -> None:
    profile = classify_template(
        DocumentType.APOSTILLE,
        "sample_data/아포스티유 샘플/미국 california 주2.jpg",
        "STATE OF NORTH CAROLINA APOSTILLE 5. at Raleigh, North Carolina",
    )

    assert profile is not None
    assert profile.template_id == "apostille.north_carolina"


def test_classify_template_detects_withholding_form() -> None:
    profile = classify_template(
        DocumentType.WITHHOLDING_TAX_FORM,
        "sample_data/국내원천소득 제한세율/국내원천소득 제한세율 적용신청서-1.png",
        "국내원천소득 제한세율 적용신청서(비거주자용)",
    )

    assert profile is not None
    assert profile.template_id == "withholding.non_resident_v1"
