from hanah_tax_ocr.parsers import (
    ApostilleParser,
    ResidencyCertificateParser,
    WithholdingTaxFormParser,
)
from hanah_tax_ocr.schemas import OCRPage, OCRResult


def build_result(lines: list[str]) -> OCRResult:
    return OCRResult(pages=[OCRPage(page_number=1, raw_text="\n".join(lines))])


def test_residency_parser_extracts_core_fields() -> None:
    parser = ResidencyCertificateParser()
    parsed = parser.parse(
        build_result(
            [
                "DEPARTMENT OF THE TREASURY",
                "INTERNAL REVENUE SERVICE",
                "Date: January 12, 2026",
                "Taxpayer: MARIA L.CHEN",
                "TIN: 987-65-4321",
                "Tax Year: 2026",
                "I certify that the above-named taxpayer is a resident of the "
                "United States of America for purposes of U.S. taxation.",
            ]
        ),
        "residency.png",
    )

    assert parsed.fields["taxpayer_name"] == "MARIA L. CHEN"
    assert parsed.fields["tin"] == "987-65-4321"
    assert parsed.fields["issue_date"] == "January 12, 2026"
    assert parsed.quality_checks["has_certification_text"] is True


def test_apostille_parser_extracts_standard_items() -> None:
    parser = ApostilleParser()
    parsed = parser.parse(
        build_result(
            [
                "APOSTILLE (Convention de La Haye du 5 octobre 1961)",
                "1. Country: UNITED STATES OF AMERICA",
                "2. This Public Document has been signed by CHONG U CHOI",
                "3. acting in the capacity of NOTARY PUBLIC COMMISSION EXPIRES 10/17/2016",
                "4. bears the seal/stamp of COUNTY OF MECKLENBURG, NORTH CAROLINA",
                "5. at Raleigh, North Carolina",
                "6. the 10TH DAY OF APRIL, 2014",
                "7. by Secretary of State or Deputy Secretary of State, State of North Carolina",
                "8. No. 5",
            ]
        ),
        "apostille.jpg",
    )

    assert parsed.fields["issuing_country"] == "UNITED STATES OF AMERICA"
    assert parsed.fields["signed_by"] == "CHONG U CHOI"
    assert parsed.fields["certificate_number"] == "5"
    assert parsed.quality_checks["filled_item_count"] == 8


def test_apostille_parser_prefers_full_text_over_noisy_regions() -> None:
    parser = ApostilleParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "APOSTILLE (Convention de La Haye du 5 octobre 1961)",
                            "1. Country: UNITED STATES OF AMERICA",
                            "2. This Public Document has been signed by CHONG U CHOI",
                            "3. acting in the capacity of "
                            "NOTARY PUBLIC COMMISSION EXPIRES 10/17/2016",
                            "4. bears the seal/stamp of COUNTY OF MECKLENBURG, NORTH CAROLINA",
                            "5. at Raleigh, North Carolina",
                            "6. the 10TH DAY OF APRIL, 2014",
                            "7. by Secretary of State or Deputy Secretary of State, "
                            "State of North Carolina",
                            "8. No. 5",
                        ]
                    ),
                )
            ],
            regions={
                "issuing_country": OCRPage(page_number=1, raw_text="ED STATES OF AMERICA"),
                "signed_by": OCRPage(page_number=1, raw_text="IOHO Y PUBLIC"),
            },
        ),
        "north_carolina_apostille.jpg",
    )

    assert parsed.fields["issuing_country"] == "UNITED STATES OF AMERICA"
    assert parsed.fields["signed_by"] == "CHONG U CHOI"


def test_withholding_parser_extracts_sample_fields() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        build_result(
            [
                "국내원천소득 제한세율 적용신청서(비거주자용)",
                "(Last Name) CHEN (First Name) MARIA (Middle Name) L",
                "주소 1234 Sunset Blvd, Apt 5B, Los Angeles, CA 90026, United States of America",
                "납세자번호 987-65-4321 거주지국 United States of America 거주지국코드 US",
                "배당소득 세율 15 %",
                "2026년 01월 12일 신청인 MARIA L. CHEN (서명 또는 인)",
            ]
        ),
        "withholding.png",
    )

    assert parsed.fields["last_name"] == "CHEN"
    assert parsed.fields["first_name"] == "MARIA"
    assert parsed.fields["middle_name"] == "L"
    assert parsed.fields["dividend_tax_rate"] == "15%"
    assert parsed.fields["signature_date"] == "2026-01-12"


def test_withholding_parser_recovers_from_region_ocr_noise() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "국내원천소득 제한세율 적용신청서(비거주자용)",
                            "주소 1234 Sunset Blvd, Apt 5B, Los Angeles, CA 90026, "
                            "United States of America",
                            "납세자번호 987-65-4321",
                            "배당소득 세율 15 %",
                            "2026년 01월 12일 신청인 MARIA L. CHEN (서명 또는 인)",
                        ]
                    ),
                )
            ],
            regions={
                "last_name": OCRPage(page_number=1, raw_text="CHEN"),
                "first_name": OCRPage(page_number=1, raw_text="irst"),
                "middle_name": OCRPage(page_number=1, raw_text="iddle"),
                "residency_country": OCRPage(
                    page_number=1,
                    raw_text="nited States 이f America 전화",
                ),
                "applicant_name": OCRPage(page_number=1, raw_text="MARIA L CHEN"),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["first_name"] == "MARIA"
    assert parsed.fields["middle_name"] == "L"
    assert parsed.fields["last_name"] == "CHEN"
    assert parsed.fields["residency_country"] == "United States of America"
    assert parsed.fields["residency_country_code"] == "US"
