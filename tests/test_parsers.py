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


def test_residency_parser_falls_back_when_region_issue_date_is_invalid() -> None:
    parser = ResidencyCertificateParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "DEPARTMENT OF THE TREASURY",
                            "INTERNAL REVENUE SERVICE",
                            "Date: January 12, 2026",
                            "Taxpayer: MARIA L.CHEN",
                            "TIN: 987-65-4321",
                            "Tax Year: 2026",
                        ]
                    ),
                )
            ],
            regions={
                "issue_date": OCRPage(
                    page_number=1,
                    raw_text="Date: Immy 1.202",
                )
            },
        ),
        "residency.png",
    )

    assert parsed.fields["issue_date"] == "January 12, 2026"


def test_residency_parser_prefers_full_text_taxpayer_when_region_is_truncated() -> None:
    parser = ResidencyCertificateParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "DEPARTMENT OF THE TREASURY",
                            "Taxpayer: UNIVERSITY OF WASHINGTON",
                            "TIN: 91-6001537",
                            "Tax Year: 2025",
                        ]
                    ),
                )
            ],
            regions={
                "taxpayer_name": OCRPage(
                    page_number=1,
                    raw_text="payer UNIVERSITY OF WA",
                )
            },
        ),
        "residency.png",
    )

    assert parsed.fields["taxpayer_name"] == "UNIVERSITY OF WASHINGTON"


def test_residency_parser_prefers_full_text_taxpayer_when_region_bleeds_into_tin() -> None:
    parser = ResidencyCertificateParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "DEPARTMENT OF THE TREASURY",
                            "Taxpayer: UNIVERSITY OF HAWAII",
                            "TIN: 99-6000354",
                            "Tax Year: 2019",
                        ]
                    ),
                )
            ],
            regions={
                "taxpayer_name": OCRPage(
                    page_number=1,
                    raw_text="UNIVERSITY TIN - .",
                )
            },
        ),
        "residency.png",
    )

    assert parsed.fields["taxpayer_name"] == "UNIVERSITY OF HAWAII"


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
    assert parsed.fields["issued_at"] == "Raleigh, North Carolina"


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


def test_withholding_parser_prefers_region_signature_date_over_header_date() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "[2026.3.20.]",
                            "2026-01-12",
                            "신청인 MARIA L. CHEN",
                        ]
                    ),
                )
            ],
            regions={
                "signature_date": OCRPage(
                    page_number=1,
                    raw_text="t.\n2026\n01\n12\nOL",
                )
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["signature_date"] == "2026-01-12"


def test_withholding_parser_falls_back_when_region_signature_date_is_invalid() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "[2026.3.20.]",
                            "2026년 01월 12일 신청인 MARIA L. CHEN",
                        ]
                    ),
                )
            ],
            regions={
                "signature_date": OCRPage(
                    page_number=1,
                    raw_text="2Q26-y1-I2",
                )
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["signature_date"] == "2026-01-12"


def test_withholding_parser_falls_back_when_region_country_code_is_invalid() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "거주지국 United States of America",
                            "거주지국코드 US",
                        ]
                    ),
                )
            ],
            regions={
                "residency_country_code": OCRPage(page_number=1, raw_text="O0"),
                "residency_country": OCRPage(
                    page_number=1,
                    raw_text="United States of America",
                ),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["residency_country_code"] == "US"
