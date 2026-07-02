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


def test_residency_parser_accepts_partial_united_states_phrase() -> None:
    parser = ResidencyCertificateParser()
    parsed = parser.parse(
        build_result(
            [
                "DEPARTMENT OF THE TREASURY",
                "INTERNAL REVENUE SERVICE",
                "Date: January 12, 2026",
                "Taxpayer: Sample 1 User",
                "TIN: 987-65-4321",
                "Tax Year: 2026",
                "I certify that the above-named taxpayer is a resident of the United States",
            ]
        ),
        "residency.png",
    )

    assert parsed.fields["taxpayer_name"] == "Sample 1 User"
    assert parsed.fields["residency_country"] == "United States of America"
    assert parsed.fields["residency_country_code"] == "US"


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


def test_residency_parser_salvages_taxpayer_and_partial_issue_date_from_neighbor_regions() -> None:
    parser = ResidencyCertificateParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "DEPARTMENT OF THE TREASURY",
                            "Date:",
                            "82021",
                            "Taxpayer:",
                            "TIN:",
                            "Tax Year:2021",
                        ]
                    ),
                )
            ],
            regions={
                "taxpayer_name": OCRPage(
                    page_number=1,
                    raw_text="Taxpayer:\nTIN",
                ),
                "tin": OCRPage(
                    page_number=1,
                    raw_text="Taxpayer:F\nTIN\nTax Year:2021",
                ),
                "issue_date": OCRPage(
                    page_number=1,
                    raw_text="Date:\n8,2021",
                ),
            },
        ),
        "legacy_residency.png",
    )

    assert parsed.fields["taxpayer_name"] == "F"
    assert parsed.fields["issue_date"] == "8, 2021"


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


def test_apostille_parser_accepts_digits_in_issued_at_and_generic_authority() -> None:
    parser = ApostilleParser()
    parsed = parser.parse(
        build_result(
            [
                "APOSTILLE (Convention de La Haye du 5 octobre 1961)",
                "1. Country: UNITED STATES OF AMERICA",
                "2. This Public Document has been signed by NOTARY SAMPLE 1.",
                "3. acting in the capacity of NOTARY PUBLIC",
                "4. bears the seal/stamp of COUNTY OF SAMPLE, STATE OF SAMPLE",
                "5. at Capital City, Sample State 1",
                "6. the 2TH DAY OF APRIL, 2021",
                "7. by Secretary of State State of Sample",
                "8. No. 5001",
            ]
        ),
        "apostille.jpg",
    )

    assert parsed.fields["signed_by"] == "NOTARY SAMPLE 1"
    assert parsed.fields["issued_at"] == "Capital City, Sample State 1"
    assert parsed.fields["issuing_authority"] == "Secretary of State State of Sample"


def test_apostille_parser_prefers_item_eight_certificate_number_over_noisy_region() -> None:
    parser = ApostilleParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "APOSTILLE",
                            "1. Country: UNITED STATES OF AMERICA",
                            "2. This Public Document has been signed by NOTARY SAMPLE 8",
                            "3. acting in the capacity of NOTARY PUBLIC",
                            "4. bears the seal/stamp of COUNTY OF SAMPLE, STATE OF SAMPLE",
                            "5. at Capital City, Sample State 8",
                            "6. the 9TH DAY OF APRIL, 2028",
                            "7. by Secretary of State State of Sample",
                            "8. No. 5008",
                        ]
                    ),
                )
            ],
            regions={
                "certificate_number": OCRPage(page_number=1, raw_text="TARY"),
            },
        ),
        "apostille.jpg",
    )

    assert parsed.fields["certificate_number"] == "5008"


def test_apostille_parser_trims_country_region_bleed_and_place_punctuation() -> None:
    parser = ApostilleParser()
    parsed = parser.parse(
        OCRResult(
            pages=[OCRPage(page_number=1, raw_text="APOSTILLE")],
            regions={
                "issuing_country": OCRPage(
                    page_number=1,
                    raw_text=(
                        "UNITED STATES OF AMERICA 3. acting in the capacity of NOTARY PUBLIC "
                        "4. bears the seal/stamp of COUNTY OF SAMPLE, STATE OF SAMPLE"
                    ),
                ),
                "issued_at": OCRPage(page_number=1, raw_text="Capital City, Sample State 5."),
            },
        ),
        "apostille.jpg",
    )

    assert parsed.fields["issuing_country"] == "UNITED STATES OF AMERICA"
    assert parsed.fields["issued_at"] == "Capital City, Sample State 5"


def test_apostille_parser_normalizes_missing_space_after_year_comma() -> None:
    parser = ApostilleParser()
    parsed = parser.parse(
        build_result(
            [
                "APOSTILLE (Convention de La Haye du 5 octobre 1961)",
                "1. Country: UNITED STATES OF AMERICA",
                "2. This Public Document has been signed by CHONG U CHOI",
                "3. acting in the capacity of NOTARY PUBLIC",
                "4. bears the seal/stamp of COUNTY OF MECKLENBURG, NORTH CAROLINA",
                "5. at Raleigh, North Carolina",
                "6. the 10TH DAY OF APRIL,2025",
                "7. by Secretary of State or Deputy Secretary of State, State of North Carolina",
                "8. No. 5",
            ]
        ),
        "apostille.jpg",
    )

    assert parsed.fields["issued_on"] == "10TH DAY OF APRIL, 2025"


def test_apostille_california_parser_prefers_full_text_over_misaligned_regions() -> None:
    parser = ApostilleParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "Statrantcalrinnia",
                            "SECRETARY OF STATE",
                            "APOSTILLE",
                            "Convention de La Haye du 5 octobre 1961",
                            "1.CountryUnited States of America",
                            "This public document",
                            "2.has been signed by",
                            "3. acting in the capacity of Deputy Registrar-Recorder/County Clerk",
                            "County of Los Angeies, State of California",
                            "4.bears the seal/stamp of the County of Los Angeles",
                            "State of California",
                            "CERTIFIED",
                            "5.At Los AngelesCalifornia",
                            "SEC",
                            "6.the 18thday of",
                            "7.by Deputy Secretary of State,State of California",
                            "8.No.",
                        ]
                    ),
                )
            ],
            regions={
                "issuing_country": OCRPage(
                    page_number=1,
                    raw_text="ry:United States of America\nublic document",
                ),
                "signed_by": OCRPage(
                    page_number=1,
                    raw_text="cting n the capacity of Deputy\nnty of Los Angees, State of Ca",
                ),
                "signer_capacity": OCRPage(
                    page_number=1,
                    raw_text="state of California\nCERTIFIED",
                ),
                "seal_owner": OCRPage(
                    page_number=1,
                    raw_text="tLos Angeles,Caltornia\ne 18th day of",
                ),
                "issued_at": OCRPage(
                    page_number=1,
                    raw_text="0.4\neal/Stamp:",
                ),
                "certificate_number": OCRPage(page_number=1, raw_text="8.No.4"),
            },
        ),
        "sample_data/아포스티유 샘플/미국 california 주.png",
    )

    assert parsed.fields["issuing_country"] == "United States of America"
    assert parsed.fields["signed_by"] is None
    assert (
        parsed.fields["signer_capacity"]
        == "Deputy Registrar-Recorder/County Clerk, County of Los Angeles, State of California"
    )
    assert parsed.fields["seal_owner"] == "County of Los Angeles, State of California"
    assert parsed.fields["issued_at"] == "Los Angeles, California"
    assert parsed.fields["issued_on"] is None
    assert parsed.fields["issuing_authority"] == "Deputy Secretary of State, State of California"
    assert parsed.fields["certificate_number"] == "4"


def test_apostille_parser_strips_signed_by_prefix_from_region_value() -> None:
    parser = ApostilleParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "APOSTILLE",
                            "Convention de La Haye du 5 octobre 1961",
                            "1. Country: UNITED STATES OF AMERICA",
                            "3. acting in the capacity of NOTARY PUBLIC",
                            "4. bears the seal/stamp of COUNTY OF SAMPLE, STATE OF SAMPLE",
                            "5. at Capital City, Sample State 11",
                            "6. the 12TH DAY OF APRIL, 2021",
                            "7. by Secretary of State State of Sample",
                            "8. No. 5011",
                        ]
                    ),
                )
            ],
            regions={
                "signed_by": OCRPage(
                    page_number=1,
                    raw_text="2. This Public Document has been signed by NOTARY SAMPLE 11",
                )
            },
        ),
        "north_carolina.png",
    )

    assert parsed.fields["signed_by"] == "NOTARY SAMPLE 11"


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
    assert (
        parsed.fields["address"]
        == "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 United States of America"
    )
    assert parsed.fields["dividend_tax_rate"] == "15%"
    assert parsed.fields["signature_date"] == "2026-01-12"


def test_withholding_parser_prefers_full_text_signature_date_when_region_year_is_implausible(
) -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "국내원천소득 제한세율 적용신청서(비거주자용)",
                            "2026년 01월 12일 신청인 MARIA L. CHEN (서명 또는 인)",
                        ]
                    ),
                )
            ],
            regions={
                "signature_date": OCRPage(
                    page_number=1,
                    raw_text="1312-02-02",
                )
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["signature_date"] == "2026-01-12"


def test_withholding_parser_extracts_single_digit_street_number_from_full_text() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        build_result(
            [
                "1 Main Street Suite 1 New York NY 10001 United States of America",
                "201-21-2001",
            ]
        ),
        "withholding.png",
    )

    assert (
        parsed.fields["address"]
        == "1 Main Street Suite 1 New York NY 10001 United States of America"
    )


def test_withholding_parser_prefers_region_address_when_full_text_contains_name_noise() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "USER",
                            "1 USER 1 Main Street Suite New York NY 10001 United States of America",
                        ]
                    ),
                )
            ],
            regions={
                "address": OCRPage(
                    page_number=1,
                    raw_text=(
                        "Main\nStreet\nSuite\n1\nNew\nYork\nNY\n10001\n"
                        "United\nStates\nof\nAmerica"
                    ),
                )
            },
        ),
        "withholding.png",
    )

    assert (
        parsed.fields["address"]
        == "1 Main Street Suite 1 New York NY 10001 United States of America"
    )


def test_withholding_parser_strips_leading_user_noise_from_full_text_address() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "USER",
                            "SAMPLE3",
                            "USER",
                            "3 Main",
                            "Street",
                            "Suite 3",
                            "New",
                            "York NY",
                            "10001",
                            "United",
                            "States",
                            "of America",
                        ]
                    ),
                )
            ],
            regions={
                "address": OCRPage(
                    page_number=1,
                    raw_text=(
                        "5\nMain\nStreet\nSuite\n3\nNew\nYork\nNY\n10001\n"
                        "United\nStates\nof\nAmerica"
                    ),
                )
            },
        ),
        "withholding.png",
    )

    assert (
        parsed.fields["address"]
        == "3 Main Street Suite 3 New York NY 10001 United States of America"
    )


def test_withholding_parser_repairs_missing_space_after_leading_street_number() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[OCRPage(page_number=1, raw_text="207-27-2007")],
            regions={
                "address": OCRPage(
                    page_number=1,
                    raw_text="7Main Street Suite 7 New York NY 10001 United States of America",
                )
            },
        ),
        "withholding.png",
    )

    assert (
        parsed.fields["address"]
        == "7 Main Street Suite 7 New York NY 10001 United States of America"
    )


def test_withholding_parser_strips_label_bleed_prefix_from_region_address() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[OCRPage(page_number=1, raw_text="withholding probe")],
            regions={
                "address": OCRPage(
                    page_number=1,
                    raw_text=(
                        "12 Last Name First Name Middle Name CHEN MARIA "
                        "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 United States of America"
                    ),
                )
            },
        ),
        "withholding.png",
    )

    assert (
        parsed.fields["address"]
        == "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 United States of America"
    )


def test_withholding_parser_prefers_region_address_when_full_text_lacks_street_marker() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "주소",
                            "2014 United States of America",
                        ]
                    ),
                )
            ],
            regions={
                "address": OCRPage(
                    page_number=1,
                    raw_text="14 Main Street Suite 14 New York NY 10001 United States of America",
                )
            },
        ),
        "withholding.png",
    )

    assert (
        parsed.fields["address"]
        == "14 Main Street Suite 14 New York NY 10001 United States of America"
    )


def test_withholding_parser_normalizes_zip_letter_confusion_in_address() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[OCRPage(page_number=1, raw_text="withholding probe")],
            regions={
                "address": OCRPage(
                    page_number=1,
                    raw_text="4 Main Street Suite 4 New York NY 10o01 United States of America",
                )
            },
        ),
        "withholding.png",
    )

    assert (
        parsed.fields["address"]
        == "4 Main Street Suite 4 New York NY 10001 United States of America"
    )


def test_withholding_parser_recovers_tin_from_leaked_address_noise() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[OCRPage(page_number=1, raw_text="withholding probe")],
            regions={
                "address": OCRPage(
                    page_number=1,
                    raw_text=(
                        "1234 Sunset Blvd Apl 5B Los Angeles CA 90026 "
                        "United States Amorica 4 5 987-65-4321 1985 -0615 United"
                    ),
                ),
                "residency_country": OCRPage(
                    page_number=1,
                    raw_text="United States of America",
                ),
                "residency_country_code": OCRPage(
                    page_number=1,
                    raw_text="US",
                ),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["tin"] == "987-65-4321"
    assert (
        parsed.fields["address"]
        == "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 United States of America"
    )


def test_withholding_parser_preserves_digits_and_rebuilds_applicant_name() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "201-21-2001",
                            "2026-01-02",
                            "SAMPLE1 A USER",
                        ]
                    ),
                )
            ],
            regions={
                "first_name": OCRPage(page_number=1, raw_text="SAMPLE1"),
                "middle_name": OCRPage(page_number=1, raw_text="A"),
                "last_name": OCRPage(page_number=1, raw_text="USER"),
                "applicant_name": OCRPage(page_number=1, raw_text="SAMPLET A USER"),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["first_name"] == "SAMPLE1"
    assert parsed.fields["applicant_name"] == "SAMPLE1 A USER"


def test_withholding_parser_prefers_applicant_first_name_when_it_preserves_digits() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "203-23-2003",
                            "2026-01-12",
                            "SAMPLE11 K USER",
                        ]
                    ),
                )
            ],
            regions={
                "first_name": OCRPage(page_number=1, raw_text="SAMPLE1T"),
                "middle_name": OCRPage(page_number=1, raw_text="K"),
                "last_name": OCRPage(page_number=1, raw_text="USER"),
                "applicant_name": OCRPage(page_number=1, raw_text="SAMPLE11\nK\nUSER"),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["first_name"] == "SAMPLE11"
    assert parsed.fields["applicant_name"] == "SAMPLE11 K USER"


def test_withholding_parser_normalizes_ocr_ambiguous_digit_suffix_in_first_name() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "220-40-2020",
                            "2026-01-21",
                            "SAMPLE2O T USER",
                        ]
                    ),
                )
            ],
            regions={
                "first_name": OCRPage(page_number=1, raw_text="SAMPLE2O"),
                "middle_name": OCRPage(page_number=1, raw_text="T"),
                "last_name": OCRPage(page_number=1, raw_text="USER"),
                "applicant_name": OCRPage(page_number=1, raw_text="SAMPLE2O\nT\nUSER"),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["first_name"] == "SAMPLE20"
    assert parsed.fields["applicant_name"] == "SAMPLE20 T USER"


def test_withholding_parser_does_not_prefer_numeric_applicant_first_name_noise() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "987-65-4321",
                            "2026-01-12",
                            "MARIA L CHEN",
                        ]
                    ),
                )
            ],
            regions={
                "first_name": OCRPage(page_number=1, raw_text="MARIA"),
                "middle_name": OCRPage(page_number=1, raw_text="L"),
                "last_name": OCRPage(page_number=1, raw_text="CHEN"),
                "applicant_name": OCRPage(page_number=1, raw_text="202 I CHEN"),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["first_name"] == "MARIA"
    assert parsed.fields["middle_name"] == "L"
    assert parsed.fields["applicant_name"] == "MARIA L. CHEN"


def test_withholding_parser_normalizes_zero_middle_initial_and_rebuilds_applicant_name() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "215-35-2015",
                            "2026-01-16",
                            "SAMPLE15 0 USER",
                        ]
                    ),
                )
            ],
            regions={
                "first_name": OCRPage(page_number=1, raw_text="SAMPLE15"),
                "middle_name": OCRPage(page_number=1, raw_text="0"),
                "last_name": OCRPage(page_number=1, raw_text="USER"),
                "applicant_name": OCRPage(page_number=1, raw_text="SAMPLE15\n0\nUSER"),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["middle_name"] == "O"
    assert parsed.fields["applicant_name"] == "SAMPLE15 O USER"


def test_withholding_parser_normalizes_one_middle_initial_and_rebuilds_applicant_name() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "209-29-2009",
                            "2026-01-10",
                            "SAMPLE9 1 USER",
                        ]
                    ),
                )
            ],
            regions={
                "first_name": OCRPage(page_number=1, raw_text="SAMPLE9"),
                "middle_name": OCRPage(page_number=1, raw_text="1"),
                "last_name": OCRPage(page_number=1, raw_text="USER"),
                "applicant_name": OCRPage(page_number=1, raw_text="SAMPLE9\n1\nUSER"),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["middle_name"] == "I"
    assert parsed.fields["applicant_name"] == "SAMPLE9 I USER"


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
    assert parsed.fields["applicant_name"] == "MARIA L. CHEN"


def test_withholding_parser_prefers_full_text_tin_over_phone_like_region_noise() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 United States of America",
                            "34",
                            "987-65-43215",
                            "+1-323-555-1234",
                            "2026년 01월 12일 신청인 MARIA L. CHEN",
                        ]
                    ),
                )
            ],
            regions={
                "tin": OCRPage(page_number=1, raw_text="+1-323\n23"),
                "applicant_name": OCRPage(page_number=1, raw_text="MARIA L CHEN"),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["tin"] == "987-65-4321"


def test_withholding_parser_prefers_exact_tin_over_later_overlong_payer_noise() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "987-65-4321",
                            "US",
                            "United States of America",
                            "123-45-67890",
                        ]
                    ),
                )
            ],
        ),
        "withholding.png",
    )

    assert parsed.fields["tin"] == "987-65-4321"


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


def test_withholding_parser_prefers_region_signature_date_over_unrelated_birthdate() -> None:
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
                            "1985-06-15",
                            "신청인 MARIA L. CHEN",
                        ]
                    ),
                )
            ],
            regions={
                "signature_date": OCRPage(
                    page_number=1,
                    raw_text="l.\n2026\n01\n12\nec L\not",
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


def test_withholding_parser_prefers_last_full_text_signature_date_over_birthdate() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "생년월일 1985-06-15",
                            "2026 01 12 바Y 또는 인 신청인 MARIA L. CHEN",
                        ]
                    ),
                )
            ],
            regions={
                "signature_date": OCRPage(
                    page_number=1,
                    raw_text="[\n2026\n1川010월1312\n22일\n년\nEGI\n이",
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


def test_withholding_parser_recovers_country_code_and_signature_date_from_compressed_noise(
) -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[
                OCRPage(
                    page_number=1,
                    raw_text="\n".join(
                        [
                            "접수일자 2026-0112",
                            "United States of Amerca",
                            "2026 년 01 12 인",
                        ]
                    ),
                )
            ],
            regions={
                "residency_country": OCRPage(
                    page_number=1,
                    raw_text="State5 이 AnerIC경",
                ),
                "residency_country_code": OCRPage(
                    page_number=1,
                    raw_text="U5",
                ),
                "signature_date": OCRPage(
                    page_number=1,
                    raw_text="2026\n년no11일12일",
                ),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["residency_country"] == "United States of America"
    assert parsed.fields["residency_country_code"] == "US"
    assert parsed.fields["signature_date"] == "2026-01-12"


def test_withholding_parser_derives_country_code_from_normalized_country() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[OCRPage(page_number=1, raw_text="United States of Amerca")],
            regions={
                "residency_country": OCRPage(
                    page_number=1,
                    raw_text="United States of Amerca",
                ),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["residency_country"] == "United States of America"
    assert parsed.fields["residency_country_code"] == "US"


def test_withholding_parser_overrides_noisy_country_code_when_country_is_us() -> None:
    parser = WithholdingTaxFormParser()
    parsed = parser.parse(
        OCRResult(
            pages=[OCRPage(page_number=1, raw_text="United States of America")],
            regions={
                "residency_country": OCRPage(
                    page_number=1,
                    raw_text="United States of America",
                ),
                "residency_country_code": OCRPage(
                    page_number=1,
                    raw_text="ST",
                ),
            },
        ),
        "withholding.png",
    )

    assert parsed.fields["residency_country"] == "United States of America"
    assert parsed.fields["residency_country_code"] == "US"
