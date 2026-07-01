from hanah_tax_ocr.normalization import normalize_address


def test_normalize_address_repairs_country_row_confusion_noise() -> None:
    value = "1234 Sunset Bivd.Apt 5BLos ArgelesCA 90026. United States of America"

    assert (
        normalize_address(value)
        == "1234 Sunset Blvd Apt 5B Los Angeles CA 90026 United States of America"
    )


def test_normalize_address_splits_city_and_state_suffix_without_harming_country_tail() -> None:
    value = "14 Main Street Suite 14 New YorkNY 10001 United States of America."

    assert (
        normalize_address(value)
        == "14 Main Street Suite 14 New York NY 10001 United States of America"
    )
