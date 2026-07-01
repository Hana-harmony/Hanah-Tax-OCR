from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_whitespace(value).strip(" ,;:")
    normalized = re.sub(r"(?<=\b[A-Z])\.(?=[A-Z])", ". ", normalized)
    normalized = re.sub(r"[^A-Za-z0-9 .,'-]", " ", normalized)
    normalized = normalize_whitespace(normalized)
    return normalized or None


def canonicalize_name(value: str | None) -> str:
    return "" if not value else re.sub(r"[^a-z0-9]", "", value.lower())


def normalize_address(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[\n\r\t,]+", " ", value)
    normalized = re.sub(r"[^A-Za-z0-9#.' -]", " ", normalized)
    normalized = normalize_whitespace(normalized)
    return normalized or None


def normalize_country(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_whitespace(value)
    compact_alpha = re.sub(r"[^A-Za-z0-9]", "", normalized).lower()
    compact_alpha = compact_alpha.translate(str.maketrans({"0": "o", "1": "i", "5": "s"}))
    if re.search(r"united states(?: of america)?", normalized, re.IGNORECASE):
        return "United States of America"
    if "nitedstates" in compact_alpha and (
        "america" in compact_alpha or "amerca" in compact_alpha
    ):
        return "United States of America"
    if re.search(r"\busa\b", normalized, re.IGNORECASE):
        return "United States of America"
    return normalized or None


def normalize_country_code(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_country(value)
    if normalized == "United States of America":
        return "US"
    compact = normalize_whitespace(value).upper()
    compact = compact.translate(str.maketrans({"0": "O", "1": "I", "5": "S"}))
    compact = re.sub(r"[^A-Z]", "", compact)
    if compact == "US":
        return "US"
    match = re.search(r"\b([A-Z]{2})\b", value.upper())
    return match.group(1) if match else None


def normalize_percentage(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"(\d{1,2})", value)
    if not match:
        return None
    return f"{match.group(1)}%"


def normalize_english_date(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_whitespace(value).strip(" ,.;:")
    match = re.search(
        r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?[,.]?\s*(\d{4})",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return normalized or None
    month, day, year = match.groups()
    return f"{month} {int(day)}, {year}"


def normalize_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"\s+", "", value)
    match = re.search(r"(\d{4})[년./-]?(\d{1,2})[월./-]?(\d{1,2})", normalized)
    if not match:
        return normalize_whitespace(value)
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def normalize_apostille_date(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = normalize_whitespace(value)
    cleaned = re.sub(r"([A-Za-z])(\d{4})", r"\1, \2", cleaned)
    cleaned = re.sub(r",\s*(\d{4})", r", \1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:")
    return cleaned or None


def normalize_apostille_authority(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.replace("StateState", "State State")
    cleaned = cleaned.replace("Deputy Secretary of StateState", "Deputy Secretary of State State")
    cleaned = re.sub(r"(?<=[A-Za-z]),(?=[A-Za-z])", ", ", cleaned)
    return normalize_whitespace(cleaned) or None
