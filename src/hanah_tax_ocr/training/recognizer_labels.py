from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def recognizer_text_for_entry(entry: Mapping[str, Any]) -> str:
    explicit = entry.get("recognizer_text")
    if isinstance(explicit, str) and explicit:
        return explicit

    text = str(entry.get("text") or "")
    if not text:
        return text

    field_name = str(entry.get("field_name") or "")
    if field_name == "issue_date":
        return text if text.lower().startswith("date:") else f"Date: {text}"

    return text


def build_signature_date_display_text(value: str) -> str:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value.strip())
    if not match:
        return value
    year, month, day = match.groups()
    return f"{year}년 {month}월 {day}일"
