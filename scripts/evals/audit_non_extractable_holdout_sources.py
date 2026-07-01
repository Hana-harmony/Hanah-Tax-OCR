from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hanah_tax_ocr.ocr import PaddleOCREngine
from hanah_tax_ocr.training.sample_dataset import build_sample_index

DEFAULT_SAMPLE_ROOT = Path("sample_data")
DEFAULT_OUTPUT_PATH = Path("evals/external_holdout/non_extractable_source_audit.json")

WITHHOLDING_TITLE_PATTERN = re.compile(r"국내원천소득.*제한세율.*적용신청서")
WITHHOLDING_NAME_LABEL_PATTERN = re.compile(r"last\s*name|first\s*name|middle\s*name", re.I)
WITHHOLDING_US_ADDRESS_PATTERN = re.compile(
    r"\b(?:street|st|road|rd|avenue|ave|blvd|boulevard|suite|apt)\b.*\b(?:United States|USA)\b",
    re.I,
)
WITHHOLDING_TIN_PATTERN = re.compile(r"\b\d{2,3}-\d{2}-\d{4}\b")
WITHHOLDING_PAYER_MARKERS = (
    "원천징수의무자",
    "대표자",
    "사업자주민등록번호",
    "제출자",
    "세무서장",
)
WITHHOLDING_INSTRUCTION_MARKERS = (
    "작성방법",
    "접수번호 및 접수일자",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit non-extractable holdout sources so missing-distribution blockers carry "
            "machine-readable evidence instead of only manual page-role labels."
        )
    )
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _normalize_text_excerpt(text: str, *, limit: int = 240) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:limit]


def _ocr_image_text(source_path: Path) -> str:
    engine = PaddleOCREngine(lang="korean")
    return engine.run(source_path).combined_text()


def _extract_pdf_text_pages(source_path: Path) -> list[str]:
    command = ["pdftotext", "-layout", str(source_path), "-"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    page_texts = [page.strip() for page in completed.stdout.split("\f")]
    return [page for page in page_texts if page]


def _withholding_text_markers(text: str) -> dict[str, bool]:
    normalized = re.sub(r"\s+", " ", text)
    return {
        "has_title": WITHHOLDING_TITLE_PATTERN.search(normalized) is not None,
        "has_name_labels": WITHHOLDING_NAME_LABEL_PATTERN.search(normalized) is not None,
        "has_instruction_header": any(
            marker in normalized for marker in WITHHOLDING_INSTRUCTION_MARKERS
        ),
        "has_payer_header": any(marker in normalized for marker in WITHHOLDING_PAYER_MARKERS),
        "has_us_address": WITHHOLDING_US_ADDRESS_PATTERN.search(normalized) is not None,
        "has_tin": WITHHOLDING_TIN_PATTERN.search(normalized) is not None,
        "has_applicant_signature": "신청인" in normalized
        and ("서명" in normalized or "sign" in normalized.lower()),
    }


def classify_withholding_page_text(text: str) -> dict[str, Any]:
    markers = _withholding_text_markers(text)
    reasons: list[str] = []
    has_filled_value_signal = markers["has_us_address"] or markers["has_tin"]

    if markers["has_payer_header"] and markers["has_instruction_header"]:
        reasons.extend(["contains_payer_header", "contains_instruction_header"])
        return {
            "classification": "back_side_payer_submission",
            "holdout_usable": False,
            "blocker_reason": "back_side_without_target_fields",
            "confidence": "high",
            "reasons": reasons,
            "markers": markers,
        }

    if markers["has_instruction_header"]:
        reasons.append("contains_instruction_header")
        return {
            "classification": "back_side_instructions",
            "holdout_usable": False,
            "blocker_reason": "back_side_without_target_fields",
            "confidence": "high",
            "reasons": reasons,
            "markers": markers,
        }

    if markers["has_payer_header"] and not markers["has_title"]:
        reasons.append("contains_payer_header_without_front_form_title")
        return {
            "classification": "back_side_payer_submission",
            "holdout_usable": False,
            "blocker_reason": "back_side_without_target_fields",
            "confidence": "medium",
            "reasons": reasons,
            "markers": markers,
        }

    if markers["has_title"] and markers["has_name_labels"]:
        if has_filled_value_signal:
            reasons.extend(["contains_front_form_title", "contains_filled_value_signals"])
            return {
                "classification": "front_filled_target_page",
                "holdout_usable": True,
                "blocker_reason": None,
                "confidence": "medium",
                "reasons": reasons,
                "markers": markers,
            }
        reasons.extend(["contains_front_form_title", "missing_filled_target_signals"])
        return {
            "classification": "front_blank_template",
            "holdout_usable": False,
            "blocker_reason": "blank_form_without_filled_entities",
            "confidence": "high",
            "reasons": reasons,
            "markers": markers,
        }

    reasons.append("no_target_page_signals")
    return {
        "classification": "unclassified_non_extractable",
        "holdout_usable": False,
        "blocker_reason": "manual_review_required",
        "confidence": "low",
        "reasons": reasons,
        "markers": markers,
    }


def build_non_extractable_source_audit(
    sample_root: Path,
    *,
    sample_index: dict[str, dict[str, object]] | None = None,
    image_text_loader: Callable[[Path], str] | None = None,
    pdf_text_loader: Callable[[Path], list[str]] | None = None,
) -> dict[str, Any]:
    sample_index = sample_index or build_sample_index(sample_root)
    image_text_loader = image_text_loader or _ocr_image_text
    pdf_text_loader = pdf_text_loader or _extract_pdf_text_pages

    unique_entries: dict[str, dict[str, object]] = {}
    for entry in sample_index.values():
        source = str(entry.get("source") or "")
        if not source or source in unique_entries:
            continue
        unique_entries[source] = dict(entry)

    audited_cases: list[dict[str, Any]] = []
    for entry in sorted(
        unique_entries.values(),
        key=lambda payload: str(payload.get("source") or ""),
    ):
        if bool(entry.get("extractable", True)):
            continue
        source_path = Path(str(entry["source"]))
        page_texts = (
            pdf_text_loader(source_path)
            if source_path.suffix.lower() == ".pdf"
            else [image_text_loader(source_path)]
        )

        page_audits = []
        for page_number, text in enumerate(page_texts, start=1):
            page_result = classify_withholding_page_text(text)
            page_audits.append(
                {
                    "page_number": page_number,
                    "classification": page_result["classification"],
                    "holdout_usable": page_result["holdout_usable"],
                    "blocker_reason": page_result["blocker_reason"],
                    "confidence": page_result["confidence"],
                    "reasons": page_result["reasons"],
                    "markers": page_result["markers"],
                    "text_excerpt": _normalize_text_excerpt(text),
                }
            )

        usable_page_numbers = [
            page["page_number"] for page in page_audits if page["holdout_usable"]
        ]
        if usable_page_numbers:
            source_classification = "contains_target_page"
            holdout_usable = True
            blocker_reason = None
        elif any(page["classification"] == "front_blank_template" for page in page_audits):
            source_classification = "blank_front_plus_back_side_only"
            holdout_usable = False
            blocker_reason = "blank_form_without_filled_entities"
        elif any(page["classification"] == "back_side_payer_submission" for page in page_audits):
            source_classification = "back_side_submission_only"
            holdout_usable = False
            blocker_reason = "back_side_without_target_fields"
        else:
            source_classification = (
                page_audits[0]["classification"]
                if page_audits
                else "unclassified_non_extractable"
            )
            holdout_usable = False
            blocker_reason = (
                page_audits[0]["blocker_reason"]
                if page_audits
                else "manual_review_required"
            )

        audited_cases.append(
            {
                "case_id": entry.get("case_id"),
                "source_path": str(source_path),
                "document_type": entry.get("document_type"),
                "declared_page_role": entry.get("page_role"),
                "declared_exclusion_reason": entry.get("exclusion_reason"),
                "page_count": len(page_audits),
                "source_classification": source_classification,
                "holdout_usable": holdout_usable,
                "blocker_reason": blocker_reason,
                "usable_page_numbers": usable_page_numbers,
                "page_audits": page_audits,
            }
        )

    return {
        "version": "2026-07-02",
        "sample_root": str(sample_root),
        "cases": audited_cases,
    }


def main() -> None:
    args = parse_args()
    payload = build_non_extractable_source_audit(args.sample_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
