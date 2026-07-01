from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_REVIEW_QUEUE_DIR = Path("data/review_queue/index")
DEFAULT_MANUAL_ANNOTATIONS_PATH = Path("evals/error_taxonomy/manual_case_annotations.json")
DEFAULT_OUTPUT_PATH = Path("evals/error_taxonomy/hard_case_manifest.json")

ROOT_CAUSE_CATALOG: dict[str, dict[str, str]] = {
    "address_label_bleed": {
        "description": "Address crop absorbed nearby name or header labels before the street body.",
        "field_groups": "english_name_org",
    },
    "address_spacing_merge": {
        "description": "Street number and address body collapsed without spacing.",
        "field_groups": "english_name_org",
    },
    "cross_document_name_alignment_error": {
        "description": (
            "Cross-document name alignment failed even when per-document OCR looked plausible."
        ),
        "field_groups": "english_name_org",
    },
    "date_crop_miss": {
        "description": "Region OCR or parser missed the date crop entirely.",
        "field_groups": "date",
    },
    "date_spacing_loss": {
        "description": "Date text lost required whitespace or punctuation during OCR.",
        "field_groups": "date",
    },
    "label_bleed_name_header": {
        "description": "Name crop included nearby labels such as First/Middle/Name headers.",
        "field_groups": "english_name_org",
    },
    "low_quality_input": {
        "description": "Low-resolution or blurry source likely reduced OCR stability.",
        "field_groups": "shared",
    },
    "mixed_korean_english_interference": {
        "description": "Korean and English text interfered within the same OCR crop.",
        "field_groups": "korean_mixed_form",
    },
    "middle_name_segmentation_ambiguity": {
        "description": "Middle-name crop absorbed adjacent first/last-name tokens.",
        "field_groups": "english_name_org",
    },
    "rate_or_checkbox_region_miss": {
        "description": "Dividend/rate area likely suffered checkbox bleed or region miss.",
        "field_groups": "numeric_tin_code",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync a root-cause taxonomy manifest from review_queue outputs."
    )
    parser.add_argument("--review-queue-dir", type=Path, default=DEFAULT_REVIEW_QUEUE_DIR)
    parser.add_argument(
        "--manual-annotations",
        type=Path,
        default=DEFAULT_MANUAL_ANNOTATIONS_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manual_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return load_json(path).get("cases", {})


def _quality_root_causes(document: dict[str, Any]) -> list[str]:
    quality_checks = document.get("quality_checks", {})
    root_causes: list[str] = []
    blur_score = quality_checks.get("blur_score")
    if quality_checks.get("low_ocr_confidence") is True:
        root_causes.append("low_quality_input")
    elif isinstance(blur_score, int | float) and blur_score < 400:
        root_causes.append("low_quality_input")
    return root_causes


def _field_root_causes(document: dict[str, Any], findings: list[dict[str, Any]]) -> list[str]:
    root_causes: set[str] = set()
    fields = document.get("fields", {})
    for finding in findings:
        field_name = str(finding.get("field_name") or "")
        code = str(finding.get("code") or "")
        value = str(fields.get(field_name) or "")

        if field_name in {"issue_date", "issued_on", "signature_date"}:
            if "missing" in code:
                root_causes.add("date_crop_miss")
            if "invalid" in code or re.search(r"[A-Za-z]+\d{4}", value):
                root_causes.add("date_spacing_loss")
        if field_name in {"first_name", "middle_name", "last_name", "applicant_name"}:
            if re.search(r"\b(irst|iddle|last|first|middle|name)\b", value, re.IGNORECASE):
                root_causes.add("label_bleed_name_header")
            if field_name == "middle_name" and len(value.split()) >= 2:
                root_causes.add("middle_name_segmentation_ambiguity")
        if field_name == "address":
            if re.search(r"^\d{1,5}[A-Za-z]", value):
                root_causes.add("address_spacing_merge")
            if re.search(
                r"\b(?:Last Name|First Name|Middle Name|USER)\b",
                value,
                re.IGNORECASE,
            ):
                root_causes.add("address_label_bleed")
        if field_name in {"residency_country", "residency_country_code"} and re.search(
            r"[가-힣]",
            value,
        ) and re.search(r"[A-Za-z]", value):
            root_causes.add("mixed_korean_english_interference")
        if field_name == "dividend_tax_rate" and ("invalid" in code or "missing" in code):
            root_causes.add("rate_or_checkbox_region_miss")
        if field_name == "taxpayer_name" and "cross_check_mismatch" in code:
            root_causes.add("cross_document_name_alignment_error")
    return sorted(root_causes)


def build_hard_case_manifest(
    review_queue_dir: Path,
    *,
    manual_annotations_path: Path | None = None,
) -> dict[str, Any]:
    manual_annotations = _manual_annotations(manual_annotations_path)
    cases: list[dict[str, Any]] = []
    root_cause_counts = Counter()

    for queue_path in sorted(review_queue_dir.glob("*.json")):
        payload = load_json(queue_path)
        findings = list(payload.get("review_result", {}).get("findings", []) or [])
        if not findings:
            continue
        case_id = str(payload.get("case_id") or queue_path.stem)
        documents = list(payload.get("documents", []) or [])
        document_type = str(documents[0].get("document_type") or "") if documents else ""

        auto_root_causes: set[str] = set()
        evidence: list[dict[str, Any]] = []
        for document in documents:
            doc_findings = [
                finding
                for finding in findings
                if finding.get("field_name") in document.get("fields", {})
            ]
            auto_root_causes.update(_quality_root_causes(document))
            auto_root_causes.update(_field_root_causes(document, doc_findings))
            evidence.append(
                {
                    "document_type": document.get("document_type"),
                    "fields": document.get("fields", {}),
                    "quality_checks": document.get("quality_checks", {}),
                    "findings": doc_findings,
                }
            )

        manual_case = manual_annotations.get(case_id, {})
        manual_root_causes = sorted(set(manual_case.get("root_causes", [])))
        combined_root_causes = sorted(auto_root_causes | set(manual_root_causes))
        for root_cause in combined_root_causes:
            root_cause_counts[root_cause] += 1

        cases.append(
            {
                "case_id": case_id,
                "document_type": document_type,
                "review_status": payload.get("review_result", {}).get("status"),
                "root_causes": combined_root_causes,
                "auto_root_causes": sorted(auto_root_causes),
                "manual_root_causes": manual_root_causes,
                "manual_notes": manual_case.get("notes"),
                "source_paths": [document.get("source_path") for document in documents],
                "evidence": evidence,
            }
        )

    return {
        "version": "2026-07-01",
        "catalog": ROOT_CAUSE_CATALOG,
        "summary": {
            "case_count": len(cases),
            "root_cause_counts": dict(sorted(root_cause_counts.items())),
        },
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    manifest = build_hard_case_manifest(
        args.review_queue_dir,
        manual_annotations_path=args.manual_annotations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
