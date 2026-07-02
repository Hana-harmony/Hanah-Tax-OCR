from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hanah_tax_ocr.training.sample_dataset import build_sample_index, normalize_path_text

from scripts.evals.audit_non_extractable_holdout_sources import (
    _extract_pdf_text_pages,
    classify_withholding_page_text,
)

DEFAULT_SAMPLE_ROOT = Path("sample_data")
DEFAULT_MANIFEST_PATH = Path("evals/external_holdout/manifest.json")
DEFAULT_OUTPUT_PATH = Path("evals/external_holdout/withholding_sample_page_audit.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit withholding sample_data pages so external-holdout mixed-language blockers "
            "carry file-level front/back/blank evidence."
        )
    )
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_case_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    indexed: dict[str, dict[str, Any]] = {}
    for section, holdout_status in (
        ("cases", "candidate"),
        ("excluded_eval_overlap_cases", "eval_overlap"),
        ("excluded_non_extractable_cases", "non_extractable"),
    ):
        for case in manifest.get(section, []):
            sample_path = normalize_path_text(str(case.get("sample_path") or ""))
            if not sample_path:
                continue
            indexed[sample_path] = {
                "holdout_status": holdout_status,
                "case_id": case.get("case_id"),
                "document_type": case.get("document_type"),
                "subset_tags": list(case.get("subset_tags", [])),
                "page_role": case.get("page_role"),
                "exclusion_reason": case.get("exclusion_reason"),
            }
    return indexed


def _source_level_classification(page_audits: list[dict[str, Any]]) -> tuple[str, bool, str | None]:
    usable_page_numbers = [page["page_number"] for page in page_audits if page["holdout_usable"]]
    if usable_page_numbers:
        return "contains_target_page", True, None
    if any(page["classification"] == "front_blank_template" for page in page_audits):
        return "blank_front_plus_back_side_only", False, "blank_form_without_filled_entities"
    if any(page["classification"] == "back_side_payer_submission" for page in page_audits):
        return "back_side_submission_only", False, "back_side_without_target_fields"
    if any(page["classification"] == "back_side_instructions" for page in page_audits):
        return "instruction_only_without_target_fields", False, "back_side_without_target_fields"
    return "unclassified_non_extractable", False, "manual_review_required"


def _metadata_page_audits(entry: dict[str, object]) -> list[dict[str, Any]] | None:
    page_role = str(entry.get("page_role") or "")
    exclusion_reason = str(entry.get("exclusion_reason") or "")
    extractable = bool(entry.get("extractable", True))

    if extractable and page_role in {"front", "front_pdf"}:
        return [
            {
                "page_number": 1,
                "classification": "front_filled_target_page",
                "holdout_usable": True,
                "blocker_reason": None,
                "confidence": "declared",
                "reasons": ["sample_dataset_extractable_front_page"],
                "markers": {
                    "page_role": page_role,
                    "extractable": extractable,
                },
                "text_excerpt": None,
                "evidence_strategy": "sample_dataset_metadata",
            }
        ]

    if page_role == "reverse_side_instructions" or exclusion_reason == "back_side_reference_page":
        return [
            {
                "page_number": 1,
                "classification": "back_side_payer_submission",
                "holdout_usable": False,
                "blocker_reason": "back_side_without_target_fields",
                "confidence": "declared",
                "reasons": ["sample_dataset_reverse_side_reference_page"],
                "markers": {
                    "page_role": page_role,
                    "extractable": extractable,
                },
                "text_excerpt": None,
                "evidence_strategy": "sample_dataset_metadata",
            }
        ]

    if (
        page_role == "blank_form_template"
        or exclusion_reason == "blank_form_without_filled_entities"
    ):
        return [
            {
                "page_number": 1,
                "classification": "front_blank_template",
                "holdout_usable": False,
                "blocker_reason": "blank_form_without_filled_entities",
                "confidence": "declared",
                "reasons": ["sample_dataset_blank_form_template"],
                "markers": {
                    "page_role": page_role,
                    "extractable": extractable,
                },
                "text_excerpt": None,
                "evidence_strategy": "sample_dataset_metadata",
            }
        ]

    return None


def build_withholding_sample_page_audit(
    sample_root: Path,
    manifest_path: Path,
    *,
    sample_index: dict[str, dict[str, object]] | None = None,
    pdf_text_loader: Callable[[Path], list[str]] | None = None,
) -> dict[str, Any]:
    sample_index = sample_index or build_sample_index(sample_root)
    pdf_text_loader = pdf_text_loader or _extract_pdf_text_pages
    manifest_cases = _manifest_case_index(manifest_path)

    unique_entries: dict[str, dict[str, object]] = {}
    for entry in sample_index.values():
        if str(entry.get("document_type") or "") != "withholding_tax_form":
            continue
        source = normalize_path_text(str(entry.get("source") or ""))
        if not source or source in unique_entries:
            continue
        unique_entries[source] = dict(entry)

    cases: list[dict[str, Any]] = []
    for entry in sorted(
        unique_entries.values(),
        key=lambda payload: str(payload.get("source") or ""),
    ):
        source_path = Path(str(entry["source"]))
        if source_path.suffix.lower() == ".pdf":
            page_texts = pdf_text_loader(source_path)
            page_audits: list[dict[str, Any]] = []
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
                        "text_excerpt": text[:240].replace("\n", " ").strip(),
                        "evidence_strategy": "pdf_text",
                    }
                )
        else:
            page_audits = _metadata_page_audits(entry) or [
                {
                    "page_number": 1,
                    "classification": "unclassified_non_extractable",
                    "holdout_usable": False,
                    "blocker_reason": "manual_review_required",
                    "confidence": "low",
                    "reasons": ["missing_sample_dataset_page_role_mapping"],
                    "markers": {
                        "page_role": entry.get("page_role"),
                        "extractable": entry.get("extractable"),
                    },
                    "text_excerpt": None,
                    "evidence_strategy": "sample_dataset_metadata_fallback",
                }
            ]

        source_classification, holdout_usable, blocker_reason = _source_level_classification(
            page_audits
        )
        manifest_case = manifest_cases.get(normalize_path_text(source_path))
        cases.append(
            {
                "source_path": str(source_path),
                "document_type": "withholding_tax_form",
                "sample_split": entry.get("split"),
                "case_id": entry.get("case_id"),
                "holdout_status": (
                    manifest_case.get("holdout_status") if manifest_case else "untracked"
                ),
                "manifest_case_id": manifest_case.get("case_id") if manifest_case else None,
                "manifest_subset_tags": (
                    manifest_case.get("subset_tags", []) if manifest_case else []
                ),
                "manifest_page_role": manifest_case.get("page_role") if manifest_case else None,
                "manifest_exclusion_reason": (
                    manifest_case.get("exclusion_reason") if manifest_case else None
                ),
                "source_classification": source_classification,
                "holdout_usable": holdout_usable,
                "blocker_reason": blocker_reason,
                "usable_page_numbers": [
                    page["page_number"] for page in page_audits if page["holdout_usable"]
                ],
                "evidence_strategy": (
                    "pdf_text"
                    if source_path.suffix.lower() == ".pdf"
                    else "sample_dataset_metadata"
                ),
                "page_audits": page_audits,
            }
        )

    return {
        "version": "2026-07-02",
        "sample_root": str(sample_root),
        "manifest_path": str(manifest_path),
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    payload = build_withholding_sample_page_audit(args.sample_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
