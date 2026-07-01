from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from hanah_tax_ocr.quality import compute_quality_metrics
from hanah_tax_ocr.training.sample_dataset import build_sample_index, normalize_path_text

DEFAULT_SAMPLE_ROOT = Path("sample_data")
DEFAULT_LABELED_ROOT = Path("data/labeled")
DEFAULT_EVAL_ROOT = Path("evals/cases")
DEFAULT_OUTPUT_ROOT = Path("evals/external_holdout")
DEFAULT_CASE_ANNOTATIONS_PATH = Path("evals/external_holdout/case_annotations.json")
CURRENT_VERSION = "2026-07-02"

REQUIRED_FIELDS_BY_DOCUMENT_TYPE: dict[str, tuple[str, ...]] = {
    "apostille": (
        "issuing_country",
        "signed_by",
        "signer_capacity",
        "seal_owner",
        "issued_at",
        "issued_on",
        "issuing_authority",
        "certificate_number",
    ),
    "residency_certificate": (
        "taxpayer_name",
        "tin",
        "tax_year",
        "issue_date",
        "residency_country",
        "residency_country_code",
    ),
    "withholding_tax_form": (
        "first_name",
        "last_name",
        "middle_name",
        "tin",
        "address",
        "residency_country",
        "residency_country_code",
        "dividend_tax_rate",
        "signature_date",
        "applicant_name",
    ),
}

MANUAL_SUBSET_TAGS: dict[str, tuple[str, ...]] = {
    "apostille_california_001": ("format_variation", "low_quality"),
    "residency_john_doe_001": ("low_quality", "format_variation"),
    "residency_legacy_001": ("crop_miss", "format_variation"),
    "residency_pdf_001": ("format_variation", "pdf_render"),
    "residency_university_hawaii_001": ("low_quality", "format_variation"),
    "withholding_hana_payer_001": ("mixed_language", "format_variation"),
    "withholding_pdf_001": ("format_variation", "pdf_render"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an external-holdout manifest that stays isolated from evals/cases."
    )
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--labeled-root", type=Path, default=DEFAULT_LABELED_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-annotations", type=Path, default=DEFAULT_CASE_ANNOTATIONS_PATH)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_case_annotations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {
        str(case_id): dict(payload)
        for case_id, payload in load_json(path).get("cases", {}).items()
    }


def _expected_fields_for_document_type(document_type: str) -> tuple[str, ...]:
    return REQUIRED_FIELDS_BY_DOCUMENT_TYPE.get(document_type, ())


def _label_status(
    document_type: str | None,
    expected_fields: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    expected_fields = expected_fields or {}
    if not document_type:
        return "needs_annotation", []
    required_fields = _expected_fields_for_document_type(document_type)
    missing_required = [
        field_name
        for field_name in required_fields
        if not expected_fields.get(field_name)
    ]
    if not expected_fields:
        return "needs_annotation", list(required_fields)
    if missing_required:
        return "partial_expected", missing_required
    return "ready", []


def _subset_tags_for_case(
    *,
    case_id: str | None,
    sample_path: Path,
    document_type: str | None,
    quality_metrics: dict[str, Any],
) -> list[str]:
    tags = set(MANUAL_SUBSET_TAGS.get(case_id or "", ()))
    suffix = sample_path.suffix.lower()
    if suffix == ".pdf":
        tags.add("pdf_render")
        tags.add("format_variation")
    width = quality_metrics.get("width")
    height = quality_metrics.get("height")
    if isinstance(width, int) and isinstance(height, int):
        if min(width, height) < 900:
            tags.add("low_quality")
    blur_score = quality_metrics.get("blur_score")
    if isinstance(blur_score, int | float) and blur_score < 400:
        tags.add("blur")
    if document_type == "withholding_tax_form":
        tags.add("mixed_language")
    if sample_path.suffix.lower() == ".jpg" and document_type in {
        "apostille",
        "residency_certificate",
    }:
        tags.add("format_variation")
    return sorted(tags)


def _priority_score(
    *,
    status: str,
    subset_tags: list[str],
    missing_required_fields: list[str],
    source_path_mismatch: bool,
) -> float:
    score = 0.0
    if status == "ready":
        score += 40.0
    elif status == "partial_expected":
        score += 28.0
    else:
        score += 20.0
    score += len(subset_tags) * 4.0
    score += len(missing_required_fields) * 2.0
    if "low_quality" in subset_tags:
        score += 8.0
    if "blur" in subset_tags or "crop_miss" in subset_tags:
        score += 6.0
    if "mixed_language" in subset_tags:
        score += 4.0
    if source_path_mismatch:
        score += 12.0
    return round(score, 2)


def build_external_holdout_manifest(
    sample_root: Path,
    labeled_root: Path,
    eval_root: Path,
    *,
    sample_index: dict[str, dict[str, str]] | None = None,
    case_annotations_path: Path | None = None,
) -> dict[str, Any]:
    sample_index = sample_index or build_sample_index(sample_root)
    case_annotations = _load_case_annotations(case_annotations_path)

    labels_by_source: dict[str, list[dict[str, Any]]] = {}
    labels_by_case_id: dict[str, list[dict[str, Any]]] = {}
    eval_sources: set[str] = set()
    eval_case_ids: set[str] = set()

    for label_path in sorted(labeled_root.rglob("label.json")):
        payload = load_json(label_path)
        source_path = payload.get("source_path")
        case_id = payload.get("case_id")
        if not source_path or not case_id:
            continue
        record = {
            "label_path": str(label_path),
            "case_id": str(case_id),
            "document_type": str(payload.get("document_type") or ""),
            "source_path": str(source_path),
            "expected_fields": dict(payload.get("expected_fields") or {}),
        }
        labels_by_source.setdefault(normalize_path_text(source_path), []).append(record)
        labels_by_case_id.setdefault(str(case_id), []).append(record)

    for expected_path in sorted(eval_root.glob("*/expected.json")):
        payload = load_json(expected_path)
        case_id = payload.get("case_id")
        if case_id:
            eval_case_ids.add(str(case_id))
        for label_record in labels_by_case_id.get(str(case_id or ""), []):
            eval_sources.add(normalize_path_text(label_record["source_path"]))

    cases: list[dict[str, Any]] = []
    required_samples: list[dict[str, Any]] = []
    excluded_overlap_cases: list[dict[str, Any]] = []
    excluded_non_extractable_cases: list[dict[str, Any]] = []

    for sample_path in sorted(path for path in sample_root.rglob("*") if path.is_file()):
        if sample_path.name == ".DS_Store":
            continue

        source_key = normalize_path_text(sample_path)
        sample_dataset_entry = sample_index.get(source_key, {})
        canonical_source = normalize_path_text(str(sample_dataset_entry.get("source") or ""))
        if canonical_source and source_key != canonical_source:
            continue
        source_variants = {
            str(variant)
            for variant in sample_dataset_entry.get("source_variants", [])
            if isinstance(variant, str) and variant
        }
        matched_label_map: dict[str, dict[str, Any]] = {}
        for variant in {source_key, *source_variants}:
            for record in labels_by_source.get(variant, []):
                matched_label_map[record["label_path"]] = record
        sample_case_id = str(sample_dataset_entry.get("case_id") or "")
        if sample_case_id:
            for record in labels_by_case_id.get(sample_case_id, []):
                matched_label_map[record["label_path"]] = record
        matched_labels = sorted(matched_label_map.values(), key=lambda record: record["label_path"])
        matched_case_ids = sorted({record["case_id"] for record in matched_labels})
        case_id = sample_case_id or (matched_case_ids[0] if matched_case_ids else "")
        has_direct_or_alias_label_match = any(
            normalize_path_text(record["source_path"]) in {source_key, *source_variants}
            for record in matched_labels
        )
        source_path_mismatch = False
        source_path_alias_match = False
        mismatched_label_paths: list[str] = []
        if sample_case_id:
            for label_record in labels_by_case_id.get(sample_case_id, []):
                normalized_label_source = normalize_path_text(label_record["source_path"])
                if normalized_label_source == source_key:
                    continue
                if normalized_label_source in source_variants:
                    source_path_alias_match = True
                    continue
                source_path_mismatch = True
                mismatched_label_paths.append(label_record["label_path"])

        quality_metrics = compute_quality_metrics(sample_path)
        subset_tags = _subset_tags_for_case(
            case_id=case_id or None,
            sample_path=sample_path,
            document_type=str(sample_dataset_entry.get("document_type") or "") or None,
            quality_metrics=quality_metrics,
        )

        if sample_dataset_entry and not bool(sample_dataset_entry.get("extractable", True)):
            excluded_non_extractable_cases.append(
                {
                    "case_id": case_id or None,
                    "sample_path": source_key,
                    "document_type": sample_dataset_entry.get("document_type"),
                    "page_role": sample_dataset_entry.get("page_role"),
                    "exclusion_reason": sample_dataset_entry.get("exclusion_reason"),
                    "label_case_ids": matched_case_ids,
                    "label_source_paths": sorted(
                        {normalize_path_text(record["source_path"]) for record in matched_labels}
                    ),
                    "source_path_mismatch": source_path_mismatch,
                    "source_path_alias_match": source_path_alias_match,
                    "mismatched_label_paths": mismatched_label_paths,
                    "subset_tags": subset_tags,
                }
            )
            continue

        if source_key in eval_sources or case_id in eval_case_ids:
            excluded_overlap_cases.append(
                {
                    "case_id": case_id or None,
                    "sample_path": source_key,
                    "document_type": sample_dataset_entry.get("document_type"),
                    "subset_tags": subset_tags,
                }
            )
            continue

        label_record = matched_labels[0] if matched_labels else None
        document_type = str(
            (label_record or {}).get("document_type")
            or sample_dataset_entry.get("document_type")
            or ""
        ) or None
        expected_fields = dict((label_record or {}).get("expected_fields") or {})
        status, missing_required_fields = _label_status(document_type, expected_fields)
        if source_path_mismatch and not has_direct_or_alias_label_match:
            status = "needs_annotation"

        subset_tags = _subset_tags_for_case(
            case_id=case_id or None,
            sample_path=sample_path,
            document_type=document_type,
            quality_metrics=quality_metrics,
        )
        priority_score = _priority_score(
            status=status,
            subset_tags=subset_tags,
            missing_required_fields=missing_required_fields,
            source_path_mismatch=source_path_mismatch,
        )
        field_observations = dict(
            case_annotations.get(case_id, {}).get("field_observations", {}) or {}
        )

        case_entry = {
            "case_id": case_id or None,
            "sample_path": source_key,
            "document_type": document_type,
            "status": status,
            "current_eval_overlap": False,
            "source_origin": "sample_data",
            "sample_dataset_split": sample_dataset_entry.get("split"),
            "label_case_ids": matched_case_ids,
            "expected_field_count": len(expected_fields),
            "missing_required_fields": missing_required_fields,
            "subset_tags": subset_tags,
            "quality_metrics": quality_metrics,
            "priority_score": priority_score,
            "expected_fields": expected_fields,
            "field_observations": field_observations,
            "source_path_mismatch": source_path_mismatch,
            "source_path_alias_match": source_path_alias_match,
            "mismatched_label_paths": mismatched_label_paths,
        }
        cases.append(case_entry)
        if status != "ready":
            required_samples.append(
                {
                    "case_id": case_id or None,
                    "sample_path": source_key,
                    "document_type": document_type,
                    "status": status,
                    "missing_required_fields": missing_required_fields,
                    "field_observations": {
                        field_name: observation
                        for field_name, observation in field_observations.items()
                        if field_name in missing_required_fields
                    },
                    "source_path_mismatch": source_path_mismatch,
                    "mismatched_label_paths": mismatched_label_paths,
                    "subset_tags": subset_tags,
                    "priority_score": priority_score,
                }
            )

    status_counts = Counter(case["status"] for case in cases)
    document_type_counts = Counter(case["document_type"] or "unknown" for case in cases)
    subset_tag_counts = Counter(tag for case in cases for tag in case["subset_tags"])

    return {
        "version": CURRENT_VERSION,
        "sample_root": str(sample_root),
        "labeled_root": str(labeled_root),
        "eval_root": str(eval_root),
        "cases": sorted(cases, key=lambda item: (-item["priority_score"], item["sample_path"])),
        "required_samples": sorted(
            required_samples,
            key=lambda item: (-item["priority_score"], item["sample_path"]),
        ),
        "summary": {
            "candidate_case_count": len(cases),
            "ready_case_count": status_counts.get("ready", 0),
            "partial_expected_case_count": status_counts.get("partial_expected", 0),
            "needs_annotation_case_count": status_counts.get("needs_annotation", 0),
            "source_path_mismatch_case_count": sum(
                1 for case in cases if case["source_path_mismatch"]
            ),
            "excluded_eval_overlap_count": len(excluded_overlap_cases),
            "excluded_non_extractable_count": len(excluded_non_extractable_cases),
            "excluded_non_extractable_label_conflict_count": sum(
                1 for case in excluded_non_extractable_cases if case["source_path_mismatch"]
            ),
            "document_type_counts": dict(sorted(document_type_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "subset_tag_counts": dict(sorted(subset_tag_counts.items())),
        },
        "excluded_eval_overlap_cases": sorted(
            excluded_overlap_cases,
            key=lambda item: item["sample_path"],
        ),
        "excluded_non_extractable_cases": sorted(
            excluded_non_extractable_cases,
            key=lambda item: item["sample_path"],
        ),
    }


def write_external_holdout(manifest: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    schema = {
        "version": CURRENT_VERSION,
        "manifest_path": "evals/external_holdout/manifest.json",
        "required_samples_path": "evals/external_holdout/required_samples.json",
        "gap_report_path": "evals/external_holdout/missing_distribution_targets.json",
        "audit_report_path": "evals/external_holdout/non_extractable_source_audit.json",
        "case_expected_path_pattern": "evals/external_holdout/cases/<case_id>/expected.json",
        "status_values": ["ready", "partial_expected", "needs_annotation"],
        "subset_tags": [
            "blur",
            "crop_miss",
            "format_variation",
            "low_quality",
            "mixed_language",
            "pdf_render",
        ],
    }
    (output_root / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "required_samples.json").write_text(
        json.dumps(manifest["required_samples"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for case in manifest["cases"]:
        case_id = case.get("case_id")
        expected_fields = case.get("expected_fields") or {}
        if not case_id or not expected_fields:
            continue
        case_dir = cases_root / str(case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_id": case_id,
            "document_type": case["document_type"],
            "source_path": case["sample_path"],
            "expected_fields": expected_fields,
            "field_observations": case.get("field_observations", {}),
            "holdout_status": case["status"],
            "subset_tags": case["subset_tags"],
        }
        (case_dir / "expected.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    manifest = build_external_holdout_manifest(
        args.sample_root,
        args.labeled_root,
        args.eval_root,
        case_annotations_path=args.case_annotations,
    )
    write_external_holdout(manifest, args.output_root)
    print(json.dumps(manifest["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
