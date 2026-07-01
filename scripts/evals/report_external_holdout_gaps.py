from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST_PATH = Path("evals/external_holdout/manifest.json")
DEFAULT_OUTPUT_PATH = Path("evals/external_holdout/missing_distribution_targets.json")
DEFAULT_AUDIT_PATH = Path("evals/external_holdout/non_extractable_source_audit.json")

TARGET_DEFINITIONS = (
    {
        "target_id": "withholding_front_filled_nonoverlap_low_quality",
        "document_type": "withholding_tax_form",
        "required_tags": ["low_quality", "mixed_language"],
        "priority": "highest",
        "needed_sample_description": (
            "내부 eval과 다른 원본의 front-side 작성 완료 withholding 샘플 1건 이상. "
            "저해상도 또는 blur가 있고, 한글 라벨과 영문 이름/주소가 함께 보여야 한다."
        ),
    },
    {
        "target_id": "withholding_front_filled_nonoverlap_crop_or_skew",
        "document_type": "withholding_tax_form",
        "required_tags": ["crop_miss"],
        "priority": "high",
        "needed_sample_description": (
            "이름/주소/서명일 중 하나가 border clipping 또는 skew 영향을 받는 작성 완료 "
            "withholding 샘플 1건 이상."
        ),
    },
    {
        "target_id": "apostille_blur_ready_case",
        "document_type": "apostille",
        "required_tags": ["blur"],
        "priority": "medium",
        "needed_sample_description": (
            "서명자/발행일이 실제로 채워져 있고 blur가 있는 apostille 샘플 1건 이상."
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize current external-holdout distribution gaps from the manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    return parser.parse_args()


def _matches_target(
    payload: dict[str, Any],
    *,
    document_type: str,
    required_tags: list[str],
) -> bool:
    if payload.get("document_type") != document_type:
        return False
    tags = set(payload.get("subset_tags", []))
    return set(required_tags).issubset(tags)


def _matches_document_type(payload: dict[str, Any], document_type: str) -> bool:
    return payload.get("document_type") == document_type


def _build_non_extractable_blocker(
    case: dict[str, Any],
    audit_case: dict[str, Any] | None,
) -> dict[str, Any]:
    blocker = {
        "case_id": case.get("case_id"),
        "page_role": case.get("page_role"),
        "exclusion_reason": case.get("exclusion_reason"),
        "source_path_mismatch": bool(case.get("source_path_mismatch")),
        "label_case_ids": case.get("label_case_ids", []),
    }
    if not audit_case:
        return blocker

    blocker["audit_source_classification"] = audit_case.get("source_classification")
    blocker["audit_holdout_usable"] = audit_case.get("holdout_usable")
    blocker["audit_blocker_reason"] = audit_case.get("blocker_reason")
    blocker["audit_page_classifications"] = [
        page.get("classification")
        for page in audit_case.get("page_audits", [])
    ]
    return blocker


def build_external_holdout_gap_report(
    manifest_path: Path,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = list(manifest.get("cases", []))
    overlap_cases = list(manifest.get("excluded_eval_overlap_cases", []))
    non_extractable_cases = list(manifest.get("excluded_non_extractable_cases", []))
    audit_cases = {}
    if audit_path is not None and audit_path.exists():
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_cases = {
            str(case.get("case_id")): case
            for case in audit_payload.get("cases", [])
            if case.get("case_id")
        }

    targets: list[dict[str, Any]] = []
    for definition in TARGET_DEFINITIONS:
        matched_cases = [
            case
            for case in cases
            if _matches_target(
                case,
                document_type=definition["document_type"],
                required_tags=definition["required_tags"],
            )
        ]
        ready_case_ids = [
            str(case["case_id"]) for case in matched_cases if case.get("status") == "ready"
        ]
        partial_case_ids = [
            str(case["case_id"])
            for case in matched_cases
            if case.get("status") == "partial_expected"
        ]
        overlap_case_ids = [
            str(case["case_id"])
            for case in overlap_cases
            if _matches_document_type(case, definition["document_type"])
        ]
        blocked_non_extractable = [
            _build_non_extractable_blocker(
                case,
                audit_cases.get(str(case.get("case_id"))),
            )
            for case in non_extractable_cases
            if _matches_document_type(case, definition["document_type"])
        ]

        if ready_case_ids:
            status = "covered"
        elif partial_case_ids:
            status = "partial_label_only"
        elif blocked_non_extractable or overlap_case_ids:
            status = "missing_source"
        else:
            status = "missing_source"

        targets.append(
            {
                "target_id": definition["target_id"],
                "document_type": definition["document_type"],
                "required_tags": definition["required_tags"],
                "status": status,
                "priority": definition["priority"],
                "evidence": {
                    "candidate_case_count_in_manifest": len(matched_cases),
                    "ready_case_ids": ready_case_ids,
                    "partial_case_ids": partial_case_ids,
                    "blocked_by_eval_overlap_case_ids": overlap_case_ids,
                    "blocked_by_non_extractable_cases": blocked_non_extractable,
                },
                "needed_sample_description": definition["needed_sample_description"],
            }
        )

    return {
        "version": "2026-07-02",
        "baseline_manifest": str(manifest_path),
        "audit_report_path": None if audit_path is None else str(audit_path),
        "targets": targets,
    }


def main() -> None:
    args = parse_args()
    payload = build_external_holdout_gap_report(args.manifest, audit_path=args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
