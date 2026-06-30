from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hanah_tax_ocr.training.data_gaps import DEFAULT_OUTPUT_PATH as DEFAULT_DATA_GAP_REPORT_PATH
from hanah_tax_ocr.training.data_gaps import load_json
from hanah_tax_ocr.training.field_crops import field_group_for

DEFAULT_REVIEW_QUEUE_DIR = Path("data/review_queue/index")
DEFAULT_LABELED_ROOT = Path("data/labeled")
DEFAULT_OUTPUT_PATH = Path("data/training/reports/review_queue_priority.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prioritize review_queue cases for manual labeling using data gap scores."
    )
    parser.add_argument(
        "--review-queue-dir",
        type=Path,
        default=DEFAULT_REVIEW_QUEUE_DIR,
    )
    parser.add_argument(
        "--data-gap-report",
        type=Path,
        default=DEFAULT_DATA_GAP_REPORT_PATH,
    )
    parser.add_argument(
        "--labeled-root",
        type=Path,
        default=DEFAULT_LABELED_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def _document_field_groups(document: dict[str, Any]) -> list[str]:
    field_names = sorted((document.get("fields") or {}).keys())
    return sorted({field_group_for(field_name) for field_name in field_names})


def build_review_queue_priority_report(
    review_queue_dir: Path,
    data_gap_report_path: Path,
    *,
    labeled_root: Path = DEFAULT_LABELED_ROOT,
) -> dict[str, Any]:
    data_gap_report = load_json(data_gap_report_path)
    priorities = data_gap_report.get("priorities", [])
    priority_by_group = {item["field_group"]: item for item in priorities}

    cases: list[dict[str, Any]] = []
    for queue_path in sorted(review_queue_dir.glob("*.json")):
        payload = load_json(queue_path)
        documents = payload.get("documents")
        if not documents:
            continue
        case_id = str(payload.get("case_id", queue_path.stem))
        if any(
            (labeled_root / str(document["document_type"]) / case_id / "label.json").exists()
            for document in documents
            if document.get("document_type")
        ):
            continue

        matched_groups = sorted(
            {
                field_group
                for document in documents
                for field_group in _document_field_groups(document)
                if field_group in priority_by_group
            },
            key=lambda field_group: (
                -priority_by_group[field_group]["priority_score"],
                field_group,
            ),
        )
        recommendations: list[str] = []
        for field_group in matched_groups:
            for recommendation in priority_by_group[field_group].get("recommendations", []):
                if recommendation not in recommendations:
                    recommendations.append(recommendation)

        review_result = payload.get("review_result", {})
        status = review_result.get("status", "unknown")
        findings = review_result.get("findings", [])
        gap_score = round(
            sum(priority_by_group[field_group]["priority_score"] for field_group in matched_groups),
            4,
        )
        status_boost = {
            "reject": 2.0,
            "needs_review": 1.0,
        }.get(status, 0.0)
        findings_boost = min(len(findings), 5) * 0.5
        priority_score = round(gap_score + status_boost + findings_boost, 4)

        cases.append(
            {
                "case_id": case_id,
                "priority_score": priority_score,
                "status": status,
                "document_types": sorted(
                    {
                        str(document.get("document_type"))
                        for document in documents
                        if document.get("document_type")
                    }
                ),
                "source_paths": sorted(
                    {
                        str(document.get("source_path"))
                        for document in documents
                        if document.get("source_path")
                    }
                ),
                "matched_field_groups": matched_groups,
                "score_breakdown": {
                    "gap_score": gap_score,
                    "status_boost": status_boost,
                    "findings_boost": findings_boost,
                },
                "findings": findings,
                "recommendations": recommendations,
                "label_targets": [
                    {
                        "document_type": document["document_type"],
                        "label_path": str(
                            labeled_root
                            / "pending_review"
                            / str(document["document_type"])
                            / case_id
                            / "label.json"
                        ),
                    }
                    for document in documents
                    if document.get("document_type")
                ],
            }
        )

    cases.sort(key=lambda item: (-item["priority_score"], item["case_id"]))
    return {
        "review_queue_dir": str(review_queue_dir),
        "data_gap_report_path": str(data_gap_report_path),
        "prioritized_case_count": len(cases),
        "priority_order": [item["case_id"] for item in cases],
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    report = build_review_queue_priority_report(
        args.review_queue_dir,
        args.data_gap_report,
        labeled_root=args.labeled_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
