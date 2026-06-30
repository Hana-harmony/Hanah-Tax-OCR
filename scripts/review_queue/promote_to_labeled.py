from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_priority_cases(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    return {
        str(case["case_id"]): case
        for case in cases
        if isinstance(case, dict) and case.get("case_id")
    }


def load_priority_order(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(case_id) for case_id in payload.get("priority_order", [])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote queued review payloads into label scaffolds for human verification."
    )
    parser.add_argument(
        "--review-queue-dir",
        type=Path,
        default=Path("data/review_queue/index"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/labeled/pending_review"),
    )
    parser.add_argument(
        "--priority-report",
        type=Path,
        default=None,
        help="Optional review_queue priority report used to promote top-ranked cases first.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to promote. Applied after priority ordering.",
    )
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def promote_review_queue(
    review_queue_dir: Path,
    output_root: Path,
    *,
    priority_report_path: Path | None = None,
    limit: int | None = None,
    case_ids: set[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    written: list[Path] = []
    priority_cases: dict[str, dict] = {}
    queue_paths = {
        queue_path.stem: queue_path
        for queue_path in sorted(review_queue_dir.glob("*.json"))
    }

    ordered_case_ids: list[str]
    if priority_report_path is not None and priority_report_path.exists():
        priority_cases = load_priority_cases(priority_report_path)
        ordered_case_ids = [
            case_id
            for case_id in load_priority_order(priority_report_path)
            if case_id in queue_paths
        ]
        ordered_case_ids.extend(
            case_id for case_id in sorted(queue_paths) if case_id not in ordered_case_ids
        )
    else:
        ordered_case_ids = sorted(queue_paths)

    promoted_case_ids: set[str] = set()
    for case_id in ordered_case_ids:
        if limit is not None and len(promoted_case_ids) >= limit:
            break
        if case_ids and case_id not in case_ids:
            continue

        queue_path = queue_paths[case_id]
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
        priority_case = priority_cases.get(case_id)
        wrote_for_case = False
        for document in payload.get("documents", []):
            document_type = document["document_type"]
            label_dir = output_root / document_type / case_id
            label_path = label_dir / "label.json"
            if label_path.exists() and not overwrite:
                continue

            label_dir.mkdir(parents=True, exist_ok=True)
            label_payload = {
                "case_id": case_id,
                "document_type": document_type,
                "source_path": document.get("source_path"),
                "dataset_split": "pending_review",
                "promotion_status": "needs_human_verification",
                "expected_status": payload.get("review_result", {}).get("status"),
                "expected_fields": document.get("fields", {}),
                "expected_quality_checks": document.get("quality_checks", {}),
                "source_findings": payload.get("review_result", {}).get("findings", []),
            }
            if priority_case is not None:
                label_payload["priority_context"] = {
                    "priority_score": priority_case.get("priority_score"),
                    "status": priority_case.get("status"),
                    "matched_field_groups": priority_case.get("matched_field_groups", []),
                    "recommendations": priority_case.get("recommendations", []),
                    "score_breakdown": priority_case.get("score_breakdown", {}),
                }
            label_path.write_text(
                json.dumps(label_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written.append(label_path)
            wrote_for_case = True
        if wrote_for_case:
            promoted_case_ids.add(case_id)
    return written


def main() -> None:
    args = parse_args()
    written = promote_review_queue(
        args.review_queue_dir,
        args.output_root,
        priority_report_path=args.priority_report,
        limit=args.limit,
        case_ids=set(args.case_id or []),
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "promoted": len(written),
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
