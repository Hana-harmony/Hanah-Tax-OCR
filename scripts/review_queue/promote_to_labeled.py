from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def promote_review_queue(
    review_queue_dir: Path,
    output_root: Path,
    *,
    case_ids: set[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    written: list[Path] = []
    for queue_path in sorted(review_queue_dir.glob("*.json")):
        case_id = queue_path.stem
        if case_ids and case_id not in case_ids:
            continue

        payload = json.loads(queue_path.read_text(encoding="utf-8"))
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
            label_path.write_text(
                json.dumps(label_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written.append(label_path)
    return written


def main() -> None:
    args = parse_args()
    written = promote_review_queue(
        args.review_queue_dir,
        args.output_root,
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
