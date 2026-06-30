from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

from scripts.ingest.bootstrap_sample_dataset import SAMPLE_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pending_review label scaffolds for uncovered sample_data files."
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=Path("data/training/reports/sample_data_coverage.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/labeled/pending_review"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_sample_index() -> dict[str, dict[str, str]]:
    return {normalize_path_text(entry["source"]): entry for entry in SAMPLE_DATASET}


def normalize_path_text(path: str | Path) -> str:
    return unicodedata.normalize("NFC", str(path))


def bootstrap_uncovered_samples(
    coverage_report_path: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    coverage_report = json.loads(coverage_report_path.read_text(encoding="utf-8"))
    uncovered_sample_paths = coverage_report.get("uncovered_sample_paths", [])
    sample_index = build_sample_index()

    written: list[Path] = []
    for sample_path in uncovered_sample_paths:
        entry = sample_index.get(normalize_path_text(sample_path))
        if entry is None:
            continue
        label_dir = output_root / entry["document_type"] / entry["case_id"]
        label_path = label_dir / "label.json"
        if label_path.exists() and not overwrite:
            continue

        label_dir.mkdir(parents=True, exist_ok=True)
        label_payload = {
            "case_id": entry["case_id"],
            "document_type": entry["document_type"],
            "source_path": entry["source"],
            "dataset_split": "pending_review",
            "promotion_status": "needs_human_verification",
            "expected_status": None,
            "expected_fields": {},
            "expected_quality_checks": {},
            "notes": [
                "Bootstrapped from sample_data coverage report.",
                "Human label review is required before this case enters the reviewed dataset.",
            ],
        }
        label_path.write_text(
            json.dumps(label_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written.append(label_path)
    return written


def main() -> None:
    args = parse_args()
    written = bootstrap_uncovered_samples(
        args.coverage_report,
        args.output_root,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "bootstrapped": len(written),
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
