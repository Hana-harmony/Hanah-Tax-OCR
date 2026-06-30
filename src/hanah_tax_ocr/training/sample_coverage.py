from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_ROOT = Path("sample_data")
DEFAULT_LABELED_ROOT = Path("data/labeled")
DEFAULT_EVAL_ROOT = Path("evals/cases")
DEFAULT_OUTPUT_PATH = Path("data/training/reports/sample_data_coverage.json")
PENDING_REVIEW_SPLIT = "pending_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report sample_data coverage across reviewed labels and eval cases."
    )
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--labeled-root", type=Path, default=DEFAULT_LABELED_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_path_text(path: str | Path) -> str:
    return unicodedata.normalize("NFC", str(path))


def build_sample_data_coverage_report(
    sample_root: Path,
    labeled_root: Path,
    eval_root: Path,
) -> dict[str, Any]:
    sample_paths = sorted(path for path in sample_root.rglob("*") if path.is_file())
    labels_by_sample: dict[str, list[dict[str, str]]] = {}
    labels_by_case_id: dict[str, list[dict[str, str]]] = {}
    pending_review_by_sample: dict[str, list[dict[str, str]]] = {}

    for label_path in sorted(labeled_root.rglob("label.json")):
        payload = load_json(label_path)
        source_path = payload.get("source_path")
        case_id = payload.get("case_id")
        document_type = payload.get("document_type")
        if not source_path or not case_id or not document_type:
            continue
        record = {
            "case_id": str(case_id),
            "document_type": str(document_type),
            "label_path": str(label_path),
        }
        source_key = normalize_path_text(source_path)
        if is_pending_review_label(label_path, labeled_root):
            pending_review_by_sample.setdefault(source_key, []).append(record)
            continue
        labels_by_sample.setdefault(source_key, []).append(record)
        labels_by_case_id.setdefault(str(case_id), []).append(record)

    eval_case_ids: set[str] = set()
    eval_by_sample: dict[str, list[str]] = {}
    for expected_path in sorted(eval_root.glob("*/expected.json")):
        payload = load_json(expected_path)
        case_id = payload.get("case_id")
        if not case_id:
            continue
        case_id = str(case_id)
        eval_case_ids.add(case_id)
        for label_record in labels_by_case_id.get(case_id, []):
            label_path = Path(label_record["label_path"])
            label_payload = load_json(label_path)
            source_path = label_payload.get("source_path")
            if source_path:
                eval_by_sample.setdefault(normalize_path_text(source_path), []).append(case_id)

    samples: list[dict[str, Any]] = []
    uncovered_sample_paths: list[str] = []
    labeled_without_eval_case_ids: list[str] = sorted(
        {
            record["case_id"]
            for records in labels_by_sample.values()
            for record in records
            if record["case_id"] not in eval_case_ids
        }
    )
    for sample_path in sample_paths:
        sample_key = normalize_path_text(sample_path)
        label_records = labels_by_sample.get(sample_key, [])
        pending_review_records = pending_review_by_sample.get(sample_key, [])
        sample_eval_case_ids = sorted(set(eval_by_sample.get(sample_key, [])))
        covered_by_labeled = bool(label_records)
        covered_by_eval = bool(sample_eval_case_ids)
        if not covered_by_labeled:
            uncovered_sample_paths.append(sample_key)
        samples.append(
            {
                "sample_path": sample_key,
                "covered_by_labeled": covered_by_labeled,
                "covered_by_eval": covered_by_eval,
                "label_case_ids": sorted({record["case_id"] for record in label_records}),
                "pending_review_case_ids": sorted(
                    {record["case_id"] for record in pending_review_records}
                ),
                "eval_case_ids": sample_eval_case_ids,
                "document_types": sorted({record["document_type"] for record in label_records}),
            }
        )

    return {
        "sample_root": str(sample_root),
        "labeled_root": str(labeled_root),
        "eval_root": str(eval_root),
        "sample_file_count": len(sample_paths),
        "covered_by_labeled_count": sum(1 for item in samples if item["covered_by_labeled"]),
        "covered_by_pending_review_count": sum(
            1 for item in samples if item["pending_review_case_ids"]
        ),
        "covered_by_eval_count": sum(1 for item in samples if item["covered_by_eval"]),
        "uncovered_sample_paths": uncovered_sample_paths,
        "labeled_without_eval_case_ids": labeled_without_eval_case_ids,
        "samples": samples,
    }


def is_pending_review_label(label_path: Path, labeled_root: Path) -> bool:
    try:
        relative_path = label_path.relative_to(labeled_root)
    except ValueError:
        return False
    return bool(relative_path.parts) and relative_path.parts[0] == PENDING_REVIEW_SPLIT


def main() -> None:
    args = parse_args()
    report = build_sample_data_coverage_report(
        args.sample_root,
        args.labeled_root,
        args.eval_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
