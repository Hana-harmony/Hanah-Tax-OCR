from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from hanah_tax_ocr.training.sample_coverage import build_sample_data_coverage_report


def test_build_sample_data_coverage_report_matches_normalized_paths(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample_data"
    labeled_root = tmp_path / "data" / "labeled"
    eval_root = tmp_path / "evals" / "cases"
    sample_dir = sample_root / "국내원천소득 제한세율"
    sample_dir.mkdir(parents=True)
    sample_path = sample_dir / "적용신청서-1.png"
    sample_path.write_bytes(b"sample")

    label_dir = labeled_root / "withholding_tax_form" / "case_001"
    label_dir.mkdir(parents=True)
    (label_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "case_001",
                "document_type": "withholding_tax_form",
                "source_path": str(sample_path),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    case_dir = eval_root / "case_001"
    case_dir.mkdir(parents=True)
    (case_dir / "expected.json").write_text(
        json.dumps({"case_id": "case_001"}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = build_sample_data_coverage_report(sample_root, labeled_root, eval_root)

    assert report["sample_file_count"] == 1
    assert report["covered_by_labeled_count"] == 1
    assert report["covered_by_pending_review_count"] == 0
    assert report["covered_by_eval_count"] == 1
    assert report["uncovered_sample_paths"] == []
    assert report["labeled_without_eval_case_ids"] == []
    assert report["samples"][0]["sample_path"] == unicodedata.normalize("NFC", str(sample_path))
    assert report["samples"][0]["label_case_ids"] == ["case_001"]
    assert report["samples"][0]["pending_review_case_ids"] == []
    assert report["samples"][0]["eval_case_ids"] == ["case_001"]
    assert report["samples"][0]["known_to_sample_dataset"] is False
    assert report["samples"][0]["sample_dataset_document_type"] is None
    assert report["samples"][0]["sample_dataset_case_id"] is None
    assert report["samples"][0]["sample_dataset_split"] is None


def test_build_sample_data_coverage_report_lists_uncovered_and_missing_eval(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample_data"
    labeled_root = tmp_path / "data" / "labeled"
    eval_root = tmp_path / "evals" / "cases"
    sample_root.mkdir(parents=True)
    uncovered_path = sample_root / "unused.pdf"
    uncovered_path.write_bytes(b"unused")
    covered_path = sample_root / "covered.jpg"
    covered_path.write_bytes(b"covered")

    label_dir = labeled_root / "apostille" / "case_002"
    label_dir.mkdir(parents=True)
    (label_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "case_002",
                "document_type": "apostille",
                "source_path": str(covered_path),
            }
        ),
        encoding="utf-8",
    )

    report = build_sample_data_coverage_report(sample_root, labeled_root, eval_root)

    assert report["covered_by_labeled_count"] == 1
    assert report["covered_by_pending_review_count"] == 0
    assert report["covered_by_eval_count"] == 0
    assert report["uncovered_sample_paths"] == [unicodedata.normalize("NFC", str(uncovered_path))]
    assert report["labeled_without_eval_case_ids"] == ["case_002"]


def test_build_sample_data_coverage_report_separates_pending_review_labels(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample_data"
    labeled_root = tmp_path / "data" / "labeled"
    eval_root = tmp_path / "evals" / "cases"
    sample_root.mkdir(parents=True)
    sample_path = sample_root / "pending_only.jpg"
    sample_path.write_bytes(b"pending")

    pending_review_dir = labeled_root / "pending_review" / "apostille" / "case_003"
    pending_review_dir.mkdir(parents=True)
    (pending_review_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "case_003",
                "document_type": "apostille",
                "source_path": str(sample_path),
            }
        ),
        encoding="utf-8",
    )

    report = build_sample_data_coverage_report(sample_root, labeled_root, eval_root)

    assert report["covered_by_labeled_count"] == 0
    assert report["covered_by_pending_review_count"] == 1
    assert report["uncovered_sample_paths"] == [unicodedata.normalize("NFC", str(sample_path))]
    assert report["samples"][0]["pending_review_case_ids"] == ["case_003"]


def test_build_sample_data_coverage_report_adds_sample_dataset_metadata_for_known_samples(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample_data"
    labeled_root = tmp_path / "data" / "labeled"
    eval_root = tmp_path / "evals" / "cases"
    known_sample = sample_root / "거주자증명서" / "6.jpg"
    known_sample.parent.mkdir(parents=True)
    known_sample.write_bytes(b"known")

    report = build_sample_data_coverage_report(sample_root, labeled_root, eval_root)

    by_path = {item["sample_path"]: item for item in report["samples"]}
    known_sample = by_path[unicodedata.normalize("NFC", str(known_sample))]
    assert known_sample["known_to_sample_dataset"] is True
    assert known_sample["sample_dataset_document_type"] == "residency_certificate"
    assert known_sample["sample_dataset_case_id"] == "residency_university_hawaii_001"
    assert known_sample["sample_dataset_split"] == "val"
