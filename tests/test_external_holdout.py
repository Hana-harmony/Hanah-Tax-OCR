from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.evals.build_external_holdout import (
    build_external_holdout_manifest,
    write_external_holdout,
)


def test_build_external_holdout_manifest_flags_partial_expected_and_eval_overlap(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample_data"
    sample_root.mkdir()
    residency_path = sample_root / "residency.png"
    apostille_path = sample_root / "apostille.png"
    Image.new("RGB", (800, 1000), "white").save(residency_path)
    Image.new("RGB", (700, 900), "white").save(apostille_path)

    labeled_root = tmp_path / "data" / "labeled"
    eval_root = tmp_path / "evals" / "cases"
    eval_case_dir = eval_root / "eval_case_001"
    eval_case_dir.mkdir(parents=True)
    (eval_case_dir / "expected.json").write_text(
        json.dumps({"case_id": "eval_case_001"}),
        encoding="utf-8",
    )

    ready_dir = labeled_root / "pending_review" / "residency_certificate" / "holdout_case_001"
    ready_dir.mkdir(parents=True)
    (ready_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "holdout_case_001",
                "document_type": "residency_certificate",
                "source_path": str(residency_path),
                "expected_fields": {
                    "taxpayer_name": "SAMPLE USER",
                    "tin": "101-11-1001",
                    "tax_year": "2026",
                    "issue_date": "January 2, 2026",
                    "residency_country": "United States of America",
                    "residency_country_code": "US",
                },
            }
        ),
        encoding="utf-8",
    )
    partial_dir = labeled_root / "pending_review" / "apostille" / "holdout_case_002"
    partial_dir.mkdir(parents=True)
    (partial_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "holdout_case_002",
                "document_type": "apostille",
                "source_path": str(apostille_path),
                "expected_fields": {
                    "issuing_country": "United States of America",
                    "certificate_number": "4",
                },
            }
        ),
        encoding="utf-8",
    )
    overlap_label_dir = labeled_root / "residency_certificate" / "eval_case_001"
    overlap_label_dir.mkdir(parents=True)
    (overlap_label_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "eval_case_001",
                "document_type": "residency_certificate",
                "source_path": str(sample_root / "overlap.png"),
                "expected_fields": {"taxpayer_name": "OVERLAP"},
            }
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (1000, 1200), "white").save(sample_root / "overlap.png")

    manifest = build_external_holdout_manifest(
        sample_root,
        labeled_root,
        eval_root,
        sample_index={
            str(residency_path): {
                "document_type": "residency_certificate",
                "case_id": "holdout_case_001",
                "split": "train",
            },
            str(apostille_path): {
                "document_type": "apostille",
                "case_id": "holdout_case_002",
                "split": "train",
            },
            str(sample_root / "overlap.png"): {
                "document_type": "residency_certificate",
                "case_id": "eval_case_001",
                "split": "test",
            },
        },
    )

    assert manifest["summary"]["candidate_case_count"] == 2
    assert manifest["summary"]["excluded_eval_overlap_count"] == 1
    status_by_case = {case["case_id"]: case["status"] for case in manifest["cases"]}
    assert status_by_case == {
        "holdout_case_001": "ready",
        "holdout_case_002": "partial_expected",
    }
    partial_case = next(case for case in manifest["cases"] if case["case_id"] == "holdout_case_002")
    assert "signed_by" in partial_case["missing_required_fields"]
    assert "low_quality" in partial_case["subset_tags"]


def test_build_external_holdout_manifest_flags_source_path_mismatch(tmp_path: Path) -> None:
    sample_root = tmp_path / "sample_data"
    sample_root.mkdir()
    sample_path = sample_root / "withholding_source_2.png"
    mismatched_label_path = sample_root / "withholding_source_1.png"
    Image.new("RGB", (1200, 1600), "white").save(sample_path)
    Image.new("RGB", (1200, 1600), "white").save(mismatched_label_path)

    labeled_root = tmp_path / "data" / "labeled"
    pending_dir = labeled_root / "pending_review" / "withholding_tax_form" / "withholding_case_001"
    pending_dir.mkdir(parents=True)
    (pending_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "withholding_case_001",
                "document_type": "withholding_tax_form",
                "source_path": str(mismatched_label_path),
                "expected_fields": {
                    "first_name": "MARIA",
                    "last_name": "CHEN",
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_external_holdout_manifest(
        sample_root,
        labeled_root,
        tmp_path / "evals" / "cases",
        sample_index={
            str(sample_path): {
                "document_type": "withholding_tax_form",
                "case_id": "withholding_case_001",
                "split": "train",
            }
        },
    )

    assert manifest["summary"]["source_path_mismatch_case_count"] == 1
    case = manifest["cases"][0]
    assert case["case_id"] == "withholding_case_001"
    assert case["status"] == "needs_annotation"
    assert case["source_path_mismatch"] is True
    assert case["mismatched_label_paths"]


def test_write_external_holdout_materializes_expected_cases(tmp_path: Path) -> None:
    manifest = {
        "cases": [
            {
                "case_id": "holdout_case_001",
                "document_type": "residency_certificate",
                "sample_path": "sample_data/residency.png",
                "status": "ready",
                "subset_tags": ["low_quality"],
                "expected_fields": {
                    "taxpayer_name": "SAMPLE USER",
                    "tin": "101-11-1001",
                    "tax_year": "2026",
                    "issue_date": "January 2, 2026",
                    "residency_country": "United States of America",
                    "residency_country_code": "US",
                },
            },
            {
                "case_id": "holdout_case_002",
                "document_type": "apostille",
                "sample_path": "sample_data/apostille.png",
                "status": "needs_annotation",
                "subset_tags": ["format_variation"],
                "expected_fields": {},
            },
        ],
        "required_samples": [],
        "summary": {},
    }

    write_external_holdout(manifest, tmp_path / "evals" / "external_holdout")

    expected_path = (
        tmp_path
        / "evals"
        / "external_holdout"
        / "cases"
        / "holdout_case_001"
        / "expected.json"
    )
    assert expected_path.exists()
    payload = json.loads(expected_path.read_text(encoding="utf-8"))
    assert payload["holdout_status"] == "ready"
    assert payload["subset_tags"] == ["low_quality"]
