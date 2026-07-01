from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.evals.build_external_holdout import (
    build_external_holdout_manifest,
    write_external_holdout,
)
from scripts.evals.report_external_holdout_gaps import build_external_holdout_gap_report


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


def test_build_external_holdout_manifest_excludes_non_extractable_and_links_alias_labels(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample_data"
    sample_root.mkdir()
    pdf_path = sample_root / "거주자증명서" / "2.pdf"
    rasterized_path = sample_root / "거주자증명서" / "2_page1.png"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.5")
    rasterized_path.write_bytes(b"png")
    reverse_side_path = (
        sample_root / "국내원천소득 제한세율" / "국내원천소득 제한세율 적용신청서-2.png"
    )
    reverse_side_path.parent.mkdir(parents=True, exist_ok=True)
    reverse_side_path.write_bytes(b"img")

    labeled_root = tmp_path / "data" / "labeled"
    label_dir = labeled_root / "pending_review" / "residency_certificate" / "residency_pdf_001"
    label_dir.mkdir(parents=True)
    (label_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "residency_pdf_001",
                "document_type": "residency_certificate",
                "source_path": str(rasterized_path),
                "expected_fields": {
                    "taxpayer_name": "UNIVERSITY OF WASHINGTON",
                    "tin": "91-6001537",
                    "tax_year": "2025",
                    "issue_date": "March 24, 2025",
                    "residency_country": "United States of America",
                    "residency_country_code": "US",
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
            str(pdf_path): {
                "source": str(pdf_path),
                "document_type": "residency_certificate",
                "case_id": "residency_pdf_001",
                "split": "test",
                "extractable": True,
                "page_role": "front_pdf",
                "source_variants": [str(pdf_path), str(rasterized_path)],
            },
            str(rasterized_path): {
                "source": str(pdf_path),
                "document_type": "residency_certificate",
                "case_id": "residency_pdf_001",
                "split": "test",
                "extractable": True,
                "page_role": "front_pdf",
                "source_variants": [str(pdf_path), str(rasterized_path)],
            },
            str(reverse_side_path): {
                "source": str(reverse_side_path),
                "document_type": "withholding_tax_form",
                "case_id": "withholding_hana_payer_001",
                "split": "train",
                "extractable": False,
                "page_role": "reverse_side_instructions",
                "exclusion_reason": "back_side_reference_page",
                "source_variants": [str(reverse_side_path)],
            },
        },
    )

    assert manifest["summary"]["candidate_case_count"] == 1
    status_by_case = {case["case_id"]: case["status"] for case in manifest["cases"]}
    assert status_by_case == {"residency_pdf_001": "ready"}
    assert manifest["cases"][0]["source_path_mismatch"] is False
    assert manifest["cases"][0]["source_path_alias_match"] is True
    assert manifest["summary"]["excluded_non_extractable_count"] == 1
    assert manifest["excluded_non_extractable_cases"][0]["case_id"] == "withholding_hana_payer_001"
    assert manifest["excluded_non_extractable_cases"][0]["page_role"] == "reverse_side_instructions"
    assert manifest["excluded_non_extractable_cases"][0]["subset_tags"] == [
        "format_variation",
        "mixed_language",
    ]


def test_build_external_holdout_manifest_tracks_non_extractable_label_conflicts(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample_data"
    sample_root.mkdir()
    sample_path = sample_root / "국내원천소득 제한세율 적용신청서-2.png"
    label_source = sample_root / "국내원천소득 제한세율 적용신청서-1.png"
    sample_path.write_bytes(b"img")
    label_source.write_bytes(b"img")

    labeled_root = tmp_path / "data" / "labeled"
    label_dir = (
        labeled_root
        / "pending_review"
        / "withholding_tax_form"
        / "withholding_hana_payer_001"
    )
    label_dir.mkdir(parents=True)
    (label_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "withholding_hana_payer_001",
                "document_type": "withholding_tax_form",
                "source_path": str(label_source),
                "expected_fields": {"first_name": "MARIA"},
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
                "source": str(sample_path),
                "document_type": "withholding_tax_form",
                "case_id": "withholding_hana_payer_001",
                "split": "train",
                "extractable": False,
                "page_role": "reverse_side_instructions",
                "exclusion_reason": "back_side_reference_page",
                "source_variants": [str(sample_path)],
            }
        },
    )

    excluded = manifest["excluded_non_extractable_cases"][0]
    assert manifest["summary"]["excluded_non_extractable_label_conflict_count"] == 1
    assert excluded["case_id"] == "withholding_hana_payer_001"
    assert excluded["source_path_mismatch"] is True
    assert excluded["label_case_ids"] == ["withholding_hana_payer_001"]
    assert excluded["mismatched_label_paths"]


def test_build_external_holdout_manifest_includes_field_observations_for_partial_cases(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample_data"
    sample_root.mkdir()
    sample_path = sample_root / "apostille.png"
    Image.new("RGB", (1041, 1312), "white").save(sample_path)

    labeled_root = tmp_path / "data" / "labeled"
    label_dir = labeled_root / "pending_review" / "apostille" / "apostille_case_001"
    label_dir.mkdir(parents=True)
    (label_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "apostille_case_001",
                "document_type": "apostille",
                "source_path": str(sample_path),
                "expected_fields": {
                    "issuing_country": "United States of America",
                    "signer_capacity": "Deputy Clerk",
                    "seal_owner": "County of Sample",
                    "issued_at": "Los Angeles, California",
                    "certificate_number": "4",
                },
            }
        ),
        encoding="utf-8",
    )
    annotations_path = tmp_path / "case_annotations.json"
    annotations_path.write_text(
        json.dumps(
            {
                "cases": {
                    "apostille_case_001": {
                        "field_observations": {
                            "signed_by": {"status": "source_blank"},
                            "issued_on": {"status": "source_blank"},
                        }
                    }
                }
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
                "source": str(sample_path),
                "document_type": "apostille",
                "case_id": "apostille_case_001",
                "split": "train",
                "extractable": True,
                "page_role": "front",
                "source_variants": [str(sample_path)],
            }
        },
        case_annotations_path=annotations_path,
    )

    case = manifest["cases"][0]
    assert case["missing_required_fields"] == ["signed_by", "issued_on", "issuing_authority"]
    assert case["field_observations"]["signed_by"]["status"] == "source_blank"
    assert case["field_observations"]["issued_on"]["status"] == "source_blank"
    assert manifest["required_samples"][0]["field_observations"] == {
        "signed_by": {"status": "source_blank"},
        "issued_on": {"status": "source_blank"},
    }


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
                "field_observations": {"issue_date": {"status": "verified"}},
            },
            {
                "case_id": "holdout_case_002",
                "document_type": "apostille",
                "sample_path": "sample_data/apostille.png",
                "status": "needs_annotation",
                "subset_tags": ["format_variation"],
                "expected_fields": {},
                "field_observations": {},
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
    assert payload["field_observations"]["issue_date"]["status"] == "verified"


def test_build_external_holdout_gap_report_surfaces_non_extractable_blockers(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cases": [],
                "excluded_eval_overlap_cases": [
                    {
                        "case_id": "withholding_maria_chen_001",
                        "document_type": "withholding_tax_form",
                        "subset_tags": ["mixed_language", "low_quality"],
                    }
                ],
                "excluded_non_extractable_cases": [
                    {
                        "case_id": "withholding_hana_payer_001",
                        "document_type": "withholding_tax_form",
                        "page_role": "reverse_side_instructions",
                        "exclusion_reason": "back_side_reference_page",
                        "source_path_mismatch": True,
                        "label_case_ids": ["withholding_hana_payer_001"],
                        "subset_tags": ["format_variation", "mixed_language"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_external_holdout_gap_report(manifest_path)

    target = next(
        item
        for item in report["targets"]
        if item["target_id"] == "withholding_front_filled_nonoverlap_low_quality"
    )
    assert target["status"] == "missing_source"
    assert target["evidence"]["blocked_by_eval_overlap_case_ids"] == ["withholding_maria_chen_001"]
    assert target["evidence"]["blocked_by_non_extractable_cases"] == [
        {
            "case_id": "withholding_hana_payer_001",
            "page_role": "reverse_side_instructions",
            "exclusion_reason": "back_side_reference_page",
            "source_path_mismatch": True,
            "label_case_ids": ["withholding_hana_payer_001"],
        }
    ]
