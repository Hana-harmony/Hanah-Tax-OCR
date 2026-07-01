from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from scripts.review_queue.bootstrap_uncovered_samples import bootstrap_uncovered_samples


def test_bootstrap_uncovered_samples_creates_pending_review_labels(tmp_path: Path) -> None:
    coverage_report_path = tmp_path / "sample_data_coverage.json"
    coverage_report_path.write_text(
        json.dumps(
            {
                "uncovered_sample_paths": [
                    unicodedata.normalize("NFD", "sample_data/거주자증명서/4.jpg"),
                    "sample_data/unknown/missing.png",
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    written = bootstrap_uncovered_samples(
        coverage_report_path,
        tmp_path / "pending_review",
    )

    assert len(written) == 1
    label_path = (
        tmp_path
        / "pending_review"
        / "residency_certificate"
        / "residency_legacy_001"
        / "label.json"
    )
    assert written == [label_path]

    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert payload["case_id"] == "residency_legacy_001"
    assert payload["document_type"] == "residency_certificate"
    assert payload["source_path"] == "sample_data/거주자증명서/4.jpg"
    assert payload["dataset_split"] == "pending_review"
    assert payload["promotion_status"] == "needs_human_verification"
    assert payload["expected_status"] is None
    assert payload["expected_fields"] == {}
    assert payload["expected_quality_checks"] == {}


def test_bootstrap_uncovered_samples_skips_existing_labels_without_overwrite(
    tmp_path: Path,
) -> None:
    coverage_report_path = tmp_path / "sample_data_coverage.json"
    coverage_report_path.write_text(
        json.dumps(
            {
                "uncovered_sample_paths": [
                    "sample_data/국내원천소득 제한세율/원본 샘플.pdf",
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    label_path = (
        tmp_path
        / "pending_review"
        / "withholding_tax_form"
        / "withholding_pdf_001"
        / "label.json"
    )
    label_path.parent.mkdir(parents=True)
    label_path.write_text(json.dumps({"case_id": "existing"}), encoding="utf-8")

    written = bootstrap_uncovered_samples(
        coverage_report_path,
        tmp_path / "pending_review",
    )

    assert written == []
    assert json.loads(label_path.read_text(encoding="utf-8")) == {"case_id": "existing"}


def test_bootstrap_uncovered_samples_prefers_alias_ocr_source_and_skips_non_extractable(
    tmp_path: Path,
) -> None:
    coverage_report_path = tmp_path / "sample_data_coverage.json"
    coverage_report_path.write_text(
        json.dumps(
            {
                "uncovered_sample_paths": [
                    "sample_data/거주자증명서/2.pdf",
                    "sample_data/국내원천소득 제한세율/국내원천소득 제한세율 적용신청서-2.png",
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    written = bootstrap_uncovered_samples(
        coverage_report_path,
        tmp_path / "pending_review",
    )

    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["case_id"] == "residency_pdf_001"
    assert payload["source_path"] == "sample_data/거주자증명서/2_page1.png"
