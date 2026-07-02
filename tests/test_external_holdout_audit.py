from __future__ import annotations

import json
from pathlib import Path

from scripts.evals.audit_non_extractable_holdout_sources import (
    build_non_extractable_source_audit,
    classify_withholding_page_text,
)
from scripts.evals.report_external_holdout_gaps import build_external_holdout_gap_report


def test_classify_withholding_page_text_uses_conservative_filled_signals() -> None:
    back_side_result = classify_withholding_page_text(
        "뒤쪽 비거주자의 제한세율 적용 신청서등을 제출합니다 "
        "원천징수의무자 하나증권 대표자성명 강성묵 "
        "사업자주민등록번호 123-45-67890 세무서장"
    )
    assert back_side_result["classification"] == "back_side_payer_submission"
    assert back_side_result["holdout_usable"] is False

    blank_front_result = classify_withholding_page_text(
        "국내원천소득 제한세율 적용신청서 "
        "Last Name First Name Middle Name "
        "주소 납세자번호 생년월일 거주지국 거주지국코드"
    )
    assert blank_front_result["classification"] == "front_blank_template"
    assert blank_front_result["holdout_usable"] is False


def test_build_non_extractable_source_audit_classifies_withholding_blockers(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample_data"
    sample_root.mkdir()
    back_side_path = sample_root / "withholding-2.png"
    pdf_path = sample_root / "withholding.pdf"
    back_side_path.write_bytes(b"img")
    pdf_path.write_bytes(b"%PDF-1.5")

    audit = build_non_extractable_source_audit(
        sample_root,
        sample_index={
            str(back_side_path): {
                "source": str(back_side_path),
                "split": "train",
                "document_type": "withholding_tax_form",
                "case_id": "withholding_hana_payer_001",
                "extractable": False,
                "page_role": "reverse_side_instructions",
                "exclusion_reason": "back_side_reference_page",
            },
            str(pdf_path): {
                "source": str(pdf_path),
                "split": "test",
                "document_type": "withholding_tax_form",
                "case_id": "withholding_pdf_001",
                "extractable": False,
                "page_role": "blank_form_template",
                "exclusion_reason": "blank_form_without_filled_entities",
            },
        },
        image_text_loader=lambda path: (
            "뒤쪽 원천징수 하나증권 대표자 사업자 주민등록번호 제출자 2026년 작성방법"
            if path == back_side_path
            else ""
        ),
        pdf_text_loader=lambda path: [
            "국내원천소득 제한세율 적용신청서 Last Name First Name Middle Name 접수번호 접수일자",
            "뒤쪽 작성방법 접수번호 및 접수일자",
        ],
    )

    audit_by_case = {case["case_id"]: case for case in audit["cases"]}
    assert audit_by_case["withholding_hana_payer_001"]["source_classification"] == (
        "back_side_submission_only"
    )
    assert audit_by_case["withholding_hana_payer_001"]["holdout_usable"] is False
    assert audit_by_case["withholding_hana_payer_001"]["page_audits"][0]["classification"] == (
        "back_side_payer_submission"
    )

    assert audit_by_case["withholding_pdf_001"]["source_classification"] == (
        "blank_front_plus_back_side_only"
    )
    assert audit_by_case["withholding_pdf_001"]["holdout_usable"] is False
    assert audit_by_case["withholding_pdf_001"]["page_audits"][0]["classification"] == (
        "front_blank_template"
    )
    assert audit_by_case["withholding_pdf_001"]["page_audits"][1]["classification"] == (
        "back_side_instructions"
    )


def test_build_external_holdout_gap_report_includes_audit_evidence(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    audit_path = tmp_path / "audit.json"
    sample_page_audit_path = tmp_path / "sample_page_audit.json"
    manifest_path.write_text(
        json.dumps(
            {
                "excluded_eval_overlap_cases": [
                    {
                        "case_id": "withholding_maria_chen_001",
                        "document_type": "withholding_tax_form",
                        "sample_path": "sample_data/withholding-1.png",
                    }
                ],
                "excluded_non_extractable_cases": [
                    {
                        "case_id": "withholding_hana_payer_001",
                        "document_type": "withholding_tax_form",
                        "sample_path": "sample_data/withholding-2.png",
                        "page_role": "reverse_side_instructions",
                        "exclusion_reason": "back_side_reference_page",
                        "source_path_mismatch": True,
                        "label_case_ids": ["withholding_hana_payer_001"],
                    }
                ],
                "cases": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "withholding_hana_payer_001",
                        "source_classification": "back_side_submission_only",
                        "holdout_usable": False,
                        "blocker_reason": "back_side_without_target_fields",
                        "page_audits": [
                            {"classification": "back_side_payer_submission"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sample_page_audit_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "source_path": "sample_data/withholding-1.png",
                        "holdout_status": "eval_overlap",
                        "source_classification": "contains_target_page",
                        "holdout_usable": True,
                        "blocker_reason": None,
                        "page_audits": [
                            {"page_number": 1, "classification": "front_filled_target_page"},
                        ],
                    },
                    {
                        "source_path": "sample_data/withholding-2.png",
                        "holdout_status": "non_extractable",
                        "source_classification": "back_side_submission_only",
                        "holdout_usable": False,
                        "blocker_reason": "back_side_without_target_fields",
                        "page_audits": [
                            {"page_number": 1, "classification": "back_side_payer_submission"},
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = build_external_holdout_gap_report(
        manifest_path,
        audit_path=audit_path,
        sample_page_audit_path=sample_page_audit_path,
    )

    assert report["audit_report_path"] == str(audit_path)
    assert report["sample_page_audit_path"] == str(sample_page_audit_path)
    target = next(
        entry
        for entry in report["targets"]
        if entry["target_id"] == "withholding_front_filled_nonoverlap_low_quality"
    )
    assert target["evidence"]["blocked_by_non_extractable_cases"] == [
        {
            "case_id": "withholding_hana_payer_001",
            "page_role": "reverse_side_instructions",
            "exclusion_reason": "back_side_reference_page",
            "source_path_mismatch": True,
            "label_case_ids": ["withholding_hana_payer_001"],
            "audit_source_classification": "back_side_submission_only",
            "audit_holdout_usable": False,
            "audit_blocker_reason": "back_side_without_target_fields",
            "audit_page_classifications": ["back_side_payer_submission"],
        }
    ]
    assert target["evidence"]["withholding_sample_page_inventory"] == [
        {
            "source_path": "sample_data/withholding-1.png",
            "holdout_status": "eval_overlap",
            "source_classification": "contains_target_page",
            "holdout_usable": True,
            "blocker_reason": None,
            "page_audits": [
                {"page_number": 1, "classification": "front_filled_target_page"},
            ],
        },
        {
            "source_path": "sample_data/withholding-2.png",
            "holdout_status": "non_extractable",
            "source_classification": "back_side_submission_only",
            "holdout_usable": False,
            "blocker_reason": "back_side_without_target_fields",
            "page_audits": [
                {"page_number": 1, "classification": "back_side_payer_submission"},
            ],
        },
    ]
