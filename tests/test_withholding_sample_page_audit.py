from __future__ import annotations

import json
from pathlib import Path

from scripts.evals.audit_withholding_sample_pages import build_withholding_sample_page_audit


def test_build_withholding_sample_page_audit_classifies_sample_inventory(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / "sample_data"
    sample_root.mkdir()
    front_path = sample_root / "withholding-1.png"
    back_path = sample_root / "withholding-2.png"
    pdf_path = sample_root / "withholding.pdf"
    front_path.write_bytes(b"img")
    back_path.write_bytes(b"img")
    pdf_path.write_bytes(b"%PDF-1.5")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "excluded_eval_overlap_cases": [
                    {
                        "case_id": "withholding_maria_chen_001",
                        "sample_path": str(front_path),
                        "document_type": "withholding_tax_form",
                        "subset_tags": ["mixed_language"],
                    }
                ],
                "excluded_non_extractable_cases": [
                    {
                        "case_id": "withholding_hana_payer_001",
                        "sample_path": str(back_path),
                        "document_type": "withholding_tax_form",
                        "page_role": "reverse_side_instructions",
                        "exclusion_reason": "back_side_reference_page",
                        "subset_tags": ["mixed_language", "format_variation"],
                    },
                    {
                        "case_id": "withholding_pdf_001",
                        "sample_path": str(pdf_path),
                        "document_type": "withholding_tax_form",
                        "page_role": "blank_form_template",
                        "exclusion_reason": "blank_form_without_filled_entities",
                        "subset_tags": ["format_variation", "pdf_render"],
                    },
                ],
                "cases": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audit = build_withholding_sample_page_audit(
        sample_root,
        manifest_path,
        sample_index={
            str(front_path): {
                "source": str(front_path),
                "document_type": "withholding_tax_form",
                "case_id": "withholding_maria_chen_001",
                "split": "train",
                "extractable": True,
                "page_role": "front",
            },
            str(back_path): {
                "source": str(back_path),
                "document_type": "withholding_tax_form",
                "case_id": "withholding_hana_payer_001",
                "split": "train",
                "extractable": False,
                "page_role": "reverse_side_instructions",
                "exclusion_reason": "back_side_reference_page",
            },
            str(pdf_path): {
                "source": str(pdf_path),
                "document_type": "withholding_tax_form",
                "case_id": "withholding_pdf_001",
                "split": "test",
                "extractable": False,
                "page_role": "blank_form_template",
                "exclusion_reason": "blank_form_without_filled_entities",
            },
        },
        pdf_text_loader=lambda _path: [
            "국내원천소득 제한세율 적용신청서 Last Name First Name Middle Name 접수번호 접수일자",
            "뒤쪽 작성방법 접수번호 및 접수일자",
        ],
    )

    by_path = {case["source_path"]: case for case in audit["cases"]}
    assert by_path[str(front_path)]["holdout_status"] == "eval_overlap"
    assert by_path[str(front_path)]["source_classification"] == "contains_target_page"
    assert by_path[str(front_path)]["holdout_usable"] is True
    assert by_path[str(front_path)]["evidence_strategy"] == "sample_dataset_metadata"

    assert by_path[str(back_path)]["holdout_status"] == "non_extractable"
    assert by_path[str(back_path)]["source_classification"] == "back_side_submission_only"
    assert by_path[str(back_path)]["holdout_usable"] is False
    assert (
        by_path[str(back_path)]["page_audits"][0]["classification"]
        == "back_side_payer_submission"
    )

    assert by_path[str(pdf_path)]["source_classification"] == "blank_front_plus_back_side_only"
    assert by_path[str(pdf_path)]["holdout_usable"] is False
    assert by_path[str(pdf_path)]["evidence_strategy"] == "pdf_text"
