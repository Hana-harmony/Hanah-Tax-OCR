from __future__ import annotations

import unicodedata
from pathlib import Path

SAMPLE_DATASET = [
    {
        "source": "sample_data/거주자증명서/4.jpg",
        "split": "train",
        "document_type": "residency_certificate",
        "case_id": "residency_legacy_001",
        "extractable": True,
        "page_role": "front",
    },
    {
        "source": "sample_data/거주자증명서/5.jpg",
        "split": "train",
        "document_type": "residency_certificate",
        "case_id": "residency_john_doe_001",
        "extractable": True,
        "page_role": "front",
    },
    {
        "source": "sample_data/거주자증명서/6.jpg",
        "split": "val",
        "document_type": "residency_certificate",
        "case_id": "residency_university_hawaii_001",
        "extractable": True,
        "page_role": "front",
    },
    {
        "source": "sample_data/거주자증명서/미국 TREASURY주.png",
        "split": "test",
        "document_type": "residency_certificate",
        "case_id": "residency_maria_chen_001",
        "extractable": True,
        "page_role": "front",
    },
    {
        "source": "sample_data/거주자증명서/2.pdf",
        "split": "test",
        "document_type": "residency_certificate",
        "case_id": "residency_pdf_001",
        "extractable": True,
        "page_role": "front_pdf",
        "preferred_ocr_source": "sample_data/거주자증명서/2_page1.png",
        "source_aliases": ["sample_data/거주자증명서/2_page1.png"],
    },
    {
        "source": "sample_data/국내원천소득 제한세율/국내원천소득 제한세율 적용신청서-1.png",
        "split": "test",
        "document_type": "withholding_tax_form",
        "case_id": "withholding_maria_chen_001",
        "extractable": True,
        "page_role": "front",
    },
    {
        "source": "sample_data/국내원천소득 제한세율/국내원천소득 제한세율 적용신청서-2.png",
        "split": "train",
        "document_type": "withholding_tax_form",
        "case_id": "withholding_hana_payer_001",
        "extractable": False,
        "page_role": "reverse_side_instructions",
        "exclusion_reason": "back_side_reference_page",
    },
    {
        "source": "sample_data/국내원천소득 제한세율/원본 샘플.pdf",
        "split": "test",
        "document_type": "withholding_tax_form",
        "case_id": "withholding_pdf_001",
        "extractable": False,
        "page_role": "blank_form_template",
        "exclusion_reason": "blank_form_without_filled_entities",
    },
    {
        "source": "sample_data/아포스티유 샘플/미국 california 주.png",
        "split": "train",
        "document_type": "apostille",
        "case_id": "apostille_california_001",
        "extractable": True,
        "page_role": "front",
    },
    {
        "source": "sample_data/아포스티유 샘플/미국 michigan 주.jpg",
        "split": "val",
        "document_type": "apostille",
        "case_id": "apostille_michigan_001",
        "extractable": True,
        "page_role": "front",
    },
    {
        "source": "sample_data/아포스티유 샘플/미국 california 주2.jpg",
        "split": "test",
        "document_type": "apostille",
        "case_id": "apostille_north_carolina_001",
        "extractable": True,
        "page_role": "front",
    },
]


def normalize_path_text(path: str | Path) -> str:
    return unicodedata.normalize("NFC", str(path))


def build_sample_index(sample_root: Path | None = None) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for entry in SAMPLE_DATASET:
        source = Path(entry["source"])
        variants = {normalize_path_text(source)}
        if sample_root is not None and not source.is_absolute():
            variants.add(normalize_path_text(sample_root.parent / source))
        for alias in entry.get("source_aliases", []):
            alias_path = Path(str(alias))
            variants.add(normalize_path_text(alias_path))
            if sample_root is not None and not alias_path.is_absolute():
                variants.add(normalize_path_text(sample_root.parent / alias_path))
        record = {
            **entry,
            "source_variants": sorted(variants),
        }
        for variant in variants:
            index[variant] = record
    return index
