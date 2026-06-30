from __future__ import annotations

import json
import sys
from pathlib import Path

from hanah_tax_ocr.schemas import DocumentType

from scripts.evals.run_eval_suite import (
    _default_ocr_lang,
    _discover_case_documents,
    _materialize_synthetic_source,
    _region_overrides_from_recognizer_root,
)
from scripts.training.export_recognizer_inference import (
    normalize_inference_dir,
    render_export_command,
)


def test_discover_case_documents_uses_labels_for_expected_cases(tmp_path: Path) -> None:
    expected_dir = tmp_path / "evals" / "cases" / "case_001"
    expected_dir.mkdir(parents=True)
    (expected_dir / "expected.json").write_text("{}", encoding="utf-8")

    label_dir = tmp_path / "data" / "labeled" / "residency_certificate" / "case_001"
    label_dir.mkdir(parents=True)
    sample_path = tmp_path / "sample_data" / "residency.png"
    sample_path.parent.mkdir(parents=True)
    sample_path.write_bytes(b"img")
    (label_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "case_001",
                "document_type": "residency_certificate",
                "source_path": str(sample_path),
            }
        ),
        encoding="utf-8",
    )

    cases = _discover_case_documents(expected_dir.parent, tmp_path / "data" / "labeled")

    assert sorted(cases) == ["case_001"]
    assert cases["case_001"][0].source_path == str(sample_path)
    assert cases["case_001"][0].document_type.value == "residency_certificate"
    assert cases["case_001"][0].ocr_lang == "en"


def test_discover_case_documents_skips_non_file_sources(tmp_path: Path) -> None:
    expected_dir = tmp_path / "evals" / "cases" / "case_001"
    expected_dir.mkdir(parents=True)
    (expected_dir / "expected.json").write_text("{}", encoding="utf-8")

    label_dir = tmp_path / "data" / "labeled" / "apostille" / "case_001"
    label_dir.mkdir(parents=True)
    (label_dir / "label.json").write_text(
        json.dumps(
            {
                "case_id": "case_001",
                "document_type": "apostille",
                "source_path": "synthetic://apostille/case_001",
            }
        ),
        encoding="utf-8",
    )

    cases = _discover_case_documents(expected_dir.parent, tmp_path / "data" / "labeled")

    assert cases == {}


def test_discover_case_documents_materializes_synthetic_sources(tmp_path: Path) -> None:
    expected_dir = tmp_path / "evals" / "cases" / "case_001"
    expected_dir.mkdir(parents=True)
    (expected_dir / "expected.json").write_text("{}", encoding="utf-8")

    label_dir = tmp_path / "data" / "labeled" / "residency_certificate" / "case_001"
    label_dir.mkdir(parents=True)
    payload = {
        "case_id": "case_001",
        "document_type": "residency_certificate",
        "source_path": "synthetic://residency_certificate/case_001",
        "expected_fields": {
            "taxpayer_name": "SAMPLE USER",
            "tin": "101-11-1001",
            "tax_year": "2021",
            "issue_date": "January 2, 2026",
            "residency_country": "United States of America",
            "residency_country_code": "US",
        },
    }
    (label_dir / "label.json").write_text(json.dumps(payload), encoding="utf-8")

    cases = _discover_case_documents(
        expected_dir.parent,
        tmp_path / "data" / "labeled",
        materialized_root=tmp_path / "materialized",
    )

    assert sorted(cases) == ["case_001"]
    materialized_path = Path(cases["case_001"][0].source_path)
    assert materialized_path.exists()
    assert materialized_path.suffix == ".png"
    assert cases["case_001"][0].ocr_lang == "en"


def test_default_ocr_lang_uses_korean_for_withholding() -> None:
    assert _default_ocr_lang(DocumentType.WITHHOLDING_TAX_FORM) == "korean"
    assert _default_ocr_lang(DocumentType.APOSTILLE) == "en"


def test_materialize_synthetic_source_renders_file(tmp_path: Path) -> None:
    payload = {
        "case_id": "apostille_regression_001",
        "document_type": "apostille",
        "source_path": "synthetic://apostille/apostille_regression_001",
        "expected_fields": {
            "issuing_country": "UNITED STATES OF AMERICA",
            "signed_by": "NOTARY SAMPLE 1",
            "signer_capacity": "NOTARY PUBLIC",
            "seal_owner": "COUNTY OF SAMPLE, STATE OF SAMPLE",
            "issued_at": "Capital City, Sample State 1",
            "issued_on": "2TH DAY OF APRIL, 2021",
            "issuing_authority": "Secretary of State State of Sample",
            "certificate_number": "5001",
        },
    }

    output_path = _materialize_synthetic_source(payload, tmp_path / "synthetic")

    assert output_path is not None
    assert Path(output_path).exists()


def test_region_overrides_from_recognizer_root_maps_field_names(tmp_path: Path) -> None:
    group_root = tmp_path / "recognizer" / "english_name_org"
    group_root.mkdir(parents=True)
    (group_root / "inference").mkdir()
    checkpoint_root = group_root / "model_output_pretrained"
    checkpoint_root.mkdir()
    (checkpoint_root / "best_accuracy.pdparams").write_text("weights", encoding="utf-8")
    (group_root / "dict.txt").write_text("A\n", encoding="utf-8")
    (group_root / "plan.json").write_text(
        json.dumps(
            {
                "field_group": "english_name_org",
                "field_names": ["taxpayer_name", "first_name"],
                "settings": {
                    "base_config": "configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml",
                    "dictionary_path": str(group_root / "dict.txt"),
                    "image_shape": "3,48,160",
                },
            }
        ),
        encoding="utf-8",
    )

    overrides = _region_overrides_from_recognizer_root(
        tmp_path / "recognizer",
        inference_subdir="inference",
        paddleocr_home=tmp_path / "PaddleOCR",
    )

    assert sorted(overrides) == ["first_name", "taxpayer_name"]
    assert overrides["taxpayer_name"]["lang"] == "en"
    assert overrides["taxpayer_name"]["ocr_version"] == "PP-OCRv3"
    assert overrides["taxpayer_name"]["rec_image_shape"] == "3,48,160"
    assert overrides["taxpayer_name"]["checkpoint_path"] == str(
        checkpoint_root / "best_accuracy"
    )
    assert overrides["taxpayer_name"]["rec_model_dir"] == str(group_root / "inference")
    assert overrides["taxpayer_name"]["rec_char_dict_path"] == str(
        (group_root / "dict.txt").resolve()
    )


def test_render_export_command_uses_plan_settings(tmp_path: Path) -> None:
    group_root = tmp_path / "recognizer" / "numeric_tin_code"
    group_root.mkdir(parents=True)
    plan_path = group_root / "plan.json"
    (group_root / "dict.txt").write_text("0\n", encoding="utf-8")
    plan_path.write_text(
        json.dumps(
            {
                "field_group": "numeric_tin_code",
                "settings": {
                    "base_config": "configs/rec/PP-OCRv3/en_PP-OCRv3_mobile_rec.yml",
                    "dictionary_path": str(group_root / "dict.txt"),
                },
            }
        ),
        encoding="utf-8",
    )

    command = render_export_command(
        plan_path,
        tmp_path / "PaddleOCR",
        checkpoint_subdir="model_output/best_accuracy",
        output_subdir="inference",
    )

    assert command.startswith(f"{sys.executable} ")
    assert "tools/export_model.py" in command
    assert "Global.use_gpu=False" in command
    assert "Global.pretrained_model=" in command
    assert "Global.save_inference_dir=" in command


def test_normalize_inference_dir_removes_pir_json_when_legacy_model_exists(tmp_path: Path) -> None:
    inference_dir = tmp_path / "inference"
    inference_dir.mkdir()
    (inference_dir / "inference.pdmodel").write_text("model", encoding="utf-8")
    (inference_dir / "inference.pdiparams").write_text("params", encoding="utf-8")
    json_path = inference_dir / "inference.json"
    json_path.write_text("{}", encoding="utf-8")

    normalize_inference_dir(inference_dir)

    assert not json_path.exists()
