from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.evals.materialize_semi_real_probe_suite import materialize_probe_suite


def test_materialize_probe_suite_writes_assets_labels_and_expected(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    Image.new("RGB", (120, 40), "white").save(source_path)

    base_label_path = tmp_path / "data" / "labeled" / "withholding_tax_form" / "base" / "label.json"
    base_label_path.parent.mkdir(parents=True)
    base_label_path.write_text(
        json.dumps(
            {
                "case_id": "base",
                "document_type": "withholding_tax_form",
                "source_path": str(source_path),
                "expected_status": "pass",
                "expected_fields": {"address": "1 Main Street"},
                "expected_quality_checks": {"signature_present": True},
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "probes": [
                    {
                        "case_id": "probe_001",
                        "base_case_id": "base",
                        "document_type": "withholding_tax_form",
                        "base_label_path": str(base_label_path),
                        "augmentation_type": "left_clip",
                        "seed": 17,
                        "augmentation_options": {"anchor": "top"},
                        "focus_fields": ["address"],
                        "failure_modes": ["crop_miss"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = materialize_probe_suite(manifest_path, tmp_path / "suite", seed=7)

    assert summary["probe_count"] == 1
    asset_path = tmp_path / "suite" / "assets" / "probe_001.png"
    label_path = (
        tmp_path / "suite" / "labeled" / "withholding_tax_form" / "probe_001" / "label.json"
    )
    expected_path = tmp_path / "suite" / "cases" / "probe_001" / "expected.json"
    assert asset_path.exists()
    assert label_path.exists()
    assert expected_path.exists()

    label_payload = json.loads(label_path.read_text(encoding="utf-8"))
    expected_payload = json.loads(expected_path.read_text(encoding="utf-8"))
    assert label_payload["source_path"] == str(asset_path)
    assert label_payload["augmentation_type"] == "left_clip"
    assert label_payload["seed"] == 17
    assert label_payload["augmentation_options"] == {"anchor": "top"}
    assert label_payload["expected_fields"]["address"] == "1 Main Street"
    assert expected_payload["failure_modes"] == ["crop_miss"]
    assert expected_payload["seed"] == 17
    assert expected_payload["augmentation_options"] == {"anchor": "top"}
