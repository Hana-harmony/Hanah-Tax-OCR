from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.training.date_synthetic import generate_synthetic_date_hard_cases
from PIL import Image


def test_generate_synthetic_date_hard_cases_appends_manifest_entries(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()

    entries = []
    for filename, field_name, text in (
        ("issue.png", "issue_date", "January 12, 2026"),
        ("signature.png", "signature_date", "2026-01-12"),
        ("apostille.png", "issued_on", "10TH DAY OF APRIL, 2014"),
    ):
        image_path = image_root / filename
        Image.new("RGB", (240, 72), "white").save(image_path)
        entries.append(
            {
                "case_id": filename,
                "document_type": "residency_certificate",
                "field_group": "date",
                "field_name": field_name,
                "text": text,
                "split": "train",
                "crop_path": str(image_path),
                "quality": {"accepted": True},
            }
        )

    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "hard_cases"
    summary = generate_synthetic_date_hard_cases(
        field_crops_root,
        output_root,
        variants_per_entry=2,
        seed=11,
    )

    assert summary["base_entry_count"] == 3
    assert summary["synthetic_entry_count"] == 6
    manifest_entries = [
        json.loads(line)
        for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(manifest_entries) == 6
    assert all(entry["augmentation_type"] == "synthetic_date" for entry in manifest_entries)
    assert all(Path(entry["crop_path"]).exists() for entry in manifest_entries)
    issue_entries = [
        entry for entry in manifest_entries if entry["field_name"] == "issue_date"
    ]
    assert issue_entries
    assert all(entry["recognizer_text"].startswith("Date: ") for entry in issue_entries)
    assert {entry["field_name"] for entry in manifest_entries} == {
        "issue_date",
        "signature_date",
        "issued_on",
    }
