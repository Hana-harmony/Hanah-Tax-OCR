import json
from pathlib import Path

from scripts.redact.mask_labels import redact_file


def test_redact_file_masks_ssn_and_ein(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    payload = {
        "tin": "123-45-6789",
        "company_ein": "12-3456789",
        "nested": {"notes": ["keep", "ssn 123-45-6789"]},
    }
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    redact_file(input_path, output_path)

    redacted = json.loads(output_path.read_text(encoding="utf-8"))
    assert redacted["tin"] == "***-**-****"
    assert redacted["company_ein"] == "**-*******"
    assert redacted["nested"]["notes"][1] == "ssn ***-**-****"
