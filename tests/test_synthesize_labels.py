import json
from pathlib import Path

from scripts.synthesize.generate_synthetic_labels import write_cases


def test_write_cases_creates_requested_count(tmp_path: Path) -> None:
    written = write_cases(tmp_path, count=3, seed=1)

    assert len(written) == 3
    first_payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert first_payload["expected_status"] == "pass"
    assert first_payload["fields"]["dividend_tax_rate"] == "15%"
