import json
from pathlib import Path

from scripts.synthesize.build_regression_suite import generate_regression_suite


def test_generate_regression_suite_writes_per_document_cases(tmp_path: Path) -> None:
    written = generate_regression_suite(
        tmp_path / "labeled",
        tmp_path / "evals",
        per_document=2,
    )

    assert len(written) == 12
    label_path = (
        tmp_path
        / "labeled"
        / "residency_certificate"
        / "residency_regression_001"
        / "label.json"
    )
    eval_path = tmp_path / "evals" / "withholding_regression_002" / "expected.json"
    assert label_path.exists()
    assert eval_path.exists()
    label_payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert label_payload["synthetic"] is True
    assert label_payload["expected_status"] == "pass"
