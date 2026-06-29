import json
from pathlib import Path

from scripts.review_queue.promote_to_labeled import promote_review_queue


def test_promote_review_queue_creates_label_scaffolds(tmp_path: Path) -> None:
    review_queue_dir = tmp_path / "review_queue"
    review_queue_dir.mkdir()
    payload = {
        "case_id": "queue_001",
        "review_result": {
            "status": "reject",
            "findings": [{"code": "required_tin_invalid", "message": "bad tin"}],
        },
        "documents": [
            {
                "document_type": "withholding_tax_form",
                "source_path": "sample.png",
                "fields": {"tin": "bad"},
                "quality_checks": {"signature_present": False},
            }
        ],
    }
    (review_queue_dir / "queue_001.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    written = promote_review_queue(review_queue_dir, tmp_path / "labeled")

    assert len(written) == 1
    label_payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert label_payload["promotion_status"] == "needs_human_verification"
    assert label_payload["expected_fields"]["tin"] == "bad"
