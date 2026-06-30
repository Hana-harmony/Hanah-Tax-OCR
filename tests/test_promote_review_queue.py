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


def test_promote_review_queue_can_follow_priority_report_with_limit(tmp_path: Path) -> None:
    review_queue_dir = tmp_path / "review_queue"
    review_queue_dir.mkdir()
    for case_id, document_type in (
        ("queue_001", "withholding_tax_form"),
        ("queue_002", "apostille"),
        ("queue_003", "residency_certificate"),
    ):
        payload = {
            "case_id": case_id,
            "review_result": {"status": "reject", "findings": []},
            "documents": [
                {
                    "document_type": document_type,
                    "source_path": f"{case_id}.png",
                    "fields": {"tin": case_id},
                    "quality_checks": {},
                }
            ],
        }
        (review_queue_dir / f"{case_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    priority_report_path = tmp_path / "review_queue_priority.json"
    priority_report_path.write_text(
        json.dumps(
            {
                "priority_order": ["queue_003", "queue_001", "queue_002"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    written = promote_review_queue(
        review_queue_dir,
        tmp_path / "labeled",
        priority_report_path=priority_report_path,
        limit=2,
    )

    assert len(written) == 2
    assert written[0].as_posix().endswith("residency_certificate/queue_003/label.json")
    assert written[1].as_posix().endswith("withholding_tax_form/queue_001/label.json")
