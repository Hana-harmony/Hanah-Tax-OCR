from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw
from hanah_tax_ocr.harness import CaseDocument, HarnessRunner
from hanah_tax_ocr.schemas import (
    DocumentType,
    OCRPage,
    OCRResult,
    OCRWordBox,
    ReviewStatus,
)


def build_ocr_result(lines: list[tuple[str, float]]) -> OCRResult:
    return OCRResult(
        pages=[
            OCRPage(
                page_number=1,
                raw_text="\n".join(text for text, _ in lines),
                words=[
                    OCRWordBox(text=text, confidence=confidence, points=[])
                    for text, confidence in lines
                ],
            )
        ]
    )


def test_harness_queues_non_pass_review(tmp_path: Path) -> None:
    image_path = tmp_path / "residency.png"
    image = Image.new("RGB", (320, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Taxpayer: Jane Doe", fill="black")
    draw.text((10, 30), "TIN: 123-45-6789", fill="black")
    image.save(image_path)

    runner = HarnessRunner(review_queue_dir=tmp_path / "review_queue")
    result = runner.run_case(
        "case_001",
        [
            CaseDocument(
                document_type=DocumentType.RESIDENCY_CERTIFICATE,
                source_path=str(image_path),
                ocr_result=build_ocr_result(
                    [
                        ("Taxpayer: Jane Doe", 0.91),
                        ("TIN: 123-45-6789", 0.92),
                    ]
                ),
            )
        ],
    )

    assert result.review_result.status == ReviewStatus.REJECT
    assert result.queued_review_path is not None
    queued_payload = json.loads(Path(result.queued_review_path).read_text(encoding="utf-8"))
    assert queued_payload["case_id"] == "case_001"


def test_harness_writes_run_result(tmp_path: Path) -> None:
    image_path = tmp_path / "withholding.png"
    image = Image.new("RGB", (400, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "First Name Jane", fill="black")
    draw.text((10, 30), "Last Name Doe", fill="black")
    draw.text((10, 50), "Address 1 Main St", fill="black")
    image.save(image_path)

    runner = HarnessRunner(review_queue_dir=tmp_path / "review_queue")
    result = runner.run_case(
        "case_002",
        [
            CaseDocument(
                document_type=DocumentType.WITHHOLDING_TAX_FORM,
                source_path=str(image_path),
                ocr_result=build_ocr_result(
                    [
                        ("First Name Jane", 0.90),
                        ("Last Name Doe", 0.89),
                        ("Address 1 Main St", 0.88),
                        ("TIN 123-45-6789", 0.91),
                        ("Country United States of America", 0.92),
                        ("Country Code US", 0.92),
                        ("15%", 0.92),
                    ]
                ),
            )
        ],
    )
    output_path = tmp_path / "run_result.json"
    runner.write_run_result(result, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["case_id"] == "case_002"
    assert payload["review_result"]["status"] in {"reject", "needs_review", "pass"}
