from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hanah_tax_ocr.harness import CaseDocument, HarnessRunner
from hanah_tax_ocr.ocr import PaddleOCREngine
from hanah_tax_ocr.schemas import DocumentType
from PIL import Image, ImageDraw, ImageFont


def _is_ocr_source_available(source_path: str) -> bool:
    path = Path(source_path)
    return path.is_file()


def _is_synthetic_source(source_path: str) -> bool:
    return source_path.startswith("synthetic://")


def _safe_marker_filename(document_type: DocumentType, case_id: str) -> str:
    if document_type == DocumentType.RESIDENCY_CERTIFICATE:
        return f"{case_id}__거주자증명서.png"
    if document_type == DocumentType.WITHHOLDING_TAX_FORM:
        return f"{case_id}__국내원천소득.png"
    return f"{case_id}__north_carolina.png"


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text_box(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    box: tuple[float, float, float, float],
    text: str,
    *,
    font_size: int = 24,
) -> None:
    width, height = image_size
    left = int(width * box[0])
    top = int(height * box[1])
    font = _load_font(font_size)
    draw.multiline_text((left, top), text, fill=(0, 0, 0), font=font, spacing=4)


def _draw_signature_and_seal(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    *,
    seal_box: tuple[float, float, float, float],
    signature_box: tuple[float, float, float, float],
) -> None:
    width, height = image_size
    seal = (
        int(width * seal_box[0]),
        int(height * seal_box[1]),
        int(width * seal_box[2]),
        int(height * seal_box[3]),
    )
    signature = (
        int(width * signature_box[0]),
        int(height * signature_box[1]),
        int(width * signature_box[2]),
        int(height * signature_box[3]),
    )
    draw.ellipse(seal, outline=(40, 40, 40), width=4)
    center_x = (seal[0] + seal[2]) // 2
    center_y = (seal[1] + seal[3]) // 2
    draw.ellipse(
        (center_x - 18, center_y - 18, center_x + 18, center_y + 18),
        outline=(40, 40, 40),
        width=3,
    )
    step = max(18, (signature[2] - signature[0]) // 6)
    points = [
        (signature[0], signature[3] - 6),
        (signature[0] + step, signature[1] + 8),
        (signature[0] + step * 2, signature[3] - 10),
        (signature[0] + step * 3, signature[1] + 12),
        (signature[0] + step * 4, signature[3] - 8),
        (signature[2], signature[1] + 10),
    ]
    draw.line(points, fill=(20, 20, 20), width=4)


def _render_residency_document(payload: dict[str, Any], output_path: Path) -> Path:
    fields = payload["expected_fields"]
    image = Image.new("RGB", (850, 1100), "white")
    draw = ImageDraw.Draw(image)
    size = image.size
    heading_font = _load_font(28)
    subheading_font = _load_font(22)
    draw.text((270, 30), "DEPARTMENT OF THE TREASURY", fill=(0, 0, 0), font=heading_font)
    draw.text((290, 68), "INTERNAL REVENUE SERVICE", fill=(0, 0, 0), font=subheading_font)
    draw.text((305, 100), "PHILADELPHIA, PA 19255", fill=(0, 0, 0), font=subheading_font)

    _draw_text_box(
        draw,
        size,
        (0.60, 0.18, 0.90, 0.24),
        f"Date: {fields['issue_date']}",
        font_size=24,
    )
    _draw_text_box(
        draw,
        size,
        (0.10, 0.25, 0.36, 0.31),
        f"Taxpayer: {fields['taxpayer_name']}",
        font_size=24,
    )
    _draw_text_box(
        draw,
        size,
        (0.10, 0.27, 0.28, 0.33),
        f"TIN: {fields['tin']}",
        font_size=22,
    )
    _draw_text_box(
        draw,
        size,
        (0.10, 0.30, 0.25, 0.36),
        f"Tax Year: {fields['tax_year']}",
        font_size=22,
    )
    body = (
        "CERTIFICATION\n"
        "I certify that the above-named taxpayer is a resident of the "
        "United States of America for purposes of U.S. taxation."
    )
    _draw_text_box(draw, size, (0.10, 0.40, 0.84, 0.58), body, font_size=24)
    _draw_signature_and_seal(
        draw,
        size,
        seal_box=(0.02, 0.02, 0.17, 0.17),
        signature_box=(0.51, 0.80, 0.71, 0.88),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _render_withholding_document(payload: dict[str, Any], output_path: Path) -> Path:
    fields = payload["expected_fields"]
    image = Image.new("RGB", (1400, 1800), "white")
    draw = ImageDraw.Draw(image)
    size = image.size
    heading_font = _load_font(34)
    draw.text(
        (180, 60),
        "국내원천소득 제한세율 적용신청서  비거주자용",
        fill=(0, 0, 0),
        font=heading_font,
    )
    _draw_text_box(
        draw,
        size,
        (0.14, 0.16, 0.30, 0.20),
        f"USER\n{fields['last_name']}",
        font_size=26,
    )
    _draw_text_box(draw, size, (0.40, 0.16, 0.54, 0.20), fields["first_name"], font_size=26)
    _draw_text_box(draw, size, (0.68, 0.16, 0.76, 0.20), fields["middle_name"], font_size=26)
    _draw_text_box(draw, size, (0.16, 0.21, 0.86, 0.25), fields["address"], font_size=24)
    _draw_text_box(draw, size, (0.14, 0.27, 0.31, 0.33), fields["tin"], font_size=24)
    _draw_text_box(draw, size, (0.56, 0.27, 0.83, 0.33), fields["residency_country"], font_size=24)
    _draw_text_box(
        draw,
        size,
        (0.86, 0.27, 0.98, 0.33),
        fields["residency_country_code"],
        font_size=24,
    )
    _draw_text_box(draw, size, (0.72, 0.39, 0.90, 0.45), fields["dividend_tax_rate"], font_size=28)
    _draw_text_box(draw, size, (0.70, 0.74, 0.96, 0.81), fields["signature_date"], font_size=28)
    _draw_text_box(draw, size, (0.40, 0.79, 0.72, 0.85), fields["applicant_name"], font_size=28)
    _draw_signature_and_seal(
        draw,
        size,
        seal_box=(0.06, 0.05, 0.15, 0.14),
        signature_box=(0.73, 0.77, 0.95, 0.84),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _render_apostille_document(payload: dict[str, Any], output_path: Path) -> Path:
    fields = payload["expected_fields"]
    image = Image.new("RGB", (1200, 1500), "white")
    draw = ImageDraw.Draw(image)
    size = image.size
    heading_font = _load_font(36)
    body_font = _load_font(24)
    draw.text((390, 70), "APOSTILLE", fill=(0, 0, 0), font=heading_font)
    draw.text((250, 120), "Convention de La Haye du 5 octobre 1961", fill=(0, 0, 0), font=body_font)

    lines = [
        f"1. Country: {fields['issuing_country']}",
        f"2. This Public Document has been signed by {fields['signed_by']}",
        f"3. acting in the capacity of {fields['signer_capacity']}",
        f"4. bears the seal/stamp of {fields['seal_owner']}",
        f"5. at {fields['issued_at']}",
        f"6. the {fields['issued_on']}",
        "7. by Secretary of State or Deputy Secretary of State, State of North Carolina",
        f"8. No. {fields['certificate_number']}",
    ]
    y = 310
    for line in lines:
        draw.text((120, y), line, fill=(0, 0, 0), font=body_font)
        y += 78
    _draw_signature_and_seal(
        draw,
        size,
        seal_box=(0.08, 0.68, 0.34, 0.96),
        signature_box=(0.56, 0.70, 0.92, 0.92),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _materialize_synthetic_source(
    payload: dict[str, Any],
    materialized_root: Path,
) -> str | None:
    source_path = str(payload.get("source_path") or "")
    if not _is_synthetic_source(source_path):
        return None
    expected_fields = payload.get("expected_fields")
    document_type = payload.get("document_type")
    case_id = str(payload.get("case_id") or "")
    if not isinstance(expected_fields, dict) or not isinstance(document_type, str) or not case_id:
        return None
    doc_type = DocumentType(document_type)
    output_path = materialized_root / _safe_marker_filename(doc_type, case_id)
    if output_path.exists():
        return str(output_path)
    if doc_type == DocumentType.RESIDENCY_CERTIFICATE:
        return str(_render_residency_document(payload, output_path))
    if doc_type == DocumentType.WITHHOLDING_TAX_FORM:
        return str(_render_withholding_document(payload, output_path))
    return str(_render_apostille_document(payload, output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OCR harness across eval cases using optional fine-tuned recognizers."
    )
    parser.add_argument(
        "--expected-root",
        type=Path,
        default=Path("evals/cases"),
    )
    parser.add_argument(
        "--labeled-root",
        type=Path,
        default=Path("data/labeled"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--recognizer-root",
        type=Path,
        default=None,
        help="Optional recognizer root containing fine-tuned per-field-group inference dirs.",
    )
    parser.add_argument(
        "--inference-subdir",
        default="inference",
        help="Subdirectory under each recognizer group containing the exported inference model.",
    )
    parser.add_argument(
        "--paddleocr-home",
        type=Path,
        default=Path("PaddleOCR"),
        help="Local PaddleOCR checkout used for checkpoint-based candidate recognizers.",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--lang", default="en")
    return parser.parse_args()


def _discover_case_documents(
    expected_root: Path,
    labeled_root: Path,
    *,
    materialized_root: Path | None = None,
    case_ids: set[str] | None = None,
) -> dict[str, list[CaseDocument]]:
    label_index: dict[str, list[dict[str, Any]]] = {}
    for label_path in sorted(labeled_root.rglob("label.json")):
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        case_id = str(payload.get("case_id") or label_path.parent.name)
        if case_ids and case_id not in case_ids:
            continue
        source_path = payload.get("source_path")
        document_type = payload.get("document_type")
        if not isinstance(source_path, str) or not isinstance(document_type, str):
            continue
        if not _is_ocr_source_available(source_path):
            if materialized_root is not None and _is_synthetic_source(source_path):
                synthetic_source_path = _materialize_synthetic_source(payload, materialized_root)
                if synthetic_source_path is not None:
                    source_path = synthetic_source_path
            if not _is_ocr_source_available(source_path):
                continue
        if not _is_ocr_source_available(source_path):
            continue
        payload = dict(payload)
        payload["source_path"] = source_path
        label_index.setdefault(case_id, []).append(payload)

    cases: dict[str, list[CaseDocument]] = {}
    for expected_path in sorted(expected_root.glob("*/expected.json")):
        case_id = expected_path.parent.name
        if case_ids and case_id not in case_ids:
            continue
        payloads = label_index.get(case_id, [])
        documents = [
            CaseDocument(
                document_type=DocumentType(str(payload["document_type"])),
                source_path=str(payload["source_path"]),
            )
            for payload in payloads
        ]
        if documents:
            documents.sort(key=lambda item: item.document_type.value)
            cases[case_id] = documents
    return cases


def _region_overrides_from_recognizer_root(
    recognizer_root: Path | None,
    *,
    inference_subdir: str,
    paddleocr_home: Path,
) -> dict[str, dict[str, Any]]:
    if recognizer_root is None:
        return {}

    overrides: dict[str, dict[str, Any]] = {}
    for plan_path in sorted(recognizer_root.glob("*/plan.json")):
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        field_group = str(plan.get("field_group") or plan_path.parent.name)
        field_names = [str(field_name) for field_name in plan.get("field_names", [])]
        settings = plan.get("settings", {})
        model_dir = recognizer_root / field_group / inference_subdir
        checkpoint_roots = [
            recognizer_root / field_group / "model_output_pretrained" / "best_accuracy",
            recognizer_root / field_group / "model_output" / "best_accuracy",
        ]
        checkpoint_path = next(
            (
                path
                for path in checkpoint_roots
                if path.with_suffix(".pdparams").exists()
            ),
            None,
        )

        if not model_dir.exists() and checkpoint_path is None:
            continue

        base_config = str(settings.get("base_config") or "")
        language = "korean" if "korean" in base_config.lower() else "en"
        dictionary_path = settings.get("dictionary_path")
        override = {
            "lang": language,
        }
        if model_dir.exists():
            override["rec_model_dir"] = str(model_dir)
        image_shape = settings.get("image_shape")
        if image_shape:
            override["rec_image_shape"] = str(image_shape)
        max_text_length = settings.get("max_text_length")
        if max_text_length:
            override["max_text_length"] = int(max_text_length)
        if "pp-ocrv3" in base_config.lower():
            override["ocr_version"] = "PP-OCRv3"
        elif "pp-ocrv4" in base_config.lower():
            override["ocr_version"] = "PP-OCRv4"
        if dictionary_path:
            override["rec_char_dict_path"] = str(Path(dictionary_path).resolve())
        if checkpoint_path is not None:
            override["checkpoint_path"] = str(checkpoint_path)
            override["base_config"] = str((paddleocr_home / str(settings["base_config"])).resolve())
            override["paddleocr_home"] = str(paddleocr_home.resolve())

        for field_name in field_names:
            overrides[field_name] = dict(override)
    return overrides


def run_eval_suite(
    expected_root: Path,
    labeled_root: Path,
    output_dir: Path,
    *,
    recognizer_root: Path | None = None,
    inference_subdir: str = "inference",
    paddleocr_home: Path = Path("PaddleOCR"),
    case_ids: set[str] | None = None,
    lang: str = "en",
) -> dict[str, Any]:
    materialized_root = output_dir / "_synthetic_inputs"
    cases = _discover_case_documents(
        expected_root,
        labeled_root,
        materialized_root=materialized_root,
        case_ids=case_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    review_queue_dir = output_dir / "review_queue"
    region_overrides = _region_overrides_from_recognizer_root(
        recognizer_root,
        inference_subdir=inference_subdir,
        paddleocr_home=paddleocr_home,
    )
    engine = PaddleOCREngine(lang=lang, region_overrides=region_overrides)
    runner = HarnessRunner(review_queue_dir=review_queue_dir, ocr_engine=engine)

    written_case_ids: list[str] = []
    missing_case_ids: list[str] = []
    for expected_path in sorted(expected_root.glob("*/expected.json")):
        case_id = expected_path.parent.name
        if case_ids and case_id not in case_ids:
            continue
        documents = cases.get(case_id)
        if not documents:
            missing_case_ids.append(case_id)
            continue
        run_result = runner.run_case(case_id, documents)
        runner.write_run_result(run_result, output_dir / f"{case_id}.json")
        written_case_ids.append(case_id)

    return {
        "expected_root": str(expected_root),
        "labeled_root": str(labeled_root),
        "output_dir": str(output_dir),
        "recognizer_root": None if recognizer_root is None else str(recognizer_root),
        "inference_subdir": inference_subdir,
        "written_case_ids": written_case_ids,
        "missing_case_ids": missing_case_ids,
        "region_override_count": len(region_overrides),
    }


def main() -> None:
    args = parse_args()
    summary = run_eval_suite(
        args.expected_root,
        args.labeled_root,
        args.output_dir,
        recognizer_root=args.recognizer_root,
        inference_subdir=args.inference_subdir,
        paddleocr_home=args.paddleocr_home,
        case_ids=set(args.case_id) or None,
        lang=args.lang,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
