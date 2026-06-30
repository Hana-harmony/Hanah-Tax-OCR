from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hanah_tax_ocr.harness import CaseDocument, HarnessRunner
from hanah_tax_ocr.ocr import PaddleOCREngine
from hanah_tax_ocr.schemas import DocumentType


def _is_ocr_source_available(source_path: str) -> bool:
    path = Path(source_path)
    return path.is_file()


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
            continue
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
    cases = _discover_case_documents(expected_root, labeled_root, case_ids=case_ids)
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
