from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from hanah_tax_ocr.schemas import OCRPage, OCRResult, OCRWordBox
from hanah_tax_ocr.template_profiles import OCRRegionSpec

REGION_FALLBACK_VERTICAL_OFFSETS: dict[str, tuple[float, ...]] = {
    "issue_date": (-0.08, -0.06, -0.04, -0.02),
}

MONTH_NAME_PATTERN = (
    r"\b(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b"
)


class _CheckpointRecognizer:
    def __init__(self, overrides: dict[str, Any]) -> None:
        self._paddleocr_home = Path(str(overrides["paddleocr_home"])).resolve()
        self._base_config = Path(str(overrides["base_config"])).resolve()
        self._checkpoint_path = Path(str(overrides["checkpoint_path"])).resolve()
        self._dict_path = Path(str(overrides["rec_char_dict_path"])).resolve()
        self._image_shape = str(overrides.get("rec_image_shape") or "3,48,320")
        self._max_text_length = int(overrides.get("max_text_length") or 32)
        self._model: Any = None
        self._ops: Any = None
        self._post_process: Any = None
        self._paddle: Any = None
        self._load()

    def _load(self) -> None:
        if str(self._paddleocr_home) not in sys.path:
            sys.path.insert(0, str(self._paddleocr_home))
        for module_name in list(sys.modules):
            if not (
                module_name == "ppocr"
                or module_name.startswith("ppocr.")
                or module_name == "tools"
                or module_name.startswith("tools.")
                or module_name == "ppstructure"
                or module_name.startswith("ppstructure.")
            ):
                continue
            module = sys.modules[module_name]
            module_file = getattr(module, "__file__", "") or ""
            if not str(module_file).startswith(str(self._paddleocr_home)):
                del sys.modules[module_name]

        import paddle
        from ppocr.data import create_operators, transform
        from ppocr.modeling.architectures import build_model
        from ppocr.postprocess import build_post_process
        from ppocr.utils.save_load import load_model
        from tools.program import load_config

        paddle.set_device("cpu")
        config = load_config(str(self._base_config))
        config["Global"]["use_gpu"] = False
        config["Global"]["pretrained_model"] = str(self._checkpoint_path)
        config["Global"]["character_dict_path"] = str(self._dict_path)
        config["Global"]["max_text_length"] = self._max_text_length
        config["Global"]["infer_img"] = self._image_shape

        post_process = build_post_process(config["PostProcess"], config["Global"])
        char_num = len(getattr(post_process, "character", []))
        if config["Architecture"]["Head"]["name"] == "MultiHead":
            out_channels_list = {
                "CTCLabelDecode": char_num,
                "SARLabelDecode": char_num + 2,
            }
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        elif char_num:
            config["Architecture"]["Head"]["out_channels"] = char_num

        model = build_model(config["Architecture"])
        load_model(config, model)
        model.eval()

        transforms: list[dict[str, Any]] = []
        for op in config["Eval"]["dataset"]["transforms"]:
            op_name = list(op)[0]
            if "Label" in op_name:
                continue
            if op_name == "RecResizeImg":
                op[op_name]["infer_mode"] = True
            elif op_name == "KeepKeys":
                op[op_name]["keep_keys"] = ["image"]
            transforms.append(op)

        config["Global"]["infer_mode"] = True
        self._ops = create_operators(transforms, config["Global"])
        self._post_process = post_process
        self._model = model
        self._paddle = paddle
        self._transform = transform

    def ocr(self, image: str | Path | np.ndarray, cls: bool = True) -> list[list[object]]:
        del cls
        if isinstance(image, str | Path):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        batch = self._transform({"image": buffer.getvalue()}, self._ops)
        images = np.expand_dims(batch[0], axis=0)
        with self._paddle.no_grad():
            preds = self._model(self._paddle.to_tensor(images))
        post_result = self._post_process(preds)

        text = ""
        confidence = 0.0
        if isinstance(post_result, list) and post_result:
            payload = post_result[0]
            if isinstance(payload, list | tuple) and len(payload) >= 2:
                text = str(payload[0]).strip()
                confidence = float(payload[1])

        width, height = pil_image.size
        box = [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
        return [[[box, (text, confidence)]]]


class PaddleOCREngine:
    """Thin wrapper around PaddleOCR with lazy import for testability."""

    def __init__(
        self,
        *,
        lang: str = "en",
        use_angle_cls: bool = True,
        show_log: bool = False,
        region_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self._kwargs = {
            "lang": lang,
            "use_angle_cls": use_angle_cls,
            "show_log": show_log,
            **kwargs,
        }
        self._region_overrides = region_overrides or {}
        self._engines: dict[tuple[tuple[str, str], ...], Any] = {}

    def _cache_key(self, overrides: dict[str, Any] | None = None) -> tuple[tuple[str, str], ...]:
        merged = {**self._kwargs, **(overrides or {})}
        return tuple(sorted((key, str(value)) for key, value in merged.items()))

    def _load(self, overrides: dict[str, Any] | None = None) -> Any:
        cache_key = self._cache_key(overrides)
        engine = self._engines.get(cache_key)
        if engine is None:
            if overrides and overrides.get("checkpoint_path"):
                engine = _CheckpointRecognizer(overrides)
            else:
                try:
                    from paddleocr import PaddleOCR
                except ImportError as exc:
                    raise RuntimeError(
                        "paddleocr is not installed. Install platform-specific paddlepaddle first, "
                        "then install this project with the [ocr] extra."
                    ) from exc
                engine = PaddleOCR(**{**self._kwargs, **(overrides or {})})
            self._engines[cache_key] = engine
        return engine

    def run(self, image_path: str | Path) -> OCRResult:
        engine = self._load()
        raw_result = engine.ocr(str(image_path), cls=self._kwargs.get("use_angle_cls", True))
        return OCRResult(pages=self._build_pages(raw_result))

    def run_regions(
        self,
        image_path: str | Path,
        region_specs: Iterable[OCRRegionSpec],
    ) -> dict[str, OCRPage]:
        path = Path(image_path)
        if path.suffix.lower() == ".pdf":
            return {}

        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            return {}

        regions: dict[str, OCRPage] = {}
        for region_spec in region_specs:
            region_engine = self._load(self._region_overrides.get(region_spec.name))
            pages = self._run_region_with_fallbacks(
                image,
                region_spec,
                region_engine,
            )
            if pages:
                regions[region_spec.name] = pages[0]
        return regions

    def _run_region_with_fallbacks(
        self,
        image: Image.Image,
        region_spec: OCRRegionSpec,
        region_engine: Any,
    ) -> list[OCRPage]:
        region_boxes = [region_spec]
        for vertical_offset in REGION_FALLBACK_VERTICAL_OFFSETS.get(region_spec.name, ()):
            shifted_top = max(0.0, region_spec.top + vertical_offset)
            shifted_bottom = min(1.0, region_spec.bottom + vertical_offset)
            if shifted_bottom <= shifted_top:
                continue
            region_boxes.append(
                OCRRegionSpec(
                    region_spec.name,
                    region_spec.left,
                    shifted_top,
                    region_spec.right,
                    shifted_bottom,
                )
            )

        for candidate_region in region_boxes:
            crop = image.crop(
                (
                    int(image.width * candidate_region.left),
                    int(image.height * candidate_region.top),
                    int(image.width * candidate_region.right),
                    int(image.height * candidate_region.bottom),
                )
            )
            best_pages: list[OCRPage] = []
            best_score = (0, 0, 0)
            for variant in self._region_variants(region_spec.name, crop):
                raw_result = region_engine.ocr(
                    np.array(variant),
                    cls=self._kwargs.get("use_angle_cls", True),
                )
                pages = self._build_pages(raw_result)
                score = self._score_region_pages(region_spec.name, pages)
                if score > best_score:
                    best_pages = pages
                    best_score = score
            if best_pages:
                return best_pages
        return []

    def _region_variants(self, region_name: str, crop: Image.Image) -> list[Image.Image]:
        variants = [crop.convert("RGB")]
        if region_name == "issue_date":
            variants.extend(
                [
                    crop.resize((crop.width * 2, crop.height * 2)).convert("RGB"),
                    ImageOps.grayscale(crop)
                    .point(lambda p: 255 if p > 180 else 0)
                    .resize((crop.width * 4, crop.height * 4))
                    .convert("RGB"),
                ]
            )
        elif region_name == "certificate_number":
            grayscale = ImageOps.grayscale(crop)
            variants = [
                grayscale.filter(ImageFilter.SHARPEN)
                .resize((crop.width * 4, crop.height * 4))
                .convert("RGB"),
                grayscale.point(lambda p: 255 if p > 180 else 0)
                .resize((crop.width * 4, crop.height * 4))
                .convert("RGB"),
                crop.resize((crop.width * 2, crop.height * 2)).convert("RGB"),
                crop.convert("RGB"),
            ]
        return variants

    def _score_region_pages(
        self,
        region_name: str,
        pages: list[OCRPage],
    ) -> tuple[int, int, int]:
        text = "\n".join(page.raw_text for page in pages if page.raw_text).strip()
        if not text:
            return (0, 0, 0)
        normalized = re.sub(r"\s+", " ", text).strip()

        if region_name == "certificate_number":
            for line in text.splitlines():
                normalized_line = re.sub(r"\s+", " ", line).strip()
                if "no" not in normalized_line.lower():
                    continue
                match = re.search(r"\bNo\.?\s*([A-Z0-9-]+)\b", normalized_line, re.IGNORECASE)
                if match and match.group(1) not in {"8", "9", "10"}:
                    return (4, len(match.group(1)), -len(normalized_line))
                return (2, len(normalized_line), -len(normalized_line))

            digits = [
                token
                for token in re.findall(r"\b\d+\b", normalized)
                if token not in {"8", "9", "10"}
            ]
            if digits:
                return (3, len(max(digits, key=len)), -len(normalized))
            return (1, len(normalized), -len(normalized))

        if region_name == "issue_date":
            has_year = re.search(r"\b\d{4}\b", normalized) is not None
            if re.search(MONTH_NAME_PATTERN, normalized, re.IGNORECASE) and has_year:
                return (4, len(normalized), -len(normalized))
            if "date" in normalized.lower() and has_year:
                return (3, len(normalized), -len(normalized))
            if has_year:
                return (2, len(normalized), -len(normalized))

        return (1, len(normalized), -len(normalized))

    def _build_pages(self, raw_result: Any) -> list[OCRPage]:
        pages: list[OCRPage] = []
        for page_number, page_lines in enumerate(raw_result or [], start=1):
            words: list[OCRWordBox] = []
            text_chunks: list[str] = []
            for line in page_lines or []:
                if len(line) < 2:
                    continue
                box, payload = line
                text = str(payload[0]).strip()
                confidence = float(payload[1]) if len(payload) > 1 else None
                if text:
                    text_chunks.append(text)
                    words.append(OCRWordBox(text=text, confidence=confidence, points=box))
            pages.append(
                OCRPage(
                    page_number=page_number,
                    words=words,
                    raw_text="\n".join(text_chunks),
                )
            )
        return pages
