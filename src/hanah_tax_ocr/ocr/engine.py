from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hanah_tax_ocr.schemas import OCRPage, OCRResult, OCRWordBox
from hanah_tax_ocr.template_profiles import OCRRegionSpec


class PaddleOCREngine:
    """Thin wrapper around PaddleOCR with lazy import for testability."""

    def __init__(
        self,
        *,
        lang: str = "en",
        use_angle_cls: bool = True,
        show_log: bool = False,
        **kwargs: Any,
    ) -> None:
        self._kwargs = {
            "lang": lang,
            "use_angle_cls": use_angle_cls,
            "show_log": show_log,
            **kwargs,
        }
        self._engine = None

    def _load(self) -> Any:
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError(
                    "paddleocr is not installed. Install platform-specific paddlepaddle first, "
                    "then install this project with the [ocr] extra."
                ) from exc
            self._engine = PaddleOCR(**self._kwargs)
        return self._engine

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

        engine = self._load()
        regions: dict[str, OCRPage] = {}
        for region_spec in region_specs:
            crop = image.crop(
                (
                    int(image.width * region_spec.left),
                    int(image.height * region_spec.top),
                    int(image.width * region_spec.right),
                    int(image.height * region_spec.bottom),
                )
            )
            raw_result = engine.ocr(
                np.array(crop),
                cls=self._kwargs.get("use_angle_cls", True),
            )
            pages = self._build_pages(raw_result)
            if pages:
                regions[region_spec.name] = pages[0]
        return regions

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
