from __future__ import annotations

import sys
import types
from pathlib import Path

import hanah_tax_ocr.ocr.engine as engine_module
from hanah_tax_ocr.ocr.engine import PaddleOCREngine
from hanah_tax_ocr.template_profiles import OCRRegionSpec
from PIL import Image


def test_paddle_ocr_engine_uses_region_overrides(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 40), "white").save(image_path)

    init_kwargs: list[dict[str, str]] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            init_kwargs.append(dict(kwargs))

        def ocr(self, _image: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            return [[([[0, 0], [1, 0], [1, 1], [0, 1]], ("OK", 0.99))]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(
            lang="en",
            region_overrides={"tin": {"rec_model_dir": "/tmp/custom-recognizer"}},
        )
        regions = engine.run_regions(
            image_path,
            [
                OCRRegionSpec("taxpayer_name", 0.0, 0.0, 0.5, 0.5),
                OCRRegionSpec("tin", 0.0, 0.5, 0.5, 1.0),
            ],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert sorted(regions) == ["taxpayer_name", "tin"]
    assert len(init_kwargs) == 2
    assert init_kwargs[0]["lang"] == "en"
    assert "rec_model_dir" not in init_kwargs[0]
    assert init_kwargs[1]["rec_model_dir"] == "/tmp/custom-recognizer"


def test_paddle_ocr_engine_uses_checkpoint_region_overrides(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 40), "white").save(image_path)

    class FakeCheckpointRecognizer:
        def __init__(self, overrides: dict[str, str]) -> None:
            self.overrides = dict(overrides)

        def ocr(self, _image: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            return [[([[0, 0], [1, 0], [1, 1], [0, 1]], ("CKPT", 0.88))]]

    original = engine_module._CheckpointRecognizer
    engine_module._CheckpointRecognizer = FakeCheckpointRecognizer
    try:
        engine = PaddleOCREngine(
            lang="en",
            region_overrides={
                "tin": {
                    "checkpoint_path": "/tmp/best_accuracy",
                    "base_config": "/tmp/config.yml",
                    "paddleocr_home": "/tmp/PaddleOCR",
                    "rec_char_dict_path": "/tmp/dict.txt",
                }
            },
        )
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("tin", 0.0, 0.0, 0.5, 0.5)],
        )
    finally:
        engine_module._CheckpointRecognizer = original

    assert regions["tin"].raw_text == "CKPT"
