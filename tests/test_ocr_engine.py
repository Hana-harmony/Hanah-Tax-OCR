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


def test_paddle_ocr_engine_retries_issue_date_region_with_upward_shift(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (100, 100), "white")
    for x in range(60, 90):
        for y in range(14, 20):
            image.putpixel((x, y), (0, 0, 0))
    image.save(image_path)

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            height = len(image_input)  # type: ignore[arg-type]
            dark_pixels = 0
            for row in image_input:  # type: ignore[assignment]
                for pixel in row:
                    if int(pixel[0]) < 64:
                        dark_pixels += 1
            if dark_pixels == 0:
                return [[]]
            return [[
                (
                    [[0, 0], [width, 0], [width, height], [0, height]],
                    ("Date: April 5, 2021", 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="en")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("issue_date", 0.60, 0.22, 0.90, 0.28)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert regions["issue_date"].raw_text == "Date: April 5, 2021"


def test_paddle_ocr_engine_retries_signed_by_region_with_upward_shift(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (100, 100), "white")
    for x in range(44, 66):
        for y in range(25, 31):
            image.putpixel((x, y), (0, 0, 0))
    image.save(image_path)

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            height = len(image_input)  # type: ignore[arg-type]
            dark_pixels = 0
            for row in image_input:  # type: ignore[assignment]
                for pixel in row:
                    if int(pixel[0]) < 64:
                        dark_pixels += 1
            if dark_pixels == 0:
                return [[]]
            return [[
                (
                    [[0, 0], [width, 0], [width, height], [0, height]],
                    ("NOTARY SAMPLE 11", 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="en")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("signed_by", 0.44, 0.38, 0.66, 0.43)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert regions["signed_by"].raw_text == "NOTARY SAMPLE 11"


def test_paddle_ocr_engine_retries_address_region_with_left_expansion(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (100, 100), "white")
    for x in range(10, 18):
        for y in range(22, 25):
            image.putpixel((x, y), (0, 0, 0))
    image.save(image_path)

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            height = len(image_input)  # type: ignore[arg-type]
            dark_pixels = 0
            for row in image_input:  # type: ignore[assignment]
                for pixel in row:
                    if int(pixel[0]) < 64:
                        dark_pixels += 1
            if dark_pixels == 0:
                return [[]]
            text = (
                "3 Main Street Suite 3 New York NY 10001 United States of America"
                if width >= 50
                else "Main Street Suite 3 New York NY 10001 United States of America"
            )
            return [[
                (
                    [[0, 0], [width, 0], [width, height], [0, height]],
                    (text, 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="korean")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("address", 0.16, 0.21, 0.86, 0.25)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert (
        regions["address"].raw_text
        == "3 Main Street Suite 3 New York NY 10001 United States of America"
    )


def test_paddle_ocr_engine_prefers_applicant_name_variant_with_three_clean_tokens(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            text = "SAMPLE2O\nT\nUSER" if width >= 60 else "SAMPLE2O\nUSER"
            return [[
                (
                    [[0, 0], [width, 0], [width, 1], [0, 1]],
                    (text, 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="korean")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("applicant_name", 0.40, 0.79, 0.72, 0.85)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert regions["applicant_name"].raw_text == "SAMPLE2O\nT\nUSER"


def test_paddle_ocr_engine_expands_applicant_name_region_and_uses_english_override(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    init_kwargs: list[dict[str, str]] = []

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            init_kwargs.append(dict(kwargs))
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            height = len(image_input)  # type: ignore[arg-type]
            if self.kwargs.get("lang") == "en" and width >= 120 and height >= 30:
                text = "SAMPLE9 I USER"
            else:
                text = "SAMPLE9 TUSER"
            return [[
                (
                    [[0, 0], [width, 0], [width, 1], [0, 1]],
                    (text, 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="korean")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("applicant_name", 0.40, 0.79, 0.72, 0.85)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert init_kwargs == [
        {"lang": "korean", "show_log": False, "use_angle_cls": True},
        {"lang": "en", "show_log": False, "use_angle_cls": True},
    ]
    assert regions["applicant_name"].raw_text == "SAMPLE9 I USER"


def test_paddle_ocr_engine_extracts_best_applicant_name_line_from_noisy_variant(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            if width >= 120:
                text = "NOISE\nMARIA L. CHEN\n1234"
            else:
                text = "MARIA\nCHEN"
            return [[
                (
                    [[0, 0], [width, 0], [width, 1], [0, 1]],
                    (text, 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="korean")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("applicant_name", 0.40, 0.79, 0.72, 0.85)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert regions["applicant_name"].raw_text == "MARIA L. CHEN"


def test_paddle_ocr_engine_prefers_thresholded_middle_name_variant(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (100, 100), "white")
    for x in range(69, 73):
        for y in range(17, 19):
            image.putpixel((x, y), (0, 0, 0))
    image.save(image_path)

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            unique_values = {
                int(pixel[0])  # type: ignore[index]
                for row in image_input  # type: ignore[assignment]
                for pixel in row
            }
            if width >= 30 and len(unique_values) <= 2:
                text = "L"
            elif width >= 30:
                text = "그"
            else:
                return [[]]
            return [[
                (
                    [[0, 0], [width, 0], [width, 1], [0, 1]],
                    (text, 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="korean")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("middle_name", 0.68, 0.16, 0.76, 0.20)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert regions["middle_name"].raw_text == "L"


def test_paddle_ocr_engine_prefers_iso_signature_date_variant(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            text = "2026년 01월 12일" if width >= 100 else "DATE"
            return [[
                (
                    [[0, 0], [width, 0], [width, 1], [0, 1]],
                    (text, 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="korean")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("signature_date", 0.70, 0.74, 0.96, 0.81)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert regions["signature_date"].raw_text == "2026년 01월 12일"


def test_paddle_ocr_engine_prefers_ordinal_issued_on_variant(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            text = "10TH DAY OF APRIL, 2014" if width >= 100 else "10TH"
            return [[
                (
                    [[0, 0], [width, 0], [width, 1], [0, 1]],
                    (text, 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="en")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("issued_on", 0.40, 0.61, 0.66, 0.67)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert regions["issued_on"].raw_text == "10TH DAY OF APRIL, 2014"


def test_paddle_ocr_engine_prefers_preprocessed_certificate_number_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = Image.new("RGB", (100, 100), "white")
    for x in range(7, 20):
        for y in range(65, 69):
            image.putpixel((x, y), (0, 0, 0))
    image.save(image_path)

    class FakePaddleOCR:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def ocr(self, image_input: object, cls: bool = True) -> list[list[object]]:
            assert cls is True
            width = len(image_input[0])  # type: ignore[index]
            height = len(image_input)  # type: ignore[arg-type]
            dark_pixels = 0
            for row in image_input:  # type: ignore[assignment]
                for pixel in row:
                    if int(pixel[0]) < 64:
                        dark_pixels += 1
            if dark_pixels == 0:
                return [[]]
            text = "8.No.4" if width >= 40 and height >= 12 else "8.No."
            return [[
                (
                    [[0, 0], [width, 0], [width, height], [0, height]],
                    (text, 0.99),
                )
            ]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    original = sys.modules.get("paddleocr")
    sys.modules["paddleocr"] = fake_module
    try:
        engine = PaddleOCREngine(lang="en")
        regions = engine.run_regions(
            image_path,
            [OCRRegionSpec("certificate_number", 0.07, 0.655, 0.20, 0.695)],
        )
    finally:
        if original is None:
            del sys.modules["paddleocr"]
        else:
            sys.modules["paddleocr"] = original

    assert regions["certificate_number"].raw_text == "8.No.4"
