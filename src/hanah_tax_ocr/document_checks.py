from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from hanah_tax_ocr.schemas import DocumentType


@dataclass(frozen=True)
class RegionRule:
    left: float
    top: float
    right: float
    bottom: float
    min_dark_ratio: float
    dark_threshold: int = 200


RESIDENCY_RULES = {
    "seal_present": RegionRule(0.02, 0.02, 0.17, 0.17, min_dark_ratio=0.06),
    "signature_present": RegionRule(0.51, 0.80, 0.71, 0.88, min_dark_ratio=0.02),
}

APOSTILLE_RULES = {
    "seal_present": RegionRule(0.06, 0.73, 0.38, 0.98, min_dark_ratio=0.08),
    "signature_present": RegionRule(0.57, 0.74, 0.90, 0.86, min_dark_ratio=0.02),
}

WITHHOLDING_RULES = {
    "signature_present": RegionRule(0.73, 0.77, 0.95, 0.84, min_dark_ratio=0.02),
}

WITHHOLDING_NO_CHECKBOX_RULES = [
    RegionRule(0.92, 0.49, 0.96, 0.52, min_dark_ratio=0.02),
    RegionRule(0.92, 0.52, 0.96, 0.55, min_dark_ratio=0.02),
    RegionRule(0.92, 0.55, 0.96, 0.58, min_dark_ratio=0.02),
    RegionRule(0.92, 0.58, 0.96, 0.61, min_dark_ratio=0.02),
    RegionRule(0.92, 0.61, 0.96, 0.64, min_dark_ratio=0.02),
    RegionRule(0.92, 0.64, 0.96, 0.67, min_dark_ratio=0.02),
    RegionRule(0.92, 0.67, 0.96, 0.70, min_dark_ratio=0.02),
]


def compute_document_checks(
    document_type: DocumentType,
    source_path: str | Path,
) -> dict[str, bool | int | None]:
    image_array = _load_grayscale_image(source_path)
    if image_array is None:
        return {
            "seal_present": None,
            "signature_present": None,
            "all_no_boxes_checked": None,
            "checked_no_box_count": None,
        }

    if document_type == DocumentType.RESIDENCY_CERTIFICATE:
        return _evaluate_region_map(image_array, RESIDENCY_RULES)

    if document_type == DocumentType.APOSTILLE:
        return _evaluate_region_map(image_array, APOSTILLE_RULES)

    if document_type == DocumentType.WITHHOLDING_TAX_FORM:
        checks = _evaluate_region_map(image_array, WITHHOLDING_RULES)
        checked_boxes = [
            _region_has_mark(image_array, rule) for rule in WITHHOLDING_NO_CHECKBOX_RULES
        ]
        checks["checked_no_box_count"] = sum(1 for flag in checked_boxes if flag)
        checks["all_no_boxes_checked"] = all(checked_boxes)
        return checks

    return {}


def _evaluate_region_map(
    image_array: np.ndarray,
    region_map: dict[str, RegionRule],
) -> dict[str, bool]:
    return {
        field_name: _region_has_mark(image_array, rule)
        for field_name, rule in region_map.items()
    }


def _region_has_mark(image_array: np.ndarray, rule: RegionRule) -> bool:
    cropped = _crop_region(image_array, rule)
    if cropped.size == 0:
        return False
    dark_ratio = float(np.mean(cropped < rule.dark_threshold))
    return dark_ratio >= rule.min_dark_ratio


def _crop_region(image_array: np.ndarray, rule: RegionRule) -> np.ndarray:
    height, width = image_array.shape
    left = max(0, min(width - 1, int(width * rule.left)))
    top = max(0, min(height - 1, int(height * rule.top)))
    right = max(left + 1, min(width, int(width * rule.right)))
    bottom = max(top + 1, min(height, int(height * rule.bottom)))
    return image_array[top:bottom, left:right]


def _load_grayscale_image(source_path: str | Path) -> np.ndarray | None:
    path = Path(source_path)
    if path.suffix.lower() == ".pdf":
        return None
    try:
        image = Image.open(path).convert("L")
    except OSError:
        return None
    return np.array(image, dtype=np.uint8)
