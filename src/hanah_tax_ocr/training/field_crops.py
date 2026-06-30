from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from hanah_tax_ocr.schemas import DocumentType
from hanah_tax_ocr.template_profiles import classify_template

MIN_BASE_TRAIN_COUNT = 12
MIN_BASE_VAL_COUNT = 3
MIN_TRAIN_SOURCE_COUNT = 3
MIN_VAL_SOURCE_COUNT = 2
TRAIN_COUNT_GAP_WEIGHT = 3.0
VAL_COUNT_GAP_WEIGHT = 2.0
TRAIN_SOURCE_GAP_WEIGHT = 2.5
VAL_SOURCE_GAP_WEIGHT = 2.0
EXACT_SOURCE_SPLIT_SEARCH_LIMIT = 14

FIELD_GROUPS: dict[str, str] = {
    "taxpayer_name": "english_name_org",
    "first_name": "english_name_org",
    "middle_name": "english_name_org",
    "last_name": "english_name_org",
    "applicant_name": "english_name_org",
    "signed_by": "english_name_org",
    "signer_capacity": "english_name_org",
    "seal_owner": "english_name_org",
    "issued_at": "english_name_org",
    "issuing_authority": "english_name_org",
    "residency_country": "english_name_org",
    "address": "english_name_org",
    "tin": "numeric_tin_code",
    "residency_country_code": "numeric_tin_code",
    "certificate_number": "numeric_tin_code",
    "tax_year": "numeric_tin_code",
    "dividend_tax_rate": "numeric_tin_code",
    "issue_date": "date",
    "signature_date": "date",
    "issued_on": "date",
}

ISSUE_DATE_VERTICAL_FALLBACK_OFFSETS = (-0.08, -0.06, -0.04, -0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export field-level crop datasets from reviewed labels."
    )
    parser.add_argument(
        "--labeled-root",
        type=Path,
        default=Path("data/labeled"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/training/field_crops"),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
    )
    parser.add_argument("--min-width", type=int, default=24)
    parser.add_argument("--min-height", type=int, default=16)
    parser.add_argument("--min-dark-ratio", type=float, default=0.01)
    parser.add_argument("--min-contrast", type=float, default=8.0)
    parser.add_argument("--max-foreground-bbox-ratio", type=float, default=0.85)
    parser.add_argument("--dense-edge-dark-ratio", type=float, default=0.45)
    parser.add_argument("--max-edge-trim-ratio", type=float, default=0.18)
    return parser.parse_args()


def field_group_for(field_name: str) -> str:
    return FIELD_GROUPS.get(field_name, "korean_mixed_form")


def split_score_for_text(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_for_group(group_key: str, val_ratio: float) -> str:
    return "val" if split_score_for_text(group_key) < val_ratio else "train"


def _ensure_split_coverage(
    group_assignments: dict[str, str],
    source_paths: list[str],
) -> None:
    unique_source_paths = sorted(
        set(source_paths),
        key=lambda source_path: (split_score_for_text(source_path), source_path),
    )
    if len(unique_source_paths) < 2:
        return

    if not any(group_assignments[source_path] == "val" for source_path in unique_source_paths):
        group_assignments[unique_source_paths[0]] = "val"
    if not any(group_assignments[source_path] == "train" for source_path in unique_source_paths):
        group_assignments[unique_source_paths[-1]] = "train"


def _field_count_by_group_for_case_document(item: dict[str, Any]) -> dict[str, int]:
    raw_counts = item.get("field_counts_by_group")
    if isinstance(raw_counts, dict):
        counts: Counter[str] = Counter()
        for field_group, count in raw_counts.items():
            if not isinstance(field_group, str) or not field_group:
                continue
            if not isinstance(count, int) or count <= 0:
                continue
            counts[field_group] += count
        if counts:
            return dict(counts)

    counts = Counter(_field_groups_for_case_document(item))
    return dict(counts)


def _field_groups_for_case_document(item: dict[str, Any]) -> list[str]:
    raw_groups = item.get("field_groups")
    if isinstance(raw_groups, str):
        return [raw_groups]
    if isinstance(raw_groups, list):
        return [
            str(group)
            for group in raw_groups
            if isinstance(group, str) and group
        ]
    raw_group = item.get("field_group")
    if isinstance(raw_group, str) and raw_group:
        return [raw_group]
    return []


def _default_case_splits(
    case_documents: list[dict[str, Any]],
    *,
    val_ratio: float,
) -> dict[str, str]:
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in case_documents:
        source_groups[item["source_path"]].append(item)

    group_assignments = {
        source_path: split_for_group(source_path, val_ratio)
        for source_path in source_groups
    }
    if 0.0 < val_ratio < 1.0:
        by_document_type: dict[str, list[str]] = defaultdict(list)
        for item in case_documents:
            by_document_type[item["document_type"]].append(item["source_path"])

        for source_paths in by_document_type.values():
            _ensure_split_coverage(group_assignments, source_paths)

        by_field_group: dict[str, list[str]] = defaultdict(list)
        for item in case_documents:
            for field_group in _field_groups_for_case_document(item):
                by_field_group[field_group].append(item["source_path"])

        for source_paths in by_field_group.values():
            _ensure_split_coverage(group_assignments, source_paths)

    return {
        str(item.get("split_key") or item["case_id"]): group_assignments[item["source_path"]]
        for item in case_documents
    }


def _score_group_split_stats(
    *,
    train_count: int,
    val_count: int,
    train_source_count: int,
    val_source_count: int,
) -> float:
    return (
        max(0, MIN_BASE_TRAIN_COUNT - train_count) * TRAIN_COUNT_GAP_WEIGHT
        + max(0, MIN_BASE_VAL_COUNT - val_count) * VAL_COUNT_GAP_WEIGHT
        + max(0, MIN_TRAIN_SOURCE_COUNT - train_source_count) * TRAIN_SOURCE_GAP_WEIGHT
        + max(0, MIN_VAL_SOURCE_COUNT - val_source_count) * VAL_SOURCE_GAP_WEIGHT
    )


def _assignment_distance(
    left: dict[str, str],
    right: dict[str, str],
) -> int:
    return sum(1 for source_path, split in left.items() if right.get(source_path) != split)


def _score_source_split_assignment(
    source_assignments: dict[str, str],
    source_profiles: dict[str, dict[str, Any]],
) -> float:
    counts_by_group: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "train_count": 0,
            "val_count": 0,
            "train_sources": set(),
            "val_sources": set(),
        }
    )

    for source_path, profile in source_profiles.items():
        split = source_assignments[source_path]
        for field_group, count in profile["field_counts_by_group"].items():
            if split == "val":
                counts_by_group[field_group]["val_count"] += count
                counts_by_group[field_group]["val_sources"].add(source_path)
            else:
                counts_by_group[field_group]["train_count"] += count
                counts_by_group[field_group]["train_sources"].add(source_path)

    return round(
        sum(
            _score_group_split_stats(
                train_count=stats["train_count"],
                val_count=stats["val_count"],
                train_source_count=len(stats["train_sources"]),
                val_source_count=len(stats["val_sources"]),
            )
            for stats in counts_by_group.values()
        ),
        4,
    )


def _has_required_split_coverage(
    source_assignments: dict[str, str],
    required_sources_by_key: dict[str, set[str]],
) -> bool:
    for source_paths in required_sources_by_key.values():
        if len(source_paths) < 2:
            continue
        if len({source_assignments[source_path] for source_path in source_paths}) < 2:
            return False
    return True


def assign_case_splits(
    case_documents: list[dict[str, Any]],
    *,
    val_ratio: float,
) -> dict[str, str]:
    if not 0.0 < val_ratio < 1.0:
        return _default_case_splits(case_documents, val_ratio=val_ratio)

    default_assignments = _default_case_splits(case_documents, val_ratio=val_ratio)
    source_assignments = {
        item["source_path"]: default_assignments[str(item.get("split_key") or item["case_id"])]
        for item in case_documents
    }
    if len(source_assignments) > EXACT_SOURCE_SPLIT_SEARCH_LIMIT:
        return default_assignments

    source_profiles: dict[str, dict[str, Any]] = {}
    sources_by_document_type: dict[str, set[str]] = defaultdict(set)
    sources_by_field_group: dict[str, set[str]] = defaultdict(set)
    for item in case_documents:
        source_path = item["source_path"]
        document_type = str(item["document_type"])
        field_counts_by_group = _field_count_by_group_for_case_document(item)

        profile = source_profiles.setdefault(
            source_path,
            {
                "field_counts_by_group": Counter(),
            },
        )
        profile["field_counts_by_group"].update(field_counts_by_group)
        sources_by_document_type[document_type].add(source_path)
        for field_group in field_counts_by_group:
            sources_by_field_group[field_group].add(source_path)

    required_sources_by_key = {
        **{
            f"document_type:{document_type}": source_paths
            for document_type, source_paths in sources_by_document_type.items()
        },
        **{
            f"field_group:{field_group}": source_paths
            for field_group, source_paths in sources_by_field_group.items()
        },
    }
    sorted_source_paths = sorted(source_profiles)
    best_source_assignments = dict(source_assignments)
    best_score = _score_source_split_assignment(best_source_assignments, source_profiles)
    best_distance = 0

    for mask in range(1 << len(sorted_source_paths)):
        candidate_assignments = {
            source_path: ("val" if (mask & (1 << index)) else "train")
            for index, source_path in enumerate(sorted_source_paths)
        }
        if not _has_required_split_coverage(candidate_assignments, required_sources_by_key):
            continue

        candidate_score = _score_source_split_assignment(candidate_assignments, source_profiles)
        candidate_distance = _assignment_distance(candidate_assignments, source_assignments)
        if (
            candidate_score < best_score
            or (candidate_score == best_score and candidate_distance < best_distance)
            or (
                candidate_score == best_score
                and candidate_distance == best_distance
                and tuple(candidate_assignments[source_path] for source_path in sorted_source_paths)
                < tuple(
                    best_source_assignments[source_path]
                    for source_path in sorted_source_paths
                )
            )
        ):
            best_source_assignments = candidate_assignments
            best_score = candidate_score
            best_distance = candidate_distance

    return {
        str(item.get("split_key") or item["case_id"]): best_source_assignments[item["source_path"]]
        for item in case_documents
    }


def discover_label_paths(labeled_root: Path) -> list[Path]:
    return sorted(labeled_root.rglob("label.json"))


def load_label(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_supported_source(source_path: str) -> bool:
    return "://" not in source_path


def _box_from_region(
    image: Image.Image,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[int, int, int, int] | None:
    box = (
        int(image.width * left),
        int(image.height * top),
        int(image.width * right),
        int(image.height * bottom),
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box


def compute_crop_quality(
    crop: Image.Image,
    *,
    min_width: int,
    min_height: int,
    min_dark_ratio: float,
    min_contrast: float,
    max_foreground_bbox_ratio: float,
    dense_edge_dark_ratio: float,
) -> dict[str, Any]:
    grayscale = crop.convert("L")
    histogram = grayscale.histogram()
    total_pixels = max(1, crop.width * crop.height)
    dark_pixels = sum(count for index, count in enumerate(histogram) if index < 200)
    dark_ratio = dark_pixels / total_pixels
    stat = ImageStat.Stat(grayscale)
    contrast = float(stat.stddev[0]) if stat.stddev else 0.0
    foreground_points = [
        (x, y)
        for y in range(crop.height)
        for x in range(crop.width)
        if grayscale.getpixel((x, y)) < 200
    ]
    if foreground_points:
        x_values = [point[0] for point in foreground_points]
        y_values = [point[1] for point in foreground_points]
        foreground_bbox = {
            "left": min(x_values),
            "top": min(y_values),
            "right": max(x_values),
            "bottom": max(y_values),
        }
        foreground_bbox_ratio = (
            (foreground_bbox["right"] - foreground_bbox["left"] + 1)
            * (foreground_bbox["bottom"] - foreground_bbox["top"] + 1)
        ) / total_pixels
        touched_edges = [
            edge_name
            for edge_name, touched in {
                "left": foreground_bbox["left"] == 0,
                "right": foreground_bbox["right"] == crop.width - 1,
                "top": foreground_bbox["top"] == 0,
                "bottom": foreground_bbox["bottom"] == crop.height - 1,
            }.items()
            if touched
        ]
    else:
        foreground_bbox = None
        foreground_bbox_ratio = 0.0
        touched_edges = []

    border_band = max(1, min(crop.width, crop.height) // 12)

    def edge_dark_ratio(coords: list[tuple[int, int]]) -> float:
        return (
            sum(1 for x, y in coords if grayscale.getpixel((x, y)) < 200)
            / max(1, len(coords))
        )

    edge_dark_ratios = {
        "left": edge_dark_ratio(
            [(x, y) for x in range(border_band) for y in range(crop.height)]
        ),
        "right": edge_dark_ratio(
            [
                (x, y)
                for x in range(crop.width - border_band, crop.width)
                for y in range(crop.height)
            ]
        ),
        "top": edge_dark_ratio(
            [(x, y) for x in range(crop.width) for y in range(border_band)]
        ),
        "bottom": edge_dark_ratio(
            [
                (x, y)
                for x in range(crop.width)
                for y in range(crop.height - border_band, crop.height)
            ]
        ),
    }
    quality_flags: list[str] = []
    if crop.width < min_width:
        quality_flags.append("too_narrow")
    if crop.height < min_height:
        quality_flags.append("too_short")
    if dark_ratio < min_dark_ratio:
        quality_flags.append("low_dark_ratio")
    if contrast < min_contrast:
        quality_flags.append("low_contrast")
    dense_edge_names = [
        edge_name
        for edge_name, ratio in edge_dark_ratios.items()
        if ratio >= dense_edge_dark_ratio
    ]
    if foreground_bbox_ratio > max_foreground_bbox_ratio and dark_ratio >= 0.35:
        quality_flags.append("foreground_fills_crop")
    if len(dense_edge_names) >= 3:
        quality_flags.append("dense_edge_content")

    return {
        "width": crop.width,
        "height": crop.height,
        "aspect_ratio": round(crop.width / max(1, crop.height), 4),
        "dark_ratio": round(dark_ratio, 6),
        "contrast": round(contrast, 4),
        "foreground_bbox_ratio": round(foreground_bbox_ratio, 6),
        "foreground_bbox": foreground_bbox,
        "touched_edges": touched_edges,
        "edge_dark_ratios": {
            edge_name: round(ratio, 6)
            for edge_name, ratio in edge_dark_ratios.items()
        },
        "quality_flags": quality_flags,
        "accepted": not quality_flags,
    }


def _dense_edge_trim_pixels(
    crop: Image.Image,
    *,
    dense_edge_dark_ratio: float,
    max_edge_trim_ratio: float,
) -> dict[str, int]:
    grayscale = crop.convert("L")
    max_left_right_trim = int(crop.width * max_edge_trim_ratio)
    max_top_bottom_trim = int(crop.height * max_edge_trim_ratio)

    def column_dark_ratio(x: int) -> float:
        return (
            sum(1 for y in range(crop.height) if grayscale.getpixel((x, y)) < 200)
            / max(1, crop.height)
        )

    def row_dark_ratio(y: int) -> float:
        return (
            sum(1 for x in range(crop.width) if grayscale.getpixel((x, y)) < 200)
            / max(1, crop.width)
        )

    trim: dict[str, int] = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    for edge_name, limit in {
        "left": max_left_right_trim,
        "right": max_left_right_trim,
        "top": max_top_bottom_trim,
        "bottom": max_top_bottom_trim,
    }.items():
        for offset in range(limit):
            if edge_name == "left":
                ratio = column_dark_ratio(offset)
            elif edge_name == "right":
                ratio = column_dark_ratio(crop.width - 1 - offset)
            elif edge_name == "top":
                ratio = row_dark_ratio(offset)
            else:
                ratio = row_dark_ratio(crop.height - 1 - offset)
            if ratio < dense_edge_dark_ratio:
                break
            trim[edge_name] += 1
    return trim


def _salvage_dense_edge_crop(
    crop: Image.Image,
    quality: dict[str, Any],
    *,
    min_width: int,
    min_height: int,
    min_dark_ratio: float,
    min_contrast: float,
    max_foreground_bbox_ratio: float,
    dense_edge_dark_ratio: float,
    max_edge_trim_ratio: float,
    max_trim_passes: int = 3,
) -> tuple[Image.Image, dict[str, Any], dict[str, int] | None]:
    if not {"foreground_fills_crop", "dense_edge_content"} & set(quality["quality_flags"]):
        return crop, quality, None

    cumulative_trim = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    candidate_crop = crop
    candidate_quality = quality

    for _ in range(max_trim_passes):
        trim = _dense_edge_trim_pixels(
            candidate_crop,
            dense_edge_dark_ratio=dense_edge_dark_ratio,
            max_edge_trim_ratio=max_edge_trim_ratio,
        )
        if not any(trim.values()):
            break

        left = trim["left"]
        top = trim["top"]
        right = candidate_crop.width - trim["right"]
        bottom = candidate_crop.height - trim["bottom"]
        if right - left < min_width or bottom - top < min_height:
            break

        candidate_crop = candidate_crop.crop((left, top, right, bottom))
        candidate_quality = compute_crop_quality(
            candidate_crop,
            min_width=min_width,
            min_height=min_height,
            min_dark_ratio=min_dark_ratio,
            min_contrast=min_contrast,
            max_foreground_bbox_ratio=max_foreground_bbox_ratio,
            dense_edge_dark_ratio=dense_edge_dark_ratio,
        )
        cumulative_trim["left"] += trim["left"]
        cumulative_trim["right"] += trim["right"]
        cumulative_trim["top"] += trim["top"]
        cumulative_trim["bottom"] += trim["bottom"]
        if candidate_quality["accepted"]:
            break

    if not candidate_quality["accepted"] or not any(cumulative_trim.values()):
        return crop, quality, None

    candidate_quality["salvage_applied"] = True
    candidate_quality["original_quality_flags"] = list(quality["quality_flags"])
    candidate_quality["salvage_strategy"] = "trim_dense_edges"
    candidate_quality["trim_pixels"] = cumulative_trim
    return candidate_crop, candidate_quality, cumulative_trim


def _salvage_low_signal_issue_date_crop(
    image: Image.Image,
    region_box: tuple[int, int, int, int],
    *,
    region_name: str,
    min_width: int,
    min_height: int,
    min_dark_ratio: float,
    min_contrast: float,
    max_foreground_bbox_ratio: float,
    dense_edge_dark_ratio: float,
) -> tuple[Image.Image, dict[str, Any], tuple[int, int, int, int] | None]:
    crop = image.crop(region_box)
    quality = compute_crop_quality(
        crop,
        min_width=min_width,
        min_height=min_height,
        min_dark_ratio=min_dark_ratio,
        min_contrast=min_contrast,
        max_foreground_bbox_ratio=max_foreground_bbox_ratio,
        dense_edge_dark_ratio=dense_edge_dark_ratio,
    )
    if (
        region_name != "issue_date"
        or quality["accepted"]
        or not {"low_dark_ratio", "low_contrast"} & set(quality["quality_flags"])
    ):
        return crop, quality, None

    image_height = image.height
    top = region_box[1] / max(1, image_height)
    bottom = region_box[3] / max(1, image_height)

    for vertical_offset in ISSUE_DATE_VERTICAL_FALLBACK_OFFSETS:
        shifted_top = max(0.0, top + vertical_offset)
        shifted_bottom = min(1.0, bottom + vertical_offset)
        if shifted_bottom <= shifted_top:
            continue
        candidate_box = (
            region_box[0],
            int(image_height * shifted_top),
            region_box[2],
            int(image_height * shifted_bottom),
        )
        candidate_crop = image.crop(candidate_box)
        candidate_quality = compute_crop_quality(
            candidate_crop,
            min_width=min_width,
            min_height=min_height,
            min_dark_ratio=min_dark_ratio,
            min_contrast=min_contrast,
            max_foreground_bbox_ratio=max_foreground_bbox_ratio,
            dense_edge_dark_ratio=dense_edge_dark_ratio,
        )
        if candidate_quality["accepted"]:
            candidate_quality["salvage_applied"] = True
            candidate_quality["original_quality_flags"] = list(quality["quality_flags"])
            candidate_quality["salvage_strategy"] = "shift_issue_date_up"
            candidate_quality["vertical_offset"] = vertical_offset
            return candidate_crop, candidate_quality, candidate_box

    return crop, quality, None


def export_field_crops(
    labeled_root: Path,
    output_root: Path,
    *,
    val_ratio: float = 0.2,
    min_width: int = 24,
    min_height: int = 16,
    min_dark_ratio: float = 0.01,
    min_contrast: float = 8.0,
    max_foreground_bbox_ratio: float = 0.85,
    dense_edge_dark_ratio: float = 0.45,
    max_edge_trim_ratio: float = 0.18,
) -> dict[str, Any]:
    manifest_entries: list[dict[str, Any]] = []
    prepared_cases: list[dict[str, Any]] = []
    counts_by_group = Counter()
    counts_by_split = Counter()
    counts_by_document = Counter()
    skipped_reasons = Counter()
    quality_flag_counts = Counter()
    accepted_count = 0
    rejected_count = 0
    salvaged_count = 0

    for label_path in discover_label_paths(labeled_root):
        payload = load_label(label_path)
        source_path_raw = payload.get("source_path")
        if not source_path_raw or not isinstance(source_path_raw, str):
            skipped_reasons["missing_source_path"] += 1
            continue
        if not _is_supported_source(source_path_raw):
            skipped_reasons["non_file_source"] += 1
            continue

        source_path = Path(source_path_raw)
        if not source_path.exists():
            skipped_reasons["source_missing"] += 1
            continue

        try:
            document_type = DocumentType(payload["document_type"])
        except (KeyError, ValueError):
            skipped_reasons["document_type_invalid"] += 1
            continue

        expected_fields = payload.get("expected_fields", {})
        if not isinstance(expected_fields, dict) or not expected_fields:
            skipped_reasons["missing_expected_fields"] += 1
            continue

        profile = classify_template(document_type, source_path)
        if profile is None or not profile.ocr_regions:
            skipped_reasons["template_without_regions"] += 1
            continue

        try:
            image = Image.open(source_path).convert("RGB")
        except OSError:
            skipped_reasons["source_unreadable"] += 1
            continue

        case_id = payload.get("case_id", label_path.parent.name)
        available_region_names = {region.name for region in profile.ocr_regions}
        field_counts_by_group = Counter(
            field_group_for(field_name)
            for field_name, expected_value in expected_fields.items()
            if expected_value not in {None, ""}
            and field_name in available_region_names
        )
        field_groups = sorted(field_counts_by_group)
        prepared_cases.append(
            {
                "split_key": f"{document_type.value}:{case_id}",
                "case_id": case_id,
                "document_type": document_type.value,
                "source_path": source_path,
                "label_path": label_path,
                "expected_fields": expected_fields,
                "field_groups": field_groups,
                "field_counts_by_group": dict(sorted(field_counts_by_group.items())),
                "profile": profile,
                "image": image,
            }
        )

    case_splits = assign_case_splits(
        [
            {
                "split_key": item["split_key"],
                "case_id": item["case_id"],
                "document_type": item["document_type"],
                "source_path": str(item["source_path"]),
                "field_groups": item["field_groups"],
                "field_counts_by_group": item["field_counts_by_group"],
            }
            for item in prepared_cases
        ],
        val_ratio=val_ratio,
    )

    for prepared_case in prepared_cases:
        split_key = prepared_case["split_key"]
        case_id = prepared_case["case_id"]
        document_type = DocumentType(prepared_case["document_type"])
        source_path = Path(prepared_case["source_path"])
        label_path = Path(prepared_case["label_path"])
        expected_fields = prepared_case["expected_fields"]
        profile = prepared_case["profile"]
        image = prepared_case["image"]
        split = case_splits[split_key]

        for region in profile.ocr_regions:
            expected_value = expected_fields.get(region.name)
            if expected_value in {None, ""}:
                continue

            box = _box_from_region(
                image,
                region.left,
                region.top,
                region.right,
                region.bottom,
            )
            if box is None:
                skipped_reasons["invalid_region_box"] += 1
                continue

            field_group = field_group_for(region.name)
            crop_dir = output_root / split / field_group / region.name
            crop_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"{case_id}__{region.name}{source_path.suffix.lower() or '.png'}"
            crop_path = crop_dir / crop_name

            crop = image.crop(box)
            quality = compute_crop_quality(
                crop,
                min_width=min_width,
                min_height=min_height,
                min_dark_ratio=min_dark_ratio,
                min_contrast=min_contrast,
                max_foreground_bbox_ratio=max_foreground_bbox_ratio,
                dense_edge_dark_ratio=dense_edge_dark_ratio,
            )
            original_box = box
            crop, quality, shifted_box = _salvage_low_signal_issue_date_crop(
                image,
                box,
                region_name=region.name,
                min_width=min_width,
                min_height=min_height,
                min_dark_ratio=min_dark_ratio,
                min_contrast=min_contrast,
                max_foreground_bbox_ratio=max_foreground_bbox_ratio,
                dense_edge_dark_ratio=dense_edge_dark_ratio,
            )
            if shifted_box is not None:
                box = shifted_box
                salvaged_count += 1
            crop, quality, trim = _salvage_dense_edge_crop(
                crop,
                quality,
                min_width=min_width,
                min_height=min_height,
                min_dark_ratio=min_dark_ratio,
                min_contrast=min_contrast,
                max_foreground_bbox_ratio=max_foreground_bbox_ratio,
                dense_edge_dark_ratio=dense_edge_dark_ratio,
                max_edge_trim_ratio=max_edge_trim_ratio,
            )
            if trim is not None:
                box = (
                    box[0] + trim["left"],
                    box[1] + trim["top"],
                    box[2] - trim["right"],
                    box[3] - trim["bottom"],
                )
                salvaged_count += 1
            crop.save(crop_path)

            manifest_entry = {
                "case_id": case_id,
                "document_type": document_type.value,
                "template_id": profile.template_id,
                "field_name": region.name,
                "field_group": field_group,
                "text": str(expected_value),
                "split": split,
                "source_path": str(source_path),
                "label_path": str(label_path),
                "crop_path": str(crop_path),
                "box": {
                    "left": box[0],
                    "top": box[1],
                    "right": box[2],
                    "bottom": box[3],
                },
                "quality": quality,
            }
            if box != original_box:
                manifest_entry["original_box"] = {
                    "left": original_box[0],
                    "top": original_box[1],
                    "right": original_box[2],
                    "bottom": original_box[3],
                }
            manifest_entries.append(manifest_entry)
            counts_by_group[field_group] += 1
            counts_by_split[split] += 1
            counts_by_document[document_type.value] += 1
            if quality["accepted"]:
                accepted_count += 1
            else:
                rejected_count += 1
                quality_flag_counts.update(quality["quality_flags"])

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in manifest_entries) + "\n"
        if manifest_entries
        else "",
        encoding="utf-8",
    )

    grouped_entries: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest_entries:
        grouped_entries[(entry["field_group"], entry["split"])].append(entry)

    for (field_group, split), entries in grouped_entries.items():
        group_manifest = output_root / "manifests" / field_group
        group_manifest.mkdir(parents=True, exist_ok=True)
        (group_manifest / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
            encoding="utf-8",
        )

    summary = {
        "manifest_path": str(manifest_path),
        "total_crops": len(manifest_entries),
        "accepted_crops": accepted_count,
        "rejected_crops": rejected_count,
        "salvaged_crops": salvaged_count,
        "unique_source_count": len({str(item["source_path"]) for item in prepared_cases}),
        "counts_by_group": dict(sorted(counts_by_group.items())),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_document_type": dict(sorted(counts_by_document.items())),
        "quality_flag_counts": dict(sorted(quality_flag_counts.items())),
        "skipped_reasons": dict(sorted(skipped_reasons.items())),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = export_field_crops(
        args.labeled_root,
        args.output_root,
        val_ratio=args.val_ratio,
        min_width=args.min_width,
        min_height=args.min_height,
        min_dark_ratio=args.min_dark_ratio,
        min_contrast=args.min_contrast,
        max_foreground_bbox_ratio=args.max_foreground_bbox_ratio,
        dense_edge_dark_ratio=args.dense_edge_dark_ratio,
        max_edge_trim_ratio=args.max_edge_trim_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
