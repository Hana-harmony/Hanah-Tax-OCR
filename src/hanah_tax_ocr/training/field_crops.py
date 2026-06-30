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
    return parser.parse_args()


def field_group_for(field_name: str) -> str:
    return FIELD_GROUPS.get(field_name, "korean_mixed_form")


def split_score_for_case(case_id: str) -> float:
    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_for_case(case_id: str, val_ratio: float) -> str:
    return "val" if split_score_for_case(case_id) < val_ratio else "train"


def assign_case_splits(
    case_documents: list[dict[str, str]],
    *,
    val_ratio: float,
) -> dict[str, str]:
    assignments = {
        item["case_id"]: split_for_case(item["case_id"], val_ratio)
        for item in case_documents
    }
    if not 0.0 < val_ratio < 1.0:
        return assignments

    by_document_type: dict[str, list[str]] = defaultdict(list)
    for item in case_documents:
        by_document_type[item["document_type"]].append(item["case_id"])

    for case_ids in by_document_type.values():
        unique_case_ids = sorted(
            set(case_ids),
            key=lambda case_id: (split_score_for_case(case_id), case_id),
        )
        if len(unique_case_ids) < 2:
            continue

        if not any(assignments[case_id] == "val" for case_id in unique_case_ids):
            assignments[unique_case_ids[0]] = "val"
        if not any(assignments[case_id] == "train" for case_id in unique_case_ids):
            assignments[unique_case_ids[-1]] = "train"

    return assignments


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
        prepared_cases.append(
            {
                "case_id": case_id,
                "document_type": document_type.value,
                "source_path": source_path,
                "label_path": label_path,
                "expected_fields": expected_fields,
                "profile": profile,
                "image": image,
            }
        )

    case_splits = assign_case_splits(
        [
            {
                "case_id": item["case_id"],
                "document_type": item["document_type"],
            }
            for item in prepared_cases
        ],
        val_ratio=val_ratio,
    )

    for prepared_case in prepared_cases:
        case_id = prepared_case["case_id"]
        document_type = DocumentType(prepared_case["document_type"])
        source_path = Path(prepared_case["source_path"])
        label_path = Path(prepared_case["label_path"])
        expected_fields = prepared_case["expected_fields"]
        profile = prepared_case["profile"]
        image = prepared_case["image"]
        split = case_splits[case_id]

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
            crop.save(crop_path)
            quality = compute_crop_quality(
                crop,
                min_width=min_width,
                min_height=min_height,
                min_dark_ratio=min_dark_ratio,
                min_contrast=min_contrast,
                max_foreground_bbox_ratio=max_foreground_bbox_ratio,
                dense_edge_dark_ratio=dense_edge_dark_ratio,
            )

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
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
