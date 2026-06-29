from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

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
    return parser.parse_args()


def field_group_for(field_name: str) -> str:
    return FIELD_GROUPS.get(field_name, "korean_mixed_form")


def split_for_case(case_id: str, val_ratio: float) -> str:
    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


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


def export_field_crops(
    labeled_root: Path,
    output_root: Path,
    *,
    val_ratio: float = 0.2,
) -> dict[str, Any]:
    manifest_entries: list[dict[str, Any]] = []
    counts_by_group = Counter()
    counts_by_split = Counter()
    counts_by_document = Counter()
    skipped_reasons = Counter()

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
        split = split_for_case(case_id, val_ratio)

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
            }
            manifest_entries.append(manifest_entry)
            counts_by_group[field_group] += 1
            counts_by_split[split] += 1
            counts_by_document[document_type.value] += 1

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
        "counts_by_group": dict(sorted(counts_by_group.items())),
        "counts_by_split": dict(sorted(counts_by_split.items())),
        "counts_by_document_type": dict(sorted(counts_by_document.items())),
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
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
