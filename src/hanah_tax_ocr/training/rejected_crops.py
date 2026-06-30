from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_FIELD_CROPS_ROOT = Path("data/training/field_crops")
DEFAULT_DATA_GAP_REPORT_PATH = Path("data/training/reports/data_gap_report.json")
DEFAULT_OUTPUT_PATH = Path("data/training/reports/rejected_field_crops.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report rejected field crops for manual review."
    )
    parser.add_argument(
        "--field-crops-root",
        type=Path,
        default=DEFAULT_FIELD_CROPS_ROOT,
    )
    parser.add_argument(
        "--data-gap-report",
        type=Path,
        default=DEFAULT_DATA_GAP_REPORT_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _priority_by_group(data_gap_report_path: Path | None) -> dict[str, dict[str, Any]]:
    if data_gap_report_path is None or not data_gap_report_path.exists():
        return {}
    report = load_json(data_gap_report_path)
    return {
        str(item["field_group"]): item
        for item in report.get("priorities", [])
        if item.get("field_group")
    }


def _review_actions(quality_flags: list[str]) -> list[str]:
    actions: list[str] = []
    if "dense_edge_content" in quality_flags:
        actions.append("inspect_seal_or_noise_overlap")
    if "foreground_fills_crop" in quality_flags:
        actions.append("tighten_region_box_or_verify_template")
    if "low_dark_ratio" in quality_flags or "low_contrast" in quality_flags:
        actions.append("check_blank_or_faint_text_crop")
    if "too_narrow" in quality_flags or "too_short" in quality_flags:
        actions.append("review_region_box_size")
    return actions or ["review_crop_quality"]


def build_rejected_field_crop_report(
    field_crops_root: Path,
    *,
    data_gap_report_path: Path | None = None,
) -> dict[str, Any]:
    priority_by_group = _priority_by_group(data_gap_report_path)
    rejected_entries = [
        entry
        for entry in load_jsonl(field_crops_root / "manifest.jsonl")
        if not entry.get("quality", {}).get("accepted", True)
    ]

    grouped_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    flag_counts = Counter()
    document_counts = Counter()
    for entry in rejected_entries:
        field_group = str(entry.get("field_group") or "unknown")
        grouped_entries[field_group].append(entry)
        flag_counts.update(entry.get("quality", {}).get("quality_flags", []))
        document_counts[str(entry.get("document_type") or "unknown")] += 1

    groups: list[dict[str, Any]] = []
    for field_group, entries in grouped_entries.items():
        priority_item = priority_by_group.get(field_group, {})
        group_flag_counts = Counter()
        for entry in entries:
            group_flag_counts.update(entry.get("quality", {}).get("quality_flags", []))
        groups.append(
            {
                "field_group": field_group,
                "priority_score": priority_item.get("priority_score"),
                "recommendations": priority_item.get("recommendations", []),
                "rejected_count": len(entries),
                "counts_by_document_type": dict(
                    sorted(
                        Counter(
                            str(entry.get("document_type") or "unknown") for entry in entries
                        ).items()
                    )
                ),
                "quality_flag_counts": dict(sorted(group_flag_counts.items())),
                "entries": [
                    {
                        "case_id": entry.get("case_id"),
                        "document_type": entry.get("document_type"),
                        "field_name": entry.get("field_name"),
                        "split": entry.get("split"),
                        "crop_path": entry.get("crop_path"),
                        "source_path": entry.get("source_path"),
                        "quality_flags": entry.get("quality", {}).get("quality_flags", []),
                        "review_actions": _review_actions(
                            entry.get("quality", {}).get("quality_flags", [])
                        ),
                    }
                    for entry in sorted(
                        entries,
                        key=lambda item: (
                            str(item.get("document_type") or ""),
                            str(item.get("case_id") or ""),
                            str(item.get("field_name") or ""),
                        ),
                    )
                ],
            }
        )

    groups.sort(
        key=lambda item: (
            -(item["priority_score"] if item["priority_score"] is not None else -1.0),
            -item["rejected_count"],
            item["field_group"],
        )
    )

    return {
        "field_crops_root": str(field_crops_root),
        "data_gap_report_path": None
        if data_gap_report_path is None
        else str(data_gap_report_path),
        "rejected_crop_count": len(rejected_entries),
        "counts_by_field_group": {
            group["field_group"]: group["rejected_count"] for group in groups
        },
        "counts_by_document_type": dict(sorted(document_counts.items())),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "groups": groups,
    }


def main() -> None:
    args = parse_args()
    report = build_rejected_field_crop_report(
        args.field_crops_root,
        data_gap_report_path=args.data_gap_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
