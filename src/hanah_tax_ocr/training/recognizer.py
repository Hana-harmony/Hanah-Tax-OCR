from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hanah_tax_ocr.training.field_crops import export_field_crops

DEFAULT_FIELD_CROPS_ROOT = Path("data/training/field_crops")
DEFAULT_RECOGNIZER_ROOT = Path("data/training/recognizer")

RECOMMENDED_SETTINGS: dict[str, dict[str, Any]] = {
    "english_name_org": {
        "base_config": "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml",
        "max_text_length": 64,
        "image_shape": "3,48,320",
        "batch_size": 32,
        "learning_rate": 0.0003,
    },
    "numeric_tin_code": {
        "base_config": "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml",
        "max_text_length": 24,
        "image_shape": "3,48,160",
        "batch_size": 64,
        "learning_rate": 0.0005,
    },
    "date": {
        "base_config": "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml",
        "max_text_length": 32,
        "image_shape": "3,48,192",
        "batch_size": 64,
        "learning_rate": 0.0004,
    },
    "korean_mixed_form": {
        "base_config": "configs/rec/PP-OCRv3/korean_PP-OCRv3_rec.yml",
        "max_text_length": 72,
        "image_shape": "3,48,320",
        "batch_size": 24,
        "learning_rate": 0.0002,
    },
}

BLOCKING_READINESS_WARNINGS = {
    "no_train_samples",
    "no_val_samples",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare PaddleOCR recognizer fine-tuning datasets from field crops."
    )
    parser.add_argument(
        "--field-crops-root",
        type=Path,
        default=DEFAULT_FIELD_CROPS_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RECOGNIZER_ROOT,
    )
    parser.add_argument(
        "--labeled-root",
        type=Path,
        default=Path("data/labeled"),
    )
    parser.add_argument(
        "--ensure-field-crops",
        action="store_true",
        help="Generate field crops first when the manifest does not exist.",
    )
    parser.add_argument(
        "--hard-cases-root",
        type=Path,
        default=Path("data/training/hard_cases"),
    )
    parser.add_argument(
        "--include-hard-cases",
        action="store_true",
        help="Merge hard-case augmented crops into the training split.",
    )
    parser.add_argument(
        "--include-rejected-crops",
        action="store_true",
        help="Include crops that failed field-crop quality checks.",
    )
    parser.add_argument(
        "--max-hard-case-ratio",
        type=float,
        default=0.5,
        help="Maximum hard-case share within each train split. Use 1.0 to disable the cap.",
    )
    return parser.parse_args()


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render or execute PaddleOCR recognizer fine-tuning commands."
    )
    parser.add_argument(
        "--plan-root",
        type=Path,
        default=DEFAULT_RECOGNIZER_ROOT,
    )
    parser.add_argument(
        "--field-group",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--paddleocr-home",
        type=Path,
        default=Path("PaddleOCR"),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated training commands instead of printing them.",
    )
    parser.add_argument(
        "--allow-unready",
        action="store_true",
        help="Allow execution even when a plan is blocked by readiness checks.",
    )
    return parser.parse_args()


def load_field_crop_manifest(field_crops_root: Path) -> list[dict[str, Any]]:
    manifest_path = field_crops_root / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def load_hard_case_manifest(hard_cases_root: Path) -> list[dict[str, Any]]:
    manifest_path = hard_cases_root / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def ensure_field_crops(field_crops_root: Path, labeled_root: Path) -> dict[str, Any]:
    manifest_path = field_crops_root / "manifest.jsonl"
    if manifest_path.exists():
        summary_path = field_crops_root / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        return {"manifest_path": str(manifest_path)}
    return export_field_crops(labeled_root, field_crops_root)


def _write_label_file(entries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    dataset_root = output_path.parent
    for entry in entries:
        relative_image = os.path.relpath(
            Path(entry["crop_path"]).resolve(),
            dataset_root.resolve(),
        )
        lines.append(f"{relative_image}\t{entry['text']}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _recommended_settings(
    field_group: str,
    sample_count: int,
    max_text_length: int,
) -> dict[str, Any]:
    settings = dict(
        RECOMMENDED_SETTINGS.get(
            field_group,
            RECOMMENDED_SETTINGS["korean_mixed_form"],
        )
    )
    settings["max_text_length"] = max(settings["max_text_length"], max_text_length)
    if sample_count < 20:
        settings["batch_size"] = max(8, settings["batch_size"] // 2)
    return settings


def _source_type_for(entry: dict[str, Any]) -> str:
    return "hard_case" if entry.get("augmentation_type") else "base"


def _source_key_for(entry: dict[str, Any]) -> str:
    return str(
        entry.get("source_path")
        or entry.get("base_crop_path")
        or entry.get("crop_path")
        or "unknown"
    )


def _counts_by(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter()
    for entry in entries:
        value = entry.get(key) or "unknown"
        counter[str(value)] += 1
    return dict(sorted(counter.items()))


def _unique_source_counts(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        value = str(entry.get(key) or "unknown")
        grouped[value].add(_source_key_for(entry))
    return dict(
        sorted((group_key, len(source_paths)) for group_key, source_paths in grouped.items())
    )


def _hard_case_variant_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for entry in entries:
        if _source_type_for(entry) != "hard_case":
            continue
        variant = str(entry.get("augmentation_type") or "unknown")
        counter[variant] += 1
    return dict(sorted(counter.items()))


def _hard_case_variant_counts_by_document_type(
    entries: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        if _source_type_for(entry) != "hard_case":
            continue
        document_type = str(entry.get("document_type") or "unknown")
        variant = str(entry.get("augmentation_type") or "unknown")
        grouped[document_type][variant] += 1
    return {
        document_type: dict(sorted(counter.items()))
        for document_type, counter in sorted(grouped.items())
    }


def _build_data_profile(
    train_entries: list[dict[str, Any]],
    val_entries: list[dict[str, Any]],
    *,
    filtered_hard_case_train_count: int = 0,
    max_hard_case_ratio: float | None = None,
) -> dict[str, Any]:
    train_base_entries = [
        entry for entry in train_entries if _source_type_for(entry) == "base"
    ]
    train_hard_case_entries = [
        entry for entry in train_entries if _source_type_for(entry) == "hard_case"
    ]
    val_base_entries = [
        entry for entry in val_entries if _source_type_for(entry) == "base"
    ]
    val_hard_case_entries = [
        entry for entry in val_entries if _source_type_for(entry) == "hard_case"
    ]
    source_counts = {
        "train": _counts_by(
            [
                {"source_type": _source_type_for(entry)}
                for entry in train_entries
            ],
            "source_type",
        ),
        "val": _counts_by(
            [
                {"source_type": _source_type_for(entry)}
                for entry in val_entries
            ],
            "source_type",
        ),
    }
    hard_case_train_count = source_counts["train"].get("hard_case", 0)
    train_count = len(train_entries)
    hard_case_train_ratio = (
        round(hard_case_train_count / train_count, 4) if train_count else 0.0
    )

    warnings: list[str] = []
    if train_count == 0:
        warnings.append("no_train_samples")
    elif train_count < 20:
        warnings.append("low_train_sample_count")
    if not val_entries:
        warnings.append("no_val_samples")
    if hard_case_train_ratio > 0.5:
        warnings.append("hard_case_dominant_train_split")
    if filtered_hard_case_train_count:
        warnings.append("hard_case_train_capped")
    hard_case_variant_counts = {
        "train": _hard_case_variant_counts(train_entries),
        "val": _hard_case_variant_counts(val_entries),
    }
    unique_hard_case_variant_counts = {
        split: len(variant_counts)
        for split, variant_counts in hard_case_variant_counts.items()
    }
    if hard_case_train_count and unique_hard_case_variant_counts["train"] < 2:
        warnings.append("low_hard_case_variant_diversity")
    if len({_source_key_for(entry) for entry in val_entries}) < 2:
        warnings.append("low_val_source_diversity")

    return {
        "counts_by_document_type": {
            "train": _counts_by(train_entries, "document_type"),
            "val": _counts_by(val_entries, "document_type"),
        },
        "counts_by_document_type_and_source": {
            "train": {
                "base": _counts_by(train_base_entries, "document_type"),
                "hard_case": _counts_by(train_hard_case_entries, "document_type"),
            },
            "val": {
                "base": _counts_by(val_base_entries, "document_type"),
                "hard_case": _counts_by(val_hard_case_entries, "document_type"),
            },
        },
        "unique_source_counts": {
            "train": len({_source_key_for(entry) for entry in train_entries}),
            "val": len({_source_key_for(entry) for entry in val_entries}),
        },
        "unique_source_counts_by_document_type": {
            "train": _unique_source_counts(train_entries, "document_type"),
            "val": _unique_source_counts(val_entries, "document_type"),
        },
        "counts_by_source_type": source_counts,
        "hard_case_variant_counts": hard_case_variant_counts,
        "hard_case_variant_counts_by_document_type": {
            "train": _hard_case_variant_counts_by_document_type(train_entries),
            "val": _hard_case_variant_counts_by_document_type(val_entries),
        },
        "unique_hard_case_variant_counts": unique_hard_case_variant_counts,
        "hard_case_train_ratio": hard_case_train_ratio,
        "filtered_hard_case_train_count": filtered_hard_case_train_count,
        "max_hard_case_ratio": max_hard_case_ratio,
        "hard_case_selection_strategy": "base_document_balance",
        "warnings": warnings,
    }


def _build_training_readiness(
    warnings: list[str],
    *,
    train_count: int,
    val_count: int,
    character_count: int,
) -> dict[str, Any]:
    normalized_warnings = list(dict.fromkeys(warnings))
    blocking_warnings = [
        warning
        for warning in normalized_warnings
        if warning in BLOCKING_READINESS_WARNINGS
    ]
    if train_count <= 0 and "no_train_samples" not in blocking_warnings:
        blocking_warnings.append("no_train_samples")
    if val_count <= 0 and "no_val_samples" not in blocking_warnings:
        blocking_warnings.append("no_val_samples")
    if character_count <= 0 and "no_train_samples" not in blocking_warnings:
        blocking_warnings.append("no_train_samples")

    advisory_warnings = [
        warning
        for warning in normalized_warnings
        if warning not in BLOCKING_READINESS_WARNINGS
    ]
    if blocking_warnings:
        status = "blocked"
    elif advisory_warnings:
        status = "review_required"
    else:
        status = "ready"

    return {
        "status": status,
        "ready_for_execution": not blocking_warnings,
        "blocking_warnings": blocking_warnings,
        "advisory_warnings": advisory_warnings,
    }


def _load_training_readiness(plan: dict[str, Any]) -> dict[str, Any]:
    training_readiness = plan.get("training_readiness")
    if isinstance(training_readiness, dict):
        return training_readiness

    settings = plan.get("settings", {})
    data_profile = plan.get("data_profile", {})
    return _build_training_readiness(
        list(data_profile.get("warnings", [])),
        train_count=int(settings.get("train_count", 0) or 0),
        val_count=int(settings.get("val_count", 0) or 0),
        character_count=int(settings.get("character_count", 0) or 0),
    )


def _entry_variant(entry: dict[str, Any]) -> str:
    return str(entry.get("augmentation_type") or "unknown")


def _entry_base_key(entry: dict[str, Any]) -> str:
    base_path = entry.get("base_crop_path") or entry.get("crop_path")
    return str(base_path or entry.get("case_id") or "unknown")


def _entry_sort_key(entry: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(entry.get("document_type") or "unknown"),
        _entry_variant(entry),
        _entry_base_key(entry),
        str(entry.get("field_name") or "unknown"),
        str(entry.get("crop_path") or "unknown"),
    )


def _select_hard_case_entries(
    hard_case_entries: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for entry in sorted(hard_case_entries, key=_entry_sort_key):
        buckets[_entry_variant(entry)][_entry_base_key(entry)].append(entry)

    selected: list[dict[str, Any]] = []
    base_usage = Counter()
    variants = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for variant in variants:
            entries_by_base = buckets[variant]
            candidate_base_keys = [
                base_key
                for base_key, entries in entries_by_base.items()
                if entries
            ]
            if not candidate_base_keys:
                continue
            candidate_base_keys.sort(key=lambda base_key: (base_usage[base_key], base_key))
            base_key = candidate_base_keys[0]
            selected.append(entries_by_base[base_key].pop(0))
            base_usage[base_key] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _allocate_hard_case_document_quotas(
    base_entries: list[dict[str, Any]],
    hard_case_entries: list[dict[str, Any]],
    allowed_hard_case_count: int,
) -> dict[str, int]:
    if allowed_hard_case_count <= 0:
        return {}

    base_counts = Counter(str(entry.get("document_type") or "unknown") for entry in base_entries)
    available_counts = Counter(
        str(entry.get("document_type") or "unknown") for entry in hard_case_entries
    )
    total_base_count = sum(base_counts.values())
    if total_base_count <= 0:
        quotas: dict[str, int] = {}
        for document_type, available_count in sorted(available_counts.items()):
            if allowed_hard_case_count <= 0:
                break
            selected_count = min(available_count, allowed_hard_case_count)
            quotas[document_type] = selected_count
            allowed_hard_case_count -= selected_count
        return quotas

    quotas = {
        document_type: min(
            available_counts.get(document_type, 0),
            int(allowed_hard_case_count * count / total_base_count),
        )
        for document_type, count in base_counts.items()
    }
    remaining = allowed_hard_case_count - sum(quotas.values())

    while remaining > 0:
        candidates = [
            document_type
            for document_type, available_count in available_counts.items()
            if quotas.get(document_type, 0) < available_count
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda document_type: (
                -(
                    allowed_hard_case_count * base_counts.get(document_type, 0) / total_base_count
                    - quotas.get(document_type, 0)
                ),
                -base_counts.get(document_type, 0),
                document_type,
            )
        )
        quotas[candidates[0]] = quotas.get(candidates[0], 0) + 1
        remaining -= 1
    return quotas


def _limit_hard_case_train_entries(
    train_entries: list[dict[str, Any]],
    *,
    max_hard_case_ratio: float | None,
) -> tuple[list[dict[str, Any]], int]:
    if max_hard_case_ratio is None or max_hard_case_ratio >= 1.0:
        return train_entries, 0
    if max_hard_case_ratio < 0.0:
        raise ValueError("max_hard_case_ratio must be greater than or equal to 0")

    base_entries = [entry for entry in train_entries if _source_type_for(entry) == "base"]
    hard_case_entries = [
        entry for entry in train_entries if _source_type_for(entry) == "hard_case"
    ]
    if not hard_case_entries:
        return train_entries, 0

    if not base_entries or max_hard_case_ratio == 0.0:
        allowed_hard_case_count = 0
    else:
        allowed_hard_case_count = int(
            len(base_entries) * max_hard_case_ratio / (1.0 - max_hard_case_ratio)
        )
    if len(hard_case_entries) <= allowed_hard_case_count:
        return train_entries, 0

    quotas = _allocate_hard_case_document_quotas(
        base_entries,
        hard_case_entries,
        allowed_hard_case_count,
    )
    selected_hard_case_entries: list[dict[str, Any]] = []
    for document_type, quota in sorted(quotas.items()):
        document_entries = [
            entry
            for entry in hard_case_entries
            if str(entry.get("document_type") or "unknown") == document_type
        ]
        selected_hard_case_entries.extend(
            _select_hard_case_entries(document_entries, quota)
        )
    selected_ids = {id(entry) for entry in selected_hard_case_entries}
    retained_entries = [
        entry
        for entry in train_entries
        if _source_type_for(entry) == "base" or id(entry) in selected_ids
    ]
    return retained_entries, len(hard_case_entries) - len(selected_hard_case_entries)


def prepare_recognizer_datasets(
    field_crops_root: Path,
    output_root: Path,
    *,
    labeled_root: Path | None = None,
    ensure_crops: bool = False,
    hard_cases_root: Path | None = None,
    include_hard_cases: bool = False,
    include_rejected_crops: bool = False,
    max_hard_case_ratio: float | None = 0.5,
) -> dict[str, Any]:
    if ensure_crops:
        if labeled_root is None:
            raise ValueError("labeled_root is required when ensure_crops is True")
        ensure_field_crops(field_crops_root, labeled_root)

    entries = load_field_crop_manifest(field_crops_root)
    if include_hard_cases and hard_cases_root is not None:
        entries.extend(load_hard_case_manifest(hard_cases_root))
    grouped_entries: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in entries:
        quality = entry.get("quality", {})
        if quality and not include_rejected_crops and not quality.get("accepted", True):
            continue
        grouped_entries[entry["field_group"]][entry["split"]].append(entry)

    summary_groups: dict[str, Any] = {}
    for field_group, splits in grouped_entries.items():
        group_root = output_root / field_group
        group_root.mkdir(parents=True, exist_ok=True)

        train_entries, filtered_hard_case_train_count = _limit_hard_case_train_entries(
            splits.get("train", []),
            max_hard_case_ratio=max_hard_case_ratio,
        )
        val_entries = splits.get("val", [])
        train_file = group_root / "train.txt"
        val_file = group_root / "val.txt"
        _write_label_file(train_entries, train_file)
        _write_label_file(val_entries, val_file)

        unique_chars = sorted(
            {
                character
                for entry in train_entries + val_entries
                for character in entry["text"]
            }
        )
        dict_path = group_root / "dict.txt"
        dict_path.write_text(
            "\n".join(unique_chars) + ("\n" if unique_chars else ""),
            encoding="utf-8",
        )

        max_text_length = max(
            (len(entry["text"]) for entry in train_entries + val_entries),
            default=0,
        )
        settings = _recommended_settings(field_group, len(train_entries), max_text_length)
        data_profile = _build_data_profile(
            train_entries,
            val_entries,
            filtered_hard_case_train_count=filtered_hard_case_train_count,
            max_hard_case_ratio=max_hard_case_ratio,
        )
        settings["dictionary_path"] = str(dict_path)
        settings["train_label_path"] = str(train_file)
        settings["val_label_path"] = str(val_file)
        settings["sample_count"] = len(train_entries) + len(val_entries)
        settings["train_count"] = len(train_entries)
        settings["val_count"] = len(val_entries)
        settings["character_count"] = len(unique_chars)
        training_readiness = _build_training_readiness(
            data_profile["warnings"],
            train_count=settings["train_count"],
            val_count=settings["val_count"],
            character_count=settings["character_count"],
        )

        plan = {
            "field_group": field_group,
            "settings": settings,
            "field_names": sorted(
                {entry["field_name"] for entry in train_entries + val_entries}
            ),
            "data_profile": data_profile,
            "training_readiness": training_readiness,
        }
        plan_path = group_root / "plan.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_groups[field_group] = {
            "train_count": len(train_entries),
            "val_count": len(val_entries),
            "character_count": len(unique_chars),
            "plan_path": str(plan_path),
            "data_profile": data_profile,
            "training_readiness": training_readiness,
        }

    summary = {
        "field_crops_root": str(field_crops_root),
        "output_root": str(output_root),
        "hard_cases_root": None if hard_cases_root is None else str(hard_cases_root),
        "include_hard_cases": include_hard_cases,
        "include_rejected_crops": include_rejected_crops,
        "max_hard_case_ratio": max_hard_case_ratio,
        "groups": dict(sorted(summary_groups.items())),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def render_training_command(plan_path: Path, paddleocr_home: Path) -> str:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    settings = plan["settings"]
    plan_root = plan_path.parent
    train_txt = Path(settings["train_label_path"]).resolve()
    val_txt = Path(settings["val_label_path"]).resolve()
    dict_txt = Path(settings["dictionary_path"]).resolve()
    model_dir = (plan_root / "model_output").resolve()
    train_py = paddleocr_home / "tools" / "train.py"
    base_config = settings["base_config"]
    return (
        f"python {shlex.quote(str(train_py))} -c {shlex.quote(base_config)} "
        f"-o Global.character_dict_path={shlex.quote(str(dict_txt))} "
        f"Global.save_model_dir={shlex.quote(str(model_dir))} "
        f"Global.max_text_length={settings['max_text_length']} "
        f"Global.infer_img={settings['image_shape']} "
        f"Optimizer.lr.learning_rate={settings['learning_rate']} "
        f"Train.loader.batch_size_per_card={settings['batch_size']} "
        f"Eval.loader.batch_size_per_card={settings['batch_size']} "
        f"Train.dataset.label_file_list=[{shlex.quote(str(train_txt))}] "
        f"Eval.dataset.label_file_list=[{shlex.quote(str(val_txt))}]"
    )


def run_training_plans(
    plan_root: Path,
    paddleocr_home: Path,
    *,
    field_groups: list[str] | None = None,
    execute: bool = False,
    allow_unready: bool = False,
) -> dict[str, str]:
    commands: dict[str, str] = {}
    blocked_groups: dict[str, list[str]] = {}
    candidate_groups = field_groups or [
        path.name
        for path in sorted(plan_root.iterdir())
        if path.is_dir() and (path / "plan.json").exists()
    ]
    for field_group in candidate_groups:
        plan_path = plan_root / field_group / "plan.json"
        if not plan_path.exists():
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        training_readiness = _load_training_readiness(plan)
        command = render_training_command(plan_path, paddleocr_home)
        commands[field_group] = command
        if execute and not allow_unready and not training_readiness["ready_for_execution"]:
            blocked_groups[field_group] = list(training_readiness["blocking_warnings"])

    if execute and blocked_groups:
        blocked_summary = ", ".join(
            f"{field_group}({', '.join(warnings)})"
            for field_group, warnings in sorted(blocked_groups.items())
        )
        raise ValueError(
            "Refusing to execute unready recognizer training plans: "
            f"{blocked_summary}. Regenerate labels or rerun with --allow-unready "
            "only for manual inspection."
        )

    if execute:
        train_py = paddleocr_home / "tools" / "train.py"
        if not train_py.exists():
            raise FileNotFoundError(
                f"Missing PaddleOCR training entrypoint: {train_py}"
            )
        for field_group in candidate_groups:
            command = commands.get(field_group)
            if command is None:
                continue
            subprocess.run(command, shell=True, check=True)
    return commands


def main() -> None:
    args = parse_args()
    summary = prepare_recognizer_datasets(
        args.field_crops_root,
        args.output_root,
        labeled_root=args.labeled_root,
        ensure_crops=args.ensure_field_crops,
        hard_cases_root=args.hard_cases_root,
        include_hard_cases=args.include_hard_cases,
        include_rejected_crops=args.include_rejected_crops,
        max_hard_case_ratio=args.max_hard_case_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False))


def run_main() -> None:
    args = parse_run_args()
    commands = run_training_plans(
        args.plan_root,
        args.paddleocr_home,
        field_groups=args.field_group or None,
        execute=args.execute,
        allow_unready=args.allow_unready,
    )
    print(json.dumps(commands, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
