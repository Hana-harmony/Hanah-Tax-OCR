from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from collections import defaultdict
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


def prepare_recognizer_datasets(
    field_crops_root: Path,
    output_root: Path,
    *,
    labeled_root: Path | None = None,
    ensure_crops: bool = False,
    hard_cases_root: Path | None = None,
    include_hard_cases: bool = False,
    include_rejected_crops: bool = False,
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

        train_entries = splits.get("train", [])
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
        settings["dictionary_path"] = str(dict_path)
        settings["train_label_path"] = str(train_file)
        settings["val_label_path"] = str(val_file)
        settings["sample_count"] = len(train_entries) + len(val_entries)
        settings["train_count"] = len(train_entries)
        settings["val_count"] = len(val_entries)
        settings["character_count"] = len(unique_chars)

        plan = {
            "field_group": field_group,
            "settings": settings,
            "field_names": sorted(
                {entry["field_name"] for entry in train_entries + val_entries}
            ),
        }
        plan_path = group_root / "plan.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_groups[field_group] = {
            "train_count": len(train_entries),
            "val_count": len(val_entries),
            "character_count": len(unique_chars),
            "plan_path": str(plan_path),
        }

    summary = {
        "field_crops_root": str(field_crops_root),
        "output_root": str(output_root),
        "hard_cases_root": None if hard_cases_root is None else str(hard_cases_root),
        "include_hard_cases": include_hard_cases,
        "include_rejected_crops": include_rejected_crops,
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
) -> dict[str, str]:
    commands: dict[str, str] = {}
    candidate_groups = field_groups or [
        path.name
        for path in sorted(plan_root.iterdir())
        if path.is_dir() and (path / "plan.json").exists()
    ]
    for field_group in candidate_groups:
        plan_path = plan_root / field_group / "plan.json"
        if not plan_path.exists():
            continue
        command = render_training_command(plan_path, paddleocr_home)
        commands[field_group] = command
        if execute:
            train_py = paddleocr_home / "tools" / "train.py"
            if not train_py.exists():
                raise FileNotFoundError(
                    f"Missing PaddleOCR training entrypoint: {train_py}"
                )
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
    )
    print(json.dumps(summary, ensure_ascii=False))


def run_main() -> None:
    args = parse_run_args()
    commands = run_training_plans(
        args.plan_root,
        args.paddleocr_home,
        field_groups=args.field_group or None,
        execute=args.execute,
    )
    print(json.dumps(commands, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
