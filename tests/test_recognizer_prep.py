from __future__ import annotations

import json
from pathlib import Path

from hanah_tax_ocr.training.recognizer import (
    prepare_recognizer_datasets,
    render_training_command,
    run_training_plans,
)


def test_prepare_recognizer_datasets_writes_group_manifests_and_plan(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()
    for image_name in ("name_001.png", "name_002.png", "tin_001.png"):
        (image_root / image_name).write_bytes(b"fake-image")

    manifest_entries = [
        {
            "case_id": "case_001",
            "field_group": "english_name_org",
            "field_name": "taxpayer_name",
            "text": "MARIA L CHEN",
            "split": "train",
            "crop_path": str(image_root / "name_001.png"),
        },
        {
            "case_id": "case_002",
            "field_group": "english_name_org",
            "field_name": "signed_by",
            "text": "CHONG U CHOI",
            "split": "val",
            "crop_path": str(image_root / "name_002.png"),
        },
        {
            "case_id": "case_003",
            "field_group": "numeric_tin_code",
            "field_name": "tin",
            "text": "987-65-4321",
            "split": "train",
            "crop_path": str(image_root / "tin_001.png"),
        },
    ]
    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in manifest_entries) + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "recognizer"
    summary = prepare_recognizer_datasets(field_crops_root, output_root)

    assert sorted(summary["groups"]) == ["english_name_org", "numeric_tin_code"]
    english_group = output_root / "english_name_org"
    assert (english_group / "train.txt").exists()
    assert (english_group / "val.txt").exists()
    plan = json.loads((english_group / "plan.json").read_text(encoding="utf-8"))
    assert plan["settings"]["character_count"] > 0
    assert "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml" in plan["settings"]["base_config"]


def test_render_training_command_uses_plan_paths(tmp_path: Path) -> None:
    group_root = tmp_path / "recognizer" / "date"
    group_root.mkdir(parents=True)
    (group_root / "train.txt").write_text("img.png\t2026-01-12\n", encoding="utf-8")
    (group_root / "val.txt").write_text("img2.png\t2026-01-13\n", encoding="utf-8")
    (group_root / "dict.txt").write_text("2\n0\n6\n-\n1\n3\n", encoding="utf-8")
    plan_payload = {
        "field_group": "date",
        "settings": {
            "base_config": "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml",
            "max_text_length": 32,
            "image_shape": "3,48,192",
            "batch_size": 64,
            "learning_rate": 0.0004,
            "dictionary_path": str(group_root / "dict.txt"),
            "train_label_path": str(group_root / "train.txt"),
            "val_label_path": str(group_root / "val.txt"),
        },
    }
    plan_path = group_root / "plan.json"
    plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    command = render_training_command(plan_path, Path("/tmp/PaddleOCR"))

    assert "python /tmp/PaddleOCR/tools/train.py" in command
    assert "Global.character_dict_path=" in command
    assert "Train.dataset.label_file_list=[" in command


def test_run_training_plans_can_execute_against_fake_paddleocr_home(tmp_path: Path) -> None:
    plan_root = tmp_path / "recognizer" / "numeric_tin_code"
    plan_root.mkdir(parents=True)
    for name in ("train.txt", "val.txt", "dict.txt"):
        (plan_root / name).write_text("sample\n", encoding="utf-8")

    plan_payload = {
        "field_group": "numeric_tin_code",
        "settings": {
            "base_config": "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml",
            "max_text_length": 24,
            "image_shape": "3,48,160",
            "batch_size": 32,
            "learning_rate": 0.0005,
            "dictionary_path": str(plan_root / "dict.txt"),
            "train_label_path": str(plan_root / "train.txt"),
            "val_label_path": str(plan_root / "val.txt"),
        },
    }
    (plan_root / "plan.json").write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paddleocr_home = tmp_path / "PaddleOCR" / "tools"
    paddleocr_home.mkdir(parents=True)
    train_py = paddleocr_home / "train.py"
    marker = tmp_path / "train_ran.txt"
    train_py.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )

    commands = run_training_plans(
        tmp_path / "recognizer",
        tmp_path / "PaddleOCR",
        field_groups=["numeric_tin_code"],
        execute=True,
    )

    assert "numeric_tin_code" in commands
    assert marker.read_text(encoding="utf-8") == "ran"
