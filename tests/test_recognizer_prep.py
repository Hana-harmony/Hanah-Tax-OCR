from __future__ import annotations

import json
from pathlib import Path

import pytest
from hanah_tax_ocr.training.recognizer import (
    _select_hard_case_entries,
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
            "document_type": "residency_certificate",
            "field_group": "english_name_org",
            "field_name": "taxpayer_name",
            "text": "MARIA L CHEN",
            "split": "train",
            "crop_path": str(image_root / "name_001.png"),
            "quality": {"accepted": True},
        },
        {
            "case_id": "case_002",
            "document_type": "apostille",
            "field_group": "english_name_org",
            "field_name": "signed_by",
            "text": "CHONG U CHOI",
            "split": "val",
            "crop_path": str(image_root / "name_002.png"),
            "quality": {"accepted": True},
        },
        {
            "case_id": "case_003",
            "document_type": "withholding_tax_form",
            "field_group": "numeric_tin_code",
            "field_name": "tin",
            "text": "987-65-4321",
            "split": "train",
            "crop_path": str(image_root / "tin_001.png"),
            "quality": {"accepted": True},
        },
    ]
    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in manifest_entries) + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "recognizer"
    summary = prepare_recognizer_datasets(field_crops_root, output_root)

    assert sorted(summary["groups"]) == ["english_name_org", "numeric_tin_code"]
    assert summary["ensure_hard_cases_manifest"] is False
    assert summary["hard_cases_sync"] is None
    english_group = output_root / "english_name_org"
    assert (english_group / "train.txt").exists()
    assert (english_group / "val.txt").exists()
    plan = json.loads((english_group / "plan.json").read_text(encoding="utf-8"))
    assert plan["settings"]["character_count"] > 0
    assert "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml" in plan["settings"]["base_config"]
    assert plan["data_profile"]["counts_by_document_type"]["train"] == {
        "residency_certificate": 1
    }
    assert plan["data_profile"]["counts_by_document_type"]["val"] == {"apostille": 1}
    assert plan["data_profile"]["unique_source_counts"] == {"train": 1, "val": 1}
    assert plan["data_profile"]["hard_case_variant_counts"] == {"train": {}, "val": {}}
    assert plan["data_profile"]["unique_hard_case_variant_counts"] == {"train": 0, "val": 0}
    assert plan["data_profile"]["filtered_stale_hard_case_count"] == 0
    assert plan["data_profile"]["hard_case_selection_strategy"] == "base_document_balance"
    assert plan["data_profile"]["hard_case_variant_floor_applied"] is False
    assert plan["training_readiness"] == {
        "status": "review_required",
        "ready_for_execution": True,
        "blocking_warnings": [],
        "advisory_warnings": ["low_train_sample_count", "low_val_source_diversity"],
    }


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


def test_prepare_recognizer_datasets_skips_rejected_crops_by_default(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()
    accepted = image_root / "accepted.png"
    rejected = image_root / "rejected.png"
    accepted.write_bytes(b"accepted")
    rejected.write_bytes(b"rejected")
    entries = [
        {
            "case_id": "case_accepted",
            "document_type": "residency_certificate",
            "field_group": "english_name_org",
            "field_name": "taxpayer_name",
            "text": "MARIA",
            "split": "train",
            "crop_path": str(accepted),
            "quality": {"accepted": True},
        },
        {
            "case_id": "case_rejected",
            "document_type": "apostille",
            "field_group": "english_name_org",
            "field_name": "signed_by",
            "text": "CHOI",
            "split": "train",
            "crop_path": str(rejected),
            "quality": {"accepted": False},
        },
    ]
    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    summary = prepare_recognizer_datasets(field_crops_root, tmp_path / "recognizer")

    assert summary["groups"]["english_name_org"]["train_count"] == 1
    assert summary["groups"]["english_name_org"]["data_profile"]["counts_by_document_type"] == {
        "train": {"residency_certificate": 1},
        "val": {},
    }
    assert "no_val_samples" in summary["groups"]["english_name_org"]["data_profile"]["warnings"]
    assert (
        summary["groups"]["english_name_org"]["data_profile"]["unique_source_counts"]["train"]
        == 1
    )
    assert summary["groups"]["english_name_org"]["data_profile"]["hard_case_variant_counts"] == {
        "train": {},
        "val": {},
    }
    assert (
        summary["groups"]["english_name_org"]["data_profile"]["filtered_stale_hard_case_count"]
        == 0
    )
    assert (
        summary["groups"]["english_name_org"]["data_profile"]["hard_case_selection_strategy"]
        == "base_document_balance"
    )
    assert summary["groups"]["english_name_org"]["training_readiness"] == {
        "status": "blocked",
        "ready_for_execution": False,
        "blocking_warnings": ["no_val_samples"],
        "advisory_warnings": ["low_train_sample_count", "low_val_source_diversity"],
    }


def test_run_training_plans_blocks_unready_group_by_default(tmp_path: Path) -> None:
    plan_root = tmp_path / "recognizer" / "date"
    plan_root.mkdir(parents=True)
    for name in ("train.txt", "val.txt", "dict.txt"):
        (plan_root / name).write_text("sample\n", encoding="utf-8")

    plan_payload = {
        "field_group": "date",
        "settings": {
            "base_config": "configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml",
            "max_text_length": 32,
            "image_shape": "3,48,192",
            "batch_size": 32,
            "learning_rate": 0.0004,
            "dictionary_path": str(plan_root / "dict.txt"),
            "train_label_path": str(plan_root / "train.txt"),
            "val_label_path": str(plan_root / "val.txt"),
            "train_count": 1,
            "val_count": 0,
            "character_count": 8,
        },
        "data_profile": {"warnings": ["no_val_samples"]},
        "training_readiness": {
            "status": "blocked",
            "ready_for_execution": False,
            "blocking_warnings": ["no_val_samples"],
            "advisory_warnings": [],
        },
    }
    (plan_root / "plan.json").write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    paddleocr_home = tmp_path / "PaddleOCR" / "tools"
    paddleocr_home.mkdir(parents=True)
    (paddleocr_home / "train.py").write_text("print('should not run')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="date\\(no_val_samples\\)"):
        run_training_plans(
            tmp_path / "recognizer",
            tmp_path / "PaddleOCR",
            field_groups=["date"],
            execute=True,
        )


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
            "train_count": 1,
            "val_count": 1,
            "character_count": 6,
        },
        "data_profile": {"warnings": []},
        "training_readiness": {
            "status": "ready",
            "ready_for_execution": True,
            "blocking_warnings": [],
            "advisory_warnings": [],
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


def test_select_hard_case_entries_spreads_across_base_crops_and_variants() -> None:
    entries = []
    for base_name in ("base_a.png", "base_b.png", "base_c.png"):
        for variant in ("left_clip", "rotate", "low_res"):
            entries.append(
                {
                    "document_type": "withholding_tax_form",
                    "case_id": base_name.removesuffix(".png"),
                    "field_name": "tin",
                    "augmentation_type": variant,
                    "base_crop_path": base_name,
                    "crop_path": f"{base_name.removesuffix('.png')}__{variant}.png",
                }
            )

    selected = _select_hard_case_entries(entries, 4)

    assert len(selected) == 4
    assert len({entry["base_crop_path"] for entry in selected}) >= 3
    assert len({entry["augmentation_type"] for entry in selected}) >= 2


def test_prepare_recognizer_datasets_tracks_hard_case_variant_coverage(tmp_path: Path) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()
    base_path = image_root / "base.png"
    base_path.write_bytes(b"base")
    base_entry = {
        "case_id": "case_001",
        "document_type": "withholding_tax_form",
        "field_group": "numeric_tin_code",
        "field_name": "tin",
        "text": "987-65-4321",
        "split": "train",
        "crop_path": str(base_path),
        "quality": {"accepted": True},
    }
    (field_crops_root / "manifest.jsonl").write_text(
        json.dumps(base_entry) + "\n",
        encoding="utf-8",
    )

    hard_cases_root = tmp_path / "hard_cases"
    hard_cases_root.mkdir()
    hard_case_entries = []
    for variant in ("left_clip", "edge_overlap"):
        hard_path = image_root / f"base__{variant}.png"
        hard_path.write_bytes(variant.encode("utf-8"))
        hard_case_entries.append(
            {
                **base_entry,
                "augmentation_type": variant,
                "base_crop_path": str(base_path),
                "crop_path": str(hard_path),
            }
        )
    (hard_cases_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in hard_case_entries) + "\n",
        encoding="utf-8",
    )

    summary = prepare_recognizer_datasets(
        field_crops_root,
        tmp_path / "recognizer",
        hard_cases_root=hard_cases_root,
        include_hard_cases=True,
        max_hard_case_ratio=1.0,
    )

    profile = summary["groups"]["numeric_tin_code"]["data_profile"]
    assert profile["hard_case_variant_counts"] == {
        "train": {"edge_overlap": 1, "left_clip": 1},
        "val": {},
    }
    assert profile["hard_case_variant_counts_by_document_type"] == {
        "train": {
            "withholding_tax_form": {"edge_overlap": 1, "left_clip": 1},
        },
        "val": {},
    }
    assert profile["unique_hard_case_variant_counts"] == {"train": 2, "val": 0}


def test_prepare_recognizer_datasets_preserves_two_variants_for_single_base_group(
    tmp_path: Path,
) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()
    base_path = image_root / "base.png"
    base_path.write_bytes(b"base")
    base_entry = {
        "case_id": "case_001",
        "document_type": "apostille",
        "field_group": "korean_mixed_form",
        "field_name": "issuing_country",
        "text": "미국",
        "split": "train",
        "crop_path": str(base_path),
        "quality": {"accepted": True},
    }
    (field_crops_root / "manifest.jsonl").write_text(
        json.dumps(base_entry) + "\n",
        encoding="utf-8",
    )

    hard_cases_root = tmp_path / "hard_cases"
    hard_cases_root.mkdir()
    hard_case_entries = []
    for variant in ("left_clip", "low_res", "overlay_patch"):
        hard_path = image_root / f"base__{variant}.png"
        hard_path.write_bytes(variant.encode("utf-8"))
        hard_case_entries.append(
            {
                **base_entry,
                "augmentation_type": variant,
                "base_crop_path": str(base_path),
                "crop_path": str(hard_path),
            }
        )
    (hard_cases_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in hard_case_entries) + "\n",
        encoding="utf-8",
    )

    summary = prepare_recognizer_datasets(
        field_crops_root,
        tmp_path / "recognizer",
        hard_cases_root=hard_cases_root,
        include_hard_cases=True,
    )

    profile = summary["groups"]["korean_mixed_form"]["data_profile"]
    assert profile["counts_by_source_type"]["train"] == {"base": 1, "hard_case": 3}
    assert profile["hard_case_variant_counts"] == {
        "train": {"left_clip": 1, "low_res": 1, "overlay_patch": 1},
        "val": {},
    }
    assert profile["unique_hard_case_variant_counts"] == {"train": 3, "val": 0}
    assert profile["filtered_hard_case_train_count"] == 0
    assert profile["hard_case_train_ratio"] == 0.75
    assert (
        profile["hard_case_selection_strategy"]
        == "base_document_balance_with_scarce_full_variant_floor"
    )
    assert profile["hard_case_variant_floor_applied"] is True
    assert "hard_case_variant_floor_applied" in profile["warnings"]
    assert "hard_case_dominant_train_split" in profile["warnings"]
    assert "low_hard_case_variant_diversity" not in profile["warnings"]
    assert "hard_case_train_capped" not in profile["warnings"]


def test_prepare_recognizer_datasets_filters_stale_hard_cases_from_non_train_bases(
    tmp_path: Path,
) -> None:
    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()
    train_base_path = image_root / "train_base.png"
    train_base_path.write_bytes(b"train")
    val_base_path = image_root / "val_base.png"
    val_base_path.write_bytes(b"val")
    entries = [
        {
            "case_id": "case_train",
            "document_type": "withholding_tax_form",
            "field_group": "numeric_tin_code",
            "field_name": "tin",
            "text": "987-65-4321",
            "split": "train",
            "crop_path": str(train_base_path),
            "quality": {"accepted": True},
        },
        {
            "case_id": "case_val",
            "document_type": "withholding_tax_form",
            "field_group": "numeric_tin_code",
            "field_name": "tin",
            "text": "987-65-4321",
            "split": "val",
            "crop_path": str(val_base_path),
            "quality": {"accepted": True},
        },
    ]
    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    hard_cases_root = tmp_path / "hard_cases"
    hard_cases_root.mkdir()
    hard_case_entries = []
    for base_path, variant in (
        (train_base_path, "left_clip"),
        (val_base_path, "rotate"),
    ):
        hard_path = image_root / f"{base_path.stem}__{variant}.png"
        hard_path.write_bytes(variant.encode("utf-8"))
        hard_case_entries.append(
            {
                "case_id": base_path.stem,
                "document_type": "withholding_tax_form",
                "field_group": "numeric_tin_code",
                "field_name": "tin",
                "text": "987-65-4321",
                "split": "train",
                "augmentation_type": variant,
                "base_crop_path": str(base_path),
                "crop_path": str(hard_path),
                "quality": {"accepted": True},
            }
        )
    (hard_cases_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in hard_case_entries) + "\n",
        encoding="utf-8",
    )

    summary = prepare_recognizer_datasets(
        field_crops_root,
        tmp_path / "recognizer",
        hard_cases_root=hard_cases_root,
        include_hard_cases=True,
        max_hard_case_ratio=1.0,
    )

    profile = summary["groups"]["numeric_tin_code"]["data_profile"]
    assert profile["counts_by_source_type"]["train"] == {"base": 1, "hard_case": 1}
    assert profile["hard_case_variant_counts"] == {
        "train": {"left_clip": 1},
        "val": {},
    }
    assert profile["filtered_stale_hard_case_count"] == 1
    assert "stale_hard_cases_filtered" in profile["warnings"]


def test_prepare_recognizer_datasets_can_generate_missing_hard_cases_when_requested(
    tmp_path: Path,
) -> None:
    from PIL import Image

    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_path = tmp_path / "base.png"
    Image.new("RGB", (120, 40), "white").save(image_path)
    field_entry = {
        "case_id": "case_001",
        "document_type": "withholding_tax_form",
        "field_group": "numeric_tin_code",
        "field_name": "tin",
        "text": "987-65-4321",
        "split": "train",
        "crop_path": str(image_path),
        "quality": {"accepted": True},
    }
    (field_crops_root / "manifest.jsonl").write_text(
        json.dumps(field_entry) + "\n",
        encoding="utf-8",
    )

    hard_cases_root = tmp_path / "hard_cases"
    summary = prepare_recognizer_datasets(
        field_crops_root,
        tmp_path / "recognizer",
        hard_cases_root=hard_cases_root,
        include_hard_cases=True,
        ensure_hard_cases_manifest=True,
    )

    assert summary["ensure_hard_cases_manifest"] is True
    assert summary["hard_cases_sync"] == {
        "status": "generated",
        "stale_entry_count": 0,
        "total_augmented_crops": 4,
    }
    assert (hard_cases_root / "manifest.jsonl").exists()
    assert summary["groups"]["numeric_tin_code"]["train_count"] == 4


def test_prepare_recognizer_datasets_can_refresh_stale_hard_cases_when_requested(
    tmp_path: Path,
) -> None:
    from PIL import Image

    field_crops_root = tmp_path / "field_crops"
    field_crops_root.mkdir()
    image_root = tmp_path / "images"
    image_root.mkdir()
    train_base = image_root / "train_base.png"
    val_base = image_root / "val_base.png"
    Image.new("RGB", (120, 40), "white").save(train_base)
    Image.new("RGB", (120, 40), "white").save(val_base)
    field_entries = [
        {
            "case_id": "case_train",
            "document_type": "withholding_tax_form",
            "field_group": "numeric_tin_code",
            "field_name": "tin",
            "text": "987-65-4321",
            "split": "train",
            "crop_path": str(train_base),
            "quality": {"accepted": True},
        },
        {
            "case_id": "case_val",
            "document_type": "withholding_tax_form",
            "field_group": "numeric_tin_code",
            "field_name": "tin",
            "text": "987-65-4321",
            "split": "val",
            "crop_path": str(val_base),
            "quality": {"accepted": True},
        },
    ]
    (field_crops_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry) for entry in field_entries) + "\n",
        encoding="utf-8",
    )

    hard_cases_root = tmp_path / "hard_cases"
    hard_cases_root.mkdir()
    stale_path = image_root / "val_base__rotate.png"
    Image.new("RGB", (120, 40), "white").save(stale_path)
    stale_entry = {
        **field_entries[1],
        "split": "train",
        "augmentation_type": "rotate",
        "base_crop_path": str(val_base),
        "crop_path": str(stale_path),
    }
    (hard_cases_root / "manifest.jsonl").write_text(
        json.dumps(stale_entry) + "\n",
        encoding="utf-8",
    )

    summary = prepare_recognizer_datasets(
        field_crops_root,
        tmp_path / "recognizer",
        hard_cases_root=hard_cases_root,
        include_hard_cases=True,
        ensure_hard_cases_manifest=True,
    )

    assert summary["hard_cases_sync"] == {
        "status": "refreshed",
        "stale_entry_count": 1,
        "total_augmented_crops": 4,
    }
    refreshed_entries = [
        json.loads(line)
        for line in (hard_cases_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {entry["base_crop_path"] for entry in refreshed_entries} == {str(train_base)}
