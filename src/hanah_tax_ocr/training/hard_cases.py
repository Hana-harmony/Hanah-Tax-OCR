from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

DEFAULT_FIELD_CROPS_ROOT = Path("data/training/field_crops")
DEFAULT_OUTPUT_ROOT = Path("data/training/hard_cases")

HARD_CASE_PROFILES: dict[str, tuple[str, ...]] = {
    "english_name_org": ("left_clip", "rotate", "low_res", "overlay_patch"),
    "numeric_tin_code": ("left_clip", "rotate", "low_res"),
    "date": ("rotate", "low_res"),
    "korean_mixed_form": ("left_clip", "low_res", "overlay_patch"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hard-case augmented field crop datasets for recognizer tuning."
    )
    parser.add_argument(
        "--field-crops-root",
        type=Path,
        default=DEFAULT_FIELD_CROPS_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260630,
    )
    return parser.parse_args()


def load_manifest(field_crops_root: Path) -> list[dict[str, Any]]:
    manifest_path = field_crops_root / "manifest.jsonl"
    entries: list[dict[str, Any]] = []
    if not manifest_path.exists():
        return entries
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _open_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _apply_left_clip(image: Image.Image) -> Image.Image:
    clip_width = max(1, int(image.width * 0.08))
    cropped = image.crop((clip_width, 0, image.width, image.height))
    canvas = Image.new("RGB", image.size, "white")
    canvas.paste(cropped.resize((image.width - clip_width, image.height)), (clip_width, 0))
    return canvas


def _apply_rotate(image: Image.Image) -> Image.Image:
    return image.rotate(4.5, expand=False, fillcolor="white")


def _apply_low_res(image: Image.Image) -> Image.Image:
    resized = image.resize(
        (max(1, int(image.width * 0.6)), max(1, int(image.height * 0.6))),
        resample=Image.Resampling.BILINEAR,
    )
    restored = resized.resize(image.size, resample=Image.Resampling.BILINEAR)
    buffer = BytesIO()
    restored.save(buffer, format="JPEG", quality=35)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _apply_overlay_patch(
    image: Image.Image,
    donors: list[dict[str, Any]],
    rng: random.Random,
) -> Image.Image:
    canvas = image.copy().convert("RGBA")
    if donors:
        donor_path = Path(rng.choice(donors)["crop_path"])
        try:
            donor = _open_image(donor_path)
            overlay = donor.resize(
                (max(8, image.width // 3), max(8, image.height // 2)),
                resample=Image.Resampling.BILINEAR,
            ).convert("RGBA")
        except OSError:
            overlay = Image.new(
                "RGBA",
                (max(8, image.width // 3), max(8, image.height // 2)),
                (255, 0, 0, 0),
            )
    else:
        overlay = Image.new(
            "RGBA",
            (max(8, image.width // 3), max(8, image.height // 2)),
            (255, 0, 0, 0),
        )

    tint = Image.new("RGBA", overlay.size, (200, 40, 40, 70))
    overlay = Image.blend(overlay, tint, 0.45)
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((0, 0, overlay.width - 1, overlay.height - 1), outline=(160, 30, 30, 180), width=2)
    position = (image.width // 5, image.height // 4)
    canvas.alpha_composite(overlay, position)
    return canvas.convert("RGB")


def augment_hard_cases(
    field_crops_root: Path,
    output_root: Path,
    *,
    seed: int = 20260630,
) -> dict[str, Any]:
    rng = random.Random(seed)
    manifest_entries = load_manifest(field_crops_root)
    train_entries = [entry for entry in manifest_entries if entry.get("split") == "train"]
    donors_by_group: dict[str, list[dict[str, Any]]] = {}
    for entry in train_entries:
        donors_by_group.setdefault(entry["field_group"], []).append(entry)

    output_manifest: list[dict[str, Any]] = []
    counts_by_group = Counter()
    counts_by_variant = Counter()

    for entry in train_entries:
        source_image = _open_image(entry["crop_path"])
        field_group = entry["field_group"]
        variants = HARD_CASE_PROFILES.get(field_group, ("low_res",))

        for variant in variants:
            if variant == "left_clip":
                augmented = _apply_left_clip(source_image)
            elif variant == "rotate":
                augmented = _apply_rotate(source_image)
            elif variant == "low_res":
                augmented = _apply_low_res(source_image)
            elif variant == "overlay_patch":
                donor_entries = donors_by_group.get("korean_mixed_form", []) + donors_by_group.get(
                    field_group,
                    [],
                )
                augmented = _apply_overlay_patch(
                    source_image,
                    donor_entries,
                    rng,
                )
            else:
                continue

            variant_dir = output_root / field_group / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            source_path = Path(entry["crop_path"])
            output_path = variant_dir / f"{source_path.stem}__{variant}{source_path.suffix}"
            augmented.save(output_path)

            output_entry = {
                **entry,
                "augmentation_type": variant,
                "base_crop_path": entry["crop_path"],
                "crop_path": str(output_path),
            }
            output_manifest.append(output_entry)
            counts_by_group[field_group] += 1
            counts_by_variant[variant] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in output_manifest)
        + ("\n" if output_manifest else ""),
        encoding="utf-8",
    )
    summary = {
        "field_crops_root": str(field_crops_root),
        "output_root": str(output_root),
        "total_augmented_crops": len(output_manifest),
        "counts_by_group": dict(sorted(counts_by_group.items())),
        "counts_by_variant": dict(sorted(counts_by_variant.items())),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = augment_hard_cases(
        args.field_crops_root,
        args.output_root,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
