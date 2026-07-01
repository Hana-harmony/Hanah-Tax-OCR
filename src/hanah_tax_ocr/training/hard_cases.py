from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

DEFAULT_FIELD_CROPS_ROOT = Path("data/training/field_crops")
DEFAULT_OUTPUT_ROOT = Path("data/training/hard_cases")

HARD_CASE_PROFILES: dict[str, tuple[str, ...]] = {
    "english_name_org": (
        "left_clip",
        "rotate",
        "low_res",
        "gaussian_blur",
        "jpeg_blocking",
        "overlay_patch",
        "stamp_shadow",
    ),
    "numeric_tin_code": (
        "left_clip",
        "rotate",
        "low_res",
        "gaussian_blur",
        "jpeg_blocking",
        "edge_overlap",
        "border_clip",
    ),
    "date": (
        "rotate",
        "low_res",
        "gaussian_blur",
        "jpeg_blocking",
        "border_clip",
    ),
    "korean_mixed_form": (
        "left_clip",
        "low_res",
        "gaussian_blur",
        "jpeg_blocking",
        "overlay_patch",
        "stamp_shadow",
    ),
}

VARIANT_FAILURE_MODES: dict[str, tuple[str, ...]] = {
    "border_clip": ("border_clipping", "crop_miss"),
    "edge_overlap": ("border_clipping", "label_bleed"),
    "gaussian_blur": ("blur",),
    "jpeg_blocking": ("compression", "low_dpi"),
    "left_clip": ("crop_miss",),
    "low_res": ("low_dpi",),
    "overlay_patch": ("label_bleed", "stamp_interference"),
    "rotate": ("skew",),
    "stamp_shadow": ("stamp_interference", "label_bleed"),
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


def _apply_gaussian_blur(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=1.1))


def _apply_jpeg_blocking(image: Image.Image) -> Image.Image:
    resized = image.resize(
        (max(1, int(image.width * 0.72)), max(1, int(image.height * 0.72))),
        resample=Image.Resampling.BILINEAR,
    )
    buffer = BytesIO()
    resized.save(buffer, format="JPEG", quality=18)
    buffer.seek(0)
    restored = Image.open(buffer).convert("RGB")
    return restored.resize(image.size, resample=Image.Resampling.BILINEAR)


def _apply_border_clip(
    image: Image.Image,
    rng: random.Random,
    *,
    anchor: str | None = None,
    clip_width_ratio: float = 0.08,
    clip_height_ratio: float = 0.12,
) -> Image.Image:
    clip_width = max(1, int(image.width * clip_width_ratio))
    clip_height = max(1, int(image.height * clip_height_ratio))
    canvas = Image.new("RGB", image.size, "white")
    resolved_anchor = anchor or rng.choice(("left", "right", "top", "bottom"))
    if resolved_anchor == "left":
        cropped = image.crop((clip_width, 0, image.width, image.height))
        canvas.paste(cropped.resize((image.width - clip_width, image.height)), (clip_width, 0))
    elif resolved_anchor == "right":
        cropped = image.crop((0, 0, image.width - clip_width, image.height))
        canvas.paste(cropped.resize((image.width - clip_width, image.height)), (0, 0))
    elif resolved_anchor == "top":
        cropped = image.crop((0, clip_height, image.width, image.height))
        canvas.paste(cropped.resize((image.width, image.height - clip_height)), (0, clip_height))
    else:
        cropped = image.crop((0, 0, image.width, image.height - clip_height))
        canvas.paste(cropped.resize((image.width, image.height - clip_height)), (0, 0))
    return canvas


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


def _apply_edge_overlap(
    image: Image.Image,
    donors: list[dict[str, Any]],
    rng: random.Random,
    *,
    anchor: str | None = None,
) -> Image.Image:
    canvas = image.copy().convert("RGBA")
    patch_width = max(10, image.width // 4)
    patch_height = max(10, image.height // 2)
    if donors:
        donor_path = Path(rng.choice(donors)["crop_path"])
        try:
            donor = _open_image(donor_path)
            patch = donor.resize(
                (patch_width, patch_height),
                resample=Image.Resampling.BILINEAR,
            ).convert("RGBA")
        except OSError:
            patch = Image.new("RGBA", (patch_width, patch_height), (255, 255, 255, 0))
    else:
        patch = Image.new("RGBA", (patch_width, patch_height), (255, 255, 255, 0))

    shade = Image.new("RGBA", patch.size, (40, 40, 40, 110))
    patch = Image.blend(patch, shade, 0.65)
    draw = ImageDraw.Draw(patch)
    step = max(4, patch.width // 6)
    for x in range(0, patch.width, step):
        draw.line(
            [(x, 0), (x, patch.height - 1)],
            fill=(15, 15, 15, 150),
            width=max(1, step // 4),
        )
    for y in range(0, patch.height, step):
        draw.line(
            [(0, y), (patch.width - 1, y)],
            fill=(20, 20, 20, 120),
            width=1,
        )

    resolved_anchor = anchor or rng.choice(("left", "right", "bottom"))
    if resolved_anchor == "left":
        position = (0, max(0, image.height // 5))
    elif resolved_anchor == "right":
        position = (image.width - patch.width, max(0, image.height // 5))
    else:
        position = (max(0, image.width // 3), image.height - patch.height)
    canvas.alpha_composite(patch, position)
    return canvas.convert("RGB")


def _apply_stamp_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    canvas = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    ellipse_width = max(12, image.width // 2)
    ellipse_height = max(12, image.height // 2)
    left = rng.randint(0, max(0, image.width - ellipse_width))
    top = rng.randint(0, max(0, image.height - ellipse_height))
    draw.ellipse(
        (left, top, left + ellipse_width, top + ellipse_height),
        outline=(176, 36, 36, 160),
        width=max(2, min(image.width, image.height) // 18),
    )
    draw.line(
        [
            (left + ellipse_width // 6, top + ellipse_height // 3),
            (left + (ellipse_width * 5) // 6, top + (ellipse_height * 2) // 3),
        ],
        fill=(176, 36, 36, 110),
        width=max(1, min(image.width, image.height) // 22),
    )
    canvas.alpha_composite(overlay)
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
    counts_by_variant_by_group: dict[str, Counter[str]] = {}

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
            elif variant == "gaussian_blur":
                augmented = _apply_gaussian_blur(source_image)
            elif variant == "jpeg_blocking":
                augmented = _apply_jpeg_blocking(source_image)
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
            elif variant == "edge_overlap":
                donor_entries = (
                    donors_by_group.get("korean_mixed_form", [])
                    + donors_by_group.get("english_name_org", [])
                    + donors_by_group.get(field_group, [])
                )
                augmented = _apply_edge_overlap(
                    source_image,
                    donor_entries,
                    rng,
                )
            elif variant == "border_clip":
                augmented = _apply_border_clip(source_image, rng)
            elif variant == "stamp_shadow":
                augmented = _apply_stamp_shadow(source_image, rng)
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
                "target_failure_modes": list(VARIANT_FAILURE_MODES.get(variant, ())),
            }
            output_manifest.append(output_entry)
            counts_by_group[field_group] += 1
            counts_by_variant[variant] += 1
            counts_by_variant_by_group.setdefault(field_group, Counter())[variant] += 1

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
        "counts_by_variant_by_group": {
            field_group: dict(sorted(counter.items()))
            for field_group, counter in sorted(counts_by_variant_by_group.items())
        },
        "variant_failure_modes": {
            variant: list(modes)
            for variant, modes in sorted(VARIANT_FAILURE_MODES.items())
        },
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
