from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEFAULT_FIELD_CROPS_ROOT = Path("data/training/field_crops")
DEFAULT_OUTPUT_ROOT = Path("data/training/hard_cases")
TARGET_FIELD_NAMES = {
    "address",
    "applicant_name",
    "first_name",
    "middle_name",
    "signed_by",
    "taxpayer_name",
}
SYNTHETIC_AUGMENTATION_PREFIX = "synthetic_english_name"
FONT_PATHS = [
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Helvetica.ttc"),
    Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
    Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
]
AMBIGUOUS_INITIALS = ("I", "J", "L", "O", "T", "A", "B", "C", "D", "E")
AMBIGUOUS_NAME_NUMBERS = (
    1,
    2,
    3,
    8,
    9,
    10,
    11,
    12,
    15,
    18,
    19,
    20,
    21,
    22,
    28,
    30,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic english-name/address hard-case crops."
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
    parser.add_argument("--variants-per-entry", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def _load_manifest_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _build_first_name(rng: random.Random) -> str:
    number = rng.choice(AMBIGUOUS_NAME_NUMBERS)
    return f"SAMPLE{number}"


def _build_middle_name(rng: random.Random) -> str:
    return rng.choice(AMBIGUOUS_INITIALS)


def _build_address(rng: random.Random) -> str:
    suite = rng.choice(AMBIGUOUS_NAME_NUMBERS)
    street_number = suite if rng.random() < 0.8 else rng.choice(AMBIGUOUS_NAME_NUMBERS)
    street = rng.choice(("Main Street", "Harbor Road", "Broadway", "Market Street"))
    city, state, postal_code = rng.choice(
        (
            ("New York", "NY", "10001"),
            ("Los Angeles", "CA", "90026"),
            ("Seattle", "WA", "98101"),
            ("Chicago", "IL", "60601"),
        )
    )
    return (
        f"{street_number} {street} Suite {suite} "
        f"{city} {state} {postal_code} United States of America"
    )


def _build_synthetic_text(field_name: str, rng: random.Random) -> tuple[str, str]:
    if field_name == "first_name":
        text = _build_first_name(rng)
        return text, text
    if field_name == "middle_name":
        text = _build_middle_name(rng)
        return text, text
    if field_name == "applicant_name":
        first_name = _build_first_name(rng)
        middle_name = _build_middle_name(rng)
        label_text = f"{first_name} {middle_name} USER"
        render_text = rng.choice(
            (
                label_text,
                f"{first_name}\n{middle_name}\nUSER",
                f"{first_name}\nUSER\n{middle_name}",
            )
        )
        return label_text, render_text
    if field_name == "signed_by":
        text = f"NOTARY SAMPLE {rng.choice(AMBIGUOUS_NAME_NUMBERS)}"
        return text, text
    if field_name == "taxpayer_name":
        number = rng.choice(AMBIGUOUS_NAME_NUMBERS)
        label_text = f"Sample {number} User"
        render_text = rng.choice((label_text, f"Sample {number}\nUser"))
        return label_text, render_text
    text = _build_address(rng)
    return text, text


def _load_font(height: int, rng: random.Random) -> ImageFont.ImageFont:
    base_size = max(18, int(height * 0.5))
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6):
        font_size = max(12, int(base_size * scale))
        candidates = [path for path in FONT_PATHS if path.exists()]
        rng.shuffle(candidates)
        for path in candidates:
            try:
                return ImageFont.truetype(str(path), size=font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def _measure_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    *,
    multiline: bool = False,
) -> tuple[int, int]:
    if multiline:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap_address_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    width: int,
) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        candidate_width, _ = _measure_text(draw, candidate, font)
        if current and candidate_width > max(24, width - 12):
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _fit_text(
    field_name: str,
    text: str,
    size: tuple[int, int],
    rng: random.Random,
) -> tuple[ImageFont.ImageFont, str]:
    probe = ImageDraw.Draw(Image.new("RGB", size, "white"))
    for scale in (1.0, 0.92, 0.84, 0.76, 0.68, 0.6):
        font = _load_font(int(size[1] * scale), rng)
        rendered_text = text
        multiline = False
        if field_name == "address":
            rendered_text = _wrap_address_text(probe, text, font, size[0])
            multiline = True
        width, height = _measure_text(probe, rendered_text, font, multiline=multiline)
        if width <= size[0] - 8 and height <= size[1] - 6:
            return font, rendered_text
    return ImageFont.load_default(), text


def _render_synthetic_crop(
    base_entry: dict[str, Any],
    render_text: str,
    rng: random.Random,
) -> Image.Image:
    base_crop = Image.open(base_entry["crop_path"]).convert("RGB")
    background = base_crop.convert("L").convert("RGB")
    background = Image.blend(background, Image.new("RGB", background.size, "white"), 0.88)
    if rng.random() < 0.4:
        background = background.filter(ImageFilter.GaussianBlur(radius=0.5))

    field_name = str(base_entry["field_name"])
    font, rendered_text = _fit_text(field_name, render_text, background.size, rng)
    draw = ImageDraw.Draw(background)
    multiline = "\n" in rendered_text
    text_width, text_height = _measure_text(draw, rendered_text, font, multiline=multiline)
    x_margin = max(4, int(background.width * 0.04))
    y_margin = max(2, int(background.height * 0.12))
    max_x = max(x_margin, background.width - text_width - x_margin)
    max_y = max(y_margin, background.height - text_height - y_margin)
    x = rng.randint(x_margin, max_x)
    y = rng.randint(y_margin, max_y)
    ink = rng.randint(12, 48)
    if multiline:
        draw.multiline_text((x, y), rendered_text, fill=(ink, ink, ink), font=font, spacing=4)
    else:
        draw.text((x, y), rendered_text, fill=(ink, ink, ink), font=font)

    if rng.random() < 0.35:
        background = background.rotate(
            rng.uniform(-2.0, 2.0),
            resample=Image.Resampling.BILINEAR,
            fillcolor="white",
        )
    if rng.random() < 0.45:
        resized = background.resize(
            (
                max(1, int(background.width * 0.72)),
                max(1, int(background.height * 0.72)),
            ),
            resample=Image.Resampling.BILINEAR,
        )
        background = resized.resize(background.size, resample=Image.Resampling.BILINEAR)
    if rng.random() < 0.4:
        background = background.filter(ImageFilter.GaussianBlur(radius=0.35))
    return background


def generate_synthetic_english_name_hard_cases(
    field_crops_root: Path,
    output_root: Path,
    *,
    variants_per_entry: int = 6,
    seed: int = 20260701,
) -> dict[str, Any]:
    rng = random.Random(seed)
    field_manifest = _load_manifest_entries(field_crops_root / "manifest.jsonl")
    base_entries = [
        entry
        for entry in field_manifest
        if entry.get("split") == "train"
        and entry.get("field_group") == "english_name_org"
        and entry.get("field_name") in TARGET_FIELD_NAMES
        and entry.get("quality", {}).get("accepted", True)
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    existing_entries = [
        entry
        for entry in _load_manifest_entries(manifest_path)
        if not str(entry.get("augmentation_type") or "").startswith(
            SYNTHETIC_AUGMENTATION_PREFIX
        )
    ]

    synthetic_entries: list[dict[str, Any]] = []
    counts_by_field_name: dict[str, int] = {field_name: 0 for field_name in TARGET_FIELD_NAMES}
    variant_dir = output_root / "english_name_org" / SYNTHETIC_AUGMENTATION_PREFIX
    variant_dir.mkdir(parents=True, exist_ok=True)

    for entry in base_entries:
        field_name = str(entry["field_name"])
        base_path = Path(entry["crop_path"])
        for variant_index in range(variants_per_entry):
            label_text, render_text = _build_synthetic_text(field_name, rng)
            image = _render_synthetic_crop(entry, render_text, rng)
            output_path = variant_dir / (
                f"{base_path.stem}__{SYNTHETIC_AUGMENTATION_PREFIX}_{variant_index:02d}.png"
            )
            image.save(output_path, format="PNG")
            synthetic_entries.append(
                {
                    **entry,
                    "text": label_text,
                    "recognizer_text": label_text,
                    "render_text": render_text,
                    "crop_path": str(output_path),
                    "base_crop_path": str(base_path),
                    "augmentation_type": SYNTHETIC_AUGMENTATION_PREFIX,
                    "quality": {
                        "accepted": True,
                        "synthetic": True,
                        "width": image.width,
                        "height": image.height,
                    },
                }
            )
            counts_by_field_name[field_name] += 1

    manifest_entries = existing_entries + synthetic_entries
    manifest_path.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in manifest_entries)
        + ("\n" if manifest_entries else ""),
        encoding="utf-8",
    )

    summary = {
        "field_crops_root": str(field_crops_root),
        "output_root": str(output_root),
        "base_entry_count": len(base_entries),
        "synthetic_entry_count": len(synthetic_entries),
        "variants_per_entry": variants_per_entry,
        "counts_by_field_name": {
            field_name: count
            for field_name, count in sorted(counts_by_field_name.items())
            if count > 0
        },
    }
    (output_root / "synthetic_english_name_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = generate_synthetic_english_name_hard_cases(
        args.field_crops_root,
        args.output_root,
        variants_per_entry=args.variants_per_entry,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
