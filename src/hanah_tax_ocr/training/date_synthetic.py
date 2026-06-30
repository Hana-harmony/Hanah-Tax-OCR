from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from hanah_tax_ocr.training.recognizer_labels import recognizer_text_for_entry

DEFAULT_FIELD_CROPS_ROOT = Path("data/training/field_crops")
DEFAULT_OUTPUT_ROOT = Path("data/training/hard_cases")
DATE_FIELD_NAMES = {"issue_date", "signature_date", "issued_on"}
SYNTHETIC_AUGMENTATION_PREFIX = "synthetic_date"
RENDER_VARIANTS = ("plain", "context_header", "context_footer", "underline_noise", "left_clip")
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
FONT_PATHS = {
    "issue_date": [
        Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
        Path("/System/Library/Fonts/Times.ttc"),
    ],
    "issued_on": [
        Path("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
        Path("/System/Library/Fonts/Times.ttc"),
    ],
    "signature_date": [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
    ],
}
CONTEXT_NOISE_SNIPPETS = {
    "issue_date": ("CERTIFICATION", "TIN:", "Tax Year: 2026"),
    "signature_date": ("신청인", "SIGNATURE", "DATE"),
    "issued_on": ("6. the", "Done at", "No. 5001"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic date hard-case crops for recognizer tuning."
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
    parser.add_argument("--variants-per-entry", type=int, default=8)
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


def _ordinal_suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "TH"
    return {1: "ST", 2: "ND", 3: "RD"}.get(day % 10, "TH")


def _build_synthetic_text(field_name: str, rng: random.Random) -> str:
    year = rng.randint(2014, 2029)
    month = rng.choice(MONTHS)
    month_index = MONTHS.index(month) + 1
    day = rng.randint(1, 28)

    if field_name == "issued_on":
        return f"{day}{_ordinal_suffix(day)} DAY OF {month.upper()}, {year}"
    if field_name == "signature_date":
        format_name = rng.choices(
            ("iso", "dots", "korean"),
            weights=(0.6, 0.25, 0.15),
            k=1,
        )[0]
        if format_name == "dots":
            return f"{year}.{month_index:02d}.{day:02d}"
        if format_name == "korean":
            return f"{year}년 {month_index:02d}월 {day:02d}일"
        return f"{year}-{month_index:02d}-{day:02d}"
    return f"{month} {day}, {year}"


def _load_font(field_name: str, height: int, rng: random.Random) -> ImageFont.ImageFont:
    font_candidates = [path for path in FONT_PATHS.get(field_name, []) if path.exists()]
    base_size = max(18, int(height * 0.44))
    if font_candidates:
        font_path = rng.choice(font_candidates)
        font_size = base_size
        while font_size >= 12:
            try:
                return ImageFont.truetype(str(font_path), size=font_size)
            except OSError:
                font_size -= 2
    return ImageFont.load_default()


def _fit_font(
    field_name: str,
    text: str,
    image_size: tuple[int, int],
    rng: random.Random,
) -> ImageFont.ImageFont:
    width, height = image_size
    for scale in (1.0, 0.92, 0.84, 0.76, 0.68, 0.6):
        font = _load_font(field_name, int(height * scale), rng)
        probe = ImageDraw.Draw(Image.new("RGB", image_size, "white"))
        bbox = probe.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= width - 12 and text_height <= height - 8:
            return font
    return ImageFont.load_default()


def _draw_context_noise(
    image: Image.Image,
    field_name: str,
    rng: random.Random,
    *,
    position: str,
) -> None:
    snippets = CONTEXT_NOISE_SNIPPETS.get(field_name, ())
    if not snippets:
        return
    draw = ImageDraw.Draw(image)
    snippet = rng.choice(snippets)
    font = _load_font(field_name, max(12, int(image.height * 0.22)), rng)
    bbox = draw.textbbox((0, 0), snippet, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = rng.randint(2, max(2, image.width - text_width - 2))
    if position == "header":
        y_min = 0
        y_max = max(0, min(int(image.height * 0.08), image.height - text_height))
    else:
        y_min = max(0, min(int(image.height * 0.72), image.height - text_height))
        y_max = max(y_min, image.height - text_height)
    y = rng.randint(y_min, y_max)
    ink = rng.randint(120, 168)
    draw.text((x, y), snippet, fill=(ink, ink, ink), font=font)


def _render_synthetic_crop(
    base_entry: dict[str, Any],
    recognizer_text: str,
    rng: random.Random,
) -> tuple[Image.Image, str]:
    field_name = str(base_entry["field_name"])
    base_crop = Image.open(base_entry["crop_path"]).convert("RGB")
    background = base_crop.convert("L").convert("RGB")
    background = Image.blend(background, Image.new("RGB", background.size, "white"), 0.9)
    if rng.random() < 0.6:
        background = background.filter(ImageFilter.GaussianBlur(radius=0.6))
    render_variant = rng.choices(
        RENDER_VARIANTS,
        weights=(0.2, 0.24, 0.22, 0.18, 0.16),
        k=1,
    )[0]

    font = _fit_font(field_name, recognizer_text, background.size, rng)
    draw = ImageDraw.Draw(background)
    bbox = draw.textbbox((0, 0), recognizer_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x_margin = max(4, int(background.width * 0.04))
    y_margin = max(2, int(background.height * 0.12))
    max_x = max(x_margin, background.width - text_width - x_margin)
    max_y = max(y_margin, background.height - text_height - y_margin)
    x = rng.randint(x_margin, max_x)
    y = rng.randint(y_margin, max_y)
    if render_variant == "left_clip":
        x = max(0, x - rng.randint(4, max(4, int(background.width * 0.08))))
    ink = rng.randint(12, 48)
    draw.text((x, y), recognizer_text, fill=(ink, ink, ink), font=font)

    if render_variant == "context_header":
        _draw_context_noise(background, field_name, rng, position="header")
    elif render_variant == "context_footer":
        _draw_context_noise(background, field_name, rng, position="footer")
    elif render_variant == "underline_noise":
        underline_y = min(background.height - 2, y + text_height + rng.randint(1, 4))
        draw.line(
            (
                max(0, x - 2),
                underline_y,
                min(background.width - 1, x + text_width + 2),
                underline_y,
            ),
            fill=(rng.randint(90, 136),) * 3,
            width=1,
        )

    if rng.random() < 0.35:
        background = background.rotate(
            rng.uniform(-2.2, 2.2),
            resample=Image.Resampling.BILINEAR,
            fillcolor="white",
        )
    if rng.random() < 0.4:
        resized = background.resize(
            (
                max(1, int(background.width * 0.72)),
                max(1, int(background.height * 0.72)),
            ),
            resample=Image.Resampling.BILINEAR,
        )
        background = resized.resize(background.size, resample=Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        background = background.filter(ImageFilter.GaussianBlur(radius=0.4))
    return background, render_variant


def generate_synthetic_date_hard_cases(
    field_crops_root: Path,
    output_root: Path,
    *,
    variants_per_entry: int = 8,
    seed: int = 20260701,
) -> dict[str, Any]:
    rng = random.Random(seed)
    field_manifest = _load_manifest_entries(field_crops_root / "manifest.jsonl")
    base_entries = [
        entry
        for entry in field_manifest
        if entry.get("split") == "train"
        and entry.get("field_group") == "date"
        and entry.get("field_name") in DATE_FIELD_NAMES
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
    counts_by_field_name: dict[str, int] = {field_name: 0 for field_name in DATE_FIELD_NAMES}
    counts_by_variant: dict[str, int] = {variant: 0 for variant in RENDER_VARIANTS}
    variant_dir = output_root / "date" / SYNTHETIC_AUGMENTATION_PREFIX
    variant_dir.mkdir(parents=True, exist_ok=True)

    for entry in base_entries:
        field_name = str(entry["field_name"])
        base_path = Path(entry["crop_path"])
        for variant_index in range(variants_per_entry):
            text = _build_synthetic_text(field_name, rng)
            recognizer_text = recognizer_text_for_entry(
                {
                    "field_name": field_name,
                    "text": text,
                }
            )
            image, render_variant = _render_synthetic_crop(entry, recognizer_text, rng)
            output_path = variant_dir / (
                f"{base_path.stem}__{SYNTHETIC_AUGMENTATION_PREFIX}_{variant_index:02d}.png"
            )
            image.save(output_path, format="PNG")
            synthetic_entries.append(
                {
                    **entry,
                    "text": text,
                    "recognizer_text": recognizer_text,
                    "crop_path": str(output_path),
                    "base_crop_path": str(base_path),
                    "augmentation_type": f"{SYNTHETIC_AUGMENTATION_PREFIX}.{render_variant}",
                    "render_variant": render_variant,
                    "quality": {
                        "accepted": True,
                        "synthetic": True,
                        "width": image.width,
                        "height": image.height,
                    },
                }
            )
            counts_by_field_name[field_name] += 1
            counts_by_variant[render_variant] += 1

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
        "counts_by_variant": {
            variant: count for variant, count in sorted(counts_by_variant.items()) if count > 0
        },
    }
    (output_root / "synthetic_date_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = generate_synthetic_date_hard_cases(
        args.field_crops_root,
        args.output_root,
        variants_per_entry=args.variants_per_entry,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
