from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment OCR training images and build manifests."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to augmentation config YAML.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def list_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)


def build_raw_manifest(split_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for image_path in list_images(split_root):
        entries.append(
            {
                "path": str(image_path),
                "document_type": image_path.parent.name,
                "split": image_path.parents[1].name,
                "source_file": image_path.name,
            }
        )
    return entries


def apply_transform(
    image: Image.Image,
    spec: dict[str, Any],
    rng: random.Random,
) -> tuple[Image.Image, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    transformed = image.copy().convert("RGB")

    rotation = rng.choice(spec["rotations"])
    metadata["rotation"] = rotation
    transformed = transformed.rotate(rotation, expand=True, fillcolor="white")

    brightness = rng.choice(spec["brightness"])
    metadata["brightness"] = brightness
    transformed = ImageEnhance.Brightness(transformed).enhance(brightness)

    contrast = rng.choice(spec["contrast"])
    metadata["contrast"] = contrast
    transformed = ImageEnhance.Contrast(transformed).enhance(contrast)

    blur_radius = rng.choice(spec["blur_radius"])
    metadata["blur_radius"] = blur_radius
    if blur_radius > 0:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    resize_scale = rng.choice(spec["resize_scale"])
    metadata["resize_scale"] = resize_scale
    if resize_scale != 1.0:
        resized = (
            max(1, int(transformed.width * resize_scale)),
            max(1, int(transformed.height * resize_scale)),
        )
        transformed = transformed.resize(resized)

    crop_ratio = rng.choice(spec["crop_ratio"])
    metadata["crop_ratio"] = crop_ratio
    if crop_ratio > 0:
        crop_x = int(transformed.width * crop_ratio)
        crop_y = int(transformed.height * crop_ratio)
        transformed = transformed.crop(
            (
                crop_x,
                crop_y,
                max(crop_x + 1, transformed.width - crop_x),
                max(crop_y + 1, transformed.height - crop_y),
            )
        )

    noise_stddev = rng.choice(spec["noise_stddev"])
    metadata["noise_stddev"] = noise_stddev
    if noise_stddev > 0:
        array = np.array(transformed).astype(np.float32)
        noise = np.random.normal(0.0, noise_stddev, array.shape)
        array = np.clip(array + noise, 0, 255).astype(np.uint8)
        transformed = Image.fromarray(array, mode="RGB")

    jpeg_quality = int(rng.choice(spec["jpeg_quality"]))
    metadata["jpeg_quality"] = jpeg_quality
    buffer = io.BytesIO()
    transformed.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    transformed = Image.open(buffer).convert("RGB")
    return transformed, metadata


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def augment_dataset(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config["seed"])
    rng = random.Random(seed)
    np.random.seed(seed)

    raw_manifest: list[dict[str, Any]] = []
    augmented_manifest: list[dict[str, Any]] = []

    for split_name, split_config in config["datasets"].items():
        source_root = Path(split_config["source_root"])
        if not source_root.exists():
            continue

        if config.get("preserve_raw_manifest", True):
            raw_manifest.extend(build_raw_manifest(source_root))

        for document_type, document_config in split_config["document_types"].items():
            if not document_config.get("enabled", False):
                continue

            input_dir = source_root / document_type
            output_dir = Path(split_config["output_root"]) / document_type
            output_dir.mkdir(parents=True, exist_ok=True)

            for image_path in list_images(input_dir):
                with Image.open(image_path) as image:
                    for variant_index in range(int(config["target_variants_per_image"])):
                        transformed, metadata = apply_transform(image, document_config, rng)
                        output_path = output_dir / f"{image_path.stem}__aug_{variant_index:02d}.jpg"
                        transformed.save(output_path, format="JPEG", quality=95)
                        augmented_manifest.append(
                            {
                                "path": str(output_path),
                                "document_type": document_type,
                                "split": split_name,
                                "source_path": str(image_path),
                                "augmentation": metadata,
                            }
                        )

    return raw_manifest, augmented_manifest


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_manifest, augmented_manifest = augment_dataset(config)
    write_jsonl(Path("data/manifests/raw_index.jsonl"), raw_manifest)
    write_jsonl(Path("data/manifests/augmented_index.jsonl"), augmented_manifest)
    print(
        json.dumps(
            {
                "raw_entries": len(raw_manifest),
                "augmented_entries": len(augmented_manifest),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
