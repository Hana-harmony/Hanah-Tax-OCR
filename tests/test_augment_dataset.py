from pathlib import Path

from PIL import Image

from scripts.augment_dataset import augment_dataset, load_config


def test_augment_dataset_creates_expected_manifest_entries(tmp_path: Path) -> None:
    source_root = tmp_path / "raw" / "train" / "residency_certificate"
    source_root.mkdir(parents=True)
    image_path = source_root / "sample.png"
    Image.new("RGB", (64, 64), "white").save(image_path)

    output_root = tmp_path / "augmented" / "train"
    config_path = tmp_path / "augmentation.yaml"
    config_path.write_text(
        """
seed: 1
target_variants_per_image: 2
preserve_raw_manifest: true
datasets:
  train:
    source_root: RAW_ROOT
    output_root: OUTPUT_ROOT
    document_types:
      residency_certificate:
        enabled: true
        rotations: [0.0]
        brightness: [1.0]
        contrast: [1.0]
        blur_radius: [0.0]
        jpeg_quality: [90]
        resize_scale: [1.0]
        crop_ratio: [0.0]
        noise_stddev: [0.0]
""".replace("RAW_ROOT", str(tmp_path / "raw" / "train")).replace("OUTPUT_ROOT", str(output_root)),
        encoding="utf-8",
    )

    config = load_config(config_path)
    raw_manifest, augmented_manifest = augment_dataset(config)

    assert len(raw_manifest) == 1
    assert len(augmented_manifest) == 2
    assert all(Path(entry["path"]).exists() for entry in augmented_manifest)
