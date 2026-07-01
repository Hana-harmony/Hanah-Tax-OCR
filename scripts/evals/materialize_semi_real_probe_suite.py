from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hanah_tax_ocr.training.hard_cases import (
    _apply_border_clip,
    _apply_edge_overlap,
    _apply_gaussian_blur,
    _apply_jpeg_blocking,
    _apply_left_clip,
    _apply_low_res,
    _apply_overlay_patch,
    _apply_rotate,
    _apply_stamp_shadow,
)
from PIL import Image

DEFAULT_MANIFEST_PATH = Path("evals/semi_real_probes/manifest.json")
DEFAULT_OUTPUT_ROOT = Path("tmp/semi_real_probe_suite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize document-level semi-real OCR probe suites "
            "from real labeled samples."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


VariantHandler = Callable[[Image.Image, list[dict[str, Any]], random.Random], Image.Image]


def _variant_handlers() -> dict[str, VariantHandler]:
    return {
        "border_clip": lambda image, _donors, rng: _apply_border_clip(image, rng),
        "edge_overlap": lambda image, donors, rng: _apply_edge_overlap(image, donors, rng),
        "gaussian_blur": lambda image, _donors, _rng: _apply_gaussian_blur(image),
        "jpeg_blocking": lambda image, _donors, _rng: _apply_jpeg_blocking(image),
        "left_clip": lambda image, _donors, _rng: _apply_left_clip(image),
        "low_res": lambda image, _donors, _rng: _apply_low_res(image),
        "overlay_patch": lambda image, donors, rng: _apply_overlay_patch(image, donors, rng),
        "rotate": lambda image, _donors, _rng: _apply_rotate(image),
        "stamp_shadow": lambda image, _donors, rng: _apply_stamp_shadow(image, rng),
    }


def _load_base_label(probe: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    base_label_path = probe.get("base_label_path")
    if base_label_path:
        path = Path(str(base_label_path))
    else:
        document_type = str(probe["document_type"])
        base_case_id = str(probe["base_case_id"])
        path = Path("data/labeled") / document_type / base_case_id / "label.json"
    payload = load_json(path)
    return payload, path


def materialize_probe_suite(
    manifest_path: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    handlers = _variant_handlers()
    rng = random.Random(seed)

    assets_root = output_root / "assets"
    labeled_root = output_root / "labeled"
    cases_root = output_root / "cases"
    assets_root.mkdir(parents=True, exist_ok=True)
    labeled_root.mkdir(parents=True, exist_ok=True)
    cases_root.mkdir(parents=True, exist_ok=True)

    materialized: list[dict[str, Any]] = []
    for probe in manifest.get("probes", []):
        case_id = str(probe["case_id"])
        augmentation_type = str(probe["augmentation_type"])
        handler = handlers[augmentation_type]
        base_label, base_label_path = _load_base_label(probe)

        source_path = Path(str(probe.get("source_path") or base_label["source_path"]))
        if not source_path.is_file():
            raise FileNotFoundError(f"Probe source does not exist: {source_path}")

        donor_source_paths = [str(path) for path in probe.get("donor_source_paths", [])]
        donors = [{"crop_path": path} for path in donor_source_paths] or [
            {"crop_path": str(source_path)}
        ]
        image = Image.open(source_path).convert("RGB")
        augmented = handler(image, donors, rng)

        asset_path = assets_root / f"{case_id}{source_path.suffix.lower() or '.png'}"
        augmented.save(asset_path)

        document_type = str(probe.get("document_type") or base_label["document_type"])
        expected_fields = dict(
            probe.get("expected_fields") or base_label.get("expected_fields") or {}
        )
        expected_quality_checks = dict(
            probe.get("expected_quality_checks") or base_label.get("expected_quality_checks") or {}
        )
        expected_status = str(
            probe.get("expected_status") or base_label.get("expected_status") or "pass"
        )

        label_payload = {
            "case_id": case_id,
            "document_type": document_type,
            "source_path": str(asset_path),
            "dataset_split": "semi_real_probe",
            "expected_status": expected_status,
            "expected_fields": expected_fields,
            "expected_quality_checks": expected_quality_checks,
            "augmentation_type": augmentation_type,
            "base_case_id": probe.get("base_case_id"),
            "base_label_path": str(base_label_path),
            "failure_modes": list(probe.get("failure_modes", [])),
            "focus_fields": list(probe.get("focus_fields", [])),
            "notes": list(probe.get("notes", [])),
        }
        label_path = labeled_root / document_type / case_id / "label.json"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            json.dumps(label_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        expected_payload = {
            "case_id": case_id,
            "document_type": document_type,
            "expected_status": expected_status,
            "expected_fields": expected_fields,
            "expected_quality_checks": expected_quality_checks,
            "augmentation_type": augmentation_type,
            "base_case_id": probe.get("base_case_id"),
            "failure_modes": list(probe.get("failure_modes", [])),
            "focus_fields": list(probe.get("focus_fields", [])),
            "notes": list(probe.get("notes", [])),
        }
        expected_path = cases_root / case_id / "expected.json"
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(
            json.dumps(expected_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        materialized.append(
            {
                "case_id": case_id,
                "document_type": document_type,
                "asset_path": str(asset_path),
                "label_path": str(label_path),
                "expected_path": str(expected_path),
                "augmentation_type": augmentation_type,
            }
        )

    summary = {
        "version": str(manifest.get("version") or "2026-07-01"),
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "probe_count": len(materialized),
        "materialized": materialized,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = materialize_probe_suite(args.manifest, args.output_root, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
