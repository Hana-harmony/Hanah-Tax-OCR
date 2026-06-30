from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from hanah_tax_ocr.training.sample_dataset import SAMPLE_DATASET


def bootstrap_sample_dataset(
    sample_root: str | Path = "sample_data",
    output_root: str | Path = "data/raw",
) -> list[dict[str, str]]:
    sample_root = Path(sample_root)
    output_root = Path(output_root)
    copied: list[dict[str, str]] = []

    for entry in SAMPLE_DATASET:
        source = Path(entry["source"])
        if not source.is_absolute():
            source = sample_root.parent / source
        if not source.exists():
            raise FileNotFoundError(f"Sample file does not exist: {source}")

        destination = (
            output_root
            / entry["split"]
            / entry["document_type"]
            / f"{entry['case_id']}{source.suffix.lower()}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source),
                "destination": str(destination),
                "split": entry["split"],
                "document_type": entry["document_type"],
                "case_id": entry["case_id"],
            }
        )

    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy tracked sample_data files into data/raw.")
    parser.add_argument("--sample-root", type=Path, default=Path("sample_data"))
    parser.add_argument("--output-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/raw_index.jsonl"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    copied = bootstrap_sample_dataset(args.sample_root, args.output_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in copied),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "copied_count": len(copied),
                "output_root": str(args.output_root),
                "manifest": str(args.manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
