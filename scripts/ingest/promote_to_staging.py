from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy selected raw files into the OCR staging area."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--document-type", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/staging"))
    return parser.parse_args()


def promote_to_staging(
    source: Path,
    document_type: str,
    output_root: Path,
) -> list[Path]:
    destination_dir = output_root / document_type
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        destination = destination_dir / path.name
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def main() -> None:
    args = parse_args()
    copied = promote_to_staging(args.source, args.document_type, args.output_root)
    print(
        json.dumps(
            {"copied": len(copied), "output_dir": str(args.output_root / args.document_type)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
