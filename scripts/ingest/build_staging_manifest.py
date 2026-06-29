from __future__ import annotations

import argparse
import json
from pathlib import Path

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index staged OCR input files into a JSONL manifest."
    )
    parser.add_argument("--source", type=Path, default=Path("data/staging"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/staging/index/staging_manifest.jsonl"),
    )
    return parser.parse_args()


def list_supported_files(source: Path) -> list[Path]:
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def build_manifest(source: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in list_supported_files(source):
        entries.append(
            {
                "path": str(path),
                "filename": path.name,
                "document_group": path.parent.name,
            }
        )
    return entries


def write_jsonl(output: Path, entries: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    entries = build_manifest(args.source)
    write_jsonl(args.output, entries)
    print(json.dumps({"entries": len(entries), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
