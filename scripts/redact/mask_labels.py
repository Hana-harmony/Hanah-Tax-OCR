from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EIN_PATTERN = re.compile(r"\b\d{2}-\d{7}\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redact sensitive values from label JSON.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        value = SSN_PATTERN.sub("***-**-****", value)
        value = EIN_PATTERN.sub("**-*******", value)
        return value
    return value


def redact_file(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    redacted = redact_value(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(redacted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    redact_file(args.input, args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
