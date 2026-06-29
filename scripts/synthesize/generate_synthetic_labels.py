from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from hanah_tax_ocr.schemas import DocumentType, ReviewStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lightweight synthetic label JSON fixtures."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/labeled/index/synthetic"),
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


def build_case(case_id: int, rng: random.Random) -> dict[str, object]:
    first_name = f"Sample{case_id}"
    last_name = "User"
    tin_prefix = rng.randint(100, 999)
    tin_mid = rng.randint(10, 99)
    tin_suffix = rng.randint(1000, 9999)
    tin = f"{tin_prefix}-{tin_mid}-{tin_suffix}"
    return {
        "case_id": f"synthetic_{case_id:03d}",
        "document_type": DocumentType.WITHHOLDING_TAX_FORM,
        "expected_status": ReviewStatus.PASS,
        "fields": {
            "first_name": first_name,
            "last_name": last_name,
            "tin": tin,
            "address": f"{case_id} Main Street",
            "residency_country": "United States of America",
            "residency_country_code": "US",
            "dividend_tax_rate": "15%",
        },
    }


def write_cases(output_dir: Path, count: int, seed: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written: list[Path] = []
    for case_id in range(1, count + 1):
        payload = build_case(case_id, rng)
        output_path = output_dir / f"{payload['case_id']}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(output_path)
    return written


def main() -> None:
    args = parse_args()
    written = write_cases(args.output_dir, args.count, args.seed)
    print(
        json.dumps(
            {"generated": len(written), "output_dir": str(args.output_dir)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
