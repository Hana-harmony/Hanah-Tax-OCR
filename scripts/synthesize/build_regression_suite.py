from __future__ import annotations

import argparse
import json
from pathlib import Path

from hanah_tax_ocr.schemas import DocumentType, ReviewStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic labeled/eval regression suites for parser "
            "normalization coverage."
        )
    )
    parser.add_argument(
        "--labeled-root",
        type=Path,
        default=Path("data/labeled"),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("evals/cases"),
    )
    parser.add_argument("--per-document", type=int, default=20)
    return parser.parse_args()


def generate_regression_suite(
    labeled_root: Path,
    eval_root: Path,
    *,
    per_document: int,
) -> list[Path]:
    written: list[Path] = []
    for index in range(1, per_document + 1):
        written.extend(
            _write_case(
                labeled_root,
                eval_root,
                case_id=f"residency_regression_{index:03d}",
                document_type=DocumentType.RESIDENCY_CERTIFICATE,
                expected_status=ReviewStatus.PASS,
                expected_fields={
                    "taxpayer_name": f"Sample {index} User",
                    "tin": f"{100 + index:03d}-{10 + index % 80:02d}-{1000 + index:04d}",
                    "tax_year": f"20{10 + index % 10:02d}",
                    "issue_date": f"January {1 + index % 28}, 2026",
                    "residency_country": "United States of America",
                    "residency_country_code": "US",
                },
                expected_quality_checks={
                    "seal_present": True,
                    "signature_present": True,
                },
            )
        )
        written.extend(
            _write_case(
                labeled_root,
                eval_root,
                case_id=f"withholding_regression_{index:03d}",
                document_type=DocumentType.WITHHOLDING_TAX_FORM,
                expected_status=ReviewStatus.PASS,
                expected_fields={
                    "first_name": f"SAMPLE{index}",
                    "last_name": "USER",
                    "middle_name": chr(64 + ((index - 1) % 20) + 1),
                    "tin": f"{200 + index:03d}-{20 + index % 70:02d}-{2000 + index:04d}",
                    "address": (
                        f"{index} Main Street Suite {index} "
                        "New York NY 10001 United States of America"
                    ),
                    "residency_country": "United States of America",
                    "residency_country_code": "US",
                    "dividend_tax_rate": "15%",
                    "signature_date": f"2026-01-{1 + index % 28:02d}",
                    "applicant_name": f"SAMPLE{index} {chr(64 + ((index - 1) % 20) + 1)} USER",
                },
                expected_quality_checks={
                    "all_no_boxes_checked": True,
                    "signature_present": True,
                },
            )
        )
        written.extend(
            _write_case(
                labeled_root,
                eval_root,
                case_id=f"apostille_regression_{index:03d}",
                document_type=DocumentType.APOSTILLE,
                expected_status=ReviewStatus.PASS,
                expected_fields={
                    "issuing_country": "UNITED STATES OF AMERICA",
                    "signed_by": f"NOTARY SAMPLE {index}",
                    "signer_capacity": "NOTARY PUBLIC",
                    "seal_owner": "COUNTY OF SAMPLE, STATE OF SAMPLE",
                    "issued_at": f"Capital City, Sample State {index}",
                    "issued_on": f"{1 + index % 28}TH DAY OF APRIL, 202{index % 10}",
                    "issuing_authority": "Secretary of State State of Sample",
                    "certificate_number": str(5000 + index),
                },
                expected_quality_checks={
                    "seal_present": True,
                    "signature_present": True,
                },
            )
        )
    return written


def _write_case(
    labeled_root: Path,
    eval_root: Path,
    *,
    case_id: str,
    document_type: DocumentType,
    expected_status: ReviewStatus,
    expected_fields: dict[str, object],
    expected_quality_checks: dict[str, object],
) -> list[Path]:
    label_dir = labeled_root / document_type.value / case_id
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / "label.json"
    label_payload = {
        "case_id": case_id,
        "document_type": document_type.value,
        "source_path": f"synthetic://{document_type.value}/{case_id}",
        "dataset_split": "synthetic_regression",
        "expected_status": expected_status.value,
        "expected_fields": expected_fields,
        "expected_quality_checks": expected_quality_checks,
        "synthetic": True,
    }
    label_path.write_text(json.dumps(label_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_dir = eval_root / case_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "expected.json"
    eval_payload = {
        "case_id": case_id,
        "document_type": document_type.value,
        "expected_status": expected_status.value,
        "expected_fields": expected_fields,
        "expected_quality_checks": expected_quality_checks,
        "synthetic": True,
    }
    eval_path.write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return [label_path, eval_path]


def main() -> None:
    args = parse_args()
    written = generate_regression_suite(
        args.labeled_root,
        args.eval_root,
        per_document=args.per_document,
    )
    print(
        json.dumps(
            {
                "generated_files": len(written),
                "per_document": args.per_document,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
