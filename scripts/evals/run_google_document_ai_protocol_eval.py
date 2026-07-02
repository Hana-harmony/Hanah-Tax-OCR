from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hanah_tax_ocr.evaluation import (
    build_field_error_report,
    compare_field_error_report_files,
)
from hanah_tax_ocr.harness import HarnessRunResult
from hanah_tax_ocr.normalization import (
    normalize_address,
    normalize_apostille_authority,
    normalize_apostille_date,
    normalize_country,
    normalize_country_code,
    normalize_english_date,
    normalize_iso_date,
    normalize_name,
    normalize_percentage,
    normalize_whitespace,
)
from hanah_tax_ocr.schemas import ExtractedDocument, ReviewResult, ReviewStatus

from scripts.evals.summarize_eval_report import summarize_eval_report

DEFAULT_CONTRACT_PATH = Path(
    "evals/sota_positioning/google_document_ai_custom_extractor_contract.json"
)
DEFAULT_EXTERNAL_MANIFEST_PATH = Path("evals/external_holdout/manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Google Document AI Custom Extractor raw JSON outputs into this "
            "repository's harness result schema and score them with the standard eval protocol."
        )
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--external-manifest", type=Path, default=DEFAULT_EXTERNAL_MANIFEST_PATH)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    return parser.parse_args()


Normalizer = Callable[[str | None], str | None]


FIELD_NORMALIZERS: dict[str, Normalizer] = {
    "taxpayer_name": normalize_name,
    "tin": normalize_whitespace,
    "tax_year": normalize_whitespace,
    "issue_date": normalize_english_date,
    "residency_country": normalize_country,
    "residency_country_code": normalize_country_code,
    "first_name": normalize_name,
    "last_name": normalize_name,
    "middle_name": normalize_name,
    "address": normalize_address,
    "dividend_tax_rate": normalize_percentage,
    "signature_date": normalize_iso_date,
    "applicant_name": normalize_name,
    "issuing_country": lambda value: (
        normalize_country(value).upper() if normalize_country(value) else None
    ),
    "signed_by": normalize_name,
    "signer_capacity": normalize_whitespace,
    "seal_owner": normalize_whitespace,
    "issued_at": normalize_whitespace,
    "issued_on": normalize_apostille_date,
    "issuing_authority": normalize_apostille_authority,
    "certificate_number": normalize_whitespace,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entity_type_alias_map(contract: dict[str, Any], document_type: str) -> dict[str, str]:
    aliases_by_field = dict((contract.get("document_type_aliases") or {}).get(document_type) or {})
    alias_map: dict[str, str] = {}
    for canonical_field, aliases in aliases_by_field.items():
        alias_map[str(canonical_field).lower()] = str(canonical_field)
        for alias in aliases or []:
            alias_map[str(alias).lower()] = str(canonical_field)
    return alias_map


def _extract_entity_text(entity: dict[str, Any]) -> str | None:
    normalized_value = dict(entity.get("normalizedValue") or {})
    for value in (
        normalized_value.get("text"),
        entity.get("mentionText"),
        entity.get("text"),
        entity.get("value"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()

    date_value = dict(normalized_value.get("dateValue") or {})
    year = date_value.get("year")
    month = date_value.get("month")
    day = date_value.get("day")
    if all(isinstance(value, int) and value > 0 for value in (year, month, day)):
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _normalize_field_value(field_name: str, value: str | None) -> str | None:
    normalizer = FIELD_NORMALIZERS.get(field_name, normalize_whitespace)
    return normalizer(value)


def _prefer_candidate(previous: str | None, candidate: str | None) -> str | None:
    if not previous:
        return candidate
    if not candidate:
        return previous
    return candidate if len(candidate) > len(previous) else previous


def normalize_google_document_ai_case_payload(
    *,
    case_id: str,
    document_type: str,
    source_path: str,
    payload: dict[str, Any],
    contract: dict[str, Any],
    contract_path: Path,
) -> HarnessRunResult:
    alias_map = _entity_type_alias_map(contract, document_type)
    fields: dict[str, Any] = {}

    for entity in payload.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("type") or "").strip().lower()
        canonical_field = alias_map.get(entity_type)
        if not canonical_field:
            continue
        normalized_value = _normalize_field_value(
            canonical_field,
            _extract_entity_text(entity),
        )
        fields[canonical_field] = _prefer_candidate(fields.get(canonical_field), normalized_value)

    extracted_document = ExtractedDocument(
        document_type=document_type,
        source_path=source_path,
        template_id="google_document_ai.custom_extractor",
        fields=fields,
        quality_checks={
            "external_system": "google_document_ai_custom_extractor",
            "adapter_contract_path": str(contract_path),
        },
        parser_warnings=[],
    )
    review_result = ReviewResult(status=ReviewStatus.PASS, findings=[], cross_check={})
    return HarnessRunResult(
        case_id=case_id,
        extracted_documents=[extracted_document],
        review_result=review_result,
        queued_review_path=None,
    )


def build_google_document_ai_protocol_eval(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(args.contract)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written_case_ids: list[str] = []
    missing_case_ids: list[str] = []
    for expected_path in sorted(args.expected_root.rglob("expected.json")):
        expected_payload = load_json(expected_path)
        case_id = str(expected_payload.get("case_id") or expected_path.parent.name)
        document_type = str(expected_payload.get("document_type") or "")
        if not document_type:
            continue

        raw_path = args.raw_dir / f"{case_id}.json"
        if not raw_path.exists():
            missing_case_ids.append(case_id)
            continue

        raw_payload = load_json(raw_path)
        run_result = normalize_google_document_ai_case_payload(
            case_id=case_id,
            document_type=document_type,
            source_path=str(raw_payload.get("source_path") or f"external://google/{case_id}"),
            payload=raw_payload,
            contract=contract,
            contract_path=args.contract,
        )
        output_path = args.output_dir / f"{case_id}.json"
        output_path.write_text(
            json.dumps(run_result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written_case_ids.append(case_id)

    report = build_field_error_report(
        args.expected_root,
        args.output_dir,
        include_missing_cases=False,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = summarize_eval_report(
        args.report_output,
        external_manifest_path=args.external_manifest,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    comparison_payload: dict[str, object] | None = None
    if args.baseline_report and args.comparison_output:
        comparison = compare_field_error_report_files(
            args.baseline_report,
            args.report_output,
        )
        args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_output.write_text(
            json.dumps(comparison.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        comparison_payload = {
            "comparison_output": str(args.comparison_output),
            "overall_status": comparison.overall_delta.status if comparison.overall_delta else None,
            "overall_exact_match_rate_delta": (
                comparison.overall_delta.exact_match_rate_delta
                if comparison.overall_delta
                else None
            ),
        }

    result = {
        "system_name": "google_document_ai_custom_extractor",
        "contract_path": str(args.contract),
        "raw_dir": str(args.raw_dir),
        "expected_root": str(args.expected_root),
        "output_dir": str(args.output_dir),
        "written_case_ids": written_case_ids,
        "missing_case_ids": missing_case_ids,
        "report_output": str(args.report_output),
        "summary_output": str(args.summary_output),
        "comparison": comparison_payload,
    }
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


def main() -> None:
    args = parse_args()
    result = build_google_document_ai_protocol_eval(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
