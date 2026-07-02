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
    "evals/sota_positioning/azure_document_intelligence_custom_extraction_contract.json"
)
DEFAULT_EXTERNAL_MANIFEST_PATH = Path("evals/external_holdout/manifest.json")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize Azure AI Document Intelligence custom extraction raw JSON outputs into "
            "this repository's harness schema and score them with the standard eval protocol."
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


def _normalize_field_value(field_name: str, value: str | None) -> str | None:
    normalizer = FIELD_NORMALIZERS.get(field_name, normalize_whitespace)
    return normalizer(value)


def _prefer_candidate(previous: str | None, candidate: str | None) -> str | None:
    if not previous:
        return candidate
    if not candidate:
        return previous
    return candidate if len(candidate) > len(previous) else previous


def _stringify_number(value: Any) -> str | None:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return None


def _flatten_address(value: dict[str, Any]) -> str | None:
    if not value:
        return None
    for key in ("content", "fullAddress", "formattedAddress"):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    ordered_parts = []
    for key in (
        "houseNumber",
        "road",
        "roadName",
        "streetAddress",
        "unit",
        "city",
        "state",
        "postalCode",
        "countryRegion",
    ):
        raw = value.get(key)
        if isinstance(raw, str) and raw.strip():
            ordered_parts.append(raw.strip())
    return " ".join(ordered_parts) if ordered_parts else None


def _extract_document_field_text(field_payload: dict[str, Any]) -> str | None:
    for key in ("valueString", "valuePhoneNumber", "valueSelectionMark", "content"):
        value = field_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("valueDate", "valueCountryRegion"):
        value = field_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("valueInteger", "valueNumber"):
        value = _stringify_number(field_payload.get(key))
        if value:
            return value

    address_value = field_payload.get("valueAddress")
    if isinstance(address_value, dict):
        return _flatten_address(address_value)

    return None


def normalize_azure_document_intelligence_case_payload(
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

    documents = payload.get("documents")
    if not isinstance(documents, list):
        documents = []
    for document_payload in documents:
        if not isinstance(document_payload, dict):
            continue
        raw_fields = document_payload.get("fields")
        if not isinstance(raw_fields, dict):
            continue
        for raw_field_name, raw_field_payload in raw_fields.items():
            if not isinstance(raw_field_payload, dict):
                continue
            canonical_field = alias_map.get(str(raw_field_name).strip().lower())
            if not canonical_field:
                continue
            normalized_value = _normalize_field_value(
                canonical_field,
                _extract_document_field_text(raw_field_payload),
            )
            fields[canonical_field] = _prefer_candidate(
                fields.get(canonical_field),
                normalized_value,
            )

    extracted_document = ExtractedDocument(
        document_type=document_type,
        source_path=source_path,
        template_id="azure_document_intelligence.custom_extraction",
        fields=fields,
        quality_checks={
            "external_system": "azure_document_intelligence_custom_extraction",
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


def build_azure_document_intelligence_protocol_eval(args: argparse.Namespace) -> dict[str, Any]:
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
        run_result = normalize_azure_document_intelligence_case_payload(
            case_id=case_id,
            document_type=document_type,
            source_path=str(raw_payload.get("source_path") or f"external://azure/{case_id}"),
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
        "system_name": "azure_document_intelligence_custom_extraction",
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
    result = build_azure_document_intelligence_protocol_eval(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
