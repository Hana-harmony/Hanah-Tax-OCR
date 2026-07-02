from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_PROTOCOL_PATH = Path("evals/benchmark_protocol.json")
DEFAULT_TABLE_TEMPLATE_PATH = Path("evals/sota_positioning/comparison_table_template.json")
DEFAULT_TARGETS_PATH = Path("evals/sota_positioning/comparison_targets.json")
DEFAULT_INTERNAL_SUMMARY_PATH = Path("evals/reports/current_tin_parser_rescue_summary.json")
DEFAULT_EXTERNAL_SUMMARY_PATH = Path(
    "evals/reports/external_holdout_tin_parser_rescue_summary.json"
)
DEFAULT_LATENCY_PATH = Path("evals/benchmark_latency_observations.json")
DEFAULT_GAP_REPORT_PATH = Path("evals/external_holdout/missing_distribution_targets.json")
DEFAULT_TAXONOMY_PATH = Path("evals/error_taxonomy/hard_case_manifest.json")
DEFAULT_OUTPUT_PATH = Path("evals/sota_positioning/current_comparison_snapshot.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a current SOTA positioning snapshot by combining the benchmark protocol, "
            "current candidate summaries, holdout gap evidence, latency observations, "
            "and planned comparison targets."
        )
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--table-template", type=Path, default=DEFAULT_TABLE_TEMPLATE_PATH)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--internal-summary", type=Path, default=DEFAULT_INTERNAL_SUMMARY_PATH)
    parser.add_argument("--external-summary", type=Path, default=DEFAULT_EXTERNAL_SUMMARY_PATH)
    parser.add_argument("--latency", type=Path, default=DEFAULT_LATENCY_PATH)
    parser.add_argument("--gap-report", type=Path, default=DEFAULT_GAP_REPORT_PATH)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _subset_metric(summary: dict[str, Any], subset_key: str, metric_key: str) -> float | None:
    subset = dict(summary.get(subset_key) or {})
    if int(subset.get("comparison_count", 0) or 0) <= 0:
        return None
    value = subset.get(metric_key)
    return None if value is None else float(value)


def _build_current_system_row(
    *,
    row: dict[str, Any],
    summary: dict[str, Any],
    summary_path: Path,
    latency_summary: dict[str, Any],
    note: str,
) -> dict[str, Any]:
    populated = dict(row)
    populated["exact_match_rate"] = float(summary.get("exact_match_rate", 0.0))
    populated["average_character_error_rate"] = float(
        summary.get("average_character_error_rate", 0.0)
    )
    populated["average_word_error_rate"] = float(summary.get("average_word_error_rate", 0.0))
    populated["document_pass_rate"] = float(summary.get("document_pass_rate", 0.0))
    populated["low_quality_exact_match_rate"] = _subset_metric(
        summary,
        "low_quality_subset",
        "exact_match_rate",
    )
    populated["low_quality_average_character_error_rate"] = _subset_metric(
        summary,
        "low_quality_subset",
        "average_character_error_rate",
    )
    populated["low_quality_average_word_error_rate"] = _subset_metric(
        summary,
        "low_quality_subset",
        "average_word_error_rate",
    )
    populated["cpu_latency_observation_seconds"] = float(
        latency_summary.get("average_case_seconds", 0.0)
    )
    populated["comparison_readiness"] = "ready"
    populated["evidence_path"] = str(summary_path)
    populated["notes"] = note
    return populated


def _build_target_rows(
    targets_payload: dict[str, Any],
    template_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    placeholder = next(
        row
        for row in template_rows
        if row.get("system_name") == "candidate_managed_service"
    )
    rows: list[dict[str, Any]] = []
    for target in targets_payload.get("targets", []):
        row = dict(placeholder)
        row["system_name"] = str(target.get("name") or "planned_target")
        row["comparison_readiness"] = str(target.get("status") or "planned")
        row["evidence_path"] = target.get("adapter_contract_path") or target.get(
            "official_reference"
        )
        row["notes"] = " ".join(
            part
            for part in [
                str(target.get("positioning_notes") or "").strip(),
                (
                    f"Official reference: {target.get('official_reference')}"
                    if target.get("official_reference")
                    else ""
                ),
                "Normalize outputs to this repository schema before scoring.",
            ]
            if part
        )
        rows.append(row)
    return rows


def _build_gap_assessment(gap_report: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for target in gap_report.get("targets", []):
        status = str(target.get("status") or "")
        if status == "covered":
            continue
        evidence = dict(target.get("evidence") or {})
        gaps.append(
            {
                "target_id": target.get("target_id"),
                "priority": target.get("priority"),
                "status": status,
                "needed_sample_description": target.get("needed_sample_description"),
                "blocked_by_eval_overlap_case_ids": evidence.get(
                    "blocked_by_eval_overlap_case_ids",
                    [],
                ),
                "blocked_by_non_extractable_cases": evidence.get(
                    "blocked_by_non_extractable_cases",
                    [],
                ),
            }
        )
    return gaps


def _top_root_causes(taxonomy_payload: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    catalog = dict(taxonomy_payload.get("catalog") or {})
    counts = dict((taxonomy_payload.get("summary") or {}).get("root_cause_counts") or {})
    ranked = sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), item[0]),
    )
    return [
        {
            "root_cause": root_cause,
            "count": count,
            "description": dict(catalog.get(root_cause) or {}).get("description"),
            "field_groups": dict(catalog.get(root_cause) or {}).get("field_groups"),
        }
        for root_cause, count in ranked[:limit]
    ]


def build_sota_comparison_snapshot(
    *,
    protocol_path: Path,
    table_template_path: Path,
    targets_path: Path,
    internal_summary_path: Path,
    external_summary_path: Path,
    latency_path: Path,
    gap_report_path: Path,
    taxonomy_path: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    table_template = load_json(table_template_path)
    targets_payload = load_json(targets_path)
    internal_summary = load_json(internal_summary_path)
    external_summary = load_json(external_summary_path)
    latency_payload = load_json(latency_path)
    gap_report = load_json(gap_report_path)
    taxonomy_payload = load_json(taxonomy_path)
    template_rows = list(table_template.get("rows", []))
    latency_summary = dict(latency_payload.get("summary") or {})

    internal_row = next(
        row
        for row in template_rows
        if row.get("system_name") == "hanah_tax_ocr_current_candidate_internal"
    )
    external_row = next(
        row
        for row in template_rows
        if row.get("system_name") == "hanah_tax_ocr_current_candidate_external_holdout"
    )

    rows = [
        _build_current_system_row(
            row=internal_row,
            summary=internal_summary,
            summary_path=internal_summary_path,
            latency_summary=latency_summary,
            note=(
                "Current best candidate under the internal regression protocol. "
                "Low-quality subset is null here because those cases live in external holdout."
            ),
        ),
        _build_current_system_row(
            row=external_row,
            summary=external_summary,
            summary_path=external_summary_path,
            latency_summary=latency_summary,
            note=(
                "Current best candidate under the isolated external holdout protocol. "
                "This is the fairer accuracy row for low-quality and format-variation claims."
            ),
        ),
        *_build_target_rows(targets_payload, template_rows),
    ]

    external_low_quality = dict(external_summary.get("low_quality_subset") or {})
    strongest_signals: list[str] = []
    if float(internal_summary.get("exact_match_rate", 0.0)) >= 1.0:
        strongest_signals.append("Internal regression exact/CER/WER is currently perfect.")
    if float(external_summary.get("exact_match_rate", 0.0)) >= 1.0:
        strongest_signals.append("External holdout exact/CER/WER is currently perfect.")
    if int(external_low_quality.get("comparison_count", 0) or 0) > 0 and float(
        external_low_quality.get("exact_match_rate", 0.0)
    ) >= 1.0:
        strongest_signals.append("Low-quality external holdout subset is currently perfect.")

    honest_risks = []
    if float(latency_summary.get("average_case_seconds", 0.0)) > 20.0:
        honest_risks.append(
            "Local CPU latency is still high enough that managed services may win on throughput."
        )
    mixed_language_subset = dict(external_summary.get("mixed_language_subset") or {})
    if int(mixed_language_subset.get("comparison_count", 0) or 0) <= 0:
        honest_risks.append(
            "Mixed-language external holdout coverage is still zero, "
            "so that claim is not yet benchmarked."
        )

    return {
        "version": "2026-07-02",
        "protocol_reference": str(protocol_path),
        "table_template_reference": str(table_template_path),
        "comparison_scope": targets_payload.get("comparison_scope"),
        "current_candidate_references": {
            "internal_summary": str(internal_summary_path),
            "external_summary": str(external_summary_path),
            "latency_observations": str(latency_path),
            "holdout_gap_report": str(gap_report_path),
            "error_taxonomy": str(taxonomy_path),
        },
        "rows": rows,
        "current_system_assessment": {
            "strongest_signals": strongest_signals,
            "coverage_gaps": _build_gap_assessment(gap_report),
            "historical_risk_root_causes": _top_root_causes(taxonomy_payload),
            "honest_risks": honest_risks,
        },
        "comparison_targets": list(targets_payload.get("targets", [])),
        "required_metrics": list(protocol.get("required_metrics", [])),
    }


def main() -> None:
    args = parse_args()
    payload = build_sota_comparison_snapshot(
        protocol_path=args.protocol,
        table_template_path=args.table_template,
        targets_path=args.targets,
        internal_summary_path=args.internal_summary,
        external_summary_path=args.external_summary,
        latency_path=args.latency,
        gap_report_path=args.gap_report,
        taxonomy_path=args.taxonomy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
