from __future__ import annotations

import json
from pathlib import Path

from scripts.evals.build_sota_comparison_snapshot import build_sota_comparison_snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_build_sota_comparison_snapshot_populates_current_rows_and_gaps(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    table_template_path = tmp_path / "template.json"
    targets_path = tmp_path / "targets.json"
    internal_summary_path = tmp_path / "internal_summary.json"
    external_summary_path = tmp_path / "external_summary.json"
    latency_path = tmp_path / "latency.json"
    gap_report_path = tmp_path / "gap_report.json"
    taxonomy_path = tmp_path / "taxonomy.json"

    _write_json(
        protocol_path,
        {
            "required_metrics": [
                "exact_match_rate",
                "average_character_error_rate",
                "average_word_error_rate",
                "document_pass_rate",
                "low_quality_subset",
                "cpu_latency_observation",
            ]
        },
    )
    _write_json(
        table_template_path,
        {
            "rows": [
                {
                    "system_name": "hanah_tax_ocr_current_candidate_internal",
                    "system_scope": "ocr+parser+normalization+review",
                    "dataset_split": "internal_regression",
                },
                {
                    "system_name": "hanah_tax_ocr_current_candidate_external_holdout",
                    "system_scope": "ocr+parser+normalization+review",
                    "dataset_split": "external_holdout",
                },
                {
                    "system_name": "candidate_managed_service",
                    "system_scope": "managed_custom_extractor_or_query_system",
                    "dataset_split": "same_protocol_only",
                },
            ]
        },
    )
    _write_json(
        targets_path,
        {
            "comparison_scope": "end_to_end",
            "targets": [
                {
                    "name": "Google Cloud Document AI Custom Extractor",
                    "status": "planned",
                    "official_reference": "https://example.com/google",
                    "positioning_notes": "Managed comparator.",
                }
            ],
        },
    )
    _write_json(
        internal_summary_path,
        {
            "exact_match_rate": 1.0,
            "average_character_error_rate": 0.0,
            "average_word_error_rate": 0.0,
            "document_pass_rate": 1.0,
            "low_quality_subset": {
                "comparison_count": 0,
                "exact_match_rate": 0.0,
                "average_character_error_rate": 0.0,
                "average_word_error_rate": 0.0,
            },
            "mixed_language_subset": {
                "comparison_count": 0,
            },
        },
    )
    _write_json(
        external_summary_path,
        {
            "exact_match_rate": 1.0,
            "average_character_error_rate": 0.0,
            "average_word_error_rate": 0.0,
            "document_pass_rate": 1.0,
            "low_quality_subset": {
                "comparison_count": 4,
                "exact_match_rate": 1.0,
                "average_character_error_rate": 0.0,
                "average_word_error_rate": 0.0,
            },
            "mixed_language_subset": {
                "comparison_count": 0,
            },
        },
    )
    _write_json(
        latency_path,
        {
            "summary": {
                "average_case_seconds": 29.4,
            }
        },
    )
    _write_json(
        gap_report_path,
        {
            "targets": [
                {
                    "target_id": "withholding_front_filled_nonoverlap_low_quality",
                    "priority": "highest",
                    "status": "missing_source",
                    "needed_sample_description": (
                        "Need one mixed-language withholding front-side sample."
                    ),
                    "evidence": {
                        "blocked_by_eval_overlap_case_ids": ["withholding_maria_chen_001"],
                        "blocked_by_non_extractable_cases": [
                            {"case_id": "withholding_pdf_001"}
                        ],
                    },
                }
            ]
        },
    )
    _write_json(
        taxonomy_path,
        {
            "catalog": {
                "mixed_korean_english_interference": {
                    "description": "Mixed-language interference.",
                    "field_groups": "korean_mixed_form",
                }
            },
            "summary": {
                "root_cause_counts": {
                    "mixed_korean_english_interference": 3,
                }
            },
        },
    )

    snapshot = build_sota_comparison_snapshot(
        protocol_path=protocol_path,
        table_template_path=table_template_path,
        targets_path=targets_path,
        internal_summary_path=internal_summary_path,
        external_summary_path=external_summary_path,
        latency_path=latency_path,
        gap_report_path=gap_report_path,
        taxonomy_path=taxonomy_path,
    )

    internal_row = next(
        row
        for row in snapshot["rows"]
        if row["system_name"] == "hanah_tax_ocr_current_candidate_internal"
    )
    assert internal_row["exact_match_rate"] == 1.0
    assert internal_row["low_quality_exact_match_rate"] is None
    assert internal_row["cpu_latency_observation_seconds"] == 29.4

    external_row = next(
        row
        for row in snapshot["rows"]
        if row["system_name"] == "hanah_tax_ocr_current_candidate_external_holdout"
    )
    assert external_row["low_quality_exact_match_rate"] == 1.0
    assert external_row["comparison_readiness"] == "ready"

    target_row = next(
        row
        for row in snapshot["rows"]
        if row["system_name"] == "Google Cloud Document AI Custom Extractor"
    )
    assert target_row["comparison_readiness"] == "planned"
    assert target_row["evidence_path"] == "https://example.com/google"

    assert snapshot["current_system_assessment"]["strongest_signals"] == [
        "Internal regression exact/CER/WER is currently perfect.",
        "External holdout exact/CER/WER is currently perfect.",
        "Low-quality external holdout subset is currently perfect.",
    ]
    assert snapshot["current_system_assessment"]["coverage_gaps"] == [
        {
            "target_id": "withholding_front_filled_nonoverlap_low_quality",
            "priority": "highest",
            "status": "missing_source",
            "needed_sample_description": "Need one mixed-language withholding front-side sample.",
            "blocked_by_eval_overlap_case_ids": ["withholding_maria_chen_001"],
            "blocked_by_non_extractable_cases": [{"case_id": "withholding_pdf_001"}],
        }
    ]
    assert snapshot["current_system_assessment"]["historical_risk_root_causes"] == [
        {
            "root_cause": "mixed_korean_english_interference",
            "count": 3,
            "description": "Mixed-language interference.",
            "field_groups": "korean_mixed_form",
        }
    ]
    assert snapshot["current_system_assessment"]["honest_risks"] == [
        "Local CPU latency is still high enough that managed services may win on throughput.",
        "Mixed-language external holdout coverage is still zero, "
        "so that claim is not yet benchmarked.",
    ]


def test_build_sota_comparison_snapshot_prefers_current_candidate_refs_file(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol.json"
    table_template_path = tmp_path / "template.json"
    targets_path = tmp_path / "targets.json"
    default_internal_summary_path = tmp_path / "default_internal_summary.json"
    default_external_summary_path = tmp_path / "default_external_summary.json"
    latency_path = tmp_path / "latency.json"
    gap_report_path = tmp_path / "gap_report.json"
    taxonomy_path = tmp_path / "taxonomy.json"
    current_candidate_refs_path = tmp_path / "current_candidate_refs.json"
    current_internal_summary_path = tmp_path / "current_internal_summary.json"
    current_external_summary_path = tmp_path / "current_external_summary.json"

    _write_json(protocol_path, {"required_metrics": []})
    _write_json(
        table_template_path,
        {
            "rows": [
                {"system_name": "hanah_tax_ocr_current_candidate_internal"},
                {"system_name": "hanah_tax_ocr_current_candidate_external_holdout"},
                {"system_name": "candidate_managed_service"},
            ]
        },
    )
    _write_json(targets_path, {"comparison_scope": "end_to_end", "targets": []})
    _write_json(
        default_internal_summary_path,
        {
            "exact_match_rate": 0.1,
            "average_character_error_rate": 0.9,
            "average_word_error_rate": 0.9,
            "document_pass_rate": 0.1,
            "low_quality_subset": {"comparison_count": 0},
            "mixed_language_subset": {"comparison_count": 0},
        },
    )
    _write_json(
        default_external_summary_path,
        {
            "exact_match_rate": 0.2,
            "average_character_error_rate": 0.8,
            "average_word_error_rate": 0.8,
            "document_pass_rate": 0.2,
            "low_quality_subset": {"comparison_count": 0},
            "mixed_language_subset": {"comparison_count": 0},
        },
    )
    _write_json(
        current_internal_summary_path,
        {
            "exact_match_rate": 1.0,
            "average_character_error_rate": 0.0,
            "average_word_error_rate": 0.0,
            "document_pass_rate": 1.0,
            "low_quality_subset": {"comparison_count": 0},
            "mixed_language_subset": {"comparison_count": 0},
        },
    )
    _write_json(
        current_external_summary_path,
        {
            "exact_match_rate": 1.0,
            "average_character_error_rate": 0.0,
            "average_word_error_rate": 0.0,
            "document_pass_rate": 1.0,
            "low_quality_subset": {
                "comparison_count": 1,
                "exact_match_rate": 1.0,
                "average_character_error_rate": 0.0,
                "average_word_error_rate": 0.0,
            },
            "mixed_language_subset": {"comparison_count": 0},
        },
    )
    _write_json(latency_path, {"summary": {"average_case_seconds": 5.0}})
    _write_json(gap_report_path, {"targets": []})
    _write_json(taxonomy_path, {"catalog": {}, "summary": {"root_cause_counts": {}}})
    _write_json(
        current_candidate_refs_path,
        {
            "internal_summary_path": str(current_internal_summary_path),
            "external_summary_path": str(current_external_summary_path),
            "latency_observations_path": str(latency_path),
            "holdout_gap_report_path": str(gap_report_path),
            "error_taxonomy_path": str(taxonomy_path),
        },
    )

    snapshot = build_sota_comparison_snapshot(
        protocol_path=protocol_path,
        table_template_path=table_template_path,
        targets_path=targets_path,
        current_candidate_refs_path=current_candidate_refs_path,
        internal_summary_path=default_internal_summary_path,
        external_summary_path=default_external_summary_path,
        latency_path=latency_path,
        gap_report_path=gap_report_path,
        taxonomy_path=taxonomy_path,
    )

    assert snapshot["current_candidate_refs_path"] == str(current_candidate_refs_path)
    assert snapshot["current_candidate_references"] == {
        "internal_summary": str(current_internal_summary_path),
        "external_summary": str(current_external_summary_path),
        "latency_observations": str(latency_path),
        "holdout_gap_report": str(gap_report_path),
        "error_taxonomy": str(taxonomy_path),
    }
    internal_row = next(
        row
        for row in snapshot["rows"]
        if row["system_name"] == "hanah_tax_ocr_current_candidate_internal"
    )
    assert internal_row["exact_match_rate"] == 1.0
