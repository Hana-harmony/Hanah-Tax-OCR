# Hanah Tax OCR

Tax document OCR and review pipeline for:

- residency certificates
- withholding tax forms
- apostilles

The current stack is Python 3.11 with PaddleOCR as the OCR backend option. Local training and experimentation can happen on macOS, but validation and deployment are expected to run in a Linux CPU environment.

## Accuracy Strategy

This repository is tuned around six practical accuracy levers:

- grow labeled and evaluation cases per document type
- split ROI templates by document layout and apostille state
- normalize parsed fields aggressively after OCR
- run OCR on field regions, not only on full pages
- recycle low-confidence outputs from `data/review_queue/`
- keep apostille parsing state-specific where layouts differ

## Layout

- `sample_data/`: sanitized sample inputs committed to git
- `data/raw/`: manually loaded raw training inputs, not committed
- `data/staging/`: preprocessed documents ready for OCR
- `data/labeled/`: reviewed labels and deterministic regression labels
- `data/review_queue/`: failed or low-confidence review outputs
- `evals/cases/`: expected outputs for case-level regression checks
- `scripts/`: ingestion, augmentation, synthesis, redaction, and queue tooling
- `src/hanah_tax_ocr/`: OCR, parsing, review, and evaluation code

## Common Commands

Install development dependencies:

```bash
python -m pip install -e .[dev]
```

Install OCR dependencies as well:

```bash
python -m pip install -e .[dev,ocr]
```

Run tests and lint:

```bash
pytest
ruff check src tests scripts
```

Run a harness review:

```bash
hanah-tax-ocr run-review \
  --case-id residency_maria_chen_001 \
  --document residency_certificate@en=sample_data/거주자증명서/미국\ TREASURY주.png \
  --output data/review_queue/index/residency_run.json
```

Promote queued failures into pending labels:

```bash
python -m scripts.review_queue.promote_to_labeled
```

Generate deterministic regression labels and eval cases:

```bash
python -m scripts.synthesize.build_regression_suite --per-document 20
```

Export reviewed labels into field crop datasets:

```bash
python -m scripts.training.export_field_crops
```

Prepare PaddleOCR recognizer fine-tuning datasets and plans:

```bash
python -m scripts.training.prepare_recognizer_finetune --ensure-field-crops
```

Field crop export writes quality metadata and marks rejected crops. The quality filter now flags crops that are too small, too blank, too low-contrast, or densely filled by edge-heavy foreground content such as seals and noise. Recognizer prep skips rejected crops by default unless `--include-rejected-crops` is set, caps train hard-case share at `0.5` unless `--max-hard-case-ratio 1.0` is used, preserves the base document-type mix when selecting capped hard cases, and each group plan includes document-type coverage, source-type counts, source-diversity counts, and hard-case ratio warnings.
Each recognizer plan also includes `training_readiness`; `run_recognizer_finetune --execute` blocks groups with no train or validation samples unless `--allow-unready` is passed for manual inspection.
Field crop splitting now keeps the same `source_path` in a single split to avoid source leakage. Validation coverage is preserved only when a document type has at least two distinct source documents.

Render per-field-group recognizer training commands:

```bash
python -m scripts.training.run_recognizer_finetune
```

Generate hard-case augmented training crops:

```bash
python -m scripts.training.augment_hard_cases
```

Rank field-group labeling gaps before collecting more base samples:

```bash
python -m scripts.training.report_data_gaps \
  --eval-report evals/current_report.json
```

Report rejected field crops that should be reviewed before the next recognizer iteration:

```bash
python -m scripts.training.report_rejected_field_crops \
  --data-gap-report data/training/reports/data_gap_report.json
```

Audit which `sample_data/` files are still missing reviewed labels or eval coverage:

```bash
python -m scripts.training.report_sample_coverage
```

The report keeps `pending_review` separate from reviewed coverage so unlabeled fixtures do not disappear just because a scaffold exists.

Prioritize `sample_data/` labels that are most likely to unblock current recognizer gaps:

```bash
python -m scripts.training.report_sample_label_priorities \
  --coverage-report data/training/reports/sample_data_coverage.json \
  --data-gap-report data/training/reports/data_gap_report.json
```

The priority output boosts pending-review samples that already have label scaffolds and highlights `val` fixtures when blocked recognizer groups are missing validation coverage.

Bootstrap `pending_review` label scaffolds for uncovered `sample_data/` files:

```bash
python -m scripts.review_queue.bootstrap_uncovered_samples \
  --coverage-report data/training/reports/sample_data_coverage.json
```

Rank queued review cases to decide which ones to label first:

```bash
python -m scripts.review_queue.report_label_priorities
```

Cases that already have reviewed labels under `data/labeled/<document_type>/<case_id>/label.json`
are skipped from this priority list.

Promote only the highest-priority queued cases into `pending_review`:

```bash
python -m scripts.review_queue.promote_to_labeled \
  --priority-report data/training/reports/review_queue_priority.json \
  --limit 2
```

Promoted label scaffolds keep the matched field groups, priority score, and recommended action list in `priority_context`.

Build a field-level CER/WER report from run results:

```bash
hanah-tax-ocr eval-report --expected-root evals/cases --actual-dir data/review_queue/index
```

Compare baseline and candidate eval reports after a recognizer change:

```bash
hanah-tax-ocr compare-eval-reports \
  --baseline evals/baseline-report.json \
  --candidate evals/candidate-report.json
```

The comparison output includes field deltas, field-group rollups, document-level rollups, and an overall weighted delta so regressions are easier to spot before promoting a candidate recognizer.

## Review Workflow

1. Run the harness on staged or sample documents.
2. Inspect low-confidence or rejected cases under `data/review_queue/index/`.
3. Promote those cases into `data/labeled/pending_review/`.
4. Verify labels manually, then move them into the reviewed dataset split.
5. Re-run regression checks before opening a PR.

## Deployment Notes

- Primary target is AWS CPU without GPU.
- PaddleOCR CPU inference is supported; keep training and heavy experimentation local.
- Commit only sanitized fixtures and deterministic labels.
