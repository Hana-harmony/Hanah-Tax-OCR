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
