# Contributing

## Development Baseline

- Python `3.11`
- Primary runtime target: Linux container on AWS CPU
- Local development on macOS is allowed, but validation should use Python 3.11 or Docker

## Branch and PR Rules

- Work in task-scoped branches only
- Keep one logical task per branch and per PR
- Use branch names such as:
  - `feat/<task-name>`
  - `fix/<task-name>`
  - `refactor/<task-name>`
  - `chore/<task-name>`
  - `docs/<task-name>`
  - `test/<task-name>`

## Commit and PR Expectations

- Keep commits intentional and reviewable
- Open a PR for each task unit instead of batching unrelated work
- PR title should follow the same prefix style as commits when possible
- PR body should include summary, background, changes, impact, and review points

## Validation Before PR

- `docker run --rm -v "$PWD":/work -w /work python:3.11-slim bash -lc "python -m pip install --quiet pytest ruff pydantic pillow numpy pyyaml && pytest && ruff check src tests scripts"`

## Harness Commands

- Build staging manifest:
  `python -m scripts.ingest.build_staging_manifest`
- Promote raw files into staging:
  `python -m scripts.ingest.promote_to_staging --source data/raw/train/residency_certificate --document-type residency_certificate`
- Run augmentation:
  `python -m scripts.augment.run --config configs/augmentation.yaml`
- Generate synthetic labels:
  `python -m scripts.synthesize.generate_synthetic_labels`
- Redact a label file:
  `python -m scripts.redact.mask_labels --input in.json --output out.json`
- Run harness review:
  `hanah-tax-ocr run-review --case-id sample-001 --document residency_certificate=sample.png`
- Evaluate harness output:
  `hanah-tax-ocr eval-case --expected evals/cases/residency_smoke_001/expected.json --actual evals/fixtures/last_run_result.json`

## Data Handling

- Do not commit raw customer documents under `data/raw/`
- Do not commit generated outputs under `data/augmented/`
- Commit only safe fixtures, configs, and deterministic test data
