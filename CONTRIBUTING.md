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

## Data Handling

- Do not commit raw customer documents under `data/raw/`
- Do not commit generated outputs under `data/augmented/`
- Commit only safe fixtures, configs, and deterministic test data
