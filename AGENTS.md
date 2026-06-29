# AGENTS.md

## Project Rules

- Use Python `3.11` as the source-of-truth runtime.
- Treat Linux Docker and AWS CPU deployment as the primary execution target.
- Keep changes scoped to one task unit whenever possible.
- Use a separate branch and PR per task unit.
- Before proposing a merge, run tests and lint in a Python 3.11 environment or Docker.
- Do not commit files under `data/raw/`, `data/augmented/`, or generated manifest JSONL files.
- Prefer maintainable fixes over temporary patches, and verify assumptions against code, tests, or logs.
