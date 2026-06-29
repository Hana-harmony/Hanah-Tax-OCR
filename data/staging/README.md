# Staging

`data/staging/` is the preprocessing handoff area between raw uploads and OCR execution.

- Put normalized files here only when they are ready for OCR.
- Do not treat this directory as a system of record.
- Files here are ignored by git by default because they may contain personal data.
