# Review Queue

`data/review_queue/` stores cases that need human follow-up.

Typical reasons:

- OCR confidence too low
- required fields missing
- signature or seal verification inconclusive
- cross-document mismatch

Each queued case should include the source file path, extracted result, review result, and a timestamp.

Promotion flow:

- queue output is written under `data/review_queue/index/`
- `python -m scripts.review_queue.report_label_priorities` ranks queued cases using the latest data gap report
- queue cases that already exist under `data/labeled/<document_type>/<case_id>/label.json` are skipped from the priority report
- `python -m scripts.review_queue.promote_to_labeled --priority-report ... --limit N` promotes only the top-ranked cases
- `python -m scripts.review_queue.promote_to_labeled` copies queue JSON into `data/labeled/pending_review/`
- promoted labels are marked for human verification before they enter the reviewed dataset
- promoted label scaffolds can include `priority_context` so reviewers see why the case was selected first
