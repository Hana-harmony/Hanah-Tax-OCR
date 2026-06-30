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
- `python -m scripts.review_queue.promote_to_labeled` copies queue JSON into `data/labeled/pending_review/`
- promoted labels are marked for human verification before they enter the reviewed dataset
