# Review Queue

`data/review_queue/` stores cases that need human follow-up.

Typical reasons:

- OCR confidence too low
- required fields missing
- signature or seal verification inconclusive
- cross-document mismatch

Each queued case should include the source file path, extracted result, review result, and a timestamp.
