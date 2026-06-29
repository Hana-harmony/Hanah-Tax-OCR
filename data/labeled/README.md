# Labeled

`data/labeled/` holds reviewed datasets with label JSON files.

Recommended per-case layout:

- `data/labeled/<document_type>/<case_id>/document.png`
- `data/labeled/<document_type>/<case_id>/label.json`

Each `label.json` should contain:

- extracted field ground truth
- expected review status
- expected rejection reasons or review findings

Additional working splits:

- `data/labeled/pending_review/`: cases promoted from the review queue and waiting for human verification
- `data/labeled/<document_type>/<case_id>/label.json`: reviewed or deterministic regression label

Keep production data out of git unless it is explicitly sanitized.
