# Training Datasets

`data/training/` stores generated training artifacts derived from reviewed labels.

Typical contents:

- `field_crops/`: field-level crop images and manifests for recognizer fine-tuning
  each entry includes quality metadata such as size, dark ratio, contrast, and acceptance flags,
  and train/val split assignment keeps document-type validation coverage when possible
- `recognizer/`: PaddleOCR recognizer train/val label files, per-group dictionaries, and plan files
  each plan/summary includes document-type coverage, source-type counts, hard-case ratio warnings,
  any capped hard-case counts applied during train split balancing,
  and the preserved base document mix for selected hard cases
- `hard_cases/`: left-clip, rotation, low-res, and overlay-based hard-case augmentations
- `reports/`: field-level OCR error maps such as CER, WER, exact-match summaries,
  field-group data gap prioritization reports for manual labeling order,
  and sample-data coverage reports for uncovered fixtures
  coverage reports separate reviewed labels from `pending_review` scaffolds

Generated images and manifests are reproducible and should generally stay out of git unless a small sanitized fixture is explicitly needed.
