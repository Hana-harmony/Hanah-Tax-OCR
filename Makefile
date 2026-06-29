PYTHON ?= python3.11
VENV ?= .venv
VENV_BIN := $(VENV)/bin

.PHONY: venv install install-ocr install-ocr-linux-cpu test lint augment ingest stage synthesize redact review eval-case docker-build

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip setuptools wheel

install: venv
	$(VENV_BIN)/pip install -e .[dev]

install-ocr: install
	@echo "Install paddlepaddle for your platform first, then:"
	$(VENV_BIN)/pip install -e .[ocr]

install-ocr-linux-cpu: install
	$(VENV_BIN)/pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
	$(VENV_BIN)/pip install -e .[ocr]

test:
	$(VENV_BIN)/pytest

lint:
	$(VENV_BIN)/ruff check src tests scripts

augment:
	$(VENV_BIN)/python scripts/augment_dataset.py --config configs/augmentation.yaml

ingest:
	$(VENV_BIN)/python -m scripts.ingest.build_staging_manifest

stage:
	@echo "Usage: $(VENV_BIN)/python -m scripts.ingest.promote_to_staging --source <raw_dir> --document-type <type>"

synthesize:
	$(VENV_BIN)/python -m scripts.synthesize.generate_synthetic_labels

redact:
	@echo "Usage: $(VENV_BIN)/python -m scripts.redact.mask_labels --input <path> --output <path>"

review:
	@echo "Usage: $(VENV_BIN)/hanah-tax-ocr run-review --case-id <id> --document <type=path> [--document <type=path>] [--output path]"

eval-case:
	@echo "Usage: $(VENV_BIN)/hanah-tax-ocr eval-case --expected <expected.json> --actual <run_result.json>"

docker-build:
	docker build -t hanah-tax-ocr .
