PYTHON ?= python3.11
VENV ?= .venv
VENV_BIN := $(VENV)/bin

.PHONY: venv install install-ocr install-ocr-linux-cpu test lint augment docker-build

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

docker-build:
	docker build -t hanah-tax-ocr .
