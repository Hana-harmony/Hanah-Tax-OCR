FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml READEME.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN pip install --upgrade pip setuptools wheel && \
    pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    pip install .[dev,ocr]

CMD ["python", "-m", "pytest"]
