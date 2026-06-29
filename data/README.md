# Data Layout

운영 기준은 `raw`와 `augmented`를 분리하고, 평가셋은 원본 위주로 유지하는 것이다.

## Directories

- `data/raw/train/<document_type>/`: 수동 적재한 원본 학습 이미지
- `data/raw/val/<document_type>/`: 검증용 원본 이미지
- `data/raw/test/<document_type>/`: 운영 전 최종 점검용 원본 이미지
- `data/augmented/train/<document_type>/`: 증강 이미지 산출물
- `data/manifests/raw_index.jsonl`: 원본 인덱스
- `data/manifests/augmented_index.jsonl`: 증강 인덱스

## Rules

- 같은 원본에서 나온 파생본은 `train`과 `val/test`에 동시에 존재하면 안 된다.
- `val/test`와 `evals/`에는 자동 증강 이미지를 넣지 않는다.
- 원본 이미지는 직접 수정하지 않고, 증강은 재생성 가능해야 한다.
