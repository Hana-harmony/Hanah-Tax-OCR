# Evaluation Layout

- `evals/cases/<case_id>/input/`: 평가 입력 파일
- `evals/cases/<case_id>/expected.json`: 기대 추출값과 판정 결과
- `evals/fixtures/`: 테스트용 고정 fixture

초기 단계에서는 문서별 샘플 몇 건만이라도 `expected.json`을 붙여서
파서와 reviewer의 회귀 테스트 기준으로 사용한다.

실무 운영 기준:

- `evals/`에는 가능하면 비식별 또는 합성 데이터만 둔다.
- 운영용 실문서는 `data/labeled/`에서 관리하고, `evals/`에는 평가에 필요한 최소 샘플만 복사한다.
- `python -m scripts.synthesize.build_regression_suite --per-document 20` 로 문서별 회귀 케이스를 확장할 수 있다.
