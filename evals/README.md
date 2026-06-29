# Evaluation Layout

- `evals/cases/<case_id>/input/`: 평가 입력 파일
- `evals/cases/<case_id>/expected.json`: 기대 추출값과 판정 결과
- `evals/fixtures/`: 테스트용 고정 fixture

초기 단계에서는 문서별 샘플 몇 건만이라도 `expected.json`을 붙여서
파서와 reviewer의 회귀 테스트 기준으로 사용한다.
