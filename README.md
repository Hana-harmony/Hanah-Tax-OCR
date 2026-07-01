# Hanah Tax OCR

다음 세금 문서를 대상으로 하는 OCR 및 검수 파이프라인입니다.

- 거주자 증명서
- 원천징수세 신고서
- 아포스티유

현재 스택은 Python `3.11`이며, OCR 백엔드 옵션으로 PaddleOCR를 사용합니다. 로컬 학습과 실험은 macOS에서 진행할 수 있지만, 검증과 배포는 Linux CPU 환경을 기준으로 합니다.

## 정확도 전략

이 저장소는 아래 여섯 가지 실무 중심 정확도 개선 축을 기준으로 구성되어 있습니다.

- 문서 유형별 라벨링 데이터와 평가 케이스를 지속적으로 늘립니다.
- 문서 레이아웃과 아포스티유 상태에 따라 ROI 템플릿을 분리합니다.
- OCR 이후 파싱된 필드를 적극적으로 정규화합니다.
- 전체 페이지뿐 아니라 필드 단위 영역에도 OCR을 수행합니다.
- `data/review_queue/`의 저신뢰 결과를 재활용합니다.
- 레이아웃 차이가 있는 아포스티유는 주(state)별 파싱 로직을 유지합니다.

## 디렉터리 구성

- `sample_data/`: Git에 포함되는 비식별화 샘플 입력 데이터
- `data/raw/`: 수동 적재한 원본 학습 입력 데이터, 커밋 금지
- `data/staging/`: OCR 수행 전 전처리가 끝난 문서
- `data/labeled/`: 검수 완료 라벨과 결정적 회귀 라벨
- `data/review_queue/`: 실패 사례 또는 저신뢰 검수 출력
- `evals/cases/`: 케이스 단위 회귀 검증용 기대 결과
- `evals/external_holdout/`: 내부 eval과 분리된 외부 비교용 홀드아웃 셋과 manifest
- `evals/error_taxonomy/`: 실패 원인 기준 hard-case taxonomy manifest
- `evals/augmentation_effects/`: semi-real hard case 증강 효과 기록
- `evals/benchmark_protocol.json`: 공식 비교 프로토콜과 승격 규칙
- `scripts/`: 적재, 증강, 합성, 비식별화, 큐 관리 도구
- `src/hanah_tax_ocr/`: OCR, 파싱, 검수, 평가 코드

## 자주 쓰는 명령어

개발 의존성 설치:

```bash
python -m pip install -e .[dev]
```

OCR 의존성까지 함께 설치:

```bash
python -m pip install -e .[dev,ocr]
```

테스트와 린트 실행:

```bash
pytest
ruff check src tests scripts
```

하네스 검수 실행:

```bash
hanah-tax-ocr run-review \
  --case-id residency_maria_chen_001 \
  --document residency_certificate@en=sample_data/거주자증명서/미국\ TREASURY주.png \
  --output data/review_queue/index/residency_run.json
```

큐에 쌓인 실패 사례를 `pending_review` 라벨로 승격:

```bash
python -m scripts.review_queue.promote_to_labeled
```

결정적 회귀 라벨과 평가 케이스 생성:

```bash
python -m scripts.synthesize.build_regression_suite --per-document 20
```

검수 완료 라벨을 필드 크롭 데이터셋으로 내보내기:

```bash
python -m scripts.training.export_field_crops
```

PaddleOCR recognizer 파인튜닝용 데이터셋과 실행 계획 준비:

```bash
python -m scripts.training.prepare_recognizer_finetune --ensure-field-crops
```

필드 크롭 내보내기 과정은 품질 메타데이터를 기록하고 제외 대상 크롭을 표시합니다. 품질 필터는 너무 작거나, 지나치게 비어 있거나, 대비가 낮거나, 인장이나 노이즈처럼 가장자리 성분이 많은 전경으로 과도하게 채워진 크롭을 탐지합니다. Recognizer 준비 단계는 기본적으로 제외된 크롭을 건너뛰며, `--include-rejected-crops`를 지정하면 포함할 수 있습니다. 또한 `--max-hard-case-ratio 1.0`을 명시하지 않는 한 학습 데이터의 hard-case 비중을 `0.5`로 제한하고, 제한된 hard case를 선택할 때 원래 문서 유형 비율을 유지합니다. 각 그룹 계획에는 문서 유형 커버리지, 소스 유형 수, 소스 다양성 수, hard-case 비율 경고가 포함됩니다.
각 recognizer 계획에는 `training_readiness`도 포함됩니다. `run_recognizer_finetune --execute`는 수동 확인용 `--allow-unready`를 주지 않는 한, 학습 또는 검증 샘플이 없는 그룹의 실행을 막습니다.
필드 크롭 분할은 동일한 `source_path`가 하나의 split에만 속하도록 유지해 source leakage를 방지합니다. 검증 데이터 커버리지는 하나의 문서 유형에 서로 다른 원본 문서가 두 건 이상 있을 때만 보장됩니다.

필드 그룹별 recognizer 학습 명령 생성:

```bash
python -m scripts.training.run_recognizer_finetune
```

hard-case 증강 학습 크롭 생성:

```bash
python -m scripts.training.augment_hard_cases
```

추가 샘플 수집 전 필드 그룹 라벨링 공백 우선순위 산출:

```bash
python -m scripts.training.report_data_gaps \
  --eval-report evals/current_report.json
```

다음 recognizer 반복 전에 검토가 필요한 제외 필드 크롭 보고서 생성:

```bash
python -m scripts.training.report_rejected_field_crops \
  --data-gap-report data/training/reports/data_gap_report.json
```

어떤 `sample_data/` 파일이 아직 검수 라벨 또는 eval 커버리지가 없는지 점검:

```bash
python -m scripts.training.report_sample_coverage
```

이 보고서는 `pending_review`를 검수 완료 커버리지와 분리해 유지하므로, 라벨 스캐폴드만 있다고 해서 미라벨 fixture가 커버된 것처럼 사라지지 않습니다.

현재 recognizer 공백을 가장 잘 메울 수 있는 `sample_data/` 라벨 우선순위 산출:

```bash
python -m scripts.training.report_sample_label_priorities \
  --coverage-report data/training/reports/sample_data_coverage.json \
  --data-gap-report data/training/reports/data_gap_report.json
```

우선순위 결과는 이미 라벨 스캐폴드가 있는 `pending_review` 샘플의 점수를 높이고, 막혀 있는 recognizer 그룹에 검증 커버리지가 부족한 경우 `val` fixture를 강조합니다.

커버되지 않은 `sample_data/` 파일에 대해 `pending_review` 라벨 스캐폴드 생성:

```bash
python -m scripts.review_queue.bootstrap_uncovered_samples \
  --coverage-report data/training/reports/sample_data_coverage.json
```

어떤 큐 적재 사례를 먼저 라벨링할지 우선순위 산출:

```bash
python -m scripts.review_queue.report_label_priorities
```

이미 `data/labeled/<document_type>/<case_id>/label.json` 아래에 검수 완료 라벨이 있는 케이스는 이 우선순위 목록에서 제외됩니다.

가장 우선순위가 높은 큐 사례만 `pending_review`로 승격:

```bash
python -m scripts.review_queue.promote_to_labeled \
  --priority-report data/training/reports/review_queue_priority.json \
  --limit 2
```

승격된 라벨 스캐폴드는 매칭된 필드 그룹, 우선순위 점수, 권장 작업 목록을 `priority_context`에 유지합니다.

실행 결과로부터 필드 단위 CER/WER 보고서 생성:

```bash
hanah-tax-ocr eval-report --expected-root evals/cases --actual-dir data/review_queue/index
```

Recognizer 변경 후 baseline과 candidate eval 보고서 비교:

```bash
hanah-tax-ocr compare-eval-reports \
  --baseline evals/baseline-report.json \
  --candidate evals/candidate-report.json
```

비교 결과에는 필드별 변화량, 필드 그룹 집계, 문서 단위 집계, 전체 가중 변화량이 포함되어 있어 candidate recognizer 승격 전에 회귀를 더 쉽게 확인할 수 있습니다.

semi-real hard case probe를 materialize, 실행, report/summary/metadata까지 한 번에 생성:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.evals.run_semi_real_probe_eval \
  --manifest evals/semi_real_probes/manifest.json \
  --suite-output-root tmp/manual_doc_probes_suite \
  --actual-output-dir tmp/manual_doc_probes_actual \
  --report-output evals/reports/current_probe_report.json \
  --summary-output evals/reports/current_probe_summary.json \
  --metadata-output evals/reports/current_probe_metadata.json \
  --case-id withholding_country_row_context_probe
```

공식 baseline 기준으로 변경 문서만 재실행하는 hybrid replay 비교:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.evals.replay_hybrid_eval \
  --baseline-actual-dir tmp/evals/current_official_v17_parser \
  --expected-root evals/cases \
  --output-dir tmp/current_candidate_replay \
  --report-output evals/reports/current_candidate_report.json \
  --summary-output evals/reports/current_candidate_summary.json \
  --baseline-report tmp/evals/current_official_v17_parser_report.json \
  --comparison-output evals/reports/current_official_v17_vs_current_candidate.json \
  --metadata-output evals/reports/current_candidate_metadata.json \
  --rerun-prefix withholding_
```

프로토콜 기준 요약 보고서 생성:

```bash
PYTHONPATH=src .venv/bin/python -m scripts.evals.summarize_eval_report \
  --report evals/reports/current_candidate_report.json \
  --external-manifest evals/external_holdout/manifest.json \
  --output evals/reports/current_candidate_summary.json
```

후보 승격 판단은 반드시 `evals/benchmark_protocol.json`의 exact match, CER, WER, field-level metrics, document pass rate, low-quality subset, CPU latency 관찰 규칙을 따릅니다.
평가 harness는 mixed Korean-English 비중이 높은 `withholding_tax_form`에 대해 기본 OCR lang을 `en`으로 사용합니다.

## 검수 워크플로

1. 스테이징 문서 또는 샘플 문서에 대해 하네스를 실행합니다.
2. `data/review_queue/index/` 아래의 저신뢰 또는 제외 사례를 확인합니다.
3. 해당 사례를 `data/labeled/pending_review/`로 승격합니다.
4. 라벨을 수동 검수한 뒤 검수 완료 데이터셋 split으로 이동합니다.
5. PR을 열기 전에 회귀 검증을 다시 실행합니다.

## 배포 메모

- 기본 배포 대상은 GPU 없는 AWS CPU 환경입니다.
- PaddleOCR의 CPU 추론은 지원하며, 학습과 무거운 실험은 로컬에서 수행하는 것을 권장합니다.
- 커밋 가능한 데이터는 비식별화 fixture와 결정적 라벨로 제한합니다.
- `data/` 아래 generated manifest JSONL은 로컬 산출물로 보고 커밋하지 않습니다.
- `data/training/reports/*.json` 같은 로컬 분석 보고서는 재생성 가능한 산출물로 보고 커밋하지 않습니다.
- 로컬 실험 산출물인 `PaddleOCR/`, `output/`, `tmp/`는 커밋하지 않습니다.
