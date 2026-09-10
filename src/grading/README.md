# 채점 및 피드백 모듈

이 디렉터리에는 채점 기준 생성, 학생 답안 채점, 짧은 피드백 생성과 대형언어모델 기반 심사자 평가 코드가 들어 있습니다.

모든 명령은 저장소 루트에서 `PYTHONPATH=src`를 설정하여 실행합니다.

## 실행 스크립트

| 스크립트 | 기능 |
| --- | --- |
| `generate_rubrics.py` | 문항별 부분점수 채점 기준 생성 |
| `run_prompt_grading.py` | 이진 및 부분점수 채점 조건 실행 |
| `run_partial_credit_baseline.py` | 부분점수 기본 비교 조건 실행 |
| `run_feedback.py` | 채점 결과를 바탕으로 짧은 피드백 생성 |
| `run_judge_evaluation.py` | 자동 채점 결과의 적절성 평가 및 비율 계산 |

## 부분점수 기본 비교 실험

```bash
PYTHONPATH=src python -m grading.run_partial_credit_baseline \
  --input data/generated_std_answer_and_partial_rubric.xlsx
```

실행이 중단되면 같은 출력 파일을 사용하여 미완료 행부터 다시 시작할 수 있습니다.

```bash
PYTHONPATH=src python -m grading.run_partial_credit_baseline \
  --input data/generated_std_answer_and_partial_rubric.xlsx \
  --resume
```

채점 결과가 모두 생성된 뒤 대형언어모델 기반 심사자 평가를 실행합니다.

```bash
PYTHONPATH=src python -m grading.run_judge_evaluation \
  --input 'results/grading_부분점수_채점_baseline(v1).xlsx'
```

심사자 평가도 `--resume` 옵션으로 이어서 실행할 수 있습니다.

## 다른 채점 및 피드백 실행

```bash
# 부분점수 프롬프트 최적화 조건
PYTHONPATH=src python -m grading.run_prompt_grading \
  --condition partial_credit_v2 \
  --input data/items.xlsx \
  --output results/partial_credit_v2.xlsx

# 채점 기준 생성
PYTHONPATH=src python -m grading.generate_rubrics \
  --input data/items_with_standards.xlsx \
  --output results/rubrics.xlsx

# 짧은 피드백 생성
PYTHONPATH=src python -m grading.run_feedback \
  --input results/partial_credit_v2.xlsx \
  --output results/feedback.xlsx
```

각 배치 스크립트는 처리 결과를 행 단위로 저장합니다. `--resume`을 사용하면 기존 출력과 입력의 문항 식별자 및 답안 순서를 확인한 뒤 미완료 행만 다시 처리합니다.
