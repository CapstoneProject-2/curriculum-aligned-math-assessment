# 교육과정 정합 중학교 수학 평가 프레임워크

논문: 「교육과정 정합 문항 생성·검증과 루브릭 기반 채점을 위한 대형언어모델 기반 평가 시스템」  
저자: 심사 중 익명성을 위해 비공개

## 프로젝트 소개

이 저장소는 중학교 수학 서술형 평가를 지원하는 대형언어모델 기반 시스템의 코드와 프롬프트를 공개하기 위해 구성되었습니다. 시스템은 교육과정에 맞는 문항을 생성하고, 생성 결과를 검증하고, 문항별 채점 기준을 생성합니다. 학생이 답안을 제출하면, 생성된 채점 기준을 바탕으로 학생 답안을 채점하고 짧은 학습 피드백을 제공합니다.

논문에서 사용한 프롬프트 원문, 실험 조건과 버전별 변경 이력을 기능별로 정리하여 논문의 실험 과정과 저장소 자료를 함께 확인할 수 있도록 합니다.

## 주요 기능

| 기능 | 설명 |
| --- | --- |
| 문항 생성 | 성취기준과 교과 지식을 활용한 수학 서술형 문항 생성 |
| 문항 검증 | 수학적·논리적 오류와 출력 형식 검증 |
| 채점 기준 생성 | 이진 채점 및 부분점수 채점을 위한 문항별 기준 생성 |
| 학생 답안 채점 | 이진 채점과 루브릭 기반 부분점수 채점 |
| 학습 피드백 | 채점 결과에 따른 짧은 학생 피드백 생성 |
| 결과 평가 | 대형언어모델 기반 심사자를 이용한 자동 채점 결과의 적절성 평가 |

## 저장소 구성

```text
prompts/
  generation/           문항 생성 프롬프트
  validation/           문항 검증 프롬프트
  rubric/               채점 기준 생성 프롬프트
  grading/              이진·부분점수 채점 프롬프트
  feedback/             학생 피드백 생성 프롬프트
  judge/                자동 채점 결과 평가 프롬프트
src/
  generation/           문항 생성 코드
  validation/           문항 검증 코드
  retrieval/            교육과정 지식 검색 코드
  grading/              채점 기준·채점·피드백·평가 코드
configs/                모델 및 실행 설정
data/                   입력 데이터 형식과 전처리 자료
results/                결과 파일과 평가지표 설명
```

## 실행 환경

- Python 3.10 이상
- OpenAI Python SDK 1.x
- pandas 2.x
- openpyxl 3.1.x
- python-dotenv 1.x
- Jinja2 3.1.x
- PyYAML 6.x
- tqdm 4.66 이상

세부 의존성 범위는 [requirements.txt](requirements.txt)에 고정되어 있습니다.

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

복사한 `.env` 파일에 `OPENAI_API_KEY`를 설정합니다. API 키와 개인 파일 경로, 비공개 모델 식별자는 저장소에 포함하지 않습니다.

채점 및 피드백 코드의 실행 방법은 [채점 모듈 안내](src/grading/README.md)를 참고하세요.

## 논문과 저장소의 대응

| 논문 항목 | 저장소 경로 |
| --- | --- |
| 논문 표 4 (7단계 사고연쇄) | `prompts/generation/v3_cot.md` |
| 논문 표 5 (3단계 검증) | `prompts/validation/`, `src/validation/` |
| 논문 그림 2 (채점 방식별 성능 비교) | `prompts/grading/`, `src/grading/`, `configs/grading.yaml`, `results/README.md` |
| 논문 표 A-4 (채점 기준 생성) | `prompts/rubric/`, `src/grading/generate_rubrics.py` |
| 논문 표 A-5 (채점 수행) | `prompts/grading/`, `src/grading/run_prompt_grading.py` |
| 논문 표 A-6 (피드백 생성) | `prompts/feedback/`, `src/grading/run_feedback.py` |
| 논문 부록 A.6 (채점 검수) | `prompts/judge/`, `src/grading/run_judge_evaluation.py` |
| 논문 부록 B (차수별 변경 이력) | `CHANGELOG.md` |
