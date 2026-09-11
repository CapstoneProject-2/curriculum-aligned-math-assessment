# common.py — 생성/검증/검색 모듈이 공통으로 쓰는 설정값과 리소스
# retrieval.py, validation.py, generation.py에서 import하여 사용한다.

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

# =========================================================
# 설정 상수 블록 (config) — 논문에 보고한 설정값은 모두 여기서 확인 가능
# =========================================================
# API 키는 저장소에 하드코딩하지 않는다.
# 실행 전 환경변수로 설정할 것: export OPENAI_API_KEY="sk-..."
# 또는 프로젝트 루트에 .env 파일을 만들고 OPENAI_API_KEY=... 를 넣을 것
# (.env는 반드시 .gitignore에 포함 — 저장소에 커밋하지 않는다)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# 기반 모델(base model). 실제 실험에서는 이 모델을 파인튜닝한 버전을 사용했으나,
# 파인튜닝 job-id·학습 조직 식별자는 익명 심사 원칙상 저장소에 공개하지 않는다.
# 재현 시 GEN_MODEL_ID, GRADING_MODEL_ID 환경변수로 실제 파인튜닝 모델 ID를 주입할 것.
BASE_MODEL = "gpt-4o-mini"
GEN_MODEL_ID = os.getenv("GEN_MODEL_ID", BASE_MODEL)
VALIDATION_MODEL = "gpt-4o-mini"

# 논문 표 4 / 저장소 정리 가이드 4번 항목에 보고한 설정값
SIMILARITY_THRESHOLD = 0.6        # 개념 검색 유사도 임계값
TOP_K_CONCEPTS = 2                # 프롬프트에 주입하는 개념·공식 수
MAX_FEWSHOT_EXAMPLES = 4          # few-shot 예시문항 수
MAX_REGEN_ATTEMPTS = 3            # 재생성 최대 시도 횟수

# 데이터 경로 (프로젝트 루트 기준 상대경로. 실행 위치에 맞게 조정할 것)
DATA_DIR = Path("../data")
PROBLEM_DATA_PATH = DATA_DIR / "수학과목문제생성데이터_10차정리.xlsx"
ACHIEVEMENT_JSON_PATH = DATA_DIR / "성취기준과_성취수준_대수.json"
UNIT_CSV_DIR = Path("./merged")

# -------------------------
# Lazy singletons (무거운 리소스는 실제로 필요할 때만 로드)
# -------------------------
_generation_llm = None
_validation_llm = None
_problem_data = None
_achievement_data = None
_achievement_level_data = None
_embedding_model = None
_embedding_matrix = None


def get_generation_llm():
    """문제 생성용 LLM (기본값 gpt-4o-mini, 파인튜닝 모델은 GEN_MODEL_ID로 주입)"""
    global _generation_llm
    if _generation_llm is None:
        _generation_llm = ChatOpenAI(model=GEN_MODEL_ID, temperature=0.7, openai_api_key=API_KEY)
    return _generation_llm


def get_validation_llm():
    """문제 검증용 LLM"""
    global _validation_llm
    if _validation_llm is None:
        _validation_llm = ChatOpenAI(model=VALIDATION_MODEL, temperature=0.1, openai_api_key=API_KEY)
    return _validation_llm


def get_problem_data() -> pd.DataFrame:
    """문제 데이터 로드 (lazy loading)"""
    global _problem_data
    if _problem_data is None:
        _problem_data = pd.read_excel(PROBLEM_DATA_PATH)
    return _problem_data


def get_achievement_dicts():
    """성취기준/성취수준 딕셔너리 로드 (lazy loading)"""
    global _achievement_data, _achievement_level_data
    if _achievement_data is None or _achievement_level_data is None:
        with open(ACHIEVEMENT_JSON_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        achievement_data = {}
        achievement_level_data = {}
        for code, item in loaded.items():
            achievement_content = (
                f"설명: {item.get('성취기준', '')}\n"
                f"성취기준 해설: {item.get('성취기준 해설', '')}\n"
                f"적용 시 고려 사항: {item.get('적용 시 고려 사항', '')}"
            ).strip()
            achievement_data[code] = achievement_content
            achievement_level_data[code] = item.get('성취수준', {})

        _achievement_data, _achievement_level_data = achievement_data, achievement_level_data
    return _achievement_data, _achievement_level_data


def get_embedding_model():
    """문장 임베딩 모델 로드 (lazy loading)"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _embedding_model


def get_problem_embeddings():
    """전체 문제 코퍼스 임베딩 행렬 (lazy loading)"""
    global _embedding_matrix
    if _embedding_matrix is None:
        df = get_problem_data()
        model = get_embedding_model()
        problem_corpus = df['question_text'].fillna("").tolist()
        _embedding_matrix = model.encode(problem_corpus, convert_to_tensor=True)
    return _embedding_matrix


def get_achievement_standard_info_from_code(standard_code):
    """성취기준 코드로부터 성취기준과 성취수준 정보를 반환"""
    if not standard_code or pd.isna(standard_code):
        return "성취기준 정보 없음", ""

    achievement_data, achievement_level_data = get_achievement_dicts()
    achievement_content = achievement_data.get(standard_code, "성취기준 정보 없음")
    achievement_levels = achievement_level_data.get(standard_code, {})

    level_text = ""
    for level, content in achievement_levels.items():
        level_text += f"{level}: {content}\n"

    return achievement_content, level_text.strip()


def get_unit_specific_conditions(unit_name):
    """단원별 특수 조건을 반환 (띄어쓰기 무시)"""
    unit_conditions = {
        "소인수분해": "소수/합성수 여부를 학생에게 직접 판별하게 하는 문제는 금지한다. 최대공약수/최소공배수를 다루는 경우, 조건을 충분히 제시하여 답이 유일하게 결정되도록 하고 정답은 자연수여야 한다.",
        "문자의사용과식": "방정식으로 해를 구하는 형태의 문제를 만들지 말고 기호의 정의, 식의 해석·변형, 항의 의미 파악 등 단원 범위 내 활동만 다룬다.",
        "단항식과 다항식의 계산": "덧셈·뺄셈·곱셈 연산만 다룬다. 일차방정식·연립일차방정식 문제로 넘어가지 않는다.",
        "일차방정식": "해가 존재하고 유일하게 결정되도록 조건을 설계한다. 실생활 맥락(개수, 가격, 인원 등)에서 등장하는 변수는 자연수로만 허용한다. 과도하게 큰 계수·복잡한 수치는 피한다.",
        "연립일차방정식": "두 식이 서로 모순되지 않도록 하고, 실생활 맥락(개수, 가격, 인원 등)의 변수는 자연수로만 허용한다. 계수·상수를 적절히 설계하여 정수해가 유일하게 나오도록 한다(무해/무수해 금지).",
        "이차방정식": "단원 범위를 벗어나는 기법 사용을 금지한다. 구한 해를 식에 대입하여 검산하고 문제풀이가 맞는지 확인하고 문제생성 한다.",
        "정수와 유리수": "반올림 요구를 남용하지 않는다(특히 둘째 자리 이상 금지). 정답 형태(정수/기약분수 등)를 명확히 지시하고, 비현실적으로 큰 수는 피한다.",
        "유리수와 순환소수": "순환마디(반복되는 자리)의 길이는 최대 3자리로 제한한다. 순환소수 표기는 명확히 하고, 반올림 요구는 금지한다.",
        "실수와 그 연산": "정답 형태(정수/분수/소수)를 명확히 지시하고, 과도한 소수 자릿수나 비현실 수치를 피한다.",
        "일차부등식": "해가 하나로 특정되도록 조건을 설계한다. 실생활 맥락(개수, 가격, 인원 등)에서는 해가 자연수로 나오도록 전제·조건을 명확히 제시한다. 범위형 문항은 단 하나의 값 또는 명확한 최솟값/최댓값으로 귀결되게 한다."
    }

    normalized_unit_name = unit_name.replace(" ", "").strip()

    if normalized_unit_name in unit_conditions:
        return unit_conditions[normalized_unit_name]

    for key, value in unit_conditions.items():
        if normalized_unit_name in key or key in normalized_unit_name:
            return value

    return "해당 단원의 특수 조건이 없습니다."
