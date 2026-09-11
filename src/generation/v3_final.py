# finalcode.py ( - 프롬프트엔지니어링 + 파인튜닝 + RAG 구조 적용 + 통합된 문제 추천)

# 환경 설정 및 데이터 로드

import os
import re
import sys
import numpy as np
import pandas as pd
import random   
import json
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from partial_scoring import MathGradingHintSystem, extract_tagged
from sentence_transformers import SentenceTransformer, util

DEVELOPMENT_MODE = True

# =========================================================
# 회차별 시드 고정 (재현성/안정성 검증용)
# 500문항(99×5회) 실험처럼 여러 회차를 독립적으로 반복할 때,
# 회차마다 "다른" 시드를 명시적으로 지정한다.
#   - 회차 간에는 서로 다른 무작위 샘플링이 이뤄져야 "독립적인 반복실험"이 됨
#   - 각 회차 자체는 동일 시드로 재실행하면 동일 결과가 재현되어야 함
# 실행 시 예: python final_code.py --seed 1  (1~5회차에 맞춰 1~5로 바꿔서 실행)
# =========================================================
if "--seed" in sys.argv:
    RUN_SEED = int(sys.argv[sys.argv.index("--seed") + 1])
else:
    RUN_SEED = 1  # 기본값. 회차별로 반드시 바꿔서 실행할 것 (1, 2, 3, 4, 5)

random.seed(RUN_SEED)
np.random.seed(RUN_SEED)
try:
    import torch
    torch.manual_seed(RUN_SEED)
except ImportError:
    pass

print(f"🎲 이번 실행의 RUN_SEED = {RUN_SEED} (결과 저장 시 파일명에 함께 기록할 것)")

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
GRADING_MODEL_ID = os.getenv("GRADING_MODEL_ID", BASE_MODEL)

# 논문 표 4 / 저장소 정리 가이드 4번 항목에 보고한 설정값
SIMILARITY_THRESHOLD = 0.6        # 개념 검색 유사도 임계값
TOP_K_CONCEPTS = 2                # 프롬프트에 주입하는 개념·공식 수
MAX_FEWSHOT_EXAMPLES = 4          # few-shot 예시문항 수
MAX_REGEN_ATTEMPTS = 3            # 재생성 최대 시도 횟수

# OpenAI 모델 초기화
generation_llm = ChatOpenAI(model=GEN_MODEL_ID, temperature=0.7, openai_api_key=API_KEY)
validation_llm = ChatOpenAI(model=VALIDATION_MODEL, temperature=0.1, openai_api_key=API_KEY)
math_system = MathGradingHintSystem()

# 문제 데이터 및 성취기준 데이터 로드
problem_data = pd.read_excel("../data/수학과목문제생성데이터_10차정리.xlsx")

with open("../data/성취기준과_성취수준_대수.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)

# 성취기준과 성취수준 정보를 포함한 데이터 구조 생성
achievement_data = {}
achievement_level_data = {}

for code, item in loaded.items():
    # 성취기준 내용 구성
    achievement_content = (
        f"설명: {item.get('성취기준', '')}" + "\n" +
        f"성취기준 해설: {item.get('성취기준 해설', '')}" + "\n" +
        f"적용 시 고려 사항: {item.get('적용 시 고려 사항', '')}"
    ).strip()
    
    achievement_data[code] = achievement_content
    
    # 성취수준 정보 저장
    achievement_level_data[code] = item.get('성취수준', {})


embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
problem_corpus = problem_data['question_text'].fillna("").tolist()
embedding_matrix = embedding_model.encode(problem_corpus, convert_to_tensor=True)

# 단원별 개념/공식 불러오기 + 예시 문제 추출
def load_unit_data(unit_name):
    path = f"./merged/{unit_name.replace(' ', '')}_통합.csv"
    df = pd.read_csv(path)
    concepts = df['concept'].dropna().tolist()
    formulas = df['formula'].dropna().tolist()
    return concepts, formulas

def retrieve_concept_formula(unit_name):
    concepts, formulas = load_unit_data(unit_name)
    return concepts, formulas  # 전체 리스트 반환

def get_achievement_standard_info_from_code(standard_code):
    """성취기준 코드로부터 성취기준과 성취수준 정보를 반환"""
    if not standard_code or pd.isna(standard_code):
        return "성취기준 정보 없음", ""
    
    achievement_content = achievement_data.get(standard_code, "성취기준 정보 없음")
    achievement_levels = achievement_level_data.get(standard_code, {})
    
    # 성취수준을 문자열로 변환
    level_text = ""
    for level, content in achievement_levels.items():
        level_text += f"{level}: {content}" + "\n"
    
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
    
    # 띄어쓰기 제거 후 매칭
    normalized_unit_name = unit_name.replace(" ", "").strip()
    
    # 정확한 매칭 시도
    if normalized_unit_name in unit_conditions:
        return unit_conditions[normalized_unit_name]
    
    # 부분 매칭 시도 (더 유연한 매칭)
    for key, value in unit_conditions.items():
        if normalized_unit_name in key or key in normalized_unit_name:
            return value
    
    return "해당 단원의 특수 조건이 없습니다."

# RAG 기반 Few-shot 예시 검색 (전략 A 기반)
def rag_search_by_unit_difficulty(unit_name, difficulty):
    """단원명 + 난이도 기준으로 문제 필터링하고 예시 4개 랜덤 추출"""
    # 엑셀에서 단원 + 난이도로 필터링
    subset = problem_data[
        (problem_data['unit_name'].str.replace(" ", "") == unit_name.replace(" ", "")) &
        (problem_data['difficulty'] == difficulty)
    ]
    
    if subset.empty:
        return pd.DataFrame()
    
    # 서로 다른 성취기준 코드를 가진 문제들을 우선 선택
    unique_standards = subset['achievement_standard_code'].dropna().unique()
    
    if len(unique_standards) >= 1:
        # 각 성취기준에서 1개씩 선택
        selected_examples = []
        for standard in unique_standards[:MAX_FEWSHOT_EXAMPLES]:
            standard_problems = subset[subset['achievement_standard_code'] == standard]
            if not standard_problems.empty:
                selected_examples.append(standard_problems.sample(1).iloc[0])
        
        # MAX_FEWSHOT_EXAMPLES개가 안 되면 나머지는 랜덤으로 채움
        remaining_count = MAX_FEWSHOT_EXAMPLES - len(selected_examples)
        if remaining_count > 0:
            remaining_problems = subset[~subset.index.isin([ex.name for ex in selected_examples])]
            if not remaining_problems.empty:
                additional = remaining_problems.sample(min(remaining_count, len(remaining_problems)))
                selected_examples.extend([additional.iloc[i] for i in range(len(additional))])
        
        return pd.DataFrame(selected_examples)
    else:
        # fallback: 아무 문제나 랜덤하게 MAX_FEWSHOT_EXAMPLES개 선택
        n_samples = min(MAX_FEWSHOT_EXAMPLES, len(subset))
        return subset.sample(n_samples)



def smart_get_few_shot_examples(unit_name, difficulty_level):
    """RAG 기반 Few-shot 예시 검색 및 관련 개념-공식 찾기 (통합 버전)
    반환: formatted_examples, concepts, formulas, allowed_codes_csv
    """
    # 1. RAG 기반 예시 문제 검색 (4개)
    few_shot_examples = rag_search_by_unit_difficulty(unit_name, difficulty_level)
    
    if few_shot_examples.empty:
        print(f"⚠️ {unit_name} 단원의 {difficulty_level} 난이도 예시 문제가 없습니다.")
        print(f"📚 개념과 공식만을 참고하여 문제를 생성합니다.")
        
        # 예시 없이 개념/공식만 사용
        concepts, formulas = load_unit_data(unit_name)
        return "[예시 문제 없음 - 개념과 공식만 참고하여 문제 생성]", concepts, formulas, ""
    
    # 2. Few-shot 예시 포맷팅 (4개)
    formatted = ""
    allowed_codes = []
    for _, row in few_shot_examples.iterrows():
        code = row.get('achievement_standard_code', '성취코드 없음') if pd.notna(row.get('achievement_standard_code')) else '성취코드 없음'
        content = row.get('achievement_standard_content', '성취기준 내용 없음') if pd.notna(row.get('achievement_standard_content')) else '성취기준 내용 없음'
        explanation = row.get('solution_text', '생략') if pd.notna(row.get('solution_text')) else '생략'

        formatted += f"<achievement_standard_code>{code}</achievement_standard_code>" + "\n"
        formatted += f"<achievement_standard_content>{content}</achievement_standard_content>" + "\n"
        formatted += f"<question_text>{row['question_text']}</question_text>" + "\n"
        formatted += f"<solution_text>{explanation}</solution_text>" + "\n"
        formatted += f"<final_answer>{row['final_answer']}</final_answer>" + "\n\n"
        if code and code != '성취코드 없음':
            allowed_codes.append(str(code).strip())

    # 3. Topic 기반 관련 개념-공식 검색 (search_concepts_by_topic 로직 통합)
    try:
        concepts_formulas_df = load_unit_data(unit_name)
        concepts_formulas_df = pd.DataFrame({'concept': concepts_formulas_df[0], 'formula': concepts_formulas_df[1]})
    except:
        # CSV 파일이 없으면 기존 방식 사용
        concepts, formulas = load_unit_data(unit_name)
        allowed_codes_csv = ", ".join(dict.fromkeys([c for c in allowed_codes if c]))
        return formatted.strip(), concepts, formulas, allowed_codes_csv
    
    # RAG 기반 개념-공식 매칭 (강화된 버전)
    all_matched_concepts_with_scores = []
    
    for _, row in few_shot_examples.iterrows():
        individual_topic = row['topic']  # 개별 topic
        
        # 개별 topic으로 유사도 계산
        concept_embeddings = embedding_model.encode(concepts_formulas_df['concept'].tolist())
        topic_embedding = embedding_model.encode([individual_topic])
        
        similarities = util.pytorch_cos_sim(topic_embedding, concept_embeddings)[0].cpu().numpy()
        
        # 임계값 기반 필터링 (RAG 강화)
        threshold = SIMILARITY_THRESHOLD
        top_indices = similarities.argsort()[-5:][::-1]  # 상위 5개 선택
        
        # 임계값을 넘는 개념만 선택
        for idx in top_indices:
            if similarities[idx] > threshold:
                matched_concept = concepts_formulas_df.iloc[idx]
                similarity_score = similarities[idx]
                
                all_matched_concepts_with_scores.append({
                    'concept': matched_concept,
                    'similarity': similarity_score
                })
    
    # 유사도 순서대로 정렬 (높은 순서)
    all_matched_concepts_with_scores.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 중복 제거 (유사도 높은 순서 유지)
    unique_concepts = []
    seen_concepts = set()
    
    for item in all_matched_concepts_with_scores:
        concept_text = item['concept']['concept']
        if concept_text not in seen_concepts:
            unique_concepts.append(item['concept'])
            seen_concepts.add(concept_text)
    
    # 상위 TOP_K_CONCEPTS개 선택 (LLM 집중도 향상)
    if len(unique_concepts) >= TOP_K_CONCEPTS:
        concepts = [unique_concepts[i]['concept'] for i in range(TOP_K_CONCEPTS)]
        formulas = [unique_concepts[i]['formula'] for i in range(TOP_K_CONCEPTS)]
    else:
        # fallback: 기존 방식 사용
        concepts, formulas = load_unit_data(unit_name)
    
    allowed_codes_csv = ", ".join(dict.fromkeys([c for c in allowed_codes if c]))
    return formatted.strip(), concepts, formulas, allowed_codes_csv






# 문제 생성 프롬프트 및 태그 파싱 함수
def parse_tags(text):
    q = re.search(r"<question_text>(.*?)</question_text>", text, re.DOTALL)
    a = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
    e = re.search(r"<solution_text>(.*?)</solution_text>", text, re.DOTALL)
    code = re.search(r"<achievement_standard_code>(.*?)</achievement_standard_code>", text, re.DOTALL)
    content = re.search(r"<achievement_standard_content>(.*?)</achievement_standard_content>", text, re.DOTALL)
    return {
        "question": q.group(1).strip() if q else "",
        "answer": a.group(1).strip() if a else "",
        "explanation": e.group(1).strip() if e else "",
        "achievement_standard_code": code.group(1).strip() if code else "",
        "achievement_standard_content": content.group(1).strip() if content else ""
    }

# 시스템 프롬프트 (역할, 규칙, 사고 단계 정의)
COT_SYSTEM_PROMPT = """
당신은 중학생 수학 교사입니다. 주어진 예시문제, 개념, 공식을 바탕으로 새로운 서술형 수학 문제를 생성하십시오.

[생각 1] 예시문제 분석
- 유형 패턴, 문제 내용(question_text), 해설 내용(solution_text), 문체 스타일을 파악하고 이를 참고하여 새로운 문제를 생성한다.
- 단 예시문제와 똑같은 문제로 생성하지 말고, 예시와 유사한 유형이지만 다른 숫자와 상황으로 문제를 새롭게 만들어야한다.
- 각 예시의 성취기준 코드를 확인하고, 생성 문제와 가장 일치하는 성취기준 1개만 선택한다.
- 반드시 제공되는 <allowed_achievement_codes> 목록 중에서만 선택하시오.

[생각 2] 개념·공식 정합성
- 주어진 개념과 공식을 기반으로 문제를 구성한다.
- 예시가 없는 경우에는 개념·공식만으로도 문제를 생성할 수 있도록 한다.

[생각 3] 수학적 구조 설계 및 해 검산 (핵심)
- 문제 조건으로부터 명확한 수학적 식을 세우고 실제로 풀어 해를 구한다.
- 실생활 문제의 경우 해가 정수나 자연수로 나오는지 확인하고, 구한 해를 원래 조건에 대입하여 검산한다.
- 해가 유일하지 않거나 해가 없는 경우 문제 조건을 다시 설계하여 수정한다.

[생각 4] 난이도 정합성
- 입력 난이도(하/중/상)에 맞춰 사고 단계와 응용 수준을 조정한다.
- **하**: 기초 개념 및 단순 계산 위주
- **중**: 개념 응용 및 두 단계 이상의 풀이
- **상**: 실생활 맥락 포함 또는 사고력·전략적 접근 요구

[생각 5] 문제 문항 구성
- 예시 스타일을 참고해 상황과 조건, 질문을 간결히 작성한다.
- **중요**: 실생활 맥락은 중·상급에서만 사용한다. 난이도가 "하"인 경우 절대 실생활 문제를 만들지 말고 순수 수학 문제만 생성한다.

[생각 6] 표현 규칙 적용
- 중학교 수준의 표현만 사용하고 명령형 종결어미(~하시오./~구하시오./~나타내시오.)를 필수로 한다.
- 존댓말과 의문형은 금지하며, 서술문은 "~이다./~나타낸다." 형태를 사용한다.

[생각 7] 단원별 조건 확인
- 해당 단원의 특수 조건을 만족하도록 문제를 구성한다.

[공통 금지사항]
- 소수점 둘째 자리 이상 반올림 금지, 모호한 지시문 금지
- 아래 <요청 단원 특수 조건> 섹션의 조건을 엄격히 적용한다.

**최종 출력**
- 7단계 사고과정을 통해 문제를 생성해보자로 시작
- 각 단계별로 2-3문장으로 설명
- 3단계에서는 계산 과정 포함
- 최종적으로 아래 태그 형식으로 출력
<achievement_standard_code>선택한 성취기준 코드 1개</achievement_standard_code>
<achievement_standard_content>선택한 성취기준 내용 1개</achievement_standard_content>
<question_text>문제 내용 + "(구체적인 문제풀이과정을 작성하시오)"</question_text>
<solution_text>풀이 설명 내용</solution_text>
<final_answer>정답 내용</final_answer>
"""

# 유저 프롬프트 (Q1-A1 → Q2-A2 구조)
COT_USER_PROMPT_TEMPLATE = """
Q1-A1 예제를 참고해 Q2에 대한 응답 A2를 작성하세요.

---

Q1:
당신은 주어진 단원과 난이도에 해당하는 문제를 7단계 사고과정을 거쳐 생성해야 합니다.
최종적으로 생성된 문제는 위 시스템 프롬프트에서 정의한 태그 형식(<question_text>, <solution_text>, <final_answer>)으로 출력하세요.

- 단원: 연립일차방정식
- 난이도: 중
- 예시문제: 주희와 민재는 같은 지역에서 자전거를 타고 출발하여 서로 다른 방향으로 이동하였다. 주희는 시속 x km로, 민재는 시속 y km로 이동하였다. 주희가 민재를 추격하는데 걸린 시간은 30분이며, 그동안 민재가 5km 더 이동하였다. 또한, 두 사람의 속력의 합은 시속 20 km이다. 주희의 시속을 구하시오.
- 개념: 상황을 식으로 표현하기, 속력과 거리 관계
- 공식: x + y = A, x - y = B → x = (A + B)/2, y = (A - B)/2, 속력 = 거리/시간
- 성취기준 코드: 9수02-13
- 성취기준 내용: 미지수가 2개인 연립일차방정식을 풀 수 있고, 이를 활용하여 문제를 해결할 수 있다.

A1: 7단계 사고과정을 통해 문제를 생성해보자.
1. 예시 분석 - 실생활 맥락, 명령형 종결어미, 정수 해 검증 방식 파악  
2. 개념·공식 정합성 - 연립방정식과 속력=거리/시간 공식 활용  
3. 수학적 구조 설계 - x+y=20, y-x=10에서 x=5, y=15로 검산  
4. 난이도 정합성 - 중급 수준, 실생활 응용  
5. 문제 문항 구성 - 자전거 추격 상황  
6. 표현 규칙 - "구하시오" 명령형 종결어미 적용  
7. 단원별 조건 - 유일한 해, 자연수 조건 만족  

<achievement_standard_code>9수02-13</achievement_standard_code>
<achievement_standard_content>연립일차방정식을 활용하여 실생활 문제를 해결할 수 있다.</achievement_standard_content>
<question_text>주희와 민재는 자전거 경주를 하고 있다. 두 사람의 속력의 합은 시속 20km이고, 속력의 차는 시속 10km이다. 민재의 속력을 구하시오. (구체적인 문제풀이과정을 작성하시오)</question_text>
<solution_text>두 사람의 속력을 각각 x, y라 하자. x + y = 20, y - x = 10을 연립하면 2y = 30, y = 15. 따라서 민재의 속력은 15km/h이다.</solution_text>
<final_answer>15km/h</final_answer>

---

Q2:
당신은 아래 조건에 따라 동일한 방식으로 새로운 문제를 생성해야 합니다.
최종적으로 생성된 문제는 태그 형식으로 출력하세요.

- 단원: {unit_name}
- 난이도: {difficulty}
- 예시문제: {few_shot_examples}
- 개념: {concepts}
- 공식: {formulas}
- 성취기준 코드: {allowed_achievement_codes}
- 성취기준 내용: {achievement_standard}
- 성취기준 해설: {achievement_levels}

A2:
"""

# 결합
problem_prompt = ChatPromptTemplate.from_messages([
    ("system", COT_SYSTEM_PROMPT),
    ("human", COT_USER_PROMPT_TEMPLATE)
])






# 난이도 조절 함수
def adjust_difficulty(current_difficulty, correct):
    order = ["하", "중", "상"]
    idx = order.index(current_difficulty)
    if correct:
        # 하→중, 중→상, 상→상
        return order[min(idx + 1, 2)]
    else:
        # 하→하, 중→하, 상→중
        return order[max(idx - 1, 0)]






# 통합된 개인화된 문제 추천 함수
def get_personalized_recommendations(wrong_problems):
    """틀린 문제 → 개인화된 추천 (통합 함수)"""
    if not wrong_problems:
        return "틀린 문제가 없습니다.", []
    
    unit_name = wrong_problems[0].get('unit_name', '')
    
    # 1. 틀린 문제 분석 및 검색 쿼리 생성
    wrong_questions = [p['question'] for p in wrong_problems]
    wrong_answers = [p['student_answer'] for p in wrong_problems]
    concepts, formulas = retrieve_concept_formula(unit_name)
    
    # 개념과 공식을 미리 문자열로 변환
    concepts_text = '\n'.join(concepts)
    formulas_text = '\n'.join(formulas)
    
    query_prompt = f"""
    다음 틀린 문제들을 분석하여 유사한 연습 문제를 찾기 위한 검색 쿼리를 생성하시오.
    
    틀린 문제들:
    {wrong_questions}
    
    학생 답안들:
    {wrong_answers}
    
    관련 개념들:
    {concepts_text}
    
    관련 공식들:
    {formulas_text}
    
    위 정보를 바탕으로 이 학생이 연습해야 할 문제 유형을 파악하고, 
    검색에 사용할 키워드나 쿼리를 생성하시오.
    """
    
    search_query = generation_llm.invoke(query_prompt)
    
    # 2. 유사도 검색
    query_vec = embedding_model.encode([search_query.content], convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_vec, embedding_matrix)[0].cpu().numpy()
    
    unit_mask = problem_data['unit_name'].str.replace(" ", "").str.contains(unit_name.replace(" ", ""), na=False)
    unit_problems = problem_data[unit_mask]
    
    if not unit_problems.empty:
        unit_indices = unit_problems.index.tolist()
        unit_similarities = similarities[unit_indices]
        top_indices = unit_similarities.argsort()[-5:][::-1]
        similar_questions = unit_problems.iloc[top_indices][['question_text', 'topic']]
    else:
        similar_questions = pd.DataFrame(columns=['question_text', 'topic'])
    
    # 3. 개인화된 추천 분석
    recommendation_prompt = f"""
    다음 틀린 문제들을 분석하여 개인화된 연습 문제 추천을 제공하시오.
    
    틀린 문제들:
    {wrong_questions}
    
    학생 답안들:
    {wrong_answers}
    
    추천할 유사 문제들:
    {similar_questions.to_string()}
    
    위 정보를 바탕으로:
    1. 학생의 약점 분석
    2. 각 추천 문제의 연습 목적
    3. 학습 순서 제안
    4. 예상 학습 효과
    """
    
    recommendation_analysis = generation_llm.invoke(recommendation_prompt)
    
    # 4. 최종 포맷팅
    wrong_list_text = "\n".join(
        f"[문제 {i+1}]" + "\n" + f"문제: {p['question']}" + "\n" + f"학생 답안: {p['student_answer']}" + "\n"
        for i, p in enumerate(wrong_problems)
    )
    
    formatted_similar_questions = ""
    for _, row in similar_questions.iterrows():
        topic = row.get('topic', '일반문제')
        question = row['question_text']
        formatted_similar_questions += f"🔹 [{topic}]  {question}" + "\n\n"
    
    return f"[틀린 문제 목록]" + "\n" + f"{wrong_list_text}" + "\n\n" + f"[개인화된 추천 분석]" + "\n" + f"{recommendation_analysis.content}" + "\n\n" + f"[추천 유사 문제들]" + "\n" + f"{formatted_similar_questions}", []


# 문제 검증 및 재생성 함수
def get_excel_problem_by_unit_difficulty(unit_name, difficulty):
    """엑셀에서 해당 단원의 난이도 문제를 가져오는 함수"""
    try:
        print(f"  🔍 엑셀에서 {unit_name} 단원의 {difficulty} 난이도 문제 검색 중...")
        
        # 단원명과 난이도로 필터링
        filtered_data = problem_data[
            (problem_data['unit_name'].str.replace(" ", "") == unit_name.replace(" ", "")) &
            (problem_data['difficulty'] == difficulty)
        ]
        
        print(f"  📊 필터링 결과: {len(filtered_data)}개 문제 발견")
        
        if filtered_data.empty:
            print(f"  ❌ 엑셀에서 {unit_name} 단원의 {difficulty} 난이도 문제를 찾을 수 없습니다.")
            # 디버깅을 위해 전체 단원 목록 출력
            available_units = problem_data['unit_name'].unique()
            print(f"  📋 사용 가능한 단원들: {list(available_units)}")
            return None
        
        # 랜덤하게 하나 선택
        selected_problem = filtered_data.sample(n=1).iloc[0]
        
        # 엑셀 데이터를 파싱된 형태로 변환
        parsed_problem = {
            "question": selected_problem['question_text'],
            "answer": selected_problem['final_answer'],
            "explanation": selected_problem['solution_text'],
            "achievement_standard_code": selected_problem['achievement_standard_code'],
            "achievement_standard_content": selected_problem['achievement_standard_content']
        }
        
        print(f"  ✅ 엑셀에서 {unit_name} 단원의 {difficulty} 난이도 문제를 가져왔습니다.")
        print(f"  📝 문제 미리보기: {parsed_problem['question'][:100]}...")
        print(f"  📝 답안: {parsed_problem['answer']}")
        print(f"  📝 성취기준 코드: {parsed_problem['achievement_standard_code']}")
        
        return parsed_problem
        
    except Exception as e:
        print(f"  ❌ 엑셀에서 문제 가져오기 실패: {e}")
        import traceback
        print(f"  📋 상세 오류: {traceback.format_exc()}")
        return None

def validate_and_regenerate_problem(unit_name, difficulty, achievement_content, achievement_levels, concepts, formulas, few_shot_examples, allowed_codes_csv="", unit_specific_conditions="", max_attempts=MAX_REGEN_ATTEMPTS):
    """생성된 문제가 풀 수 있고 조건에 모순이 없는지 LLM으로 검증하고, 필요시 재생성
    
    반환값: (parsed, is_valid, result, attempt_log)
    attempt_log: 각 시도(1~3회)마다 생성된 문제 초안과 어느 검증 단계에서
                 막혔는지를 기록한 리스트. 교수님께 "검증 과정 중 어디서
                 걸러졌는지" 보여드릴 때 이 리스트를 그대로 엑셀 시트로 저장하면 됨.
    """
    attempt_log = []
    
    for attempt in range(max_attempts):
        print(f"  🔍 문제 검증 시도 {attempt + 1}/{max_attempts}...")
        
        # 문제 생성
        inputs = {
            "achievement_standard": achievement_content,
            "achievement_levels": achievement_levels,
            "concepts": "\n".join(concepts),
            "formulas": "\n".join(formulas),
            "few_shot_examples": few_shot_examples,
            "difficulty": difficulty,
            "unit_name": unit_name, # 단원 이름 추가
            "allowed_achievement_codes": allowed_codes_csv,
            "unit_specific_conditions": unit_specific_conditions
        }
        
        chain = problem_prompt | generation_llm | StrOutputParser()
        result = chain.invoke(inputs)
        parsed = parse_tags(result)
        
        # 1단계: LLM 기반 문제 정합성 검증
        llm_validation_passed = False
        llm_detail = {}
        try:
            llm_validation_result, llm_detail = validate_problem_with_llm(parsed, unit_name, difficulty, parsed["answer"], parsed["explanation"])
            if llm_validation_result:
                print(f"  ✅ LLM 정합성 검증 통과!")
                llm_validation_passed = True
            else:
                print(f"  ❌ LLM 정합성 검증 실패: 문제가 풀리지 않거나 조건에 모순이 있습니다.")
        except Exception as e:
            print(f"  ⚠️ LLM 정합성 검증 중 오류 발생: {e}")
            llm_detail = {"source": "outer_exception", "error": str(e), "failed_items": ["LLM 검증 호출 자체 실패"]}
        
        # 2단계: 형식성 검증 (1단계 통과 여부와 관계없이 항상 실행)
        format_result = {}
        try:
            format_result = basic_validation_fallback(parsed["question"], parsed["answer"], parsed["explanation"], unit_name, difficulty)
            if format_result.get("is_valid", False):
                print(f"  ✅ 형식성 검증 통과! (신뢰도: {format_result.get('confidence', '낮음')})")
                format_validation_passed = True
            else:
                print(f"  ❌ 형식성 검증 실패: {format_result.get('error', '알 수 없는 오류')}")
                format_validation_passed = False
        except Exception as e:
            print(f"  ⚠️ 형식성 검증 중 오류 발생: {e}")
            format_validation_passed = False
            format_result = {"error": str(e)}
        
        # [추가] 이번 시도의 결과를 attempt_log에 기록
        # → 문제가 시도마다 어떻게 바뀌었는지, 어느 검증 단계(정합성/형식성)에서
        #   막혔는지, 정합성 검증이면 6개 세부항목 중 무엇 때문인지까지 남음
        # [추가] 정합성 검증의 6개 세부항목 개별 결과 (없으면 빈 값으로)
        checks = llm_detail.get("validation_checks", {})

        attempt_log.append({
            "시도번호": attempt + 1,
            "생성된_문제": parsed.get("question", ""),
            "생성된_정답": parsed.get("answer", ""),
            "생성된_해설": parsed.get("explanation", ""),
            "LLM_실제풀이답": llm_detail.get("actual_solution", ""),
            "LLM_계산과정": llm_detail.get("calculation_steps", ""),
            "정합성_통과": llm_validation_passed,
            "정합성_실패항목": ", ".join(llm_detail.get("failed_items", [])) if not llm_validation_passed else "",
            "세부_답일치": checks.get("step1_answer_matches", ""),
            "세부_해설타당": checks.get("step1_explanation_valid", ""),
            "세부_해존재유일": checks.get("step2_solution_exists", ""),
            "세부_조건충분": checks.get("step3_conditions_sufficient", ""),
            "세부_단원조건": checks.get("step4_unit_conditions_met", ""),
            "세부_난이도적절": checks.get("step5_difficulty_appropriate", ""),
            "형식성_통과": format_validation_passed,
            "형식성_실패사유": format_result.get("error", "") if not format_validation_passed else "",
            "최종처리": "",  # 아래에서 채워짐 (채택 / 재생성 / 엑셀대체 / 미검증사용)
        })
        
        # 검증 결과에 따른 처리 (논문 기준: 둘 다 통과해야 함)
        if format_validation_passed and llm_validation_passed:
            print(f"  ✅ 모든 검증 통과! 문제 사용 가능")
            print(f"  📊 검증 결과: 정합성=통과, 형식성=통과")
            attempt_log[-1]["최종처리"] = "채택"
            return parsed, True, result, attempt_log
        elif format_validation_passed and not llm_validation_passed:
            print(f"  ❌ 검증 실패: 형식성만 통과, 정합성 실패")
            print(f"  📊 검증 결과: 정합성=실패, 형식성=통과")
        elif not format_validation_passed and llm_validation_passed:
            print(f"  ❌ 검증 실패: 정합성만 통과, 형식성 실패")
            print(f"  📊 검증 결과: 정합성=통과, 형식성=실패")
        else:
            print(f"  ❌ 검증 실패: 둘 다 실패")
            print(f"  📊 검증 결과: 정합성=실패, 형식성=실패")
        
        # 3단계: 재생성 또는 종료
        if attempt < max_attempts - 1:
            print(f"  🔄 문제 재생성 중... (시도 {attempt + 1}/{max_attempts})")
            attempt_log[-1]["최종처리"] = "재생성"
            
        else:
            print(f"  ⚠️ 최대 시도 횟수(3회) 초과. 3단계: 엑셀 데이터 대체를 시도합니다.")
            excel_problem = get_excel_problem_by_unit_difficulty(unit_name, difficulty)
            if excel_problem:
                print(f"  ✅ 엑셀 문제로 대체 완료 (검증된 기존 문항 사용)")
                print(f"  📋 대체 사유: 생성 문항이 3회 검증 실패")
                # 엑셀 문제임을 표시하기 위해 특별한 키 추가
                excel_problem['is_excel_replacement'] = True
                attempt_log[-1]["최종처리"] = "3회 실패 → 엑셀대체"
                return excel_problem, True, None, attempt_log  # 엑셀 문제는 7단계 사고과정 없음
            else:
                print(f"  ⚠️ 해당 조건의 엑셀 문제가 없어 생성 문항을 유지합니다.")
                print(f"  🚨 경고: 검증 미통과 문항 사용 (정합성·형식성 검증 실패)")
                print(f"  📋 사용자에게 투명하게 알림: 이 문제는 검증을 통과하지 못했습니다.")
                attempt_log[-1]["최종처리"] = "3회 실패 → 대체본 없어 미검증 사용"
                return parsed, False, result, attempt_log
    
    return None, False, None, attempt_log

# 검증 프롬프트 템플릿 (ChatPromptTemplate 사용)
validation_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    수학 문제 검증 전문가입니다. 다음 5단계로 검증하세요:
    
    1) 실제 풀이: 문제를 직접 풀어 정답과 계산 과정을 구하세요. 그 다음 다음을 확인하세요:
       - 당신이 계산한 답과 제공된 답이 일치하는가?
       - 제공된 해설의 수학적 내용이 당신의 계산과 의미적으로 일치하는가?
    2) 해 존재/유일성: 수학 문제에 해가 존재하고 유일한지 확인하세요
    3) 조건 충분성: 제시된 조건으로 논리적 풀이가 가능한지 확인하세요
    4) 단원별 조건: <요청 단원 특수 조건>을 만족하는지 확인하세요
    5) 난이도 적절성: 하/중/상 가이드에 맞는지 확인하세요
    
    오직 아래 JSON만 출력하세요:
    {json_schema}
    """),
    ("human", """
    [문제]
    {question}
    
    [문제 정보]
    단원: {unit_name}
    난이도: {difficulty}
    
    [제공된 정답]
    {provided_answer}
    
    [제공된 해설]
    {provided_explanation}
    
    <요청 단원 특수 조건>
    {unit_specific_conditions}
    </요청 단원 특수 조건>
    
    위 문제를 풀어보고 검증하세요.
    """)
])

def validate_problem_with_llm(parsed_problem, unit_name, difficulty, provided_answer=None, provided_explanation=None):
    """
    LLM을 사용하여 문제를 실제로 풀어서 답이 맞는지 검증하는 함수
    """
    question = parsed_problem["question"]
    answer = provided_answer if provided_answer else parsed_problem["answer"]
    explanation = provided_explanation if provided_explanation else parsed_problem["explanation"]
    
    try:
        # 단원별 특수 조건 가져오기
        unit_specific_conditions = get_unit_specific_conditions(unit_name)
        
        # JSON 스키마 정의 (5개 검증 절차 + 답 비교 로직 + 풀이 검증)
        json_schema = """{
  "step1_actual_solution": "LLM이 계산한 정답",
  "step1_calculation_steps": "LLM의 계산 과정",
  "step1_answer_matches": true/false,                    // LLM 계산 답과 제공된 답이 일치하는가?
  "step1_explanation_valid": true/false,                 // 제공된 해설이 LLM 계산과 의미적으로 일치하는가?
  "step2_solution_exists": true/false,                   // 수학 문제에 해가 존재하고 유일한가?
  "step3_conditions_sufficient": true/false,             // 제시된 조건으로 논리적 풀이가 가능한가?
  "step4_unit_conditions_met": true/false,               // 단원별 특수 조건을 만족하는가?
  "step5_difficulty_appropriate": true/false,            // 난이도가 적절한가?
  "overall_valid": true/false                            // 전체 검증 통과 여부
}"""
        
        # 재검증 프롬프트 생성 (문제와 제공된 정답/해설을 함께 검증)
        validation_inputs = {
            "question": question,
            "unit_name": unit_name,
            "difficulty": difficulty,
            "provided_answer": answer,
            "provided_explanation": explanation,
            "unit_specific_conditions": unit_specific_conditions,
            "json_schema": json_schema
        }
        
        chain = validation_prompt | validation_llm | StrOutputParser()
        response = chain.invoke(validation_inputs)
        
        # 응답 내용 확인 및 디버깅
        print(f"  🔍 LLM 검증 응답: {response[:200]}...")
        
        # JSON 파싱 시도
        try:
            result = json.loads(response)
        except json.JSONDecodeError as json_error:
            print(f"  ❌ JSON 파싱 실패: {json_error}")
            print(f"  📝 원본 응답: {response}")
            
            # JSON 파싱 실패 시 기본 검증으로 fallback
            print(f"  ⚠️ JSON 파싱 실패로 기본 검증으로 전환")
            fallback_result = basic_validation_fallback(question, answer, explanation, unit_name, difficulty)
            detail = {"source": "json_parse_error", "raw_response": response[:300], "failed_items": ["LLM 검증 응답 JSON 파싱 실패"]}
            return fallback_result.get("is_valid", False), detail
        
        # 검증 결과 확인 - 5개 검증 절차 모두 통과해야 함
        validation_checks = {
            "step1_answer_matches": result.get("step1_answer_matches", False),
            "step1_explanation_valid": result.get("step1_explanation_valid", False),
            "step2_solution_exists": result.get("step2_solution_exists", False),
            "step3_conditions_sufficient": result.get("step3_conditions_sufficient", False),
            "step4_unit_conditions_met": result.get("step4_unit_conditions_met", False),
            "step5_difficulty_appropriate": result.get("step5_difficulty_appropriate", False)
        }
        
        # 모든 검증 조건을 통과해야 함
        is_valid = all(validation_checks.values())
        
        # step1 관련 정보는 별도로 저장 (디버깅용)
        actual_solution = result.get("step1_actual_solution", "")
        calculation_steps = result.get("step1_calculation_steps", "")
        
        # [추가] 어느 세부 검증 항목에서 실패했는지 사람이 읽을 수 있는 문자열로 정리
        # (기존엔 이 정보가 print()로만 찍히고 버려졌음 — 이제 detail로 함께 반환하여
        #  엑셀 로그에 "몇 번째 시도가 어느 항목에서 막혔는지"를 남길 수 있게 함)
        step_label = {
            "step1_answer_matches": "정답 불일치",
            "step1_explanation_valid": "해설-계산 불일치",
            "step2_solution_exists": "해 없음/비유일",
            "step3_conditions_sufficient": "조건 불충분",
            "step4_unit_conditions_met": "단원별 특수조건 위반",
            "step5_difficulty_appropriate": "난이도 부적절",
        }
        failed_items = [label for key, label in step_label.items() if not validation_checks.get(key, False)]
        detail = {
            "source": "llm_validation",
            "validation_checks": validation_checks,
            "failed_items": failed_items,          # 예: ["단원별 특수조건 위반", "해 없음/비유일"]
            "actual_solution": actual_solution,
            "calculation_steps": calculation_steps,
        }
        
        # 검증 실패 개수 확인
        failed_checks = sum(1 for check, passed in validation_checks.items() if not passed)
        if failed_checks > 0:
            print(f"  ❌ 검증 실패: {failed_checks}개 항목 실패 → {', '.join(failed_items)}")
            print(f"    📝 LLM 계산 답: {actual_solution}")
            print(f"    📝 계산 과정: {calculation_steps}")
            for check_name, check_result in validation_checks.items():
                status = "✅" if check_result else "❌"
                step_name = check_name.replace("step", "").replace("_", " ").title()
                print(f"    {status} {step_name}: {check_result}")
        else:
            print(f"  ✅ 모든 검증 항목 통과!")
            print(f"    📝 LLM 계산 답: {actual_solution}")
            print(f"    📝 계산 과정: {calculation_steps}")
        
        if not is_valid:
            print(f"  ❌ 문제 검증 실패")
        else:
            print(f"  ✅ 문제 검증 통과!")
        
        return is_valid, detail
        
    except Exception as e:
        print(f"  ❌ LLM 검증 중 예외 발생: {e}")
        # LLM 검증 실패 시 기본 검증으로 fallback
        # [버그 수정] basic_validation_fallback은 dict({"is_valid":..., ...})를 반환하는데
        # 이 함수는 다른 정상 경로에서 bool을 반환하므로, 호출부의 `if llm_validation_result:`가
        # dict를 받으면 내용과 무관하게 항상 True로 판정되는 문제가 있었다.
        # is_valid 값만 꺼내 bool로 통일해서 반환한다.
        fallback_result = basic_validation_fallback(question, answer, explanation, unit_name, difficulty)
        detail = {"source": "exception", "error": str(e), "failed_items": ["LLM 검증 중 오류"]}
        return fallback_result.get("is_valid", False), detail

def basic_validation_fallback(question, answer, explanation, unit_name, difficulty):
    """LLM 검증 실패 시 기본 검증으로 대체 - 간단한 형식 검증만 수행"""
    
    print(f"  🔍 기본 검증 시작...")
    
    # 기본적인 검증만 수행
    if not question or not answer or not explanation:
        print(f"  ❌ 기본 검증 실패: 문제, 정답, 해설 중 누락된 요소가 있습니다.")
        return {"is_valid": False, "error": "문제, 정답, 해설 중 누락된 요소가 있습니다.", "confidence": "높음"}
    
    # 문제에 명령형 지시문이 있는지 확인 (더 유연하게)
    command_keywords = ["구하시오", "계산하시오", "설명하시오", "나타내시오", "작성하시오", "구하여라", "오", "라", "요", "하세요", "하라", "하시오"]
    has_command = any(keyword in question for keyword in command_keywords)
    
    # 문제 길이 확인 (엄격하게 유지 - 너무 길면 의미없는 문제)
    if len(question) < 20 or len(question) > 500:
        print(f"  ❌ 기본 검증 실패: 문제 길이가 적절하지 않습니다. (길이: {len(question)})")
        return {"is_valid": False, "error": "문제 길이가 적절하지 않습니다.", "confidence": "보통"}
    
    # 기본 검증 통과 조건을 더 유연하게
    if not has_command:
        print(f"  ⚠️ 명령형 지시문이 없지만 문제 내용으로 판단하여 통과")
    
    # 핵심 요소만 있으면 통과
    if question and answer and explanation:
        print(f"  ✅ 기본 검증 통과! (핵심 요소 존재)")
        return {"is_valid": True, "error": "", "confidence": "보통"}


# =========================================================
# 배치 생성 함수 (11개 단원 × 3난이도 × 3문제 = 99문항 자동 생성)
# 기존 start_session()의 적응형(사람이 문제 풀며 난이도 조정) 로직은 제거하고,
# 1차/2차 코드와 동일한 "고정 3문제 × 3난이도" 배치 구조로 재구성.
# 검증 루프(validate_and_regenerate_problem)는 그대로 사용하되,
# 검증 과정이 결과 엑셀에도 남도록 로그용 컬럼을 추가함.
# =========================================================
def generate_batch_problems(unit_list, run_seed=None):
    all_data = []
    all_logs = []  # 시도별(최대 3회) 상세 검증 로그 — "결과" 시트와 별도로 저장

    for unit_name in unit_list:
        print(f"\n📘 단원: {unit_name}")

        for difficulty in ["하", "중", "상"]:
            # [추가] 같은 단원+난이도 안에서 이미 채택된 문제 텍스트를 기록해서
            # 중복 생성을 방지한다. (원래 코드는 1/2/3번째 문제를 서로 독립적으로
            # 생성해서, 같은 예시 풀을 참고하다 보니 완전히 동일한 문제가
            # 나오는 경우가 있었음 — seed1 결과에서 실제로 확인된 문제)
            generated_questions_in_slot = []

            for i in range(3):  # 난이도당 3문제
                problem_index = i + 1
                print(f"\n--- {unit_name} / {difficulty} / {problem_index}번째 문제 ---")

                # RAG 기반 예시 및 개념-공식 검색 (기존 로직 그대로 재사용)
                few_shot_examples, smart_concepts, smart_formulas, allowed_codes_csv = smart_get_few_shot_examples(unit_name, difficulty)

                if smart_concepts and smart_formulas:
                    concepts, formulas = smart_concepts, smart_formulas
                else:
                    concepts, formulas = retrieve_concept_formula(unit_name)

                # 성취기준 코드/내용 조회
                unit_rows = problem_data[
                    problem_data['unit_name'].str.replace(" ", "").str.contains(unit_name.replace(" ", ""), na=False)
                ]
                if not unit_rows.empty:
                    standard_code = unit_rows.iloc[0].get('achievement_standard_code', '')
                    achievement_content, achievement_levels = get_achievement_standard_info_from_code(standard_code)
                else:
                    achievement_content, achievement_levels = "성취기준 정보 없음", ""

                unit_specific_conditions = get_unit_specific_conditions(unit_name)

                # [추가] 중복이 아닌 문제가 나올 때까지 최대 3회(원래 검증루프 1세트당)
                # 더 시도한다. 매 시도마다 검증 루프(최대 3회 재생성)가 통째로 다시 도므로
                # 최악의 경우 시도 횟수가 늘어날 수 있음 — 비용은 크게 늘지 않음(문항당 몇 회 수준).
                max_duplicate_retries = 3
                combined_attempt_log = []
                for dup_attempt in range(max_duplicate_retries):
                    parsed, is_valid, cot_result, attempt_log = validate_and_regenerate_problem(
                        unit_name,
                        difficulty,
                        achievement_content,
                        achievement_levels,
                        concepts,
                        formulas,
                        few_shot_examples,
                        allowed_codes_csv,
                        unit_specific_conditions
                    )

                    question_text = (parsed or {}).get("question", "")
                    is_duplicate = question_text in generated_questions_in_slot

                    # [수정] 검증 자체는 통과했지만 이미 나온 문제와 중복이라 폐기하는 경우,
                    # 내부 검증 로직이 이미 "채택"이라고 써놓은 라벨을 "폐기(중복)"으로 덮어써서
                    # 실제로 최종 채택된 것과 혼동되지 않게 한다.
                    if is_duplicate and attempt_log:
                        attempt_log[-1]["최종처리"] = "폐기(중복)"

                    combined_attempt_log.extend(attempt_log)

                    if not is_duplicate:
                        break  # 중복 아님 → 채택
                    print(f"  🔁 같은 단원·난이도 내 중복 문제 감지 — 재생성 (중복재시도 {dup_attempt + 1}/{max_duplicate_retries})")
                else:
                    print(f"  ⚠️ {max_duplicate_retries}회 재시도에도 중복 회피 실패 — 중복 상태로 그대로 사용")

                # [추가] 중복재시도를 포함한 "시도번호"를 1부터 끊기지 않는 연속 번호로 다시 매긴다.
                # (기존엔 검증루프 내부 재시도 번호가 중복재시도마다 1로 리셋되어,
                #  로그만 보면 "1번 시도"가 여러 번 나오는 것처럼 헷갈렸음)
                for idx, log_entry in enumerate(combined_attempt_log):
                    log_entry["시도번호"] = idx + 1

                attempt_log = combined_attempt_log

                if parsed is None:
                    print(f"  🚨 생성 완전 실패 — 이 슬롯은 결측으로 남김")
                    parsed = {"question": "", "answer": "", "explanation": "", "rubric": []}
                    is_valid = False

                generated_questions_in_slot.append(parsed.get("question", ""))

                all_data.append({
                    "run_seed": run_seed,
                    "단원명": unit_name,
                    "난이도": difficulty,
                    "문제번호": problem_index,
                    "문제": parsed.get("question", ""),
                    "정답": parsed.get("answer", ""),
                    "해설": parsed.get("explanation", ""),
                    "검증_통과여부": is_valid,
                    "엑셀_대체여부": parsed.get("is_excel_replacement", False),
                    "총_시도횟수": len(attempt_log),
                })

                # 이 문제에 대한 모든 시도(1~3회 × 중복재시도)를 로그 테이블에 단원/난이도/문제번호와 함께 누적
                for log_entry in attempt_log:
                    all_logs.append({
                        "run_seed": run_seed,
                        "단원명": unit_name,
                        "난이도": difficulty,
                        "문제번호": problem_index,
                        **log_entry,
                    })

    return pd.DataFrame(all_data), pd.DataFrame(all_logs)


if __name__ == "__main__":

    unit_list = [
        "소인수분해",
        "문자의사용과식",
        "단항식과다항식의계산",
        "일차방정식",
        "연립일차방정식",
        "이차방정식",
        "정수와유리수",
        "유리수와순환소수",
        "실수와그연산",
        "일차부등식",
        "다항식의곱셈과인수분해",
    ]

    df_result, df_log = generate_batch_problems(unit_list, run_seed=RUN_SEED)

    output_path = f"샘플문제3차_seed{RUN_SEED}.xlsx"
    # 한 엑셀 파일 안에 두 개 시트로 저장:
    #  - "결과": 최종 채택된 99문항 (지금까지 보던 형태)
    #  - "검증로그": 문항별로 1~3번째 시도에서 각각 무엇이 생성됐고
    #               정합성/형식성 중 어디서, 어떤 세부항목 때문에 막혔는지 전부 기록
    #
    # [버그 수정 v2] 파인튜닝 모델이 \frac{2}{3} 같은 LaTeX 수식을 답하는 경우,
    # 문자열 안에 \f, \v 등이 실제 제어문자(폼피드 등)로 남아있으면
    # openpyxl이 "IllegalCharacterError"를 내며 저장에 실패한다.
    # DataFrame 전체(NaN 포함 모든 값)를 순회하며 모든 문자열 셀을 강제로 정제한다.
    import re as _re_sanitize
    _CONTROL_CHAR_RE = _re_sanitize.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    def _sanitize_for_excel(df):
        df = df.copy()
        n_cleaned = 0
        for col in df.columns:
            def _clean(x):
                nonlocal n_cleaned
                if isinstance(x, str):
                    new_x = _CONTROL_CHAR_RE.sub("", x)
                    if new_x != x:
                        n_cleaned += 1
                    return new_x
                return x
            df[col] = df[col].apply(_clean)
        print(f"  🧹 엑셀 저장 전 정제: '{df.columns[0] if len(df.columns) else ''}' 등 포함 테이블에서 {n_cleaned}개 셀의 제어문자 제거")
        return df

    df_result = _sanitize_for_excel(df_result)
    df_log = _sanitize_for_excel(df_log)

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_result.to_excel(writer, sheet_name="결과", index=False)
            df_log.to_excel(writer, sheet_name="검증로그", index=False)
    except Exception as e:
        # [최후 안전장치] 그래도 실패하면, 모든 셀을 강제로 문자열화 + 재정제해서 다시 시도
        print(f"  ⚠️ 1차 저장 실패({e}), 전체 셀 강제 정제 후 재시도합니다...")
        def _force_clean_all(df):
            df = df.copy()
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: _CONTROL_CHAR_RE.sub("", str(x)) if pd.notna(x) else x
                )
            return df
        df_result = _force_clean_all(df_result)
        df_log = _force_clean_all(df_log)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_result.to_excel(writer, sheet_name="결과", index=False)
            df_log.to_excel(writer, sheet_name="검증로그", index=False)
        print(f"  ✅ 재시도로 저장 성공")

    n_total = len(df_result)
    n_valid = int(df_result["검증_통과여부"].sum())
    n_excel_fallback = int(df_result["엑셀_대체여부"].sum())
    n_first_try = int((df_log[df_log["최종처리"] == "채택"]["시도번호"] == 1).sum())
    # [버그 수정] "검증_통과여부"엔 엑셀대체 문항도 True로 포함돼 있어(둘 다 유효 채택),
    # 자체검증만의 순수 통과 건수는 n_valid에서 엑셀대체 건수를 빼야 함.
    # 기존 식은 이걸 또 빼서 음수가 나오는 버그가 있었음.
    n_self_validated = n_valid - n_excel_fallback
    n_unverified = n_total - n_valid
    print(f"\n✅ 엑셀 저장 완료: {output_path} (시트: 결과 / 검증로그)")
    print(f"📊 총 {n_total}문항 | 자체검증통과 {n_self_validated}건 | 엑셀대체 {n_excel_fallback}건 | 미검증사용 {n_unverified}건")
    print(f"📊 1회 시도 만에 통과: {n_first_try}건 / {n_total}건 ({n_first_try/n_total*100:.1f}%)")
