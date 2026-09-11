# retrieval.py — RAG 기반 검색 모듈
# 단원별 개념·공식 검색, few-shot 예시문항 검색(코사인 유사도), 코퍼스 대체,
# 개인화된 문제 추천에 해당하는 코드를 이 파일에 모았다.
# 원본: src/generation/v3_final.py (단일 파일)에서 기능별로 분리한 것으로,
# 로직은 원본과 동일하다.

import sys
from pathlib import Path

# common.py는 ../generation/ 에 있음 (src/generation/common.py)
sys.path.append(str(Path(__file__).resolve().parent.parent / "generation"))

import pandas as pd
from sentence_transformers import util

from common import (
    SIMILARITY_THRESHOLD,
    TOP_K_CONCEPTS,
    MAX_FEWSHOT_EXAMPLES,
    UNIT_CSV_DIR,
    get_generation_llm,
    get_problem_data,
    get_embedding_model,
    get_problem_embeddings,
)


# -------------------------
# 단원별 개념/공식
# -------------------------
def load_unit_data(unit_name):
    """단원별 개념/공식 CSV 로드"""
    path = UNIT_CSV_DIR / f"{unit_name.replace(' ', '')}_통합.csv"
    df = pd.read_csv(path)
    concepts = df['concept'].dropna().tolist()
    formulas = df['formula'].dropna().tolist()
    return concepts, formulas


def retrieve_concept_formula(unit_name):
    """단원별 개념/공식 가져오기 (전체 리스트 반환)"""
    return load_unit_data(unit_name)


# -------------------------
# RAG 기반 Few-shot 예시 검색 (전략 A 기반)
# -------------------------
def rag_search_by_unit_difficulty(unit_name, difficulty):
    """단원명 + 난이도 기준으로 문제 필터링하고 예시 최대 MAX_FEWSHOT_EXAMPLES개 추출"""
    problem_data = get_problem_data()
    subset = problem_data[
        (problem_data['unit_name'].str.replace(" ", "") == unit_name.replace(" ", "")) &
        (problem_data['difficulty'] == difficulty)
    ]

    if subset.empty:
        return pd.DataFrame()

    # 서로 다른 성취기준 코드를 가진 문제들을 우선 선택
    unique_standards = subset['achievement_standard_code'].dropna().unique()

    if len(unique_standards) >= 1:
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
    # 1. RAG 기반 예시 문제 검색
    few_shot_examples = rag_search_by_unit_difficulty(unit_name, difficulty_level)

    if few_shot_examples.empty:
        print(f"⚠️ {unit_name} 단원의 {difficulty_level} 난이도 예시 문제가 없습니다.")
        print(f"📚 개념과 공식만을 참고하여 문제를 생성합니다.")

        # 예시 없이 개념/공식만 사용
        concepts, formulas = load_unit_data(unit_name)
        return "[예시 문제 없음 - 개념과 공식만 참고하여 문제 생성]", concepts, formulas, ""

    # 2. Few-shot 예시 포맷팅
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

    # 3. Topic 기반 관련 개념-공식 검색
    try:
        concepts_formulas_df = load_unit_data(unit_name)
        concepts_formulas_df = pd.DataFrame({'concept': concepts_formulas_df[0], 'formula': concepts_formulas_df[1]})
    except Exception:
        # CSV 파일이 없으면 기존 방식 사용
        concepts, formulas = load_unit_data(unit_name)
        allowed_codes_csv = ", ".join(dict.fromkeys([c for c in allowed_codes if c]))
        return formatted.strip(), concepts, formulas, allowed_codes_csv

    # RAG 기반 개념-공식 매칭 (코사인 유사도)
    embedding_model = get_embedding_model()
    all_matched_concepts_with_scores = []

    for _, row in few_shot_examples.iterrows():
        individual_topic = row['topic']

        concept_embeddings = embedding_model.encode(concepts_formulas_df['concept'].tolist())
        topic_embedding = embedding_model.encode([individual_topic])

        similarities = util.pytorch_cos_sim(topic_embedding, concept_embeddings)[0].cpu().numpy()

        # 임계값 기반 필터링
        threshold = SIMILARITY_THRESHOLD
        top_indices = similarities.argsort()[-5:][::-1]  # 상위 5개 선택

        for idx in top_indices:
            if similarities[idx] > threshold:
                matched_concept = concepts_formulas_df.iloc[idx]
                similarity_score = similarities[idx]

                all_matched_concepts_with_scores.append({
                    'concept': matched_concept,
                    'similarity': similarity_score
                })

    # 유사도 순서대로 정렬
    all_matched_concepts_with_scores.sort(key=lambda x: x['similarity'], reverse=True)

    # 중복 제거
    unique_concepts = []
    seen_concepts = set()
    for item in all_matched_concepts_with_scores:
        concept_text = item['concept']['concept']
        if concept_text not in seen_concepts:
            unique_concepts.append(item['concept'])
            seen_concepts.add(concept_text)

    # 상위 TOP_K_CONCEPTS개 선택
    if len(unique_concepts) >= TOP_K_CONCEPTS:
        concepts = [unique_concepts[i]['concept'] for i in range(TOP_K_CONCEPTS)]
        formulas = [unique_concepts[i]['formula'] for i in range(TOP_K_CONCEPTS)]
    else:
        # fallback: 기존 방식 사용
        concepts, formulas = load_unit_data(unit_name)

    allowed_codes_csv = ", ".join(dict.fromkeys([c for c in allowed_codes if c]))
    return formatted.strip(), concepts, formulas, allowed_codes_csv


# -------------------------
# 코퍼스 대체 (검증 3회 실패 시 사용)
# -------------------------
def get_excel_problem_by_unit_difficulty(unit_name, difficulty):
    """엑셀 코퍼스에서 해당 단원·난이도의 검증된 기존 문제를 가져오는 함수"""
    try:
        print(f"  🔍 엑셀에서 {unit_name} 단원의 {difficulty} 난이도 문제 검색 중...")
        problem_data = get_problem_data()

        filtered_data = problem_data[
            (problem_data['unit_name'].str.replace(" ", "") == unit_name.replace(" ", "")) &
            (problem_data['difficulty'] == difficulty)
        ]

        print(f"  📊 필터링 결과: {len(filtered_data)}개 문제 발견")

        if filtered_data.empty:
            print(f"  ❌ 엑셀에서 {unit_name} 단원의 {difficulty} 난이도 문제를 찾을 수 없습니다.")
            available_units = problem_data['unit_name'].unique()
            print(f"  📋 사용 가능한 단원들: {list(available_units)}")
            return None

        selected_problem = filtered_data.sample(n=1).iloc[0]

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


# -------------------------
# 개인화된 문제 추천
# -------------------------
# 참고: 이 기능(오답 기반 재추천)은 시스템에 설계·구현되어 있으나,
# 논문에서는 정량적 검증 대상에 포함하지 않았다(심사 답변서 참고).
# 부록 A/B의 정식 프롬프트 명세에도 포함하지 않았다.
def get_personalized_recommendations(wrong_problems):
    """틀린 문제 → 개인화된 추천 (통합 함수)"""
    if not wrong_problems:
        return "틀린 문제가 없습니다.", []

    unit_name = wrong_problems[0].get('unit_name', '')

    wrong_questions = [p['question'] for p in wrong_problems]
    wrong_answers = [p['student_answer'] for p in wrong_problems]
    concepts, formulas = retrieve_concept_formula(unit_name)

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

    generation_llm = get_generation_llm()
    search_query = generation_llm.invoke(query_prompt)

    embedding_model = get_embedding_model()
    embedding_matrix = get_problem_embeddings()
    problem_data = get_problem_data()

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

    wrong_list_text = "\n".join(
        f"[문제 {i+1}]" + "\n" + f"문제: {p['question']}" + "\n" + f"학생 답안: {p['student_answer']}" + "\n"
        for i, p in enumerate(wrong_problems)
    )

    formatted_similar_questions = ""
    for _, row in similar_questions.iterrows():
        topic = row.get('topic', '일반문제')
        question = row['question_text']
        formatted_similar_questions += f"🔹 [{topic}]  {question}" + "\n\n"

    return (
        f"[틀린 문제 목록]" + "\n" + f"{wrong_list_text}" + "\n\n" +
        f"[개인화된 추천 분석]" + "\n" + f"{recommendation_analysis.content}" + "\n\n" +
        f"[추천 유사 문제들]" + "\n" + f"{formatted_similar_questions}"
    ), []
