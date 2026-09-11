# validation.py — 문항 검증 모듈
# 정합성 검증(LLM이 직접 풀어서 재검증)과 형식성 검증(규칙 기반)을 담당한다.
# 프롬프트 원문: prompts/validation/consistency.md, prompts/validation/format.md
# 원본: src/generation/v3_final.py (단일 파일)에서 기능별로 분리한 것으로,
# 로직은 원본과 동일하다.

import sys
import json
from pathlib import Path

# common.py는 ../generation/ 에 있음 (src/generation/common.py)
sys.path.append(str(Path(__file__).resolve().parent.parent / "generation"))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import get_validation_llm, get_unit_specific_conditions


# -------------------------
# 정합성 검증 프롬프트 템플릿
# 원문: prompts/validation/consistency.md
# -------------------------
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
    LLM을 사용하여 문제를 실제로 풀어서 답이 맞는지 검증하는 함수 (정합성 검증)
    """
    question = parsed_problem["question"]
    answer = provided_answer if provided_answer else parsed_problem["answer"]
    explanation = provided_explanation if provided_explanation else parsed_problem["explanation"]

    try:
        unit_specific_conditions = get_unit_specific_conditions(unit_name)

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

        validation_inputs = {
            "question": question,
            "unit_name": unit_name,
            "difficulty": difficulty,
            "provided_answer": answer,
            "provided_explanation": explanation,
            "unit_specific_conditions": unit_specific_conditions,
            "json_schema": json_schema
        }

        validation_llm = get_validation_llm()
        chain = validation_prompt | validation_llm | StrOutputParser()
        response = chain.invoke(validation_inputs)

        print(f"  🔍 LLM 검증 응답: {response[:200]}...")

        try:
            result = json.loads(response)
        except json.JSONDecodeError as json_error:
            print(f"  ❌ JSON 파싱 실패: {json_error}")
            print(f"  📝 원본 응답: {response}")
            print(f"  ⚠️ JSON 파싱 실패로 기본 검증으로 전환")
            fallback_result = basic_validation_fallback(question, answer, explanation, unit_name, difficulty)
            detail = {"source": "json_parse_error", "raw_response": response[:300], "failed_items": ["LLM 검증 응답 JSON 파싱 실패"]}
            return fallback_result.get("is_valid", False), detail

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

        actual_solution = result.get("step1_actual_solution", "")
        calculation_steps = result.get("step1_calculation_steps", "")

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
            "failed_items": failed_items,
            "actual_solution": actual_solution,
            "calculation_steps": calculation_steps,
        }

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
        fallback_result = basic_validation_fallback(question, answer, explanation, unit_name, difficulty)
        detail = {"source": "exception", "error": str(e), "failed_items": ["LLM 검증 중 오류"]}
        return fallback_result.get("is_valid", False), detail


def basic_validation_fallback(question, answer, explanation, unit_name, difficulty):
    """LLM 검증 실패 시 기본 검증으로 대체 (형식성 검증)
    원문/판정 기준: prompts/validation/format.md
    """
    print(f"  🔍 기본 검증 시작...")

    if not question or not answer or not explanation:
        print(f"  ❌ 기본 검증 실패: 문제, 정답, 해설 중 누락된 요소가 있습니다.")
        return {"is_valid": False, "error": "문제, 정답, 해설 중 누락된 요소가 있습니다.", "confidence": "높음"}

    command_keywords = ["구하시오", "계산하시오", "설명하시오", "나타내시오", "작성하시오", "구하여라", "오", "라", "요", "하세요", "하라", "하시오"]
    has_command = any(keyword in question for keyword in command_keywords)

    if len(question) < 20 or len(question) > 500:
        print(f"  ❌ 기본 검증 실패: 문제 길이가 적절하지 않습니다. (길이: {len(question)})")
        return {"is_valid": False, "error": "문제 길이가 적절하지 않습니다.", "confidence": "보통"}

    if not has_command:
        print(f"  ⚠️ 명령형 지시문이 없지만 문제 내용으로 판단하여 통과")

    if question and answer and explanation:
        print(f"  ✅ 기본 검증 통과! (핵심 요소 존재)")
        return {"is_valid": True, "error": "", "confidence": "보통"}
