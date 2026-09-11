# generation.py — 문항 생성 메인 파이프라인
# 7단계 CoT 프롬프트로 문항을 생성하고, 검증-재생성 루프(validation.py)와
# RAG 검색(retrieval.py)을 오케스트레이션하여 최종 문항을 산출한다.
# 프롬프트 원문: prompts/generation/v3_cot.md
# 원본: 이 파일 + common.py + ../retrieval/retrieval.py + ../validation/validation.py는
# 원래 하나의 파일(v3_final.py)이었던 것을 기능별로 분리한 것이다. 로직은 원본과 동일하다.

import re
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common import (
    MAX_REGEN_ATTEMPTS,
    get_generation_llm,
    get_problem_data,
    get_achievement_standard_info_from_code,
    get_unit_specific_conditions,
)

# retrieval.py, validation.py는 각각 ../retrieval/, ../validation/ 에 있음
sys.path.append(str(Path(__file__).resolve().parent.parent / "retrieval"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "validation"))

from retrieval import smart_get_few_shot_examples, retrieve_concept_formula, get_excel_problem_by_unit_difficulty
from validation import validate_problem_with_llm, basic_validation_fallback

DEVELOPMENT_MODE = True

# =========================================================
# 회차별 시드 고정 (재현성/안정성 검증용)
# 500문항(99×5회) 실험처럼 여러 회차를 독립적으로 반복할 때,
# 회차마다 "다른" 시드를 명시적으로 지정한다.
#   - 회차 간에는 서로 다른 무작위 샘플링이 이뤄져야 "독립적인 반복실험"이 됨
#   - 각 회차 자체는 동일 시드로 재실행하면 동일 결과가 재현되어야 함
# 실행 시 예: python generation.py --seed 1  (1~5회차에 맞춰 1~5로 바꿔서 실행)
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


# -------------------------
# 태그 파싱
# -------------------------
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


# -------------------------
# 문항 생성 CoT 프롬프트 (원문: prompts/generation/v3_cot.md)
# -------------------------
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

problem_prompt = ChatPromptTemplate.from_messages([
    ("system", COT_SYSTEM_PROMPT),
    ("human", COT_USER_PROMPT_TEMPLATE)
])


# -------------------------
# 난이도 조절 함수 (학습 적응 단계에서 사용)
# -------------------------
def adjust_difficulty(current_difficulty, correct):
    order = ["하", "중", "상"]
    idx = order.index(current_difficulty)
    if correct:
        # 하→중, 중→상, 상→상
        return order[min(idx + 1, 2)]
    else:
        # 하→하, 중→하, 상→중
        return order[max(idx - 1, 0)]


# -------------------------
# 생성-검증-재생성 루프 (오케스트레이션)
# validation.py의 판정 함수를 호출하고, 실패 시 retrieval.py의 코퍼스 대체로 넘어간다.
# -------------------------
def validate_and_regenerate_problem(unit_name, difficulty, achievement_content, achievement_levels, concepts, formulas, few_shot_examples, allowed_codes_csv="", unit_specific_conditions="", max_attempts=MAX_REGEN_ATTEMPTS):
    """생성된 문제가 풀 수 있고 조건에 모순이 없는지 검증하고, 필요시 재생성

    반환값: (parsed, is_valid, result, attempt_log)
    attempt_log: 각 시도(1~3회)마다 생성된 문제 초안과 어느 검증 단계에서
                 막혔는지를 기록한 리스트. 교수님께 "검증 과정 중 어디서
                 걸러졌는지" 보여드릴 때 이 리스트를 그대로 엑셀 시트로 저장하면 됨.
    """
    attempt_log = []
    generation_llm = get_generation_llm()

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
            "unit_name": unit_name,
            "allowed_achievement_codes": allowed_codes_csv,
            "unit_specific_conditions": unit_specific_conditions
        }

        chain = problem_prompt | generation_llm | StrOutputParser()
        result = chain.invoke(inputs)
        parsed = parse_tags(result)

        # 1단계: LLM 기반 정합성 검증 (validation.py)
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

        # 2단계: 형식성 검증 (validation.py, 1단계 통과 여부와 관계없이 항상 실행)
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
        elif not format_validation_passed and llm_validation_passed:
            print(f"  ❌ 검증 실패: 정합성만 통과, 형식성 실패")
        else:
            print(f"  ❌ 검증 실패: 둘 다 실패")

        # 3단계: 재생성 또는 종료
        if attempt < max_attempts - 1:
            print(f"  🔄 문제 재생성 중... (시도 {attempt + 1}/{max_attempts})")
            attempt_log[-1]["최종처리"] = "재생성"
        else:
            print(f"  ⚠️ 최대 시도 횟수({max_attempts}회) 초과. 3단계: 코퍼스 대체를 시도합니다.")
            excel_problem = get_excel_problem_by_unit_difficulty(unit_name, difficulty)
            if excel_problem:
                print(f"  ✅ 코퍼스 문제로 대체 완료 (검증된 기존 문항 사용)")
                excel_problem['is_excel_replacement'] = True
                attempt_log[-1]["최종처리"] = "3회 실패 → 엑셀대체"
                return excel_problem, True, None, attempt_log
            else:
                print(f"  ⚠️ 해당 조건의 코퍼스 문제가 없어 생성 문항을 유지합니다.")
                print(f"  🚨 경고: 검증 미통과 문항 사용 (정합성·형식성 검증 실패)")
                attempt_log[-1]["최종처리"] = "3회 실패 → 대체본 없어 미검증 사용"
                return parsed, False, result, attempt_log

    return None, False, None, attempt_log


# =========================================================
# 배치 생성 함수 (11개 단원 × 3난이도 × 3문제 = 99문항 자동 생성)
# =========================================================
def generate_batch_problems(unit_list, run_seed=None):
    all_data = []
    all_logs = []  # 시도별(최대 3회) 상세 검증 로그 — "결과" 시트와 별도로 저장

    for unit_name in unit_list:
        print(f"\n📘 단원: {unit_name}")

        for difficulty in ["하", "중", "상"]:
            # 같은 단원+난이도 안에서 이미 채택된 문제 텍스트를 기록해서 중복 생성을 방지
            generated_questions_in_slot = []

            for i in range(3):  # 난이도당 3문제
                problem_index = i + 1
                print(f"\n--- {unit_name} / {difficulty} / {problem_index}번째 문제 ---")

                # RAG 기반 예시 및 개념-공식 검색 (retrieval.py)
                few_shot_examples, smart_concepts, smart_formulas, allowed_codes_csv = smart_get_few_shot_examples(unit_name, difficulty)

                if smart_concepts and smart_formulas:
                    concepts, formulas = smart_concepts, smart_formulas
                else:
                    concepts, formulas = retrieve_concept_formula(unit_name)

                # 성취기준 코드/내용 조회
                problem_data = get_problem_data()
                unit_rows = problem_data[
                    problem_data['unit_name'].str.replace(" ", "").str.contains(unit_name.replace(" ", ""), na=False)
                ]
                if not unit_rows.empty:
                    standard_code = unit_rows.iloc[0].get('achievement_standard_code', '')
                    achievement_content, achievement_levels = get_achievement_standard_info_from_code(standard_code)
                else:
                    achievement_content, achievement_levels = "성취기준 정보 없음", ""

                unit_specific_conditions = get_unit_specific_conditions(unit_name)

                # 중복이 아닌 문제가 나올 때까지 최대 3회 더 시도
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

                    if is_duplicate and attempt_log:
                        attempt_log[-1]["최종처리"] = "폐기(중복)"

                    combined_attempt_log.extend(attempt_log)

                    if not is_duplicate:
                        break  # 중복 아님 → 채택
                    print(f"  🔁 같은 단원·난이도 내 중복 문제 감지 — 재생성 (중복재시도 {dup_attempt + 1}/{max_duplicate_retries})")
                else:
                    print(f"  ⚠️ {max_duplicate_retries}회 재시도에도 중복 회피 실패 — 중복 상태로 그대로 사용")

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
    #  - "결과": 최종 채택된 99문항
    #  - "검증로그": 문항별로 1~3번째 시도에서 각각 무엇이 생성됐고
    #               정합성/형식성 중 어디서, 어떤 세부항목 때문에 막혔는지 전부 기록
    #
    # [버그 수정] 파인튜닝 모델이 \frac{2}{3} 같은 LaTeX 수식을 답하는 경우,
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
    n_self_validated = n_valid - n_excel_fallback
    n_unverified = n_total - n_valid
    print(f"\n✅ 엑셀 저장 완료: {output_path} (시트: 결과 / 검증로그)")
    print(f"📊 총 {n_total}문항 | 자체검증통과 {n_self_validated}건 | 엑셀대체 {n_excel_fallback}건 | 미검증사용 {n_unverified}건")
    print(f"📊 1회 시도 만에 통과: {n_first_try}건 / {n_total}건 ({n_first_try/n_total*100:.1f}%)")
