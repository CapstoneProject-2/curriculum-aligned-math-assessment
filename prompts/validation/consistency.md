버전: v2, v3 공통
사용 시기: 2차·3차 실험 (생성 후 재검증 단계)
모델: gpt-4o-mini (validation_llm, temperature 0.1)
직전 버전 대비 변경: 해당없음 (v2에서 도입된 이후 v3까지 프롬프트 내용 동일)
변경 이유: 생성 모델이 만든 문제를 별도 모델이 직접 풀어보게 하여, 정답·해설·조건의 논리적 정합성을 독립적으로 재검증하기 위함

## 시스템 프롬프트

```
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
```

## 사용자 프롬프트

```
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
```

## JSON 출력 스키마 (`{json_schema}` 자리에 주입되는 값)

```json
{
  "step1_actual_solution": "LLM이 계산한 정답",
  "step1_calculation_steps": "LLM의 계산 과정",
  "step1_answer_matches": true,
  "step1_explanation_valid": true,
  "step2_solution_exists": true,
  "step3_conditions_sufficient": true,
  "step4_unit_conditions_met": true,
  "step5_difficulty_appropriate": true,
  "overall_valid": true
}
```

## 비고

- 대응 코드: `src/generation/v3_final.py`의 `validate_problem_with_llm()` 함수, `validation_prompt` 템플릿.
- 판정 기준: `step1_answer_matches` ~ `step5_difficulty_appropriate` 6개 항목이 **모두 true**여야 정합성 검증 통과로 판정한다 (`is_valid = all(validation_checks.values())`).
- 정합성 검증 통과 여부는 형식성 검증(`format.md` 참고)과 AND 조건으로 결합되어 최종 채택 여부가 결정된다 (v3 기준, v2는 OR 조건이었음 — `prompts/generation/v2_rag.md`, `v3_cot.md` 비고 참고).
- 응답이 JSON 파싱에 실패하거나 API 호출 자체가 예외로 실패하면, `basic_validation_fallback()`(형식성 검증)의 결과로 대체 판정한다.
