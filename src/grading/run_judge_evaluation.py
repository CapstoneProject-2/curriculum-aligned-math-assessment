"""Run or resume LLM-as-a-Judge evaluation after grading has completed."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from grading.common import (
    ROOT, load_config, load_prompt, make_resume_frame,
    read_table, render_user_prompt, request_completion, require_api_key,
    require_columns, write_table,
)


JUDGE_COLUMNS = [
    "question_id", "grade_level_name", "unit_name", "topic", "difficulty",
    "achievement_standard_code", "question_text", "solution_text", "final_answer",
    "rubric", "student_answer", "answer_number", "std_answer_label",
    "grading_result", "grading_conclusion", "score", "judge_response",
    "judge_conclusion",
]
REQUIRED = {
    "grade_level_name", "unit_name", "difficulty", "question_text", "solution_text",
    "final_answer", "rubric", "student_answer", "grading_result",
}


def extract_single_judge_conclusion(response: object) -> str | None:
    """Return a verdict only when exactly one valid Judge tag is present."""
    if not isinstance(response, str):
        return None
    conclusions = re.findall(
        r"<judging_conclusion>(.*?)</judging_conclusion>",
        response,
        flags=re.DOTALL,
    )
    if len(conclusions) != 1:
        return None
    conclusion = conclusions[0].strip()
    return conclusion if conclusion in {"적절", "부적절"} else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completed grading workbook")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "judge_grading_부분점수_채점_baseline(v1).xlsx")
    parser.add_argument("--model", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-interval", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    grading = read_table(args.input)
    require_columns(grading, REQUIRED)
    excluded = config["evaluation"]["excluded_unit"]
    grading = grading.loc[grading["unit_name"] != excluded].reset_index(drop=True)
    expected_rows = config["evaluation"]["expected_rows_after_exclusion"]
    if len(grading) != expected_rows:
        raise ValueError(f"'{excluded}' 제외 후 {expected_rows}행이어야 합니다: {len(grading)}행")
    incomplete = grading["grading_result"].isna() | grading["grading_result"].eq("")
    if incomplete.any():
        raise RuntimeError(f"채점 미완료 답안이 {incomplete.sum()}개 있습니다. 먼저 채점을 완료하세요.")

    if args.resume and args.output.exists():
        existing = read_table(args.output)
        if len(existing) != expected_rows:
            require_columns(existing, {"unit_name"})
            existing = existing.loc[existing["unit_name"] != excluded].reset_index(drop=True)
            if len(existing) != expected_rows:
                raise ValueError(
                    f"기존 Judge 출력에서 '{excluded}' 제외 후 {expected_rows}행이어야 합니다: {len(existing)}행"
                )
            write_table(existing, args.output)

    result = make_resume_frame(grading, args.output, JUDGE_COLUMNS, args.resume)
    # Revalidate saved responses so earlier first-tag-only parsing does not
    # mark a response containing multiple conclusions as complete.
    result["judge_conclusion"] = result["judge_response"].map(
        extract_single_judge_conclusion
    )
    pending = result.index[
        ~result["judge_conclusion"].isin(["적절", "부적절"])
    ]
    print(f"Judge 대상: {len(result)}개 / 이번 실행: {len(pending)}개")
    if len(pending):
        require_api_key()
        system, user = load_prompt(ROOT / "prompts" / "judge" / "llm_as_judge.md")
        client = OpenAI()
        model = args.model or config["judge"]["model"]
        for index in tqdm(pending, desc="Judge 평가"):
            response: str | None = None
            conclusion: str | None = None
            for attempt in range(1, 4):
                try:
                    response = request_completion(
                        client,
                        model=model,
                        system_prompt=system,
                        user_prompt=render_user_prompt(user, result.loc[index]),
                        temperature=config["judge"]["temperature"],
                        max_tokens=config["judge"]["max_completion_tokens"],
                    )
                    conclusion = extract_single_judge_conclusion(response)
                    if conclusion is not None:
                        break
                    print(
                        f"\n행 {index} Judge 형식 실패 ({attempt}/3): "
                        "유효한 judging_conclusion 태그가 정확히 하나여야 합니다."
                    )
                except Exception as error:
                    print(f"\n행 {index} Judge 요청 실패 ({attempt}/3): {error}")
            result.loc[index, "judge_response"] = response
            result.loc[index, "judge_conclusion"] = conclusion
            write_table(result, args.output)
            if args.request_interval:
                time.sleep(args.request_interval)

    valid = result["judge_conclusion"].isin(["적절", "부적절"])
    adequate = result["judge_conclusion"].eq("적절")
    if valid.any():
        print(f"Judge 논리적 적절성: {adequate.sum() / valid.sum() * 100:.2f}% ({adequate.sum()}/{valid.sum()})")
    print(f"유효하지 않거나 누락된 Judge 판정: {(~valid).sum()}개")
    print(f"Judge 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
