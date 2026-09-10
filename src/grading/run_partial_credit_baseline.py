"""Run or resume the zero-shot partial-credit baseline on a local workbook."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from grading.common import (
    ROOT, extract_tagged, load_config, load_prompt, make_resume_frame,
    read_table, render_user_prompt, request_completion, require_api_key,
    require_columns, write_table,
)


GRADING_COLUMNS = [
    "question_id", "grade_level", "unit_order", "difficulty_level", "question_step",
    "grade_level_name", "unit_name", "topic", "eval_area", "content_area",
    "difficulty", "achievement_standard_code", "achievement_standard_content",
    "achievement_standard_description", "question_text", "solution_text",
    "final_answer", "rubric", "student_answer", "answer_number",
    "std_answer_label", "grading_result", "grading_conclusion", "score",
]
REQUIRED = set(GRADING_COLUMNS[:-3])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "grading_부분점수_채점_baseline(v1).xlsx")
    parser.add_argument("--model", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-interval", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    source = read_table(args.input)
    require_columns(source, REQUIRED)
    excluded = config["evaluation"]["excluded_unit"]
    source = source.loc[source["unit_name"] != excluded].reset_index(drop=True)
    expected_rows = config["evaluation"]["expected_rows_after_exclusion"]
    if len(source) != expected_rows:
        raise ValueError(f"'{excluded}' 제외 후 {expected_rows}행이어야 합니다: {len(source)}행")

    # Earlier interrupted runs may have been started before the 360-row filter
    # was added. Normalize that checkpoint once, then use the ordinary resume
    # identity checks below.
    if args.resume and args.output.exists():
        existing = read_table(args.output)
        if len(existing) != expected_rows:
            require_columns(existing, {"unit_name"})
            existing = existing.loc[existing["unit_name"] != excluded].reset_index(drop=True)
            if len(existing) != expected_rows:
                raise ValueError(
                    f"기존 출력에서 '{excluded}' 제외 후 {expected_rows}행이어야 합니다: {len(existing)}행"
                )
            write_table(existing, args.output)

    result = make_resume_frame(source, args.output, GRADING_COLUMNS, args.resume)
    pending = result.index[
        result["grading_result"].isna()
        | result["grading_result"].eq("")
        | result["score"].isna()
        | result["score"].eq("")
        | result["grading_conclusion"].isna()
        | result["grading_conclusion"].eq("")
    ]
    print(f"채점 대상: {len(result)}개 / 이번 실행: {len(pending)}개")
    if not len(pending):
        return

    require_api_key()
    system, user = load_prompt(ROOT / "prompts" / "grading" / "partial_credit_v1.md")
    client = OpenAI()
    condition = config["grading_conditions"]["partial_credit_v1"]
    model = args.model or condition["model"]
    for index in tqdm(pending, desc="부분점수 베이스라인 채점"):
        try:
            response = request_completion(client, model=model, system_prompt=system,
                user_prompt=render_user_prompt(user, result.loc[index]),
                temperature=condition["temperature"], max_tokens=condition["max_tokens"])
            result.loc[index, "grading_result"] = response
            result.loc[index, "score"] = extract_tagged(response, "score")
            result.loc[index, "grading_conclusion"] = extract_tagged(response, "grading_conclusion")
        except Exception as error:  # blank cells remain eligible for --resume
            print(f"\n행 {index} 채점 실패: {error}")
        write_table(result, args.output)
        if args.request_interval:
            time.sleep(args.request_interval)
    print(f"채점 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
