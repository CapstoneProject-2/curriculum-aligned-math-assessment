"""Run a selected public binary or partial-credit grading prompt with resume."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from grading.common import (
    ROOT, extract_tagged, load_config, load_prompt, make_resume_frame,
    read_table, render_user_prompt, request_completion, require_api_key,
    require_columns, write_table,
)


PROMPTS = {
    "binary_v1": ROOT / "prompts" / "grading" / "binary_v1.md",
    "binary_v2": ROOT / "prompts" / "grading" / "binary_v2.md",
    "partial_credit_v2": ROOT / "prompts" / "grading" / "partial_credit_v2.md",
}
REQUIRED = {"grade_level_name", "unit_name", "difficulty", "question_text", "final_answer", "rubric", "student_answer"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=PROMPTS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-interval", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = read_table(args.input)
    require_columns(source, REQUIRED)
    is_partial = args.condition.startswith("partial_credit")
    columns = list(source.columns)
    for name in ("grading_result", "grading_conclusion") + (("score",) if is_partial else ()):
        if name not in columns:
            columns.append(name)
    result = make_resume_frame(source, args.output, columns, args.resume)
    pending = result.index[result["grading_result"].isna() | result["grading_result"].eq("")]
    print(f"채점 대상: {len(result)}개 / 이번 실행: {len(pending)}개")
    if not len(pending):
        return

    config = load_config()
    require_api_key()
    system, user = load_prompt(PROMPTS[args.condition])
    client = OpenAI()
    condition = config["grading_conditions"][args.condition]
    model = args.model or condition["model"]
    for index in tqdm(pending, desc=f"{args.condition} 채점"):
        try:
            response = request_completion(client, model=model, system_prompt=system,
                user_prompt=render_user_prompt(user, result.loc[index]),
                temperature=condition["temperature"], max_tokens=condition["max_tokens"])
            result.loc[index, "grading_result"] = response
            result.loc[index, "grading_conclusion"] = extract_tagged(response, "grading_conclusion")
            if is_partial:
                result.loc[index, "score"] = extract_tagged(response, "score")
        except Exception as error:
            print(f"\n행 {index} 채점 실패: {error}")
        write_table(result, args.output)
        if args.request_interval:
            time.sleep(args.request_interval)
    print(f"채점 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
