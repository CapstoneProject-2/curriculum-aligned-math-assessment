"""Generate short feedback from a grading workbook with resume checkpoints."""

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


REQUIRED = {"grade_level_name", "unit_name", "difficulty", "question_text", "final_answer", "rubric", "student_answer", "grading_result"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    columns = list(source.columns)
    for name in ("feedback_result", "feedback"):
        if name not in columns:
            columns.append(name)
    result = make_resume_frame(source, args.output, columns, args.resume)
    pending = result.index[result["feedback_result"].isna() | result["feedback_result"].eq("")]
    print(f"피드백 대상: {len(result)}개 / 이번 실행: {len(pending)}개")
    if not len(pending):
        return

    config = load_config()
    require_api_key()
    system, user = load_prompt(ROOT / "prompts" / "feedback" / "feedback.md")
    client = OpenAI()
    model = args.model or config["feedback"]["model"]
    for index in tqdm(pending, desc="짧은 피드백 생성"):
        try:
            response = request_completion(client, model=model, system_prompt=system,
                user_prompt=render_user_prompt(user, result.loc[index]),
                temperature=config["feedback"]["temperature"], max_tokens=config["feedback"]["max_tokens"])
            result.loc[index, "feedback_result"] = response
            result.loc[index, "feedback"] = extract_tagged(response, "feedback")
        except Exception as error:
            print(f"\n행 {index} 피드백 실패: {error}")
        write_table(result, args.output)
        if args.request_interval:
            time.sleep(args.request_interval)
    print(f"피드백 결과 저장: {args.output}")


if __name__ == "__main__":
    main()
