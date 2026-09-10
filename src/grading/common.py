"""Shared, public utilities for prompt-based assessment runs."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict[str, Any]:
    with (ROOT / "configs" / "grading.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def require_api_key() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY를 .env 또는 환경 변수에 설정하세요.")


def load_prompt(path: Path) -> tuple[str, str]:
    """Read the two executable sections of a prompt Markdown file."""
    text = path.read_text(encoding="utf-8")
    system = re.search(r"^## 시스템 프롬프트\s*\n(.*?)(?=^## 사용자 프롬프트\s*$)", text, re.M | re.S)
    user = re.search(r"^## 사용자 프롬프트\s*\n(.*)$", text, re.M | re.S)
    if not system or not user:
        raise ValueError(f"프롬프트에 시스템/사용자 섹션이 없습니다: {path}")
    def strip_fence(section: str) -> str:
        section = section.strip()
        fenced = re.fullmatch(r"```(?:text|markdown)?\s*\n(.*?)\n```", section, re.S)
        return fenced.group(1) if fenced else section
    return strip_fence(system.group(1)), strip_fence(user.group(1))


def render_user_prompt(template: str, row: pd.Series) -> str:
    values = {name: "" if pd.isna(value) else str(value) for name, value in row.items()}
    return Template(template).render(**values)


def request_completion(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # Reasoning models accept max_completion_tokens, whereas earlier chat
    # models use max_tokens. Temperature is intentionally omitted for o-series.
    if model.startswith("o"):
        request["max_completion_tokens"] = max_tokens
    else:
        request["temperature"] = temperature
        request["max_tokens"] = max_tokens
    response = client.chat.completions.create(**request)
    return (response.choices[0].message.content or "").strip()


def extract_tagged(text: object, tag: str) -> str | None:
    if not isinstance(text, str):
        return None
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"지원하지 않는 입력 형식입니다: {path}")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Atomically checkpoint a CSV/XLSX output after each completed row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise ValueError(f"지원하지 않는 출력 형식입니다: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, prefix=f".{path.stem}.", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if suffix == ".csv":
            frame.to_csv(temporary_path, index=False)
        else:
            frame.to_excel(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def require_columns(frame: pd.DataFrame, names: set[str]) -> None:
    missing = sorted(names - set(frame.columns))
    if missing:
        raise ValueError(f"입력 파일에 필요한 열이 없습니다: {', '.join(missing)}")


def make_resume_frame(source: pd.DataFrame, output: Path, columns: list[str], resume: bool) -> pd.DataFrame:
    """Reuse a prior output only when it identifies the same input rows."""
    result = source.copy()
    for column in columns:
        if column not in result:
            result[column] = pd.NA
    result = result.reindex(columns=columns)
    if not (resume and output.exists()):
        return result

    existing = read_table(output)
    key_columns = [name for name in ("question_id", "answer_number") if name in source.columns]
    if key_columns and all(name in existing.columns for name in key_columns):
        if not existing[key_columns].astype(str).equals(source[key_columns].astype(str)):
            raise ValueError("--resume 출력 파일의 답안 순서가 현재 입력 파일과 다릅니다.")
    for column in columns:
        if column in existing.columns and column not in source.columns:
            result[column] = existing[column].values
    return result
