"""Safe extraction of a Plan from raw LLM text. Never raises; returns a ParseResult."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from backend.app.schemas.plan import Plan

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


@dataclass
class ParseResult:
    plan: Optional[Plan]
    error: Optional[str]
    json_obj: Optional[Any] = None

    @property
    def ok(self) -> bool:
        return self.plan is not None


def extract_json_text(text: str) -> Optional[str]:
    """Pull the first JSON object out of text (handles code fences and prose around it)."""
    if not text or not text.strip():
        return None
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = _FENCE.search(s)
    if m and m.group(1).strip().startswith("{"):
        return m.group(1).strip()
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for idx in range(start, len(s)):
        ch = s[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start: idx + 1]
    return None  # unbalanced braces (for example, output cut off at the token limit)


def parse_json_object(text: str) -> tuple[Optional[dict], Optional[str]]:
    raw = extract_json_text(text)
    if raw is None:
        return None, "no complete JSON object found in the output"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
    if not isinstance(obj, dict):
        return None, "top-level JSON value is not an object"
    return obj, None


def parse_plan(text: str) -> ParseResult:
    obj, err = parse_json_object(text)
    if obj is None:
        return ParseResult(None, err)
    try:
        plan = Plan.model_validate(obj)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", ()))
        return ParseResult(None, f"schema error at '{loc}': {first.get('msg')}", obj)
    return ParseResult(plan, None, obj)
