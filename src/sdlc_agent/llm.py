from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any

import requests


@dataclass(frozen=True)
class LlmResponse:
    content: str
    raw: dict[str, Any]


class LlmError(RuntimeError):
    pass


def chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout_sec: int = 60,
    retries: int = 2,
) -> LlmResponse:
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code >= 400:
                raise LlmError(f"LLM HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            content = _extract_content(data)
            return LlmResponse(content=content, raw=data)
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(1 + attempt)
    raise LlmError(str(last_error))


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise LlmError("Empty LLM response")

    candidates: list[str] = [text]
    if "```" in text:
        stripped = _strip_code_fence(text)
        if stripped:
            candidates.append(stripped)

    balanced = _extract_balanced_object(text)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        for attempt in (candidate, _repair_json(candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    raise LlmError("No valid JSON object found in LLM response")


def _extract_content(data: dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LlmError(f"Invalid LLM response schema: {exc}") from exc


def _strip_code_fence(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _extract_balanced_object(text: str) -> str | None:
    depth = 0
    in_string = False
    escape = False
    start = None
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _repair_json(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return _escape_newlines_in_strings(text)


def _escape_newlines_in_strings(text: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
                out.append(ch)
                continue
            if ch == "\\":
                escape = True
                out.append(ch)
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)
    return "".join(out)
