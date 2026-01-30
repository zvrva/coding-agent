from __future__ import annotations

from dataclasses import dataclass
import json
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LlmError("No JSON object found in LLM response")
    return json.loads(text[start:end + 1])


def _extract_content(data: dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise LlmError(f"Invalid LLM response schema: {exc}") from exc
