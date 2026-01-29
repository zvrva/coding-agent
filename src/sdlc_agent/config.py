from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class Settings:
    github_token: str
    github_api_base: str
    codestral_api_key: str
    codestral_api_base: str
    codestral_model: str
    max_iterations: int
    test_timeout_sec: int
    pip_timeout_sec: int
    default_test_cmd: str


def load_settings() -> Settings:
    _load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("AGENT_GITHUB_TOKEN")
    if not github_token:
        github_token = _get_required("GITHUB_TOKEN")
    codestral_api_key = _get_required("CODESTRAL_API_KEY")

    return Settings(
        github_token=github_token,
        github_api_base=_get_optional("GITHUB_API_BASE", "https://api.github.com"),
        codestral_api_key=codestral_api_key,
        codestral_api_base=_get_optional("CODESTRAL_API_BASE", "https://api.mistral.ai/v1"),
        codestral_model=_get_optional("CODESTRAL_MODEL", "codestral-latest"),
        max_iterations=_get_int("MAX_ITERATIONS", 5),
        test_timeout_sec=_get_int("TEST_TIMEOUT_SEC", 900),
        pip_timeout_sec=_get_int("PIP_TIMEOUT_SEC", 600),
        default_test_cmd=_get_optional("DEFAULT_TEST_CMD", "python -m pytest -q"),
    )


def _load_dotenv() -> None:
    env_path = Path(os.getenv("ENV_FILE", ".env"))
    if not env_path.is_file():
        return

    try:
        content = env_path.read_text(encoding="utf-8")
    except Exception:
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _get_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required env var: {name}")
    return value


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Invalid int for {name}: {raw}") from None


def _get_optional(name: str, default: str) -> str:
    value = os.getenv(name)
    if not value:
        return default
    return value
