from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess
import time
import tomllib
from typing import Iterable


_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


@dataclass
class CommandResult:
    cmd: str
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float


def run_cmd(cmd: str, cwd: Path, timeout_sec: int | None = None) -> CommandResult:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        env=_clean_env(os.environ),
    )
    duration = time.time() - start
    return CommandResult(
        cmd=cmd,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        duration_sec=duration,
    )


def detect_install_cmd(repo_path: Path) -> list[str] | None:
    pyproject = repo_path / "pyproject.toml"
    if pyproject.exists():
        return [
            'python -m pip install -e ".[dev]"',
            "python -m pip install -e .",
            "python -m pip install .",
        ]

    req_dev = repo_path / "requirements-dev.txt"
    if req_dev.exists():
        return [f"python -m pip install -r {req_dev.name}"]

    req = repo_path / "requirements.txt"
    if req.exists():
        return [f"python -m pip install -r {req.name}"]

    return None


def detect_quality_commands(repo_path: Path) -> list[str]:
    pyproject = _load_pyproject(repo_path)
    deps = _collect_declared_deps(repo_path, pyproject)

    has_ruff = _has_pyproject_tool(pyproject, "ruff") or _has_ruff_config(repo_path)
    has_black = _has_pyproject_tool(pyproject, "black")
    has_mypy = _has_pyproject_tool(pyproject, "mypy") or _has_mypy_config(repo_path)

    has_ruff = has_ruff or ("ruff" in deps)
    has_black = has_black or ("black" in deps)
    has_mypy = has_mypy or ("mypy" in deps)

    commands: list[str] = []
    if has_ruff:
        commands.append("ruff check .")
    if has_black:
        commands.append("black --check .")
    if has_mypy:
        commands.append("mypy .")
    return commands


def run_quality_checks(repo_path: Path, timeout_sec: int | None = None) -> list[CommandResult]:
    results: list[CommandResult] = []
    for cmd in detect_quality_commands(repo_path):
        results.append(run_cmd(cmd, cwd=repo_path, timeout_sec=timeout_sec))
    return results


def _load_pyproject(repo_path: Path) -> dict:
    path = repo_path / "pyproject.toml"
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _has_pyproject_tool(pyproject: dict, tool_name: str) -> bool:
    tool = pyproject.get("tool", {})
    return tool_name in tool


def _has_ruff_config(repo_path: Path) -> bool:
    if (repo_path / ".ruff.toml").exists() or (repo_path / "ruff.toml").exists():
        return True
    return False


def _has_mypy_config(repo_path: Path) -> bool:
    if (repo_path / "mypy.ini").exists():
        return True
    setup_cfg = repo_path / "setup.cfg"
    if setup_cfg.exists() and _file_contains_section(setup_cfg, "mypy"):
        return True
    tox_ini = repo_path / "tox.ini"
    if tox_ini.exists() and _file_contains_section(tox_ini, "mypy"):
        return True
    return False


def _file_contains_section(path: Path, section: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    pattern = rf"^\s*\[{re.escape(section)}\]\s*$"
    return re.search(pattern, text, flags=re.MULTILINE) is not None


def _collect_declared_deps(repo_path: Path, pyproject: dict) -> set[str]:
    deps: set[str] = set()

    project = pyproject.get("project", {})
    for dep in project.get("dependencies", []):
        name = _parse_req_name(dep)
        if name:
            deps.add(name)
    for extra_deps in project.get("optional-dependencies", {}).values():
        for dep in extra_deps:
            name = _parse_req_name(dep)
            if name:
                deps.add(name)

    for req_file in ("requirements-dev.txt", "requirements.txt"):
        path = repo_path / req_file
        if not path.exists():
            continue
        for name in _iter_requirements(path):
            deps.add(name)

    return deps


def _iter_requirements(path: Path) -> Iterable[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    names: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r") or line.startswith("--"):
            continue
        name = _parse_req_name(line)
        if name:
            names.append(name)
    return names


def _parse_req_name(req_line: str) -> str | None:
    match = _REQ_NAME_RE.match(req_line)
    if not match:
        return None
    name = match.group(1).lower().replace("_", "-")
    return name


def _clean_env(env: dict) -> dict:
    clean = dict(env)
    clean.pop("PIP_REQUIRE_VIRTUALENV", None)
    return clean
