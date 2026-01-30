from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile
from typing import Iterable

from .config import Settings
from .github import GitHubClient
from .llm import chat, extract_json
from .runner import detect_install_cmd, run_cmd
from .state import State, can_iterate, new_state, next_iteration, parse_state, render_state


PROMPT_CODE_PATCH = """Ты — код-агент. Внеси изменения, чтобы исправить задачу.
Верни ТОЛЬКО JSON со схемой:
{
  "files": [
    {"path": "relative/path.py", "content": "FULL FILE CONTENTS"}
  ],
  "summary": "короткое описание"
}
Правила:
- Меняй только нужные файлы.
- Для каждого изменённого файла дай полный контент.
- Никакого markdown. Только JSON.
"""


PROMPT_CODE_FIX = """Ты — код-агент. Исправь ошибки в коде по логам тестов.
Верни ТОЛЬКО JSON со схемой:
{
  "files": [
    {"path": "relative/path.py", "content": "FULL FILE CONTENTS"}
  ],
  "summary": "что исправлено"
}
Правила:
- Исправь ошибки, указанные в логах.
- Для каждого изменённого файла дай полный контент.
- Никакого markdown. Только JSON.
"""


@dataclass(frozen=True)
class CodeResult:
    pr_number: int | None
    branch: str
    iteration: int
    summary: str


def run_code_agent(settings: Settings, agent_repo: str, issue_number: int) -> CodeResult:
    gh = GitHubClient(settings.github_token, settings.github_api_base)
    issue = gh.get_issue(agent_repo, issue_number)

    target_repo = _parse_target_repo(issue.body or "")
    if not target_repo:
        target_repo = agent_repo

    branch = f"sdlc/issue-{issue_number}"
    pr = gh.find_open_pr_by_branch(target_repo, branch)
    state = _load_or_init_state(gh, pr, target_repo, issue.html_url, settings.max_iterations)
    if not can_iterate(state):
        gh.post_comment(agent_repo, issue_number, "Лимит итераций исчерпан.")
        return CodeResult(pr_number=pr.number if pr else None, branch=branch, iteration=state.iteration, summary="max-iterations")

    state = next_iteration(state, verdict="in_progress")
    iteration = state.iteration

    def schedule_retry(reason: str) -> None:
        if iteration >= state.max_iterations:
            return
        payload = {
            "repo": target_repo,
            "issue_number": issue_number,
            "reason": reason,
        }
        try:
            gh.dispatch_event(agent_repo, "issue_opened", payload)
        except Exception as exc:
            gh.post_comment(agent_repo, issue_number, f"?? ??????? ????????????? ??????: {exc}")


    with tempfile.TemporaryDirectory(prefix="sdlc-agent-") as tmp:
        repo_path = _clone_repo(target_repo, settings.github_token, tmp)
        _checkout_branch(repo_path, branch)

        install_cmds = detect_install_cmd(repo_path)
        if install_cmds:
            _install_dependencies(repo_path, install_cmds, settings.pip_timeout_sec)

        context = _build_context(repo_path, issue.body or "")
        messages = [
            {"role": "system", "content": PROMPT_CODE_PATCH},
            {"role": "user", "content": context},
        ]
        llm_resp = chat(
            settings.codestral_api_base,
            settings.codestral_api_key,
            settings.codestral_model,
            messages,
            temperature=0.2,
            max_tokens=2048,
            timeout_sec=60,
        )
        data = extract_json(llm_resp.content)
        summary = str(data.get("summary", ""))
        changed_files = _apply_file_changes(repo_path, data)

        if not changed_files:
            message = "No files changed by LLM. PR not created."
            gh.post_comment(agent_repo, issue_number, message)
            if pr:
                gh.post_comment(target_repo, pr.number, message)
            schedule_retry("no-files")
            return CodeResult(pr_number=pr.number if pr else None, branch=branch, iteration=iteration, summary="no-files")

        if not _has_changes(repo_path):
            message = "No changes after applying patch. PR not created."
            gh.post_comment(agent_repo, issue_number, message)
            if pr:
                gh.post_comment(target_repo, pr.number, message)
            schedule_retry("no-changes")
            return CodeResult(pr_number=pr.number if pr else None, branch=branch, iteration=iteration, summary="no-changes")

        commit_res = _commit_all(repo_path, branch, iteration, summary)
        if commit_res.returncode != 0:
            message = "Commit failed or no changes to commit. PR not created."
            gh.post_comment(agent_repo, issue_number, message + "\n" + commit_res.stderr.strip())
            if pr:
                gh.post_comment(target_repo, pr.number, message)
            schedule_retry("no-commit")
            return CodeResult(pr_number=pr.number if pr else None, branch=branch, iteration=iteration, summary="no-commit")

        _push(repo_path)

        pr_body = _build_pr_body(issue, summary)
        pr = gh.create_or_update_pr(
            target_repo,
            branch=branch,
            title=f"[SDLC] Issue #{issue_number}: {issue.title}",
            body=pr_body,
        )

        state = next_iteration(state, verdict="pending_review")
        gh.upsert_state_comment(target_repo, pr.number, render_state(state))
        gh.set_attempt_label(target_repo, pr.number, iteration, state.max_iterations)
        gh.post_comment(target_repo, pr.number, _format_run_report_skipped())

        return CodeResult(pr_number=pr.number, branch=branch, iteration=iteration, summary=summary)


def _parse_target_repo(text: str) -> str | None:
    patterns = [
        r"(?im)^\s*Target repo\s*:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s*$",
        r"(?im)^\s*Целевой репозиторий\s*:\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s*$",
    ]
    for pat in patterns:
        match = re.search(pat, text or "")
        if match:
            return match.group(1)
    return None


def _clone_repo(target_repo: str, token: str, tmp_dir: str) -> Path:
    repo_name = target_repo.split("/")[-1]
    clone_url = f"https://github.com/{target_repo}.git"
    run_cmd(f"git clone {clone_url}", cwd=Path(tmp_dir))
    repo_path = Path(tmp_dir) / repo_name
    auth_url = f"https://x-access-token:{token}@github.com/{target_repo}.git"
    run_cmd(f"git remote set-url origin {auth_url}", cwd=repo_path)
    return repo_path


def _checkout_branch(repo_path: Path, branch: str) -> None:
    run_cmd("git fetch origin", cwd=repo_path)
    default_branch = _get_default_branch(repo_path)
    res = run_cmd(f"git checkout -B {branch} origin/{default_branch}", cwd=repo_path)
    if res.returncode != 0:
        res = run_cmd(f"git checkout -B {branch} origin/main", cwd=repo_path)
        if res.returncode != 0:
            run_cmd(f"git checkout -B {branch} origin/master", cwd=repo_path)


def _install_dependencies(repo_path: Path, cmds: list[str], timeout_sec: int) -> None:
    last = None
    for cmd in cmds:
        last = run_cmd(cmd, cwd=repo_path, timeout_sec=timeout_sec)
        if last.returncode == 0:
            return
    raise RuntimeError(f"Dependency install failed: {last.stderr if last else 'unknown'}")


def _build_pytest_fallback_cmd(repo_path: Path) -> str:
    repo_str = str(repo_path)
    src_str = str(repo_path / "src")
    code = (
        f"import sys; sys.path.insert(0, \"{repo_str}\"); "
        f"sys.path.insert(0, \"{src_str}\"); "
        "import pytest; "
        "raise SystemExit(pytest.main([\"-q\"]))"
    )
    return f"python -c '{code}'"


def _build_fix_context(repo_path: Path, issue_text: str, error_text: str) -> str:
    issue_text = (issue_text or "").strip()
    error_text = (error_text or "").strip()
    files = _select_context_files(repo_path, issue_text)
    parts = [
        "ISSUE:",
        issue_text,
        "",
        "ERRORS:",
        error_text[:8000],
        "",
        "FILES:",
    ]
    for path in files:
        rel = path.relative_to(repo_path)
        content = _read_file_limited(path, limit=4000)
        parts.append(f"\n# {rel}\n{content}\n")
    return "\n".join(parts)


def _build_context(repo_path: Path, issue_text: str) -> str:
    issue_text = (issue_text or "").strip()
    files = _select_context_files(repo_path, issue_text)
    parts = [
        "ISSUE:",
        issue_text,
        "",
        "REPO_SUMMARY:",
        _summarize_repo(repo_path),
        "",
        "FILES:",
    ]
    for path in files:
        rel = path.relative_to(repo_path)
        content = _read_file_limited(path, limit=4000)
        parts.append(f"\n# {rel}\n{content}\n")
    return "\n".join(parts)


def _select_context_files(repo_path: Path, issue_text: str) -> list[Path]:
    candidates = _collect_candidates(repo_path)
    mandatory = _collect_mandatory_files(repo_path)
    ranked = sorted(
        candidates,
        key=lambda p: _score_file(p, repo_path, issue_text),
        reverse=True,
    )
    selected: list[Path] = []
    seen = set()
    for path in mandatory + ranked:
        if path in seen:
            continue
        seen.add(path)
        selected.append(path)
    return selected[:12] if len(selected) <= 12 else selected


def _collect_candidates(repo_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for p in repo_path.rglob("*"):
        if not p.is_file():
            continue
        if _skip_path(p):
            continue
        if p.suffix in {".py", ".md", ".txt"}:
            candidates.append(p)
    return candidates


def _skip_path(path: Path) -> bool:
    parts = set(path.parts)
    if ".venv" in parts or "__pycache__" in parts or ".git" in parts:
        return True
    if "node_modules" in parts or ".tox" in parts or ".mypy_cache" in parts:
        return True
    return False


def _score_file(path: Path, repo_path: Path, issue_text: str) -> int:
    rel = path.relative_to(repo_path).as_posix()
    score = 0

    if rel.startswith("tests/"):
        score += 5
    if rel.startswith("src/") or "/src/" in rel:
        score += 3

    tokens = _issue_tokens(issue_text)
    for token in tokens:
        if token in rel.lower():
            score += 3

    if path.suffix == ".py":
        score += 2
    if path.name.lower() == "readme.md":
        score += 1

    return score


def _issue_tokens(text: str) -> set[str]:
    tokens = set()
    for raw in re.split(r"[^A-Za-z0-9_.-]+", text.lower()):
        if len(raw) >= 3:
            tokens.add(raw)
    return tokens


def _read_file_limited(path: Path, limit: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(content) <= limit:
        return content
    return content[:limit] + "\n... [truncated]\n"


def _summarize_repo(repo_path: Path) -> str:
    top = []
    for p in repo_path.iterdir():
        if p.name.startswith("."):
            continue
        top.append(p.name + ("/" if p.is_dir() else ""))
    top = sorted(top)
    return "Top-level: " + ", ".join(top)


def _apply_file_changes(repo_path: Path, data: dict) -> list[str]:
    files = data.get("files") or []
    changed: list[str] = []
    for item in files:
        rel = str(item.get("path", "")).strip()
        content = item.get("content")
        if not rel or content is None:
            continue
        full_path = (repo_path / rel).resolve()
        if not str(full_path).startswith(str(repo_path.resolve())):
            continue
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        changed.append(rel)

    return changed


def _commit_all(repo_path: Path, branch: str, iteration: int, summary: str):
    run_cmd("git add .", cwd=repo_path)
    msg = f"SDLC(iter={iteration}): {summary or 'apply changes'}"
    return run_cmd(f"git commit -m \"{msg}\"", cwd=repo_path)


def _push(repo_path: Path) -> None:
    run_cmd("git push -u origin HEAD", cwd=repo_path)


def _build_pr_body(issue, summary: str) -> str:
    body = [
        "## SDLC Agent PR",
        f"- Source issue: {issue.html_url}",
        f"- Summary: {summary or 'n/a'}",
        "",
        "### Issue context",
        issue.body or "",
    ]
    return "\n".join(body)


def _load_or_init_state(
    gh: GitHubClient,
    pr,
    target_repo: str,
    issue_url: str,
    max_iterations: int,
) -> State:
    if pr:
        comment = gh.find_state_comment(target_repo, pr.number)
        if comment:
            parsed = parse_state(comment.body)
            if parsed:
                return parsed
    return new_state(target_repo=target_repo, source_issue_url=issue_url, max_iterations=max_iterations)


def _format_run_report(quality_results, test_result) -> str:
    lines = ["## SDLC Agent run report"]
    if quality_results:
        lines.append("### Quality checks")
        for res in quality_results:
            lines.append(_format_command(res))
    else:
        lines.append("### Quality checks")
        lines.append("No quality tools detected (ruff/black/mypy).")

    lines.append("### Tests")
    lines.append(_format_command(test_result))
    return "\n".join(lines)


def _format_command(res) -> str:
    status = "OK" if res.returncode == 0 else f"FAIL ({res.returncode})"
    return "\n".join(
        [
            f"- `{res.cmd}` -> {status} in {res.duration_sec:.1f}s",
            "```",
            (res.stdout or res.stderr or "").strip()[:4000],
            "```",
        ]
    )


def _collect_mandatory_files(repo_path: Path) -> list[Path]:
    paths: list[Path] = []
    for base in (
        repo_path / "python_utils_demo",
        repo_path / "src" / "python_utils_demo",
        repo_path / "tests",
    ):
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.is_file() and not _skip_path(path):
                paths.append(path)
    return paths


def _get_default_branch(repo_path: Path) -> str:
    res = run_cmd("git symbolic-ref refs/remotes/origin/HEAD", cwd=repo_path)
    if res.returncode == 0:
        ref = (res.stdout or "").strip()
        if ref.startswith("refs/remotes/origin/"):
            return ref.split("/")[-1]
    return "main"


def _has_changes(repo_path: Path) -> bool:
    res = run_cmd("git status --porcelain", cwd=repo_path)
    return bool((res.stdout or "").strip())


def _format_run_report_skipped() -> str:
    return "\n".join([
        "## SDLC Agent run report",
        "Изменения подготовлены. Проверки выполнит review-agent.",
    ])
