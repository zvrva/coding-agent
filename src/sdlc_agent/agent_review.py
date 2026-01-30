from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Any

from github.GithubException import GithubException

from .config import Settings
from .github import GitHubClient
from .llm import chat, extract_json
from .runner import detect_install_cmd, run_cmd, run_quality_checks
from .state import parse_state, render_state


PROMPT_REVIEW = """\u0422\u044b - \u0440\u0435\u0432\u044c\u044e\u0435\u0440. \u041f\u0440\u043e\u0432\u0435\u0440\u044c PR \u043f\u043e Issue, diff \u0438 \u043b\u043e\u0433\u0430\u043c.
\u0412\u0435\u0440\u043d\u0438 \u0422\u041e\u041b\u042c\u041a\u041e JSON:
{
  "verdict": "approve" | "changes_requested",
  "summary": "\u043a\u0440\u0430\u0442\u043a\u043e",
  "blocking": ["\u043f\u0440\u043e\u0431\u043b\u0435\u043c\u044b"],
  "notes": ["\u0441\u043e\u043c\u043d\u0435\u043d\u0438\u044f"]
}
\u041f\u0440\u0430\u0432\u0438\u043b\u0430:
- \u0415\u0441\u043b\u0438 \u0442\u0435\u0441\u0442\u044b \u0438\u043b\u0438 \u0447\u0435\u043a\u0438 \u0443\u043f\u0430\u043b\u0438 - verdict changes_requested.
- \u0411\u043b\u043e\u043a\u0438\u0440\u0443\u0439 \u0442\u043e\u043b\u044c\u043a\u043e \u043f\u0440\u0438 \u044f\u0432\u043d\u044b\u0445 \u0434\u043e\u043a\u0430\u0437\u0430\u0442\u0435\u043b\u044c\u0441\u0442\u0432\u0430\u0445.
- \u0415\u0441\u043b\u0438 \u0441\u043e\u043c\u043d\u0435\u043d\u0438\u044f - notes.
- \u0422\u043e\u043b\u044c\u043a\u043e JSON, \u0431\u0435\u0437 markdown."""

PROMPT_GENERATE_TESTS = """\u0422\u044b - \u0442\u0435\u0441\u0442-\u0430\u0433\u0435\u043d\u0442. \u0421\u0433\u0435\u043d\u0435\u0440\u0438\u0440\u0443\u0439 pytest-\u0442\u0435\u0441\u0442\u044b \u0434\u043b\u044f \u043f\u0440\u043e\u0432\u0435\u0440\u043a\u0438 \u0438\u0441\u043f\u0440\u0430\u0432\u043b\u0435\u043d\u0438\u044f.
\u0412\u0435\u0440\u043d\u0438 \u0422\u041e\u041b\u042c\u041a\u041e JSON:
{
  "files": [
    {"path": "tests/test_generated_agent.py", "content": "FULL FILE CONTENTS"}
  ],
  "summary": "\u0447\u0442\u043e \u0434\u043e\u0431\u0430\u0432\u043b\u0435\u043d\u043e"
}
\u041f\u0440\u0430\u0432\u0438\u043b\u0430:
- \u0415\u0441\u043b\u0438 \u0442\u0435\u0441\u0442\u044b \u0443\u0436\u0435 \u0435\u0441\u0442\u044c, \u043d\u043e\u0432\u044b\u0435 \u043c\u0438\u043d\u0438\u043c\u0430\u043b\u044c\u043d\u044b\u0435.
- \u0422\u043e\u043b\u044c\u043a\u043e JSON."""


@dataclass(frozen=True)
class ReviewResult:
    verdict: str
    summary: str
    blocking: list[str]
    notes: list[str]


@dataclass(frozen=True)
class TestRun:
    result: object | None
    note: str
    generated_files: list[str]


def run_review_agent(settings: Settings, target_repo: str, pr_number: int) -> ReviewResult:
    gh = GitHubClient(settings.github_token, settings.github_api_base)
    pr = gh.get_pull(target_repo, pr_number)
    diff = gh.get_pr_diff(target_repo, pr_number)
    files = gh.get_pr_files(target_repo, pr_number)

    issue_text = ""
    agent_issue_repo = None
    agent_issue_number = None
    st = None
    state_comment = gh.find_state_comment(target_repo, pr_number)
    if state_comment:
        st = parse_state(state_comment.body)
        if st and st.source_issue_url:
            issue = gh.get_issue_by_url(st.source_issue_url)
            issue_text = issue.body or ""
            agent_issue_repo = issue.repository.full_name
            agent_issue_number = issue.number

    quality_results = []
    test_run = TestRun(result=None, note="", generated_files=[])

    with tempfile.TemporaryDirectory(prefix="sdlc-review-") as tmp:
        repo_path = _clone_repo(pr.head.repo.full_name, settings.github_token, tmp)
        _checkout_ref(repo_path, pr.head.ref)

        install_cmds = detect_install_cmd(repo_path)
        if install_cmds:
            _install_dependencies(repo_path, install_cmds, settings.pip_timeout_sec)
            quality_results = run_quality_checks(repo_path, timeout_sec=settings.test_timeout_sec)
            test_run = _run_or_generate_tests(repo_path, issue_text, diff, files, settings)
        else:
            test_run = _run_or_generate_tests(repo_path, issue_text, diff, files, settings)

    context = _build_review_context(issue_text, diff, files, quality_results, test_run)
    messages = [
        {"role": "system", "content": PROMPT_REVIEW},
        {"role": "user", "content": context},
    ]
    llm_resp = chat(
        settings.codestral_api_base,
        settings.codestral_api_key,
        settings.codestral_model,
        messages,
        temperature=0.2,
        max_tokens=1024,
        timeout_sec=60,
    )
    data = extract_json(llm_resp.content)
    result = _parse_review_result(data)

    tests_failed = test_run.result is not None and getattr(test_run.result, "returncode", 0) != 0
    tests_ok = test_run.result is not None and getattr(test_run.result, "returncode", 0) == 0
    quality_ok = all(getattr(r, "returncode", 0) == 0 for r in quality_results) if quality_results else True

    if tests_failed or not quality_ok:
        result = ReviewResult(
            verdict="changes_requested",
            summary=result.summary,
            blocking=result.blocking,
            notes=result.notes,
        )
    elif tests_ok and quality_ok and result.blocking:
        merged_notes = list(result.notes) + [f"\u041f\u0440\u043e\u0432\u0435\u0440\u044c \u0432\u0440\u0443\u0447\u043d\u0443\u044e: {item}" for item in result.blocking]
        result = ReviewResult(
            verdict="approve",
            summary=result.summary,
            blocking=[],
            notes=merged_notes,
        )
    elif tests_ok and quality_ok and not result.blocking:
        result = ReviewResult(
            verdict="approve",
            summary=result.summary,
            blocking=result.blocking,
            notes=result.notes,
        )

    review_body = _format_review_comment(result, quality_results, test_run)
    event = "APPROVE" if result.verdict == "approve" else "REQUEST_CHANGES"
    if event == "REQUEST_CHANGES":
        reviewer_login = gh.get_current_user_login()
        if pr.user and reviewer_login and pr.user.login == reviewer_login:
            event = "COMMENT"
    try:
        gh.post_review(target_repo, pr_number, review_body, event)
    except GithubException as exc:
        if event == "REQUEST_CHANGES" and "Review Can not request changes on your own pull request" in str(exc):
            gh.post_review(target_repo, pr_number, review_body, "COMMENT")
        else:
            raise

    if state_comment:
        st = parse_state(state_comment.body)
        if st:
            updated = st.__class__(
                target_repo=st.target_repo,
                source_issue_url=st.source_issue_url,
                iteration=st.iteration,
                max_iterations=st.max_iterations,
                last_verdict=result.verdict,
                created_at=st.created_at,
            )
            gh.upsert_state_comment(target_repo, pr_number, render_state(updated))

    if st and st.source_issue_url:
        issue = gh.get_issue_by_url(st.source_issue_url)
        attempt_comment = _format_attempt_comment(pr.html_url, st.iteration, st.max_iterations, result, quality_results, test_run)
        gh.post_comment(issue.repository.full_name, issue.number, attempt_comment)

    if result.verdict == "changes_requested" and st and st.iteration < st.max_iterations and agent_issue_number:
        dispatch_repo = os.getenv("AGENT_REPO") or agent_issue_repo
        if dispatch_repo:
            gh.dispatch_event(dispatch_repo, "issue_opened", {"repo": st.target_repo, "issue_number": agent_issue_number})

    return result


def _clone_repo(target_repo: str, token: str, tmp_dir: str) -> Path:
    repo_name = target_repo.split("/")[-1]
    clone_url = f"https://github.com/{target_repo}.git"
    run_cmd(f"git clone {clone_url}", cwd=Path(tmp_dir))
    repo_path = Path(tmp_dir) / repo_name
    auth_url = f"https://x-access-token:{token}@github.com/{target_repo}.git"
    run_cmd(f"git remote set-url origin {auth_url}", cwd=repo_path)
    return repo_path


def _checkout_ref(repo_path: Path, ref: str) -> None:
    run_cmd("git fetch origin", cwd=repo_path)
    run_cmd(f"git checkout {ref}", cwd=repo_path)


def _install_dependencies(repo_path: Path, cmds: list[str], timeout_sec: int) -> None:
    last = None
    for cmd in cmds:
        last = run_cmd(cmd, cwd=repo_path, timeout_sec=timeout_sec)
        if last.returncode == 0:
            return
    raise RuntimeError(f"Dependency install failed: {last.stderr if last else 'unknown'}")


def _run_or_generate_tests(
    repo_path: Path,
    issue_text: str,
    diff: str,
    files: list[dict[str, Any]],
    settings: Settings,
) -> TestRun:
    generated_files: list[str] = []
    if not _has_tests(repo_path):
        generated_files = _generate_tests(repo_path, issue_text, diff, files, settings)

    test_result = _run_tests(repo_path, settings)
    note = "" if test_result else "\u0422\u0435\u0441\u0442\u044b \u043d\u0435 \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u043b\u0438\u0441\u044c."
    return TestRun(result=test_result, note=note, generated_files=generated_files)


def _generate_tests(
    repo_path: Path,
    issue_text: str,
    diff: str,
    files: list[dict[str, Any]],
    settings: Settings,
) -> list[str]:
    context = _build_test_context(issue_text, diff, files, repo_path)
    messages = [
        {"role": "system", "content": PROMPT_GENERATE_TESTS},
        {"role": "user", "content": context},
    ]
    llm_resp = chat(
        settings.codestral_api_base,
        settings.codestral_api_key,
        settings.codestral_model,
        messages,
        temperature=0.2,
        max_tokens=1024,
        timeout_sec=60,
    )
    data = extract_json(llm_resp.content)
    return _apply_generated_tests(repo_path, data)


def _apply_generated_tests(repo_path: Path, data: dict[str, Any]) -> list[str]:
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


def _build_review_context(
    issue_text: str,
    diff: str,
    files: list[dict[str, Any]],
    quality_results,
    test_run: TestRun,
) -> str:
    parts = ["ISSUE:", issue_text.strip(), "", "DIFF:", diff[:10000], "", "FILES:"]
    for f in files:
        parts.append(f"- {f.get("filename")} ({f.get("status")} +{f.get("additions")}/-{f.get("deletions")})")
    parts.append("")
    parts.append("QUALITY:")
    if quality_results:
        for res in quality_results:
            parts.append(_format_command(res))
    else:
        parts.append("No quality tools detected.")
    parts.append("")
    parts.append("TESTS:")
    if test_run.result is not None:
        parts.append(_format_command(test_run.result))
    else:
        parts.append(test_run.note or "Tests not run.")
    if test_run.generated_files:
        parts.append("")
        parts.append("GENERATED_TESTS:")
        for name in test_run.generated_files:
            parts.append(f"- {name}")
    return "\n".join(parts)

def _build_test_context(issue_text: str, diff: str, files: list[dict[str, Any]], repo_path: Path) -> str:
    parts = ["ISSUE:", issue_text.strip(), "", "DIFF:", diff[:8000], "", "FILES:"]
    for f in files:
        parts.append(f"- {f.get("filename")}")
    return "\n".join(parts)

def _build_pytest_fallback_cmd(repo_path: Path) -> str:
    repo_str = str(repo_path)
    src_str = str(repo_path / "src")
    code = (
        f"import sys; sys.path.insert(0, '{repo_str}'); "
        f"sys.path.insert(0, '{src_str}'); "
        "import pytest; "
        "raise SystemExit(pytest.main(['-q']))"
    )
    return f"python -c \"{code}\""


def _needs_pytest_install(res) -> bool:
    text = (res.stdout + "\n" + res.stderr).lower()
    return "no module named pytest" in text or "pytest: not found" in text

def _build_test_commands(default_cmd: str, fallback_cmd: str) -> list[str]:
    cmds = []
    for cmd in [default_cmd, "python -m pytest -q", "pytest -q", fallback_cmd]:
        if cmd and cmd not in cmds:
            cmds.append(cmd)
    return cmds


def _run_tests(repo_path: Path, settings: Settings):
    pythonpath_parts = [str(repo_path)]
    src_dir = repo_path / "src"
    if src_dir.exists():
        pythonpath_parts.append(str(src_dir))
    existing_pp = os.environ.get("PYTHONPATH")
    if existing_pp:
        pythonpath_parts.append(existing_pp)
    pythonpath_value = os.pathsep.join(pythonpath_parts)
    test_env = {"PYTHONPATH": pythonpath_value}

    fallback_cmd = _build_pytest_fallback_cmd(repo_path)
    commands = _build_test_commands(settings.default_test_cmd, fallback_cmd)

    last = None
    installed_pytest = False
    for cmd in commands:
        res = run_cmd(cmd, cwd=repo_path, timeout_sec=settings.test_timeout_sec, extra_env=test_env)
        last = res
        if res.returncode == 0:
            return res
        if not installed_pytest and _needs_pytest_install(res):
            run_cmd("python -m pip install pytest", cwd=repo_path, timeout_sec=settings.pip_timeout_sec)
            installed_pytest = True
    return last


def _has_tests(repo_path: Path) -> bool:
    if (repo_path / "tests").exists():
        return True
    for p in repo_path.rglob("test_*.py"):
        if p.is_file():
            return True
    return False


def _format_review_comment(result: ReviewResult, quality_results, test_run: TestRun) -> str:
    not_run = "?? ???????????"
    lines = ["## SDLC Agent Review", f"**Verdict:** {result.verdict}", ""]
    if result.summary:
        lines.append(f"**Summary:** {result.summary}")
        lines.append("")
    if result.blocking:
        lines.append("### Blocking issues")
        for item in result.blocking:
            lines.append(f"- {item}")
        lines.append("")
    if result.notes:
        lines.append("### Notes")
        for item in result.notes:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("### Checks")
    if quality_results:
        for res in quality_results:
            lines.append(_format_command(res))
    else:
        lines.append("- Quality checks: not detected")
    if test_run.result is not None:
        lines.append(_format_command(test_run.result))
    else:
        lines.append(f"- ?????: {test_run.note or not_run}")
    if test_run.generated_files:
        lines.append("- ??????????????? ?????: " + ", ".join(test_run.generated_files))
    return "\n".join(lines)

def _format_attempt_comment(pr_url: str, attempt: int, max_attempts: int, result: ReviewResult, quality_results, test_run: TestRun) -> str:
    not_run = "?? ???????????"
    lines = [
        f"??????? {attempt}/{max_attempts}",
        f"PR: {pr_url}",
        f"???????: {result.verdict}",
    ]
    if result.summary:
        lines.append(f"????: {result.summary}")
    if result.blocking:
        lines.append("???????:")
        for item in result.blocking:
            lines.append(f"- {item}")
    if result.notes:
        lines.append("???????:")
        for item in result.notes:
            lines.append(f"- {item}")
    lines.append("????????:")
    if quality_results:
        for res in quality_results:
            lines.append(_format_command(res))
    else:
        lines.append("- ????????: ?? ???????")
    if test_run.result is not None:
        lines.append(_format_command(test_run.result))
    else:
        lines.append(f"- ?????: {test_run.note or not_run}")
    if test_run.generated_files:
        lines.append("- ??????????????? ?????: " + ", ".join(test_run.generated_files))
    return "\n".join(lines)

def _format_command(res) -> str:
    status = "OK" if res.returncode == 0 else f"FAIL ({res.returncode})"
    output = (res.stdout or res.stderr or "").strip()[:2000]
    return "\n".join(
        [
            f"- `{res.cmd}` -> {status}",
            "```",
            output,
            "```",
        ]
    )
