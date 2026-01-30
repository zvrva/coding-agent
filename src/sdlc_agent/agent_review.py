from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from github.GithubException import GithubException

from .config import Settings
from .github import GitHubClient
from .llm import chat, extract_json
from .state import parse_state, render_state


PROMPT_REVIEW = """Ты — ревьюер. Проверь PR на соответствие задаче.
Верни ТОЛЬКО JSON со схемой:
{
  "verdict": "approve" | "changes_requested",
  "summary": "краткое резюме",
  "blocking": ["список блокирующих проблем"],
  "notes": ["дополнительные замечания"]
}
Правила:
- Если тесты падают, verdict должен быть changes_requested.
- Если требования не выполнены, verdict должен быть changes_requested.
- Никакого markdown. Только JSON.
"""


@dataclass(frozen=True)
class ReviewResult:
    verdict: str
    summary: str
    blocking: list[str]
    notes: list[str]


def run_review_agent(settings: Settings, target_repo: str, pr_number: int) -> ReviewResult:
    gh = GitHubClient(settings.github_token, settings.github_api_base)
    pr = gh.get_pull(target_repo, pr_number)
    diff = gh.get_pr_diff(target_repo, pr_number)
    files = gh.get_pr_files(target_repo, pr_number)

    issue_text = ""
    state_comment = gh.find_state_comment(target_repo, pr_number)
    if state_comment:
        st = parse_state(state_comment.body)
        if st and st.source_issue_url:
            issue = gh.get_issue_by_url(st.source_issue_url)
            issue_text = issue.body or ""

    context = _build_review_context(issue_text, diff, files)
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

    review_body = _format_review_comment(result)
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

    return result


def _build_review_context(issue_text: str, diff: str, files: list[dict[str, Any]]) -> str:
    parts = ["ISSUE:", issue_text.strip(), "", "DIFF:", diff[:10000], "", "FILES:"]
    for f in files:
        parts.append(
            f"- {f.get('filename')} ({f.get('status')} +{f.get('additions')}/-{f.get('deletions')})"
        )
    return "\n".join(parts)


def _parse_review_result(data: dict[str, Any]) -> ReviewResult:
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in {"approve", "changes_requested"}:
        verdict = "changes_requested"
    summary = str(data.get("summary", "")).strip()
    blocking = [str(x) for x in data.get("blocking", [])]
    notes = [str(x) for x in data.get("notes", [])]
    return ReviewResult(verdict=verdict, summary=summary, blocking=blocking, notes=notes)


def _format_review_comment(result: ReviewResult) -> str:
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
    return "\n".join(lines)
