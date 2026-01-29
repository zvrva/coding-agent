from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import re
from typing import Any


STATE_MARKER = "SDLC_AGENT_STATE"
_STATE_RE = re.compile(rf"<!--\s*{STATE_MARKER}:\s*(\{{.*?\}})\s*-->", re.DOTALL)


@dataclass(frozen=True)
class State:
    target_repo: str
    source_issue_url: str
    iteration: int
    max_iterations: int
    last_verdict: str
    created_at: str


def new_state(target_repo: str, source_issue_url: str, max_iterations: int) -> State:
    return State(
        target_repo=target_repo,
        source_issue_url=source_issue_url,
        iteration=0,
        max_iterations=max_iterations,
        last_verdict="pending",
        created_at=_now_iso(),
    )


def parse_state(text: str) -> State | None:
    match = _STATE_RE.search(text or "")
    if not match:
        return None
    payload = match.group(1)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return _state_from_dict(data)


def render_state(state: State) -> str:
    payload = json.dumps(_state_to_dict(state), ensure_ascii=True, separators=(",", ":"))
    return f"<!-- {STATE_MARKER}: {payload} -->"


def next_iteration(state: State, verdict: str) -> State:
    return replace(
        state,
        iteration=state.iteration + 1,
        last_verdict=verdict,
    )


def can_iterate(state: State) -> bool:
    return state.iteration < state.max_iterations


def _state_from_dict(data: dict[str, Any]) -> State:
    return State(
        target_repo=str(data.get("target_repo", "")),
        source_issue_url=str(data.get("source_issue_url", "")),
        iteration=int(data.get("iteration", 0)),
        max_iterations=int(data.get("max_iterations", 5)),
        last_verdict=str(data.get("last_verdict", "unknown")),
        created_at=str(data.get("created_at", "")),
    )


def _state_to_dict(state: State) -> dict[str, Any]:
    return {
        "target_repo": state.target_repo,
        "source_issue_url": state.source_issue_url,
        "iteration": state.iteration,
        "max_iterations": state.max_iterations,
        "last_verdict": state.last_verdict,
        "created_at": state.created_at,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
