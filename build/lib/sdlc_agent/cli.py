from __future__ import annotations

import argparse
import os
import sys

from .agent_code import run_code_agent
from .agent_review import run_review_agent
from .config import load_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sdlc-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    code = sub.add_parser("code", help="Run code agent for an Issue")
    code.add_argument("--issue", type=int, required=True, help="Issue number in agent repo")
    code.add_argument(
        "--agent-repo",
        default=os.getenv("GITHUB_REPOSITORY", ""),
        help="Agent repo full name (owner/name)",
    )

    review = sub.add_parser("review", help="Run review agent for a PR")
    review.add_argument("--repo", required=True, help="Target repo full name (owner/name)")
    review.add_argument("--pr", type=int, required=True, help="Pull request number")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "code":
        if not args.agent_repo:
            parser.error("Missing --agent-repo or GITHUB_REPOSITORY")
        run_code_agent(settings, agent_repo=args.agent_repo, issue_number=args.issue)
        return 0

    if args.command == "review":
        run_review_agent(settings, target_repo=args.repo, pr_number=args.pr)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
