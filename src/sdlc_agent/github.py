from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Any

import requests
from github import Github


from .state import STATE_MARKER


@dataclass(frozen=True)
class RepoRef:
    full_name: str


class GitHubClient:
    def __init__(self, token: str, api_base: str) -> None:
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._gh = Github(login_or_token=token, base_url=self._api_base)

    def get_repo(self, full_name: str):
        return self._gh.get_repo(full_name)

    def dispatch_event(self, repo_full: str, event_type: str, payload: dict) -> None:
        url = f"{self._api_base}/repos/{repo_full}/dispatches"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
        }
        data = {"event_type": event_type, "client_payload": payload}
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"Dispatch failed: {resp.status_code} {resp.text}")

    def set_attempt_label(self, repo_full: str, pr_number: int, iteration: int, max_iterations: int) -> None:
        repo = self.get_repo(repo_full)
        issue = repo.get_issue(pr_number)
        labels = list(issue.get_labels())
        pattern = r"^(attempt|???????) \d+/\d+$"
        for lbl in labels:
            if re.match(pattern, lbl.name, flags=re.IGNORECASE):
                issue.remove_from_labels(lbl)
        label_name = f"??????? {iteration}/{max_iterations}"
        try:
            repo.get_label(label_name)
        except Exception:
            repo.create_label(name=label_name, color="ededed")
        issue.add_to_labels(label_name)

    def get_issue(self, repo_full: str, number: int):
        return self.get_repo(repo_full).get_issue(number)

    def get_pull(self, repo_full: str, number: int):
        return self.get_repo(repo_full).get_pull(number)

    def get_issue_by_url(self, url: str):
        repo_full, number = _parse_issue_url(url)
        return self.get_issue(repo_full, number)

    def create_branch(self, repo_full: str, branch_name: str, from_branch: str | None = None) -> None:
        repo = self.get_repo(repo_full)
        base_branch = from_branch or repo.default_branch
        base_ref = repo.get_git_ref(f"heads/{base_branch}")
        repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=base_ref.object.sha)

    def create_or_update_pr(
        self,
        repo_full: str,
        branch: str,
        title: str,
        body: str,
        base_branch: str | None = None,
    ):
        repo = self.get_repo(repo_full)
        base = base_branch or repo.default_branch

        pr = self.find_open_pr_by_branch(repo_full, branch)
        if pr:
            pr.edit(title=title, body=body)
            return pr

        return repo.create_pull(title=title, body=body, base=base, head=branch)

    def post_comment(self, repo_full: str, issue_or_pr_number: int, body: str) -> None:
        issue = self.get_issue(repo_full, issue_or_pr_number)
        issue.create_comment(body)

    def post_review(self, repo_full: str, pr_number: int, body: str, event: str) -> None:
        pr = self.get_pull(repo_full, pr_number)
        pr.create_review(body=body, event=event)

    def find_state_comment(self, repo_full: str, pr_number: int):
        pr = self.get_pull(repo_full, pr_number)
        for comment in pr.get_issue_comments():
            if STATE_MARKER in comment.body:
                return comment
        return None

    def upsert_state_comment(self, repo_full: str, pr_number: int, body: str) -> None:
        comment = self.find_state_comment(repo_full, pr_number)
        if comment:
            comment.edit(body)
        else:
            self.post_comment(repo_full, pr_number, body)

    def get_pr_diff(self, repo_full: str, pr_number: int) -> str:
        pr = self.get_pull(repo_full, pr_number)
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github.v3.diff",
        }
        resp = requests.get(pr.diff_url, headers=headers, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"Failed to fetch PR diff: {resp.status_code} {resp.text}")
        return resp.text

    def get_pr_files(self, repo_full: str, pr_number: int) -> list[dict[str, Any]]:
        pr = self.get_pull(repo_full, pr_number)
        files = []
        for f in pr.get_files():
            files.append(
                {
                    "filename": f.filename,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                    "changes": f.changes,
                    "patch": f.patch or "",
                }
            )
        return files

    def find_open_pr_by_branch(self, repo_full: str, branch: str):
        repo = self.get_repo(repo_full)
        head = f"{repo.owner.login}:{branch}"
        pulls = repo.get_pulls(state="open", head=head)
        for pr in pulls:
            return pr
        return None

    def get_current_user_login(self) -> str:
        return self._gh.get_user().login


def _parse_issue_url(url: str) -> tuple[str, int]:
    parts = url.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid issue URL: {url}")
    repo_full = "/".join(parts[-4:-2])
    number = int(parts[-1])
    return repo_full, number
