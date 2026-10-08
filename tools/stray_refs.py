#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Archives branches/tags that are still based on the pre-rewrite history.

The torch_tpu git history was rewritten on 2026-10-07 to drop ~2 GiB of golden
`.gz` blobs. Any branch or tag that is pushed from a clone of the *old*
history contains the old root commit (`PRE_REWRITE_ROOT`) and drags those blobs
back into every default `git clone`. This script, run by
`.github/workflows/stray_refs.yml` on every push and on a schedule, finds such
refs and moves them out of the way while telling the pusher how to recover.

Contract:
- Inputs:
    Environment variables:
      GITHUB_TOKEN: Token used for git fetch/push and the REST API. Needs
        `contents: write`, `issues: write` and `pull-requests: write`. The
        default `GITHUB_TOKEN` is enough for unprotected refs; restoring a
        protected branch needs a token that can bypass its ruleset.
      GITHUB_REPOSITORY: 'owner/repo'.
      GITHUB_EVENT_NAME: 'push', 'schedule' or 'workflow_dispatch'.
      GITHUB_EVENT_PATH: Path to the event payload JSON. For 'push' events
        the payload supplies the pusher and the pre-push SHA directly.
      GITHUB_API_URL / GITHUB_SERVER_URL: (Optional) API and web base URLs.
      RUNNER_TEMP: (Optional) Where the scratch repository is created.
      STRAY_REFS_DRY_RUN: (Optional) If 'true', plan and `git push --dry-run`
        but post nothing.

- Behaviour:
    1. A scratch bare repository receives a commits-only partial fetch
       (`--filter=tree:0`) of every branch and tag: a few MiB, never the blobs.
    2. Every `refs/heads/*` / `refs/tags/*` that contains `PRE_REWRITE_ROOT`
       is a stray ref.
    3. For each stray ref at SHA S, one atomic `git push` guarded by
       `--force-with-lease` on both sides creates `refs/archive/<name>` at S
       (fast-forwarding an existing archive ref, or using
       `refs/archive/<name>-<sha12>` if an unrelated one exists) and deletes
       the branch/tag. Nothing already archived is ever overwritten and no
       objects are uploaded because S is already on the remote.
    4. `main`, `master` and `release/*` are never archived. If such a branch
       was *force-pushed* from a rewritten commit to a pre-rewrite one and the
       pre-rewrite SHA is already archived, the push is undone by resetting
       the branch to its previous SHA via the refs API. Otherwise an
       "ACTION NEEDED" issue is opened and nothing is changed.
    5. The pusher is notified once per (ref, SHA): as a comment on the ref's
       open pull request if any, else via an issue that @-mentions them. Bots
       are not notified for plain archives.

- Return code:
    Exit 0 if every stray ref was handled; 1 if any ref failed or needs a human.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Callable, Final, Mapping
import urllib.error
import urllib.parse
import urllib.request

PRE_REWRITE_ROOT: Final[str] = "0d0db441625684fa4f795f38498e373daf2c51a7"
ARCHIVE_PREFIX: Final[str] = "refs/archive"
SCAN_PREFIX: Final[str] = "refs/scan"
FETCH_FILTER: Final[str] = "tree:0"
PROTECTED_BRANCHES: Final[frozenset[str]] = frozenset({"main", "master"})
PROTECTED_PREFIXES: Final[tuple[str, ...]] = ("release/",)
ISSUE_LABEL: Final[str] = "stray-pre-rewrite-ref"
USER_AGENT: Final[str] = "TorchTPU-Stray-Refs"

# `(method, path, body) -> parsed JSON`. Replaceable in tests.
ApiFn = Callable[[str, str, Any], Any]


# --------------------------------------------------------------------------
# git plumbing
# --------------------------------------------------------------------------


def _git_base(token: str | None) -> list[str]:
  cmd = ["git", "-c", "safe.bareRepository=all"]
  if token:
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    cmd += ["-c", f"http.extraheader=AUTHORIZATION: basic {basic}"]
  return cmd


def _git(
  git_dir: pathlib.Path,
  *args: str,
  token: str | None = None,
  check: bool = True,
) -> str:
  proc = subprocess.run(
    [*_git_base(token), f"--git-dir={git_dir}", *args],
    stdout=subprocess.PIPE,
    stderr=None if check else subprocess.DEVNULL,
    text=True,
    check=check,
  )
  return proc.stdout.strip()


def list_refs(git_dir: pathlib.Path, prefix: str) -> dict[str, str]:
  out = _git(
    git_dir, "for-each-ref", "--format=%(refname) %(objectname)", prefix
  )
  return dict(line.split(" ", 1) for line in out.splitlines() if line)


def refs_containing(
  git_dir: pathlib.Path, prefix: str, commit: str
) -> dict[str, str]:
  if not commit_exists(git_dir, commit):
    return {}
  out = _git(
    git_dir,
    "for-each-ref",
    f"--contains={commit}",
    "--format=%(refname) %(objectname)",
    prefix,
  )
  return dict(line.split(" ", 1) for line in out.splitlines() if line)


def commit_exists(git_dir: pathlib.Path, sha: str) -> bool:
  return bool(
    _git(
      git_dir,
      "rev-parse",
      "--verify",
      "--quiet",
      f"{sha}^{{commit}}",
      check=False,
    )
  )


def is_ancestor(git_dir: pathlib.Path, ancestor: str, descendant: str) -> bool:
  proc = subprocess.run(
    [
      *_git_base(None),
      f"--git-dir={git_dir}",
      "merge-base",
      "--is-ancestor",
      ancestor,
      descendant,
    ],
    check=False,
    stderr=subprocess.DEVNULL,
  )
  return proc.returncode == 0


def ensure_scratch_repo(scratch: pathlib.Path, remote_url: str) -> pathlib.Path:
  """Creates a bare scratch repo whose `origin` fetches into refs/scan/*."""
  if not (scratch / "HEAD").is_file():
    scratch.mkdir(parents=True, exist_ok=True)
    subprocess.run(
      [*_git_base(None), "init", "--bare", "--quiet", str(scratch)], check=True
    )
  if _git(scratch, "remote", "get-url", "origin", check=False):
    _git(scratch, "remote", "set-url", "origin", remote_url)
  else:
    _git(scratch, "remote", "add", "--no-tags", "origin", remote_url)
  _git(scratch, "config", "remote.origin.tagOpt", "--no-tags")
  _git(scratch, "config", "--unset-all", "remote.origin.fetch", check=False)
  for src in ("heads", "tags"):
    _git(
      scratch,
      "config",
      "--add",
      "remote.origin.fetch",
      f"+refs/{src}/*:{SCAN_PREFIX}/{src}/*",
    )
  return scratch


def fetch_snapshot(git_dir: pathlib.Path, token: str | None) -> None:
  """Commits-only fetch of all branches/tags plus the archived old main."""
  _git(
    git_dir,
    "fetch",
    "--quiet",
    "--prune",
    f"--filter={FETCH_FILTER}",
    "origin",
    token=token,
  )
  _git(
    git_dir,
    "fetch",
    "--quiet",
    f"--filter={FETCH_FILTER}",
    "origin",
    f"+{ARCHIVE_PREFIX}/main:{SCAN_PREFIX}/archive/main",
    token=token,
    check=False,
  )


def fetch_commit(git_dir: pathlib.Path, sha: str, token: str | None) -> bool:
  """Fetches an unadvertised commit (e.g. the `before` of a force-push)."""
  if commit_exists(git_dir, sha):
    return True
  _git(
    git_dir,
    "fetch",
    "--quiet",
    f"--filter={FETCH_FILTER}",
    "origin",
    sha,
    token=token,
    check=False,
  )
  return commit_exists(git_dir, sha)


def pre_rewrite_root(git_dir: pathlib.Path) -> str | None:
  """`PRE_REWRITE_ROOT`, or the root of the archived main for test repos."""
  if commit_exists(git_dir, PRE_REWRITE_ROOT):
    return PRE_REWRITE_ROOT
  out = _git(
    git_dir,
    "rev-list",
    "--max-parents=0",
    f"{SCAN_PREFIX}/archive/main",
    check=False,
  )
  return out.splitlines()[0] if out else None


# --------------------------------------------------------------------------
# GitHub REST API
# --------------------------------------------------------------------------


def make_api(token: str, api_url: str) -> ApiFn:
  """Returns an `ApiFn` backed by urllib."""
  headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "User-Agent": USER_AGENT,
    "X-GitHub-Api-Version": "2022-11-28",
  }

  def api(method: str, path: str, body: Any = None) -> Any:
    data = None
    hdrs = dict(headers)
    if body is not None:
      data = json.dumps(body).encode("utf-8")
      hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(
      f"{api_url}/{path}", data=data, headers=hdrs, method=method
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
      raw = resp.read()
    return json.loads(raw) if raw.strip() else None

  return api


def _is_bot(login: str) -> bool:
  return login.endswith("[bot]")


@dataclasses.dataclass(frozen=True)
class PushInfo:
  actor: str
  forced: bool
  before: str
  after: str
  timestamp: str


def push_from_event(
  event_name: str, payload: Mapping[str, Any], refname: str, sha: str
) -> PushInfo | None:
  """Attribution straight from a `push` event payload, when it matches."""
  if event_name != "push" or payload.get("ref") != refname:
    return None
  if payload.get("after") != sha or payload.get("deleted"):
    return None
  sender = payload.get("sender") or {}
  pusher = payload.get("pusher") or {}
  actor = sender.get("login") or pusher.get("name") or "unknown"
  head = payload.get("head_commit") or {}
  return PushInfo(
    actor=actor,
    forced=bool(payload.get("forced")),
    before=payload.get("before", ""),
    after=sha,
    timestamp=head.get("timestamp") or "",
  )


def push_from_activity(
  api: ApiFn, repo: str, refname: str, sha: str
) -> PushInfo | None:
  """Attribution via the repository activity API (schedule / other refs)."""
  q = urllib.parse.urlencode({"ref": refname, "per_page": 30})
  try:
    entries = api("GET", f"repos/{repo}/activity?{q}", None) or []
  except urllib.error.HTTPError:
    return None
  for entry in entries:
    if entry.get("after") == sha:
      return PushInfo(
        actor=entry["actor"]["login"],
        forced=entry["activity_type"] == "force_push",
        before=entry["before"],
        after=sha,
        timestamp=entry["timestamp"],
      )
  return None


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


class Action(enum.Enum):
  ARCHIVE = "archive"
  FAST_FORWARD_ARCHIVE = "ff-archive"
  ARCHIVE_SUFFIXED = "archive-suffixed"
  DELETE_ONLY = "delete-only"
  PROTECTED = "protected"


@dataclasses.dataclass(frozen=True)
class Migration:
  refname: str
  sha: str
  archive_ref: str
  archive_lease: str  # Expected remote SHA of archive_ref; "" = must not exist.
  action: Action
  old_base: str | None  # Merge base with the pre-rewrite main, if known.

  @property
  def short_name(self) -> str:
    if self.refname.startswith("refs/heads/"):
      return self.refname.removeprefix("refs/heads/")
    return self.refname.removeprefix("refs/")


def is_protected(refname: str) -> bool:
  if not refname.startswith("refs/heads/"):
    return False
  branch = refname.removeprefix("refs/heads/")
  return branch in PROTECTED_BRANCHES or branch.startswith(PROTECTED_PREFIXES)


def ls_remote(
  git_dir: pathlib.Path, pattern: str, token: str | None
) -> dict[str, str]:
  out = _git(git_dir, "ls-remote", "origin", pattern, token=token)
  refs: dict[str, str] = {}
  for line in out.splitlines():
    sha, ref = line.split(None, 1)
    if not ref.endswith("^{}"):
      refs[ref] = sha
  return refs


def plan_migrations(
  git_dir: pathlib.Path, token: str | None
) -> list[Migration]:
  root = pre_rewrite_root(git_dir)
  if root is None:
    return []
  stray: dict[str, str] = {}
  for src in ("heads", "tags"):
    for scan_ref, sha in refs_containing(
      git_dir, f"{SCAN_PREFIX}/{src}/", root
    ).items():
      stray[scan_ref.replace(f"{SCAN_PREFIX}/", "refs/", 1)] = sha
  if not stray:
    return []

  remote_archived = ls_remote(git_dir, f"{ARCHIVE_PREFIX}/*", token)
  old_main = f"{SCAN_PREFIX}/archive/main"
  have_old_main = commit_exists(git_dir, old_main)

  plans: list[Migration] = []
  for refname, sha in sorted(stray.items()):
    if refname.startswith("refs/heads/"):
      archive_ref = f"{ARCHIVE_PREFIX}/{refname.removeprefix('refs/heads/')}"
    else:
      archive_ref = f"{ARCHIVE_PREFIX}/{refname.removeprefix('refs/')}"
    old_base = None
    if have_old_main:
      old_base = _git(git_dir, "merge-base", old_main, sha, check=False) or None
    existing = remote_archived.get(archive_ref)
    if is_protected(refname):
      action, lease = Action.PROTECTED, existing or ""
    elif existing is None:
      action, lease = Action.ARCHIVE, ""
    elif existing == sha:
      action, lease = Action.DELETE_ONLY, existing
    elif is_ancestor(git_dir, existing, sha):
      action, lease = Action.FAST_FORWARD_ARCHIVE, existing
    else:
      archive_ref = f"{archive_ref}-{sha[:12]}"
      if remote_archived.get(archive_ref) == sha:
        action, lease = Action.DELETE_ONLY, sha
      elif archive_ref in remote_archived:
        raise RuntimeError(f"{archive_ref} exists with an unexpected SHA")
      else:
        action, lease = Action.ARCHIVE_SUFFIXED, ""
    plans.append(Migration(refname, sha, archive_ref, lease, action, old_base))
  return plans


def push_command(
  git_dir: pathlib.Path,
  m: Migration,
  token: str | None,
  *,
  dry_run: bool,
) -> list[str]:
  scan_ref = m.refname.replace("refs/", f"{SCAN_PREFIX}/", 1)
  cmd = [
    *_git_base(token),
    f"--git-dir={git_dir}",
    "push",
    "--no-verify",
    "--atomic",
    f"--force-with-lease={m.refname}:{m.sha}",
  ]
  refspecs = [f":{m.refname}"]
  if m.action is not Action.DELETE_ONLY:
    cmd.append(f"--force-with-lease={m.archive_ref}:{m.archive_lease}")
    refspecs.insert(0, f"{scan_ref}:{m.archive_ref}")
  if dry_run:
    cmd.append("--dry-run")
  return [*cmd, "origin", *refspecs]


def restore_protected_branch(
  api: ApiFn,
  git_dir: pathlib.Path,
  repo: str,
  m: Migration,
  push: PushInfo | None,
  token: str | None,
  *,
  dry_run: bool,
) -> str | None:
  """Undoes a stale force-push to a protected branch; returns restored SHA."""
  if push is None or not push.forced or not push.before:
    return None
  if m.archive_lease != m.sha:
    return None  # The pushed SHA carries new work; a human must look.
  if not fetch_commit(git_dir, push.before, token):
    return None
  root = pre_rewrite_root(git_dir)
  if root is None or is_ancestor(git_dir, root, push.before):
    return None
  if not dry_run:
    current = api("GET", f"repos/{repo}/git/ref/{m.refname[5:]}", None)
    if current["object"]["sha"] != m.sha:
      return None
    api(
      "PATCH",
      f"repos/{repo}/git/refs/{m.refname[5:]}",
      {"sha": push.before, "force": True},
    )
  return push.before


# --------------------------------------------------------------------------
# Notification
# --------------------------------------------------------------------------


class Outcome(enum.Enum):
  ARCHIVED = "archived"
  RESTORED = "restored"
  NEEDS_HUMAN = "needs-human"


def _marker(m: Migration) -> str:
  return f"<!-- stray-ref {m.refname}@{m.sha} -->"


def owner_hint(m: Migration) -> str:
  name = m.short_name
  local = name.removeprefix("tags/")
  lines = [
    "# Get the archived commit back as a local branch:",
    f"git fetch origin {m.archive_ref}:refs/heads/{local}-old",
  ]
  if m.old_base and m.refname.startswith("refs/heads/"):
    lines += [
      "# Or replay your commits onto the rewritten history:",
      f"git fetch origin && git rebase --onto origin/main"
      f" {m.old_base[:12]} {name}",
    ]
  return "\n".join(lines)


def compose_notification(
  m: Migration,
  push: PushInfo | None,
  outcome: Outcome,
  *,
  restored_to: str | None = None,
  server_url: str = "https://github.com",
  repo: str = "",
) -> tuple[str, str]:
  name = m.short_name
  root = PRE_REWRITE_ROOT[:12]
  mention = f"@{push.actor}" if push and not _is_bot(push.actor) else "Hi"
  when = f" on {push.timestamp}" if push and push.timestamp else ""
  if outcome is Outcome.RESTORED:
    assert restored_to is not None
    title = f"Protected branch `{name}` was reset to pre-rewrite history"
    what = (
      f"`{name}` was force-pushed{when} to `{m.sha[:12]}`, a commit from the"
      " **pre-rewrite** torch_tpu history (it contains the old root commit"
      f" `{root}`). It has been reset back to `{restored_to[:12]}`, the"
      " commit it pointed at before that push."
    )
  elif outcome is Outcome.NEEDS_HUMAN:
    title = (
      f"ACTION NEEDED: protected branch `{name}` is on pre-rewrite history"
    )
    what = (
      f"`{name}` (pushed{when}) is at `{m.sha[:12]}`, which contains the"
      f" pre-rewrite root commit `{root}`. This is a protected branch and"
      " the push cannot be undone automatically, so **nothing has been"
      " changed**. A repository admin needs to archive the new commits and"
      " reset the branch onto the rewritten history."
    )
  else:
    title = f"Stray pre-rewrite ref `{name}` was archived"
    what = (
      f"`{name}` (pushed{when} at `{m.sha[:12]}`) was based on the"
      " **pre-rewrite** torch_tpu history: it contains the old root commit"
      f" `{root}`, which drags the removed golden `.gz` blobs back into"
      f" every default clone. It has been moved to `{m.archive_ref}` and"
      f" removed from `{m.refname.rsplit('/', 1)[0]}/`."
    )
  body = [
    f"{mention} — this is an automated notice from the torch_tpu"
    " history-rewrite sweeper"
    f" ([workflow]({server_url}/{repo}/actions/workflows/stray_refs.yml)).",
    "",
    what,
    "",
  ]
  if outcome is Outcome.NEEDS_HUMAN:
    body.append("Please do not push to this branch again until it is fixed.")
  else:
    body += [
      "Nothing is lost: the exact commit you pushed is preserved under"
      " `refs/archive/`. The simplest fix is to re-clone the repository;"
      " to recover work from the old branch:",
      "",
      "```",
      owner_hint(m),
      "```",
    ]
  body += [
    "",
    "Please make sure every clone you push from has been re-cloned or"
    " reset onto the rewritten history. Reply here if something looks"
    " wrong.",
    "",
    _marker(m),
  ]
  return title, "\n".join(body)


def already_notified(
  api: ApiFn, repo: str, m: Migration, pr_number: int | None
) -> bool:
  marker = _marker(m)
  if pr_number is not None:
    comments = (
      api("GET", f"repos/{repo}/issues/{pr_number}/comments?per_page=100", None)
      or []
    )
    return any(marker in (c.get("body") or "") for c in comments)
  q = urllib.parse.urlencode({
    "labels": ISSUE_LABEL,
    "state": "all",
    "per_page": 100,
  })
  issues = api("GET", f"repos/{repo}/issues?{q}", None) or []
  return any(marker in (i.get("body") or "") for i in issues)


def find_open_pr(api: ApiFn, repo: str, branch: str) -> int | None:
  owner = repo.split("/", 1)[0]
  q = urllib.parse.urlencode({"state": "open", "head": f"{owner}:{branch}"})
  prs = api("GET", f"repos/{repo}/pulls?{q}", None) or []
  return prs[0]["number"] if prs else None


def notify(
  api: ApiFn,
  repo: str,
  m: Migration,
  push: PushInfo | None,
  outcome: Outcome,
  *,
  restored_to: str | None = None,
  server_url: str = "https://github.com",
  dry_run: bool = False,
) -> bool:
  if push is not None and _is_bot(push.actor) and outcome is Outcome.ARCHIVED:
    return False
  pr_number = None
  if m.refname.startswith("refs/heads/"):
    pr_number = find_open_pr(api, repo, m.short_name)
  if already_notified(api, repo, m, pr_number):
    return False
  title, body = compose_notification(
    m,
    push,
    outcome,
    restored_to=restored_to,
    server_url=server_url,
    repo=repo,
  )
  if dry_run:
    print(f"  [dry-run] would notify: {title}")
    return True
  if pr_number is not None:
    api("POST", f"repos/{repo}/issues/{pr_number}/comments", {"body": body})
    print(f"  Commented on PR #{pr_number}.")
    return True
  params: dict[str, Any] = {
    "title": title,
    "body": body,
    "labels": [ISSUE_LABEL],
  }
  if push is not None and not _is_bot(push.actor):
    params["assignees"] = [push.actor]
  try:
    issue = api("POST", f"repos/{repo}/issues", params)
  except urllib.error.HTTPError as err:
    if err.code != 422:
      raise
    params.pop("assignees", None)  # Not a collaborator.
    issue = api("POST", f"repos/{repo}/issues", params)
  print(f"  Opened issue #{issue['number']}.")
  return True


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SweepStats:
  stray_refs_found: int = 0
  refs_archived: int = 0
  protected_restored: int = 0
  refs_failed: int = 0
  notifications_sent: int = 0


def sweep(
  git_dir: pathlib.Path,
  api: ApiFn,
  repo: str,
  *,
  token: str | None,
  event_name: str = "schedule",
  payload: Mapping[str, Any] | None = None,
  server_url: str = "https://github.com",
  dry_run: bool = False,
) -> SweepStats:
  payload = payload or {}
  fetch_snapshot(git_dir, token)
  plans = plan_migrations(git_dir, token)
  archived = restored = failed = notified = 0

  for m in plans:
    print(f"{m.refname} @ {m.sha[:12]} -> {m.action.value} {m.archive_ref}")
    push = push_from_event(event_name, payload, m.refname, m.sha)
    if push is None:
      push = push_from_activity(api, repo, m.refname, m.sha)
    if push:
      print(
        f"  pushed by {push.actor}"
        f" ({'force ' if push.forced else ''}push, before"
        f" {push.before[:12]})"
      )
    restored_to: str | None = None

    if m.action is Action.PROTECTED:
      restored_to = restore_protected_branch(
        api, git_dir, repo, m, push, token, dry_run=dry_run
      )
      if restored_to is None:
        failed += 1
        outcome = Outcome.NEEDS_HUMAN
        print(
          "  PROTECTED branch on pre-rewrite history; needs a human.",
          file=sys.stderr,
        )
      else:
        restored += 1
        outcome = Outcome.RESTORED
        print(f"  Reset to {restored_to[:12]} (undid stale force-push).")
    else:
      cmd = push_command(git_dir, m, token, dry_run=dry_run)
      if subprocess.run(cmd, check=False).returncode != 0:
        failed += 1
        print("  FAILED to archive; remote unchanged.", file=sys.stderr)
        continue
      archived += 1
      outcome = Outcome.ARCHIVED

    if notify(
      api,
      repo,
      m,
      push,
      outcome,
      restored_to=restored_to,
      server_url=server_url,
      dry_run=dry_run,
    ):
      notified += 1

  return SweepStats(len(plans), archived, restored, failed, notified)


def main() -> int:
  token = os.environ.get("GITHUB_TOKEN")  # PY_ENVIRON_OK=ci
  repo = os.environ.get("GITHUB_REPOSITORY")  # PY_ENVIRON_OK=ci
  event_name = os.environ.get("GITHUB_EVENT_NAME", "")  # PY_ENVIRON_OK=ci
  event_path = os.environ.get("GITHUB_EVENT_PATH")  # PY_ENVIRON_OK=ci
  api_url = os.environ.get(  # PY_ENVIRON_OK=ci
    "GITHUB_API_URL", "https://api.github.com"
  )
  server_url = os.environ.get(  # PY_ENVIRON_OK=ci
    "GITHUB_SERVER_URL", "https://github.com"
  )
  runner_temp = os.environ.get("RUNNER_TEMP")  # PY_ENVIRON_OK=ci
  dry_run = os.environ.get(  # PY_ENVIRON_OK=ci
    "STRAY_REFS_DRY_RUN", ""
  ).lower() in ("1", "true", "yes")

  if not (token and repo):
    print("GITHUB_TOKEN and GITHUB_REPOSITORY are required", file=sys.stderr)
    return 1
  payload: dict[str, Any] = {}
  if event_path and pathlib.Path(event_path).is_file():
    payload = json.loads(pathlib.Path(event_path).read_text(encoding="utf-8"))
  if event_name == "push" and payload.get("deleted"):
    print("Branch deletion; nothing to do.")
    return 0

  scratch = (
    pathlib.Path(runner_temp or tempfile.gettempdir()) / "stray_refs.git"
  )
  git_dir = ensure_scratch_repo(scratch, f"{server_url}/{repo}.git")
  api = make_api(token, api_url)
  mode = "DRY RUN" if dry_run else "LIVE"
  print(f"[{mode}] sweeping {repo} ({event_name})")
  stats = sweep(
    git_dir,
    api,
    repo,
    token=token,
    event_name=event_name,
    payload=payload,
    server_url=server_url,
    dry_run=dry_run,
  )
  print(
    f"[{mode}] stray={stats.stray_refs_found}"
    f" archived={stats.refs_archived} restored={stats.protected_restored}"
    f" failed={stats.refs_failed} notified={stats.notifications_sent}"
  )
  return 1 if stats.refs_failed else 0


if __name__ == "__main__":
  sys.exit(main())
