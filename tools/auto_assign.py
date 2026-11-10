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

"""Auto-assigns GitHub PR reviewers and issue owners from BUILD expert lists.

Runs directly from the checked-out repository root in GitHub Actions (not built
or executed by Bazel) using only the Python standard library and `GITHUB_TOKEN`.
Reads `# Domain experts for this package:` blocks from exported `BUILD` and
`BUILD.bazel` files, filters exclusions and authors, and picks the least-loaded
near-top expert across open PRs and issues.

Usage:
  GITHUB_TOKEN=... GITHUB_EVENT_PATH=event.json \\
    GITHUB_REPOSITORY=google-pytorch/torch_tpu \\
    GITHUB_EVENT_NAME=pull_request_target \\
    ./tools/auto_assign.py --exclude-file .github/auto_assign_excludes.txt

Inputs:
  Environment variables:
    GITHUB_TOKEN: Bearer token for GitHub REST API authentication.
    GITHUB_EVENT_PATH: Path to the GitHub Actions JSON event payload file.
    GITHUB_REPOSITORY: Repository slug in `owner/repo` format.
    GITHUB_EVENT_NAME: Event name (`issues`, `pull_request`, or
      `pull_request_target`).
    GITHUB_API_URL: Optional base API URL (defaults to
      `https://api.github.com`).
  Exclude file format:
    Newline-delimited GitHub handles (case-insensitive); `#` starts comments.
"""

import argparse
import collections
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Final
import urllib.error
import urllib.parse
import urllib.request

_BUILD_NAMES: Final[tuple[str, ...]] = ("BUILD", "BUILD.bazel")
_EXPERTS_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"# Domain experts for this package:\n((?:# [^\n]*(?:\n|\Z))+)"
)
_COPYBARA_CL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?m)^\s*(?:Generated from|Exports)\s+cl/\d+\b"
)
_REPO_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"\b((?:csrc|src/torch_tpu|tests|perflab|examples|bazel|build_files|ci|"
    r"docker|docs|shims|tools|_internal)/[A-Za-z0-9_./-]+)"
)
_BACKTICK_CODE_RE: Final[re.Pattern[str]] = re.compile(r"`([^`]+)`")
_QUALIFIED_OP_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:torch\.|aten::|f\.|nn\.functional\.)([a-z0-9_-]+)\b"
)
_OP_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\b([a-z0-9_-]{3,})\b")
_NEXT_LINK_RE: Final[re.Pattern[str]] = re.compile(r'<([^>]+)>;\s*rel="next"')

_DEPENDABOT_REVIEWER: Final[str] = "chizhang529"
_NEAR_TOP_FRACTION: Final[float] = 1 / 3
_MAX_ISSUE_PROD_PATHS: Final[int] = 4
_MAX_ISSUE_TEST_PATHS: Final[int] = 2
_EXPLICIT_PATH_WEIGHT: Final[float] = 3.0

_OP_NAME_STOPWORDS: Final[frozenset[str]] = frozenset({
    # go/keep-sorted start
    "all",
    "any",
    "cat",
    "clamp",
    "dot",
    "dynamic",
    "equal",
    "experimental",
    "eye",
    "fill",
    "flip",
    "in",
    "index",
    "is",
    "lerp",
    "linear",
    "logical",
    "max",
    "min",
    "mm",
    "norm",
    "normal",
    "pooling",
    "prod",
    "put",
    "random",
    "resize",
    "roll",
    "round",
    "set",
    "sort",
    "stride",
    "take",
    "threshold",
    "to",
    "uniform",
    "unique",
    "view",
    "where",
    # go/keep-sorted end
})


def _compile_keyword_pattern(keywords: Sequence[str]) -> re.Pattern[str]:
  alternatives = "|".join(re.escape(k) for k in keywords)
  return re.compile(rf"(?<![a-z0-9_])(?:{alternatives})(?![a-z0-9_])")


_SUBSYSTEM_RULES: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (_compile_keyword_pattern(("pallas",)), "src/torch_tpu/_internal/pallas"),
    (
        _compile_keyword_pattern((
            "distributed",
            "fsdp",
            "ddp",
            "dtensor",
            "c10d",
            "all_reduce",
            "all_gather",
        )),
        "csrc/distributed",
    ),
    (
        _compile_keyword_pattern(
            ("dynamo", "torch.compile", "inductor", "aot_autograd")
        ),
        "src/torch_tpu/_internal/compile",
    ),
    (_compile_keyword_pattern(("export",)), "src/torch_tpu/_internal/export"),
    (
        _compile_keyword_pattern(("profiler", "xprof", "profiling")),
        "src/torch_tpu/_internal/profiler",
    ),
    (_compile_keyword_pattern(("pjrt",)), "csrc/pjrt"),
    (_compile_keyword_pattern(("mlir", "stablehlo")), "csrc/mlir"),
    (_compile_keyword_pattern(("torchax",)), "perflab/torchax"),
    (_compile_keyword_pattern(("eager",)), "csrc/eager"),
)

_OP_FALLBACK_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:aten::\w+|torch\.\w+|nn\.functional|linalg|ops?|kernels?|"
    r"dtypes?|tensors?)\b"
)

_PR_ACTIONS: Final[frozenset[str]] = frozenset({
    # go/keep-sorted start
    "opened",
    "ready_for_review",
    # go/keep-sorted end
})
_ISSUE_ACTIONS: Final[frozenset[str]] = frozenset({
    # go/keep-sorted start
    "opened",
    "reopened",
    "transferred",
    # go/keep-sorted end
})


@dataclass(frozen=True)
class ActionConfig:
  """Configuration controlling path lookup, filtering, and dry-run behavior."""

  # Path to the repository checkout root used for BUILD and directory lookups.
  repo_root: Path
  # Lowercase GitHub logins excluded from assignment recommendations.
  exclude_logins: Set[str] = frozenset()
  # Minimum score ratio relative to the top candidate for load balancing.
  near_top: float = _NEAR_TOP_FRACTION
  # When True, prints the JSON recommendation without calling GitHub POST APIs.
  dry_run: bool = False


@dataclass(frozen=True)
class GitHubClient:
  """Minimal GitHub REST API client configuration."""

  # Base REST API URL (e.g. `https://api.github.com`).
  api_url: str
  # Repository slug in `owner/repo` format.
  repo: str
  # Bearer token used in the `Authorization` header.
  token: str
  # Per-request socket timeout in seconds.
  timeout: float = 30.0

  def headers(self) -> Mapping[str, str]:
    """Returns standard authenticated GitHub REST API request headers."""
    return {
        "Authorization": f"Bearer {self.token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "TorchTPU-Auto-Assign",
        "X-GitHub-Api-Version": "2022-11-28",
    }

  def _validate_same_origin(self, url: str) -> None:
    expected = urllib.parse.urlsplit(self.api_url)
    actual = urllib.parse.urlsplit(url)
    if (actual.scheme, actual.netloc) != (expected.scheme, expected.netloc):
      raise ValueError(
          f"Refusing cross-origin pagination URL {url!r} (expected origin"
          f" {expected.scheme}://{expected.netloc})."
      )

  def get_paginated(self, path: str) -> Sequence[Any]:
    """Fetches and concatenates JSON pages following `Link: rel=\"next\"`."""
    results: list[Any] = []
    next_url: str | None = f"{self.api_url.rstrip('/')}/{path.lstrip('/')}"
    while next_url:
      self._validate_same_origin(next_url)
      req = urllib.request.Request(next_url, headers=self.headers())
      with urllib.request.urlopen(req, timeout=self.timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, list):
          results.extend(data)
        else:
          results.append(data)
        link_header = resp.headers.get("Link") if resp.headers else ""
      match = _NEXT_LINK_RE.search(link_header or "")
      next_url = match.group(1) if match else None
    return results

  def post_json(self, path: str, body: Mapping[str, Any]) -> Any:
    """Sends an authenticated POST request with a JSON payload."""
    url = f"{self.api_url.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={**self.headers(), "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
      raw = resp.read().decode("utf-8")
      return json.loads(raw) if raw else None


@dataclass(frozen=True)
class Candidate:
  """A scored domain expert eligible for assignment."""

  # GitHub login handle preserved with its original casing.
  github: str
  # Cumulative expertise score across weighted paths.
  score: float
  # Sorted BUILD package paths relative to `ActionConfig.repo_root`.
  packages: Sequence[str]
  # Count of currently open assigned/requested PRs and assigned issues.
  open_load: int = 0


def _is_bot(user: Mapping[str, Any] | None) -> bool:
  if not user:
    return False
  login = user.get("login") or ""
  return user.get("type") == "Bot" or login.endswith("[bot]")


def load_excludes(path: Path) -> Set[str]:
  """Reads lowercase GitHub handles from a newline-delimited exclude file."""
  handles: set[str] = set()
  with open(path, encoding="utf-8") as fp:
    for line in fp:
      entry = line.split("#", 1)[0].strip().lower()
      if entry:
        handles.add(entry)
  return frozenset(handles)


def _find_experts_cached(
    repo_root: Path,
    rel_path: str,
    cache: dict[Path, tuple[str, ...] | None],
) -> tuple[str, Sequence[str]] | None:
  resolved_root = repo_root.resolve()
  clean_rel = rel_path.strip("/")
  candidate_path = (resolved_root / clean_rel).resolve()
  if (
      candidate_path != resolved_root
      and resolved_root not in candidate_path.parents
  ):
    return None

  current = candidate_path if candidate_path.is_dir() else candidate_path.parent
  while current == resolved_root or resolved_root in current.parents:
    if current not in cache:
      cache[current] = None
      for build_name in _BUILD_NAMES:
        build_file = current / build_name
        if build_file.is_file():
          with open(build_file, encoding="utf-8") as fp:
            match = _EXPERTS_BLOCK_RE.search(fp.read())
          if match:
            handles = tuple(
                line[2:].strip()
                for line in match.group(1).splitlines()
                if line[2:].strip()
            )
            cache[current] = handles
            break
    experts = cache[current]
    if experts:
      rel_pkg = (
          "."
          if current == resolved_root
          else current.relative_to(resolved_root).as_posix()
      )
      return rel_pkg, experts
    if current == resolved_root:
      break
    current = current.parent
  return None


def find_experts(
    repo_root: Path, rel_path: str
) -> tuple[str, Sequence[str]] | None:
  """Returns `(package_rel_path, handles)` for the nearest enclosing BUILD."""
  return _find_experts_cached(repo_root, rel_path, {})


def _normalize_repo_path(path: str) -> str:
  cleaned = path.strip().rstrip(".,:;)`'\"").lstrip("/")
  if cleaned.startswith("_internal/"):
    return f"src/torch_tpu/{cleaned}"
  return cleaned


def _matching_op_dirs(ops_dir: Path, op_search_text: str) -> Sequence[str]:
  if not ops_dir.is_dir():
    return ()
  op_dirs = {child.name for child in ops_dir.iterdir() if child.is_dir()}
  qualified = set(_QUALIFIED_OP_RE.findall(op_search_text))
  bare = set(_OP_TOKEN_RE.findall(op_search_text))
  matched = (op_dirs & qualified) | ((op_dirs - _OP_NAME_STOPWORDS) & bare)
  return tuple(f"csrc/ops/{name}" for name in sorted(matched))


def infer_issue_weights(
    issue: Mapping[str, Any], repo_root: Path
) -> Mapping[str, float]:
  """Infers weighted repository paths from an issue's title, body, and labels.

  Returns an empty mapping when no specific area can be identified so callers
  leave ambiguous issues unassigned for manual triage.
  """
  title = issue.get("title") or ""
  body = issue.get("body") or ""
  labels = " ".join(
      label.get("name", "")
      for label in issue.get("labels") or ()
      if isinstance(label, Mapping)
  )
  text = f"{title} {body}"
  combined = f"{text} {labels}".lower()

  prod_paths: list[str] = []
  test_paths: list[str] = []
  for match in _REPO_PATH_RE.finditer(text):
    normalized = _normalize_repo_path(match.group(1))
    bucket = test_paths if normalized.startswith("tests/") else prod_paths
    if normalized not in bucket:
      bucket.append(normalized)

  weights: dict[str, float] = {
      path: _EXPLICIT_PATH_WEIGHT for path in prod_paths[:_MAX_ISSUE_PROD_PATHS]
  }
  for pattern, target_path in _SUBSYSTEM_RULES:
    if pattern.search(combined) and target_path not in weights:
      weights[target_path] = 1.0

  # Match csrc/ops/<op> against title and inline code spans to avoid false
  # hits on generic English words in issue prose.
  code_text = " ".join(_BACKTICK_CODE_RE.findall(body))
  op_search_text = f"{title} {code_text}".lower()
  for op_path in _matching_op_dirs(repo_root / "csrc" / "ops", op_search_text):
    weights.setdefault(op_path, 1.0)

  if not weights and _OP_FALLBACK_RE.search(combined):
    weights["csrc/ops"] = 1.0
  if not weights:
    for path in test_paths[:_MAX_ISSUE_TEST_PATHS]:
      weights[path] = 1.0
  return weights


def score_candidates(
    weights: Mapping[str, float],
    config: ActionConfig,
    author: str | None,
) -> Sequence[Candidate]:
  """Scores eligible domain experts from weighted repository paths."""
  excluded = {login.lower() for login in config.exclude_logins}
  if author:
    excluded.add(author.lower())

  scores: collections.Counter[str] = collections.Counter()
  evidence: dict[str, set[str]] = collections.defaultdict(set)
  cache: dict[Path, tuple[str, ...] | None] = {}

  for path, weight in weights.items():
    found = _find_experts_cached(config.repo_root, path, cache)
    if not found:
      continue
    pkg, experts = found
    norm = sum(1 / (idx + 1) for idx in range(len(experts)))
    for idx, handle in enumerate(experts):
      scores[handle] += weight * (1 / (idx + 1)) / norm
      evidence[handle].add(pkg)

  eligible = [
      (handle, score)
      for handle, score in scores.most_common()
      if handle.lower() not in excluded and not handle.endswith("[bot]")
  ]
  if not eligible:
    return ()
  cutoff = config.near_top * eligible[0][1] - 1e-12
  return tuple(
      Candidate(
          github=handle,
          score=score,
          packages=tuple(sorted(evidence[handle])),
      )
      for handle, score in eligible
      if score >= cutoff
  )


def count_open_loads(
    client: GitHubClient, logins: Set[str]
) -> Mapping[str, int]:
  """Counts open PR review/assignee and open issue workload per login."""
  lowered = {login.lower() for login in logins}
  loads = {login: 0 for login in logins}
  if not lowered:
    return loads

  pulls = client.get_paginated(
      f"/repos/{client.repo}/pulls?state=open&per_page=100"
  )
  issues = client.get_paginated(
      f"/repos/{client.repo}/issues?state=open&per_page=100"
  )

  def _participants(item: Mapping[str, Any], *keys: str) -> set[str]:
    users: set[str] = set()
    for key in keys:
      for entry in item.get(key) or ():
        if isinstance(entry, Mapping) and entry.get("login"):
          users.add(entry["login"].lower())
    return users

  pr_sets = [
      _participants(pr, "requested_reviewers", "assignees") for pr in pulls
  ]
  issue_sets = [
      _participants(issue, "assignees")
      for issue in issues
      if "pull_request" not in issue
  ]

  for login in logins:
    key = login.lower()
    loads[login] = sum(key in ps for ps in pr_sets) + sum(
        key in iset for iset in issue_sets
    )
  return loads


def rank_by_load(
    candidates: Sequence[Candidate], loads: Mapping[str, int]
) -> Sequence[Candidate]:
  """Sorts candidates by lowest open workload, breaking ties by higher score."""
  updated = [
      Candidate(
          github=c.github,
          score=c.score,
          packages=c.packages,
          open_load=loads.get(c.github, 0),
      )
      for c in candidates
  ]
  return tuple(sorted(updated, key=lambda c: (c.open_load, -c.score)))


def _pr_has_reviews(
    client: GitHubClient, pr_number: int, author_login: str
) -> bool:
  reviews = client.get_paginated(
      f"/repos/{client.repo}/pulls/{pr_number}/reviews?per_page=100"
  )
  for review in reviews:
    if review.get("state") == "PENDING":
      continue
    reviewer = ((review.get("user") or {}).get("login") or "").lower()
    if reviewer and reviewer != author_login.lower():
      return True
  return False


def _select_pr_candidates(
    client: GitHubClient, payload: Mapping[str, Any], config: ActionConfig
) -> tuple[int, Sequence[Candidate]] | None:
  action = payload.get("action")
  pr = payload.get("pull_request")
  if action not in _PR_ACTIONS or not isinstance(pr, Mapping):
    return None
  number = pr.get("number")
  if not number or pr.get("state") == "closed" or pr.get("draft"):
    return None

  user = pr.get("user") or {}
  author = user.get("login") or ""
  body = pr.get("body") or ""
  if author == "copybara-service[bot]" or _COPYBARA_CL_RE.search(body):
    return None
  if (
      pr.get("requested_reviewers")
      or pr.get("requested_teams")
      or pr.get("assignees")
  ):
    return None
  if _pr_has_reviews(client, number, author):
    return None

  is_dependabot = "dependabot" in author.lower()
  if not is_dependabot and _is_bot(user):
    return None

  excluded = {login.lower() for login in config.exclude_logins}
  if is_dependabot and _DEPENDABOT_REVIEWER.lower() not in excluded:
    return number, (
        Candidate(github=_DEPENDABOT_REVIEWER, score=1.0, packages=(".",)),
    )

  files = client.get_paginated(
      f"/repos/{client.repo}/pulls/{number}/files?per_page=100"
  )
  weights: collections.Counter[str] = collections.Counter()
  for file_info in files:
    filename = file_info.get("filename")
    if filename:
      additions = file_info.get("additions") or 0
      deletions = file_info.get("deletions") or 0
      weights[filename] += math.sqrt(1 + additions + deletions)
  return number, score_candidates(weights, config, author)


def _select_issue_candidates(
    payload: Mapping[str, Any], config: ActionConfig
) -> tuple[int, Sequence[Candidate]] | None:
  action = payload.get("action")
  issue = payload.get("issue")
  if action not in _ISSUE_ACTIONS or not isinstance(issue, Mapping):
    return None
  number = issue.get("number")
  if not number or issue.get("state") == "closed":
    return None

  user = issue.get("user")
  if _is_bot(user):
    return None
  if issue.get("assignees") or issue.get("assignee"):
    return None

  weights = infer_issue_weights(issue, config.repo_root)
  if not weights:
    return None
  author = (user or {}).get("login") or ""
  return number, score_candidates(weights, config, author)


def _apply_assignment(
    client: GitHubClient,
    is_pr: bool,
    number: int,
    candidates: Sequence[Candidate],
) -> str | None:
  endpoint = (
      f"/repos/{client.repo}/pulls/{number}/requested_reviewers"
      if is_pr
      else f"/repos/{client.repo}/issues/{number}/assignees"
  )
  field = "reviewers" if is_pr else "assignees"
  for candidate in candidates:
    try:
      client.post_json(endpoint, {field: [candidate.github]})
      return candidate.github
    except urllib.error.HTTPError as err:
      if err.code == 422:
        print(
            f"::warning::Skipping unassignable handle '{candidate.github}' on "
            f"{'PR' if is_pr else 'issue'} #{number} (HTTP 422).",
            file=sys.stderr,
        )
        continue
      raise
  return None


def process_event(
    event_name: str,
    payload: Mapping[str, Any],
    client: GitHubClient,
    config: ActionConfig,
) -> str | None:
  """Selects and optionally applies a reviewer or assignee for an event."""
  if event_name in ("pull_request", "pull_request_target"):
    is_pr = True
    selection = _select_pr_candidates(client, payload, config)
  elif event_name == "issues":
    is_pr = False
    selection = _select_issue_candidates(payload, config)
  else:
    return None

  if not selection:
    return None
  number, candidates = selection
  if not candidates:
    return None

  if len(candidates) > 1 or config.dry_run:
    loads = count_open_loads(client, {c.github for c in candidates})
    ranked = rank_by_load(candidates, loads)
  else:
    ranked = candidates

  if config.dry_run:
    chosen = ranked[0]
    print(
        json.dumps({
            "kind": "pr" if is_pr else "issue",
            "number": number,
            "chosen": chosen.github,
            "score": round(chosen.score, 2),
            "open_load": chosen.open_load,
            "packages": chosen.packages,
        })
    )
    return chosen.github

  assigned = _apply_assignment(client, is_pr, number, ranked)
  if assigned:
    kind = "reviewer" if is_pr else "assignee"
    print(f"Assigned {assigned} as {kind} on #{number}.")
  return assigned


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line options for the GitHub auto-assigner."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--repo-root",
      type=Path,
      default=Path.cwd(),
      help="Path to the repository checkout root.",
  )
  parser.add_argument(
      "--exclude-file",
      type=Path,
      default=None,
      help="Optional newline-delimited file of GitHub handles to exclude.",
  )
  parser.add_argument(
      "--exclude",
      nargs="*",
      default=(),
      help="Additional GitHub handles to exclude.",
  )
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Compute recommendation without mutating GitHub.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  """Reads GitHub Actions event inputs and applies auto-assignment."""
  args = _parse_args(argv)
  token = os.environ.get("GITHUB_TOKEN")  # PY_ENVIRON_OK=ci
  event_path = os.environ.get("GITHUB_EVENT_PATH")  # PY_ENVIRON_OK=ci
  repo = os.environ.get("GITHUB_REPOSITORY")  # PY_ENVIRON_OK=ci
  event_name = os.environ.get("GITHUB_EVENT_NAME")  # PY_ENVIRON_OK=ci
  api_url = os.environ.get(  # PY_ENVIRON_OK=ci
      "GITHUB_API_URL", "https://api.github.com"
  )

  missing = [
      name
      for name, val in (
          ("GITHUB_TOKEN", token),
          ("GITHUB_EVENT_PATH", event_path),
          ("GITHUB_REPOSITORY", repo),
          ("GITHUB_EVENT_NAME", event_name),
      )
      if not val
  ]
  if missing:
    print(
        f"Missing required environment variable(s): {', '.join(missing)}.",
        file=sys.stderr,
    )
    return 1

  assert token and event_path and repo and event_name

  if not args.repo_root.is_dir():
    print(
        f"Repository root directory does not exist: {args.repo_root}.",
        file=sys.stderr,
    )
    return 1

  excludes: set[str] = {handle.lower() for handle in args.exclude}
  if args.exclude_file is not None:
    excludes |= load_excludes(args.exclude_file)

  payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
  client = GitHubClient(api_url=api_url, repo=repo, token=token)
  config = ActionConfig(
      repo_root=args.repo_root,
      exclude_logins=frozenset(excludes),
      dry_run=args.dry_run,
  )
  process_event(event_name, payload, client, config)
  return 0


if __name__ == "__main__":
  sys.exit(main())
