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

"""Shared helpers and constants for TorchTPU quality-champion PR workflows."""

from collections.abc import Mapping, Sequence
import json
import pathlib
from typing import Any
import urllib.request

STATUS_CONTEXT = "assignee-is-quality-champion"
DEFAULT_CHAMPIONS_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / ".github"
    / "quality_champions.txt"
)


def load_champions(
    path: pathlib.Path = DEFAULT_CHAMPIONS_PATH,
) -> Sequence[str]:
  """Loads GitHub logins from the quality champions config file.

  Args:
    path: Path to `.github/quality_champions.txt`.

  Returns:
    Tuple of unique GitHub logins in file order, preserving original casing.
  """
  champions: list[str] = []
  seen: set[str] = set()
  for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.split("#", 1)[0].strip()
    if line and line.lower() not in seen:
      seen.add(line.lower())
      champions.append(line)
  return tuple(champions)


def github_request(
    url: str,
    token: str,
    method: str = "GET",
    data: Mapping[str, Any] | None = None,
    *,
    user_agent: str = "torch-tpu-pr-automation",
) -> Any:
  """Executes a GitHub REST API request and returns parsed JSON if present.

  Args:
    url: Full GitHub REST API URL.
    token: GitHub API bearer token.
    method: HTTP method (`GET`, `POST`, `DELETE`, etc.).
    data: Optional JSON request body.
    user_agent: Value for the `User-Agent` header.

  Returns:
    Parsed JSON response body, or `None` when the response body is empty.
  """
  body = json.dumps(data).encode("utf-8") if data is not None else None
  headers = {
      "Accept": "application/vnd.github+json",
      "Authorization": f"Bearer {token}",
      "User-Agent": user_agent,
      "X-GitHub-Api-Version": "2022-11-28",
  }
  if data is not None:
    headers["Content-Type"] = "application/json"
  req = urllib.request.Request(url, data=body, method=method, headers=headers)
  with urllib.request.urlopen(req, timeout=30) as resp:
    raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else None
