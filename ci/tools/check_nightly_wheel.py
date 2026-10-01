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

"""Verifies that uv resolves torch_tpu to a fresh nightly wheel."""

import argparse
from collections.abc import Sequence
import datetime
import re
import subprocess
import sys
from typing import Final
import urllib.parse

DEFAULT_INDEX_URL: Final[str] = (
    "https://us-python.pkg.dev/ml-oss-artifacts-transient/"
    "torch-tpu-virtual-registry/simple/"
)
DEFAULT_PYTHON_VERSION: Final[str] = "3.12"
DEFAULT_PYTHON_PLATFORM: Final[str] = "x86_64-manylinux_2_31"
DEFAULT_MAX_AGE_HOURS: Final[float] = 8.0

_RESOLVED_PACKAGE_RE: Final[re.Pattern[str]] = re.compile(
    r"^torch[-_]tpu==(?P<version>\S+)$", re.IGNORECASE
)
_NIGHTLY_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^\d+\.\d+\.\d+\.dev(?P<timestamp>\d{14})$"
)


def get_gcloud_access_token() -> str:
  """Fetches an OAuth2 access token from gcloud."""
  proc = subprocess.run(
      ["gcloud", "auth", "print-access-token"],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    stderr = proc.stderr.strip()
    raise RuntimeError(
        "Failed to obtain access token via `gcloud auth print-access-token`"
        f" (exit code {proc.returncode}): {stderr}"
    )
  token = proc.stdout.strip()
  if not token:
    raise RuntimeError("`gcloud auth print-access-token` returned empty token.")
  return token


def build_authenticated_index_url(index_url: str, token: str | None) -> str:
  """Injects oauth2accesstoken credentials into index_url if not present."""
  if not token:
    return index_url
  parsed = urllib.parse.urlsplit(index_url)
  if parsed.username or "@" in parsed.netloc:
    return index_url
  netloc = f"oauth2accesstoken:{token}@{parsed.netloc}"
  return urllib.parse.urlunsplit(
      (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
  )


def _redact_url(url: str) -> str:
  """Redacts credentials from a URL for safe logging."""
  return re.sub(r"(://[^:/@]+:)[^@]+(@)", r"\1***\2", url)


def resolve_latest_wheel_version(
    index_url: str,
    python_version: str = DEFAULT_PYTHON_VERSION,
    python_platform: str = DEFAULT_PYTHON_PLATFORM,
) -> str:
  """Resolves the latest torch_tpu version using `uv pip compile`."""
  cmd = [
      "uv",
      "pip",
      "compile",
      "-",
      "--no-config",
      "--no-deps",
      "--refresh",
      "--prerelease=allow",
      "--python-version",
      python_version,
      "--python-platform",
      python_platform,
      "--index-url",
      index_url,
  ]
  proc = subprocess.run(
      cmd,
      input="torch_tpu\n",
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    stderr = _redact_url(proc.stderr.strip())
    raise RuntimeError(
        f"`uv pip compile` failed for Python {python_version} (exit code"
        f" {proc.returncode}):\n{stderr}"
    )

  for raw_line in proc.stdout.splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    match = _RESOLVED_PACKAGE_RE.match(line)
    if match:
      return match.group("version")

  stdout = _redact_url(proc.stdout.strip())
  raise RuntimeError(
      "Could not find resolved `torch-tpu==<version>` line in `uv pip compile`"
      f" output:\n{stdout}"
  )


def parse_nightly_version_timestamp(version: str) -> datetime.datetime:
  """Extracts the UTC build timestamp from a `.dev<YYYYMMDDHHMMSS>` version."""
  match = _NIGHTLY_VERSION_RE.match(version)
  if not match:
    raise ValueError(
        f"Resolved version '{version}' is not a nightly dev wheel matching"
        " '<major>.<minor>.<patch>.dev<YYYYMMDDHHMMSS>'. A non-nightly or"
        " malformed wheel in the registry may be taking precedence."
    )
  timestamp_str = match.group("timestamp")
  dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
  return dt.replace(tzinfo=datetime.timezone.utc)


def compute_freshness_cutoff(
    now_utc: datetime.datetime,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> datetime.datetime:
  """Returns the minimum acceptable wheel timestamp in UTC.

  Nightly wheels are stamped in UTC via `date -u +%Y%m%d%H%M%S` in
  `ci/build_wheel.sh` and `ci/build_wheels.sh`, and built every 5 hours in UTC
  (`0 0 */5 * * *`). A nightly wheel is considered fresh if it was built since
  midnight (00:00 UTC) of the current day OR within the last `max_age_hours`
  hours, whichever cutoff is earlier.

  Args:
    now_utc: Current timezone-aware UTC datetime.
    max_age_hours: Maximum age window in hours compared against midnight UTC.

  Returns:
    The earliest acceptable UTC datetime for the wheel build timestamp.
  """
  midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
  age_cutoff_utc = now_utc - datetime.timedelta(hours=max_age_hours)
  return min(midnight_utc, age_cutoff_utc)


def check_wheel_freshness(
    version: str,
    now_utc: datetime.datetime | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> str:
  """Validates that a resolved wheel version is a fresh nightly build.

  Args:
    version: The resolved `torch_tpu` version string.
    now_utc: Current timezone-aware UTC datetime (defaults to current UTC time).
    max_age_hours: Maximum age window in hours compared against midnight UTC.

  Returns:
    A human-readable status message describing the fresh wheel.

  Raises:
    ValueError: If `version` is not a `.dev<YYYYMMDDHHMMSS>` nightly version or
      if its build timestamp is older than the freshness cutoff.
  """
  if now_utc is None:
    now_utc = datetime.datetime.now(datetime.timezone.utc)

  wheel_time = parse_nightly_version_timestamp(version)
  cutoff = compute_freshness_cutoff(now_utc, max_age_hours=max_age_hours)
  age = now_utc - wheel_time
  age_hours = age.total_seconds() / 3600.0

  if wheel_time < cutoff:
    raise ValueError(
        f"Resolved nightly wheel 'torch-tpu=={version}' is stale: built at"
        f" {wheel_time.isoformat()} ({age_hours:.1f}h ago; current time"
        f" {now_utc.isoformat()}), which is older than the cutoff"
        f" {cutoff.isoformat()} (earlier of 00:00 UTC today or"
        f" {max_age_hours:g}h ago)."
    )

  return (
      f"Resolved fresh nightly wheel 'torch-tpu=={version}' built at"
      f" {wheel_time.isoformat()} ({age_hours:.1f}h ago; cutoff"
      f" {cutoff.isoformat()})."
  )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "Verify that uv resolves torch_tpu from Artifact Registry to a fresh"
          " nightly wheel."
      )
  )
  parser.add_argument(
      "--index-url",
      default=DEFAULT_INDEX_URL,
      help="Python package index URL to query.",
  )
  parser.add_argument(
      "--python-version",
      default=DEFAULT_PYTHON_VERSION,
      help="Target Python version passed to `uv pip compile`.",
  )
  parser.add_argument(
      "--python-platform",
      default=DEFAULT_PYTHON_PLATFORM,
      help="Target Python platform passed to `uv pip compile`.",
  )
  parser.add_argument(
      "--max-age-hours",
      type=float,
      default=DEFAULT_MAX_AGE_HOURS,
      help=(
          "Maximum age in hours (compared with 00:00 UTC today; the earlier of"
          " the two is used as the cutoff)."
      ),
  )
  parser.add_argument(
      "--no-gcloud-auth",
      action="store_true",
      help="Do not invoke `gcloud auth print-access-token` for the index URL.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = parse_args(argv)

  try:
    token = None if args.no_gcloud_auth else get_gcloud_access_token()
    auth_index_url = build_authenticated_index_url(args.index_url, token)
    version = resolve_latest_wheel_version(
        index_url=auth_index_url,
        python_version=args.python_version,
        python_platform=args.python_platform,
    )
    message = check_wheel_freshness(
        version=version,
        max_age_hours=args.max_age_hours,
    )
  except (RuntimeError, ValueError) as exc:
    print(f"::error::{exc}", file=sys.stderr)
    return 1

  print(message)
  return 0


if __name__ == "__main__":
  sys.exit(main())
