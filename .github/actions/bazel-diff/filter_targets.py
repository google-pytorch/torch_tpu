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

"""Filters a list of Bazel targets based on .bazelrc tags and Bazel query."""

import os
import pathlib
import re
import subprocess

from absl import app
from absl import logging


def get_test_tag_filters(bazelrc_path: pathlib.Path, bazel_config: str) -> str:
  """Extracts the --test_tag_filters for a specific config from .bazelrc."""
  if not bazelrc_path.exists():
    return ""

  pattern = re.compile(rf"^test:{bazel_config}\s+--test_tag_filters=(.+)$")
  with open(bazelrc_path, "r") as f:
    for line in f:
      match = pattern.match(line.strip())
      if match:
        return match.group(1).strip()
  return ""


def build_bazel_query(impacted_targets: list[str], tag_filters: str) -> str:
  """Constructs the Bazel query string based on positive and negative tags."""
  if not tag_filters:
    targets_str = " ".join(impacted_targets)
    return f"tests(set({targets_str}))"

  pos_queries = []
  neg_queries = []

  for tag in tag_filters.split(","):
    tag = tag.strip()
    if not tag:
      continue

    if tag.startswith("-"):
      clean_tag = tag[1:]
      neg_queries.append(f"attr('tags', '[\\[ ]{clean_tag}[,\\]]', $impacted)")
    else:
      pos_queries.append(f"attr('tags', '[\\[ ]{tag}[,\\]]', $impacted)")

  inner_query = " union ".join(pos_queries)
  if inner_query:
    inner_query = f"({inner_query})"
  else:
    inner_query = "$impacted"

  for nq in neg_queries:
    inner_query += f" except {nq}"

  targets_str = " ".join(impacted_targets)
  return f"let impacted = tests(set({targets_str})) in {inner_query}"


def main(_):
  workspace_dir_str = os.environ.get("TORCH_TPU_WORKSPACE_DIR")
  bazel_config = os.environ.get("TORCH_TPU_BAZEL_CONFIG")

  if not workspace_dir_str or not bazel_config:
    raise app.UsageError(
        "Both TORCH_TPU_WORKSPACE_DIR and TORCH_TPU_BAZEL_CONFIG environment"
        " variables must be set."
    )

  workspace_dir = pathlib.Path(workspace_dir_str)

  impacted_file = workspace_dir / "impacted_targets.txt"
  bazelrc_file = workspace_dir / ".bazelrc"

  if not impacted_file.exists() or impacted_file.stat().st_size == 0:
    logging.info("No impacted targets found.")
    return

  with open(impacted_file, "r") as f:
    impacted_targets = f.read().split()

  if not impacted_targets:
    return

  tag_filters = get_test_tag_filters(bazelrc_file, bazel_config)
  logging.info("Sanitized TEST_TAG_FILTERS: '%s'", tag_filters)

  query = build_bazel_query(impacted_targets, tag_filters)
  logging.info("Full Executable Bazel Query:\n%s", query)

  logging.info("Executing query to finalize targets...")
  try:
    result = subprocess.run(
        ["bazel", "query", "--keep_going", query],
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    filtered_targets = result.stdout.strip()
  except subprocess.CalledProcessError as e:
    logging.error(
        "Bazel query failed (this might be expected if targets were"
        " dropped): %s",
        e.stderr,
    )
    filtered_targets = e.stdout.strip()

  with open(impacted_file, "w") as f:
    if filtered_targets:
      f.write(filtered_targets + "\n")


if __name__ == "__main__":
  app.run(main)
