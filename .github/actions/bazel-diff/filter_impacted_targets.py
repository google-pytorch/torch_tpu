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

"""This script uses bazel query to filter the bazel-diff target list against the Bazel tag filters."""

import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

from absl import app
from absl import logging


def parse_tag_filters(tag_filters_line):
  """Parses a Bazel --test_tag_filters string into required and excluded sets."""
  if not tag_filters_line:
    return set(), set()

  tags = tag_filters_line.split("--test_tag_filters=", 1)[1].strip().split(",")
  required_tags = {t for t in tags if not t.startswith("-")}
  excluded_tags = {t[1:] for t in tags if t.startswith("-")}
  return required_tags, excluded_tags


def get_bazel_tag_filters(bazel_config, extra_flags):
  """Queries Bazel for the active tag filters on the given config."""
  cmd = [
      "bazel",
      "canonicalize-flags",
      "--for_command=test",
      f"--config={bazel_config}",
  ]
  if extra_flags:
    cmd.extend(shlex.split(extra_flags))

  result = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if result.returncode != 0:
    logging.error("Failed to read bazel flags:\n%s", result.stderr)
    sys.exit(1)

  tag_filters_line = next(
      (
          line
          for line in result.stdout.split("\n")
          if "--test_tag_filters=" in line
      ),
      None,
  )
  return parse_tag_filters(tag_filters_line)


def get_targets_metadata_xml(targets, query_file):
  """Uses bazel query to fetch the XML metadata for a list of targets."""
  with open(query_file, "w") as f:
    f.write(f"set({' '.join(targets)})")

  query_cmd = ["bazel", "query", f"--query_file={query_file}", "--output=xml"]
  query_result = subprocess.run(
      query_cmd, capture_output=True, text=True, check=False
  )

  if query_result.returncode != 0:
    logging.warning("bazel query returned non-zero:\n%s", query_result.stderr)

  return query_result.stdout


def filter_targets(xml_content, required_tags, excluded_tags):
  """Filters a Bazel query XML output based on tag sets."""
  root = ET.fromstring(xml_content)
  filtered_targets = []

  for rule in root.findall("rule"):
    rule_name = rule.get("name")
    rule_class = rule.get("class")

    if not rule_name:
      continue
    if not rule_class or not rule_class.endswith("_test"):
      filtered_targets.append(rule_name)
      continue

    list_node = rule.find("list[@name='tags']")
    tags = set()
    if list_node is not None:
      for string_node in list_node.findall("string"):
        tags.add(string_node.get("value"))

    # Target must have ALL required tags.
    if required_tags and not required_tags.issubset(tags):
      continue
    # Target must NOT have ANY excluded tags.
    if excluded_tags.intersection(tags):
      continue

    filtered_targets.append(rule_name)

  return filtered_targets


def main(argv):
  if len(argv) < 3:
    logging.error(
        "Usage: filter_impacted_targets.py <impacted_targets.txt> "
        "<bazel_config> [extra_flags]"
    )
    sys.exit(1)

  impacted_file = argv[1]
  bazel_config = argv[2]
  extra_flags = argv[3] if len(argv) > 3 else ""

  # 1. Determines which tags to filter.
  required_tags, excluded_tags = get_bazel_tag_filters(
      bazel_config, extra_flags
  )

  if not required_tags and not excluded_tags:
    logging.info(
        "No --test_tag_filters found for this config. No filtering needed."
    )
    return

  logging.info(
      "Applying tag filters -> Required: %s, Excluded: %s",
      required_tags,
      excluded_tags,
  )

  # 2. Reads the initial targets from bazel-diff output.
  with open(impacted_file, "r") as f:
    targets = [line.strip() for line in f if line.strip()]

  if not targets:
    return

  # 3. Fetches metadata and filters.
  query_file = impacted_file + ".query"
  xml_content = get_targets_metadata_xml(targets, query_file)
  filtered_targets = filter_targets(xml_content, required_tags, excluded_tags)

  # 4. Overwrites targets file and cleans up query file.
  with open(impacted_file, "w") as f:
    for t in filtered_targets:
      f.write(t + "\n")

  if os.path.exists(query_file):
    os.remove(query_file)

  logging.info(
      "Filtered %d targets down to %d using Bazel tag filters.",
      len(targets),
      len(filtered_targets),
  )


if __name__ == "__main__":
  app.run(main)
