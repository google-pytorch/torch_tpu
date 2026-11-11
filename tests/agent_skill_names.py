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

"""Naming rules for TorchTPU agent skills, shared by the checks that enforce them.

A skill's name is the `name` in the YAML frontmatter of its SKILL.md and has
the form `tt-<verb-phrase>`:

*   `tt-` prefix: agents load skills from many places into one namespace, and a
    skill with the same name from elsewhere silently shadows ours.
*   A verb phrase naming the task, starting with an imperative verb (e.g.
    `tt-write-docs`, not `tt-docs-review` or `tt-writing-docs`). Only the
    common "-ing" mistake is checked.
*   Lowercase words joined by hyphens: the Agent Skills spec
    (https://agentskills.io) allows only `a-z`, `0-9` and `-` in names.

This module uses only the Python standard library, so CI scripts can use it
without installing dependencies.
"""

from collections.abc import Sequence
import pathlib
import re

_NAME_RE = re.compile(r"tt-[a-z0-9]+(?:-[a-z0-9]+)*")
_FRONTMATTER_NAME_RE = re.compile(r"^name:\s*(\S+)\s*$", re.MULTILINE)


def frontmatter_name(skill_md: pathlib.Path) -> str | None:
  """Returns the `name` in the YAML frontmatter of `skill_md`, if any."""
  text = skill_md.read_text(encoding="utf-8")
  if not text.startswith("---"):
    return None
  parts = text.split("---", 2)
  if len(parts) < 3:
    return None
  match = _FRONTMATTER_NAME_RE.search(parts[1])
  return match.group(1) if match else None


def name_errors(name: str) -> Sequence[str]:
  """Returns the problems with skill name `name`; empty if it is valid."""
  if not _NAME_RE.fullmatch(name):
    return [
        f"Skill name {name!r} must be `tt-` followed by lowercase words"
        " joined by hyphens."
    ]
  if name.split("-")[1].endswith("ing"):
    return [
        f"Skill name {name!r} must start with an imperative verb after `tt-`"
        " (e.g. `tt-write-docs`, not `tt-writing-docs`)."
    ]
  return []
