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

"""Validates the names and directories of the agent skills in .agents/skills/.

Every skill lives in `.agents/skills/<name>/SKILL.md`, where `<name>` is the
`name` in the SKILL.md frontmatter. The name rules are in
tests/agent_skill_names.py; in addition, the directory must equal the name, as
the Agent Skills spec (https://agentskills.io) requires.

Run from the repository root: `python3 -m ci.tools.check_agent_skills`.
"""

import argparse
import pathlib
import sys

from tests import agent_skill_names


def skill_errors(skill_dir: pathlib.Path) -> list[str]:
  """Returns the naming errors of the skill in `skill_dir`."""
  skill_md = skill_dir / "SKILL.md"
  if not skill_md.is_file():
    return [f"{skill_dir} has no SKILL.md."]
  name = agent_skill_names.frontmatter_name(skill_md)
  if name is None:
    return [f"{skill_md} has no `name` in its frontmatter."]
  errors = [
      f"{skill_md}: {error}" for error in agent_skill_names.name_errors(name)
  ]
  if skill_dir.name != name:
    errors.append(
        f"Skill directory {skill_dir} must be named after the skill:"
        f" .agents/skills/{name}/."
    )
  return errors


def validate_agent_skills(repo_root: pathlib.Path) -> bool:
  """Validates every skill under `repo_root`/.agents/skills/."""
  skills_dir = repo_root / ".agents" / "skills"
  if not skills_dir.is_dir():
    return True
  errors = []
  for skill_dir in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
    # Git does not track empty directories; skip leftovers from moves.
    if any(skill_dir.iterdir()):
      errors.extend(skill_errors(skill_dir))
  for error in errors:
    print(f"::error::{error}", file=sys.stderr)
  if errors:
    print(
        "See tests/agent_skill_names.py and ci/tools/check_agent_skills.py for"
        " the skill naming convention.",
        file=sys.stderr,
    )
  return not errors


def _default_repo_root() -> pathlib.Path:
  script_root = pathlib.Path(__file__).resolve().parents[2]
  if (script_root / "ci").is_dir():
    return script_root
  return pathlib.Path.cwd()


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Validate agent skill names and directories."
  )
  parser.add_argument(
      "--repo-root",
      type=pathlib.Path,
      default=_default_repo_root(),
      help="Path to the repository root directory.",
  )
  args = parser.parse_args(argv)

  if not validate_agent_skills(args.repo_root):
    return 1

  print("Agent skill names validated successfully.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
