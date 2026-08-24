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

"""Custom presubmit convention checker for TorchTPU OSS repository.

This tool validates coding conventions (such as banned Bazel rules or C++
patterns) that mirror presubmit rules in METADATA.

It supports incremental checking (`check_delta_only=True`) by parsing unified
diffs from either `git diff -U0` (in Git/GitHub Actions CI) or `hg diff -U0`
(in CitC/Fig workspaces), ensuring developers and PRs are only alerted to
violations on new or modified lines.
"""

import argparse
import collections
from collections.abc import Mapping, Sequence
import concurrent.futures
import dataclasses
import os
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
try:
  from torch_tpu.oss.ci.tools import custom_lint_rules
except ImportError:
  import custom_lint_rules  # pytype: disable=import-error

Rule = custom_lint_rules.Rule
RULES = custom_lint_rules.RULES


@dataclasses.dataclass(frozen=True)
class CandidateFile:
  """A candidate file paired with matching rules to avoid redundant evaluation."""

  file_path: pathlib.Path
  rel_path: str
  matching_rules: Sequence[Rule]


@dataclasses.dataclass(frozen=True)
class Violation:
  """Represents a rule violation in a specific file and line."""

  rule_id: str
  file_path: str
  line_number: int
  line_content: str
  message: str


def normalize_path(path: pathlib.Path, root_dir: pathlib.Path) -> str:
  """Returns path relative to root_dir as POSIX string, or str(path) if not relative."""
  try:
    return path.relative_to(root_dir).as_posix()
  except ValueError:
    pass
  try:
    return path.resolve().relative_to(root_dir.resolve()).as_posix()
  except ValueError:
    return path.as_posix()


_G3 = ''
_TP_PREFIX = ''


def find_vcs_root(
    start_path: pathlib.Path | None = None,
) -> pathlib.Path | None:
  """Finds the root directory of the VCS repository (git, hg, or internal depot)."""
  candidate = (start_path or pathlib.Path.cwd()).resolve()
  for p in [candidate] + list(candidate.parents):
    if (p / '.git').exists() or (p / '.hg').exists():
      return p
    if _G3 and p.name == _G3:
      return p
  return None


KNOWN_VCS_PREFIXES = (
    (
        f'//depot/{_G3}/{_TP_PREFIX}/',
        f'depot/{_G3}/{_TP_PREFIX}/',
        f'{_G3}/{_TP_PREFIX}/',
        f'{_TP_PREFIX}/',
    )
    if _TP_PREFIX
    else ()
)


def normalize_diff_path(
    path: str,
    root_dir: pathlib.Path | None = None,
) -> str:
  """Normalizes a VCS diff file path to be relative to torch_tpu root."""
  # Strip standard VCS diff prefixes ('a/' or 'b/')
  if path.startswith('b/') or path.startswith('a/'):
    path = path[2:]

  # 1. Strip known VCS / monorepo depot prefixes
  for prefix in KNOWN_VCS_PREFIXES:
    if path.startswith(prefix):
      path = path[len(prefix) :]
      break

  # 2. If root_dir is provided and path is absolute, normalize relative to root_dir
  if root_dir is not None:
    p = pathlib.Path(path)
    if p.is_absolute():
      try:
        return p.resolve().relative_to(root_dir.resolve()).as_posix()
      except ValueError:
        return path

    # 3. If relative, try resolving against VCS root, cwd, or root_dir
    vcs_root = find_vcs_root(root_dir)
    bases = (
        [vcs_root, pathlib.Path.cwd(), root_dir]
        if vcs_root
        else [pathlib.Path.cwd(), root_dir]
    )
    for base in bases:
      if base is not None:
        candidate = (base / path).resolve()
        try:
          return candidate.relative_to(root_dir.resolve()).as_posix()
        except ValueError:
          pass

  return path


def parse_unified_diff(
    diff_text: str,
    root_dir: pathlib.Path | None = None,
) -> Mapping[str, set[int]]:
  """Parses unified diff (-U0) and maps file relative path to modified line numbers."""
  modified_lines: dict[str, set[int]] = collections.defaultdict(set)
  current_file: str | None = None

  for line in diff_text.splitlines():
    if line.startswith('+++ '):
      # line[4:] skips the 4-character header prefix '+++ '
      header = line[4:].strip()
      # Unified diff headers append metadata (timestamp, file mode) after the file path,
      # separated by a tab (\t), 2+ spaces (\s{2,}), or space before ISO dates (\s+\d{4}-\d{2}-\d{2}).
      path_part = re.split(r'\t|\s{2,}|\s+\d{4}-\d{2}-\d{2}', header)[0].strip()
      # Normalize path (handles stripping 'a/' or 'b/' prefix and monorepo prefixes)
      normalized = normalize_diff_path(path_part, root_dir=root_dir)
      # Ignore deleted files pointing to /dev/null
      if normalized in ('/dev/null', 'dev/null', ''):
        current_file = None
      else:
        current_file = normalized
        modified_lines[current_file]
    elif line.startswith('@@ ') and current_file is not None:
      # Match unified diff hunk header: @@ -old_start,count +new_start,count @@
      # Group 1 captures the target file start line (+new_start).
      # Group 2 captures the target file line count (defaults to 1 if omitted).
      match = re.search(r'\+(\d+)(?:,(\d+))?', line)
      if match:
        start_line = int(match.group(1))
        count = int(match.group(2)) if match.group(2) is not None else 1
        for line_num in range(start_line, start_line + count):
          modified_lines[current_file].add(line_num)
  return modified_lines


def get_modified_lines_from_vcs(
    base_sha: str | None = None,
    root_dir: pathlib.Path | None = None,
) -> Mapping[str, set[int]] | None:
  """Returns mapping of modified line numbers via git/hg diff, or None if VCS diff fails."""
  root_dir = find_torch_tpu_root(root_dir)
  # First try git diff against base_sha using triple-dot to diff against merge-base
  git_cmd = ['git', 'diff', '-U0', '--no-color']
  if base_sha:
    clean_base = base_sha.split('..')[0]
    diff_target = (
        clean_base if clean_base.endswith('...') else f'{clean_base}...'
    )
    git_cmd.append(diff_target)
  try:
    result = subprocess.run(
        git_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
      return parse_unified_diff(result.stdout, root_dir=root_dir)
  except FileNotFoundError:
    pass

  # If git failed or is not available, try hg diff -U0 (CitC/Fig environments)
  try:
    result = subprocess.run(
        ['hg', 'diff', '-U0'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode == 0:
      return parse_unified_diff(result.stdout, root_dir=root_dir)
  except FileNotFoundError:
    pass

  return None


def get_commit_message(base_sha: str | None) -> str:
  """Retrieves current CL/commit messages to check for disable tags."""
  cmds: list[list[str]] = []
  if base_sha:
    clean_base = base_sha.rstrip('.')
    cmds.append(['git', 'log', f'{clean_base}..HEAD', '--format=%B'])
  cmds.extend([
      ['git', 'log', '-1', '--format=%B'],
      ['hg', 'log', '-l', '1', '--template', '{desc}'],
  ])

  for cmd in cmds:
    try:
      result = subprocess.run(cmd, capture_output=True, text=True, check=False)
      if result.returncode == 0 and result.stdout.strip():
        return result.stdout
    except FileNotFoundError:
      continue

  return ''


def is_rule_disabled(rule: Rule, commit_message: str) -> bool:
  """Returns True if any of the rule's disable_tags are present in the commit message."""
  return any(tag in commit_message for tag in rule.disable_tags)


DEFAULT_EXCLUDED_PATH_REGEX = (
    r'(^|/)(custom_lint\.py|custom_lint_rules\.py|custom_lint\.sh|custom_lint_test\.py|'
    r'custom_lint_presubmit\.py|custom_lint_presubmit_test\.py|copy\.bara\.sky)$|'
    r'.*\.(orig|bak)$|.*\.del\..*'
)
DEFAULT_EXCLUDED_PATH_RE = re.compile(DEFAULT_EXCLUDED_PATH_REGEX)


def is_path_excluded(rel_path: str) -> bool:
  """Returns True if the file path is globally excluded from custom lint checks."""
  if DEFAULT_EXCLUDED_PATH_RE.search(rel_path):
    return True
  if rel_path.startswith('examples/') or '/examples/' in rel_path:
    return True
  return False


def check_file_content(
    content: str,
    lines: Sequence[str],
    rel_path: str,
    rule: Rule,
    modified_lines: Mapping[str, set[int]] | None,
) -> Sequence[Violation]:
  """Checks file content against a rule and returns any violations."""
  if is_path_excluded(rel_path):
    return []
  if not rule.matches_path(rel_path):
    return []

  if modified_lines is None and getattr(rule, 'operations', ()):
    return []

  if modified_lines is not None and rule.check_delta_only:
    if rel_path not in modified_lines:
      return []

  # Fast path: if prohibited pattern doesn't occur anywhere in content, exit early.
  if not re.search(rule.prohibited_regex, content):
    return []

  violations: list[Violation] = []
  delta_lines = (
      modified_lines.get(rel_path) if modified_lines is not None else None
  )

  for line_idx, line in enumerate(lines, start=1):
    if rule.check_delta_only and delta_lines is not None:
      if line_idx not in delta_lines:
        continue

    match = re.search(rule.prohibited_regex, line)
    window = None
    if not match and line_idx < len(lines):
      window = f'{line}\n{lines[line_idx]}'
      m = re.search(rule.prohibited_regex, window)
      if m and m.start() <= len(line):
        match = m

    if match:
      if rule.suppression_regex:
        if re.search(rule.suppression_regex, line):
          continue
        if window is None and line_idx < len(lines):
          window = f'{line}\n{lines[line_idx]}'
        if window and re.search(rule.suppression_regex, window):
          continue

      violations.append(
          Violation(
              rule_id=rule.id,
              file_path=rel_path,
              line_number=line_idx,
              line_content=line.strip(),
              message=rule.failure_message,
          )
      )
  return violations


def find_candidate_files(
    root_dir: pathlib.Path, rules: Sequence[Rule]
) -> Sequence[CandidateFile]:
  """Finds all files under root_dir that match at least one rule's path_regex."""
  candidates: list[CandidateFile] = []
  ignore_dirs = {
      '.git',
      '.hg',
      '.nox',
      'bazel-bin',
      'bazel-out',
      'bazel-testlogs',
      'blaze-bin',
      'blaze-out',
      'blaze-testlogs',
      'blaze-genfiles',
      'venv',
      '.venv',
  }

  for dirpath, dirnames, filenames in os.walk(root_dir):
    dirnames[:] = [
        d
        for d in dirnames
        if d not in ignore_dirs
        and not d.startswith('.')
        and not d.startswith('bazel-')
        and not d.startswith('blaze-')
    ]

    for filename in filenames:
      full_path = pathlib.Path(dirpath) / filename
      rel_path = normalize_path(full_path, root_dir)
      if is_path_excluded(rel_path):
        continue
      matching = [r for r in rules if r.matches_path(rel_path)]
      if matching:
        candidates.append(
            CandidateFile(
                file_path=full_path,
                rel_path=rel_path,
                matching_rules=matching,
            )
        )
  return candidates


def find_torch_tpu_root(
    start_path: pathlib.Path | None = None,
) -> pathlib.Path:
  """Finds the root directory of the torch_tpu package/repository."""

  def is_torch_tpu_root(p: pathlib.Path) -> bool:
    if p.name == 'torch_tpu' and (p / 'METADATA').exists():
      return True
    if (p / 'pyproject.toml').exists() and (p / 'oss').exists():
      return True
    if (p / 'pyproject.toml').exists():
      return True
    return False

  # 1. First try start_path if explicitly provided
  if start_path:
    resolved_start = start_path.resolve()
    if _TP_PREFIX:
      torch_tpu_sub = resolved_start / _TP_PREFIX
      if torch_tpu_sub.exists() and is_torch_tpu_root(torch_tpu_sub):
        return torch_tpu_sub
    for p in [resolved_start] + list(resolved_start.parents):
      if is_torch_tpu_root(p):
        return p

  # 2. Next try relative to __file__ location
  current = pathlib.Path(__file__).resolve()
  for parent in current.parents:
    if is_torch_tpu_root(parent):
      return parent

  # 3. Fall back to current working directory
  candidate = (start_path or pathlib.Path.cwd()).resolve()
  for p in [candidate] + list(candidate.parents):
    if is_torch_tpu_root(p):
      return p

  return candidate


def _check_candidate(
    task: tuple[CandidateFile, Mapping[str, set[int]] | None],
) -> list[Violation]:
  """Worker function to evaluate all matching rules on a single CandidateFile."""
  candidate, modified_lines = task
  if not candidate.file_path.is_file():
    return []
  try:
    content = candidate.file_path.read_text(encoding='utf-8')
  except (OSError, UnicodeDecodeError):
    return []
  lines = content.splitlines()
  violations: list[Violation] = []
  for rule in candidate.matching_rules:
    violations.extend(
        check_file_content(
            content, lines, candidate.rel_path, rule, modified_lines
        )
    )
  return violations


def format_summary_report(
    violations: Sequence[Violation],
    files_checked: int,
) -> str:
  """Formats violations into a grouped summary report matching custom_linter_report.txt."""
  by_rule: dict[str, list[Violation]] = collections.defaultdict(list)
  for v in violations:
    by_rule[v.rule_id].append(v)

  sorted_groups = sorted(
      by_rule.values(), key=lambda g: (-len(g), g[0].rule_id)
  )
  out_lines = [
      '=' * 60,
      f' SUMMARY: {len(violations)} errors, 0 warnings',
      (
          f' Files checked: {files_checked}   Findings: {len(violations)}  '
          f' Rules with findings: {len(sorted_groups)}'
      ),
      '=' * 60,
      '',
      '   FINDINGS  RULE',
  ]
  for g in sorted_groups:
    out_lines.append(f'  {len(g):9d}  {g[0].rule_id}')
  out_lines.append('')

  for g in sorted_groups:
    by_path: dict[str, list[Violation]] = collections.defaultdict(list)
    for v in g:
      by_path[v.file_path].append(v)
    out_lines.extend([
        '#' * 60,
        f'# {g[0].rule_id}  [ERROR]  {len(g)} findings in {len(by_path)} files',
        '#' * 60,
        'Message:',
    ])
    for msg_line in g[0].message.split('\n'):
      out_lines.append(f'  {msg_line}')
    out_lines.append('')
    for path, path_violations in sorted(by_path.items()):
      out_lines.append(f'  {path}  ({len(path_violations)})')
      for pv in path_violations:
        out_lines.append(f'      {pv.file_path}:{pv.line_number}')
    out_lines.append('')

  return '\n'.join(out_lines) + '\n'


def run_checks(
    root_dir: pathlib.Path,
    files: Sequence[str] | None = None,
    base_sha: str | None = None,
    check_all: bool = False,
    jobs: int | None = None,
    report_file: str | None = None,
) -> int:
  """Runs convention checks and prints violations."""
  root_dir = find_torch_tpu_root(root_dir)
  rules = RULES
  commit_message = '' if (check_all and files) else get_commit_message(base_sha)
  active_rules = [r for r in rules if not is_rule_disabled(r, commit_message)]

  if not active_rules:
    print(
        'INFO: All custom convention rules disabled by CL/commit description.'
    )
    return 0

  modified_lines = (
      None
      if check_all
      else get_modified_lines_from_vcs(base_sha, root_dir=root_dir)
  )

  t0 = time.perf_counter()
  if files:
    candidate_files: list[CandidateFile] = []
    for f in files:
      p = pathlib.Path(f)
      if not p.is_absolute():
        cwd_path = pathlib.Path.cwd() / p
        root_path = root_dir / p
        p = (
            root_path
            if not cwd_path.exists() and root_path.exists()
            else cwd_path
        )
      p = p.resolve()
      if p.is_dir():
        candidate_files.extend(find_candidate_files(p, active_rules))
      elif p.is_file():
        rel_path = normalize_path(p, root_dir)
        if is_path_excluded(rel_path):
          continue
        matching = [r for r in active_rules if r.matches_path(rel_path)]
        if matching:
          candidate_files.append(
              CandidateFile(
                  file_path=p,
                  rel_path=rel_path,
                  matching_rules=matching,
              )
          )
  elif modified_lines is not None and all(
      r.check_delta_only for r in active_rules
  ):
    candidate_files = []
    for rel_path in sorted(modified_lines):
      if is_path_excluded(rel_path):
        continue
      full_path = root_dir / rel_path
      if not full_path.is_file():
        continue
      matching = [r for r in active_rules if r.matches_path(rel_path)]
      if matching:
        candidate_files.append(
            CandidateFile(
                file_path=full_path,
                rel_path=rel_path,
                matching_rules=matching,
            )
        )
  else:
    candidate_files = list(find_candidate_files(root_dir, active_rules))
  t_walk = time.perf_counter() - t0

  detected_cpus = os.cpu_count() or 1
  effective_jobs = (
      jobs
      if jobs is not None
      else (detected_cpus if len(candidate_files) > 32 else 1)
  )

  t_eval_start = time.perf_counter()
  all_violations: list[Violation] = []
  if effective_jobs > 1 and len(candidate_files) > 1:
    tasks = [(c, modified_lines) for c in candidate_files]
    chunksize = max(1, len(tasks) // (effective_jobs * 4))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=effective_jobs
    ) as executor:
      for file_violations in executor.map(
          _check_candidate, tasks, chunksize=chunksize
      ):
        all_violations.extend(file_violations)
  else:
    for candidate in candidate_files:
      all_violations.extend(_check_candidate((candidate, modified_lines)))
  t_eval = time.perf_counter() - t_eval_start
  t_total = time.perf_counter() - t0

  timing_summary = (
      f'INFO: Checked {len(candidate_files)} files across {len(active_rules)}'
      f' rules using {effective_jobs}/{detected_cpus} CPU worker(s) in'
      f' {t_total:.3f}s (walk={t_walk:.3f}s, eval={t_eval:.3f}s)'
  )

  if report_file:
    report_path = pathlib.Path(report_file)
    report_path.write_text(
        format_summary_report(all_violations, len(candidate_files)),
        encoding='utf-8',
    )
    print(f'Report written to {report_path} (view with: less {report_path})')

  if not all_violations:
    print(timing_summary)
    print('INFO: Custom TorchTPU convention checks passed successfully.')
    return 0

  print('=' * 65)
  print(
      f'ERROR: {len(all_violations)} custom convention check violation(s)'
      ' found:'
  )
  print('=' * 65)
  for v in all_violations:
    print(f'[{v.rule_id}] {v.file_path}:{v.line_number}')
    print(f'  >   {v.line_content}')
    print(f'  {v.message}')
    print('-' * 65)
  print(f'Total violations: {len(all_violations)}')
  print(timing_summary)
  print('=' * 65)
  return 1


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description='Custom TorchTPU OSS presubmit convention checker.'
  )
  parser.add_argument(
      'files',
      nargs='*',
      help='Optional specific files or directories to check.',
  )
  parser.add_argument(
      '--base-sha',
      default='origin/main',
      help='Base git SHA or tracking branch for incremental diffs.',
  )
  parser.add_argument(
      '--all',
      action='store_true',
      dest='check_all',
      help='Check all lines regardless of git/hg diffs.',
  )
  parser.add_argument(
      '-j',
      '--jobs',
      type=int,
      default=None,
      help=(
          'Number of parallel worker processes (default: auto-scales to CPU'
          ' count when checking >32 files).'
      ),
  )
  parser.add_argument(
      '-o',
      '--report-file',
      default=None,
      help=(
          'Optional path to write a grouped summary report (e.g.'
          ' /tmp/custom_linter_report.txt).'
      ),
  )
  args = parser.parse_args(argv)

  working_dir = os.environ.get('BUILD_WORKING_DIRECTORY')
  root_dir = pathlib.Path(working_dir) if working_dir else pathlib.Path.cwd()
  return run_checks(
      root_dir=root_dir,
      files=args.files,
      base_sha=args.base_sha,
      check_all=args.check_all,
      jobs=args.jobs,
      report_file=args.report_file,
  )


if __name__ == '__main__':
  sys.exit(main())
