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

"""Translates Bazel test targets to Python test files and runs pytest."""

import argparse
from collections.abc import Iterable, Sequence
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Final

BASE_TAG_EXCLUDES: Final[tuple[str, ...]] = (
    "manual",
    "notest",
    "nobuild",
    "nopresubmit",
    "wheel_test_excluded",
)

CPU_TAG_EXCLUDES: Final[tuple[str, ...]] = (
    "requires-tpu.*",
    "requires-gpu.*",
    "requires-viper.*",
    "requires-ghost.*",
    "cuda-build-test",
)

TPU_TAG_EXCLUDES: Final[tuple[str, ...]] = ("requires-gpu.*",)

# Constructed dynamically to avoid Copybara intra-repo rewrite false positives.
_EXAMPLES_TARGET: Final[str] = f"//{'examples'}/..."

TESTS_INCLUSION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(tests|examples)/.*\.py$"
)

# TODO: ecalubaquib - remove exclusion once tests are runnable and fixed.
BASE_TESTS_EXCLUSIONS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^examples/benchmarks/"),
    re.compile(r"^tests/distributed/"),
    re.compile(r"^tests/fused_adagrad_test\.py$"),
    re.compile(r"^tests/index_put_op_unit_test\.py$"),
    re.compile(r"^tests/ops_test\.py$"),
    re.compile(r"^tests/tracing_test\.py$"),
    re.compile(r"^tests/profiler/"),
    re.compile(r"^tests/.*lint_test\.py$"),
    re.compile(r"^tests/cpp_style_test\.py$"),
    re.compile(r"^tests/native_functions_test\.py$"),
)

CPU_TESTS_EXCLUSIONS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(^|[_/])(tpu|gpu)_test\.py$"),
    re.compile(r"(^|/)tpu_.*_test\.py$"),
    re.compile(r"^tests/compile/debug_test\.py$"),
)

TPU_TESTS_EXCLUSIONS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(^|[_/])(cpu|gpu)_test\.py$"),
    re.compile(r"(^|/)cpu_.*_test\.py$"),
    re.compile(r"^tests/pallas/"),
    re.compile(r"^tests/compile/compile_with_distributed_test\.py$"),
    re.compile(r"^tests/compile/model_coverage_hf_test\.py$"),
    re.compile(r"^tests/compile/model_coverage_timm_test\.py$"),
    re.compile(r"^tests/compiler_option_test\.py$"),
    re.compile(r"^tests/errors_test\.py$"),
    re.compile(r"^tests/tpu_backend_config_test\.py$"),
    re.compile(r"^tests/tpu_errors_test\.py$"),
    re.compile(r"^tests/fused_adam_test\.py$"),
    re.compile(r"^tests/fused_sgd_test\.py$"),
    re.compile(r"^tests/fused_adamw_test\.py$"),
    re.compile(r"^tests/fused_moving_avg_obs_fq_helper_test\.py$"),
    re.compile(r"^tests/compile/activation_checkpointing_mem_test\.py$"),
)


def infer_device(
    bazel_config: str | None = None, explicit_device: str | None = None
) -> str:
  """Infers target device ('cpu' or 'tpu') from config or explicit parameter."""
  if explicit_device:
    device = explicit_device.strip().lower()
    if device in ("cpu", "tpu"):
      return device
    raise ValueError(f"Invalid explicit device '{explicit_device}'.")
  if bazel_config and "tpu" in bazel_config.lower():
    return "tpu"
  return "cpu"


def infer_tpu_generation(bazel_config: str | None) -> str | None:
  """Extracts TPU generation ('v5', 'v6', 'v7') from bazel config if present."""
  if not bazel_config:
    return None
  match = re.search(
      r"(?:^|[^a-z0-9])(v5e?|v6e?|v7x?)(?:$|[^a-z0-9])", bazel_config.lower()
  )
  return match.group(1)[:2] if match else None


def get_tag_excludes_regex(device: str, tpu_gen: str | None = None) -> str:
  """Returns a pipe-delimited regex pattern of Bazel tags to exclude."""
  tags = list(BASE_TAG_EXCLUDES)
  if device == "cpu":
    tags.extend(CPU_TAG_EXCLUDES)
  elif device == "tpu":
    tags.extend(TPU_TAG_EXCLUDES)
    tags.append(f"fails-on-tpu-{tpu_gen}" if tpu_gen else "fails-on-tpu.*")
  return "|".join(tags)


def sanitize_targets(targets: Iterable[str]) -> list[str]:
  """Filters out empty lines, comments, and external repository targets."""
  sanitized = []
  for target in targets:
    line = target.strip()
    if not line or line.startswith("#"):
      continue
    if line.startswith("@@//"):
      line = line[2:]
    elif line.startswith("@//"):
      line = line[1:]
    elif line.startswith("@"):
      continue
    sanitized.append(line)
  return sanitized


def build_bazel_query(
    mode: str,
    targets: Sequence[str] | None,
    tag_excludes_regex: str,
    positive_tag: str | None = None,
) -> str:
  """Generates Bazel query expression for mapping targets to source files."""
  if mode == "full":
    all_targets = f"//tests/... + {_EXAMPLES_TARGET}"
    base = f"kind('.*test', {all_targets})"
    if positive_tag:
      base = f"attr('tags', '{positive_tag}', {base})"
    return (
        f"labels(srcs, {base} except attr('tags', '{tag_excludes_regex}',"
        f" {all_targets}))"
    )
  if mode == "diff":
    sanitized = sanitize_targets(targets or ())
    if not sanitized:
      return ""
    joined = "\n    ".join(sanitized)
    base = "kind('.*test', $impacted)"
    if positive_tag:
      base = f"attr('tags', '{positive_tag}', {base})"
    return (
        f"let impacted = set(\n    {joined}\n) in labels(srcs, {base} except"
        f" attr('tags', '{tag_excludes_regex}', $impacted))"
    )
  raise ValueError(f"Unknown mode '{mode}'.")


def convert_bazel_label_to_path(label: str) -> str | None:
  """Converts a Bazel label (e.g. //tests:foo_test.py) to a relative path."""
  cleaned = re.sub(r"^@{0,2}//", "", label.strip()).replace(":", "/")
  normalized = re.sub(r"/+", "/", cleaned)
  return normalized or None


def filter_test_files(
    labels_or_paths: Iterable[str],
    device: str,
    workspace_dir: pathlib.Path | None = None,
    check_exists: bool = True,
) -> list[str]:
  """Filters and deduplicates test files according to patterns and device."""
  exclusions = list(BASE_TESTS_EXCLUSIONS)
  exclusions.extend(
      CPU_TESTS_EXCLUSIONS if device == "cpu" else TPU_TESTS_EXCLUSIONS
  )
  seen: set[str] = set()
  filtered: list[str] = []
  for item in labels_or_paths:
    relative_path = convert_bazel_label_to_path(item)
    if not relative_path or not TESTS_INCLUSION_PATTERN.match(relative_path):
      continue
    if any(pattern.search(relative_path) for pattern in exclusions):
      continue
    if (
        check_exists
        and workspace_dir
        and not (workspace_dir / relative_path).is_file()
    ):
      continue
    if relative_path not in seen:
      seen.add(relative_path)
      filtered.append(relative_path)
  return filtered


def run_bazel_query(query: str, workspace_dir: pathlib.Path) -> list[str]:
  """Executes bazel query and returns matching labels."""
  if not query.strip():
    return []
  with tempfile.NamedTemporaryFile(
      mode="w", suffix=".txt", delete=False
  ) as query_file:
    query_file.write(query)
    query_path = pathlib.Path(query_file.name)
  try:
    result = subprocess.run(
        ["bazel", "query", f"--query_file={query_path}", "--output=label"],
        cwd=workspace_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode in (3, 4):
      return []
    if result.returncode != 0:
      raise RuntimeError(
          f"Bazel query failed ({result.returncode}):\n{result.stderr}"
      )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]
  finally:
    query_path.unlink(missing_ok=True)


def _is_executable_file(path: pathlib.Path) -> bool:
  """Returns True if path exists, is a regular file, and is executable."""
  return path.is_file() and os.access(path, os.X_OK)


def resolve_pytest_cmd(
    workspace_dir: pathlib.Path, pytest_binary: str | None = None
) -> list[str]:
  """Resolves the pytest command to execute within the workspace environment."""
  del workspace_dir
  if pytest_binary:
    return [pytest_binary]
  if "VIRTUAL_ENV" in os.environ:
    venv_bin = pathlib.Path(os.environ["VIRTUAL_ENV"]) / "bin"
    if _is_executable_file(venv_bin / "python"):
      return [str(venv_bin / "python"), "-m", "pytest"]
    if _is_executable_file(venv_bin / "pytest"):
      return [str(venv_bin / "pytest")]
  tmp_python = pathlib.Path("/tmp/venv/bin/python")
  if _is_executable_file(tmp_python):
    return [str(tmp_python), "-m", "pytest"]
  system_pytest = shutil.which("pytest")
  return [system_pytest] if system_pytest else [sys.executable, "-m", "pytest"]


def run_pytest(
    test_files: Sequence[str],
    workspace_dir: pathlib.Path,
    device: str,
    extra_pytest_flags: str = "",
    pytest_binary: str | None = None,
) -> int:
  """Executes pytest sequentially across resolved test files."""
  if not test_files:
    print("No Python test files matched. Skipping pytest execution.")
    return 0
  print(f"Found {len(test_files)} impacted Python test file(s).")
  print("--- Impacted Test Files List ---")
  for test_file in test_files:
    print(test_file)
  print("--------------------------------")

  base_cmd = resolve_pytest_cmd(workspace_dir, pytest_binary) + [
      "--verbose",
      "--import-mode=importlib",
  ]
  if extra_pytest_flags:
    base_cmd.extend(shlex.split(extra_pytest_flags))

  overall_exit_code = 0
  for test_file in test_files:
    file_env = os.environ.copy()
    if device == "cpu":
      if "autoload_disabled_backend_test" in test_file:
        file_env.pop("TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND", None)
      else:
        file_env.setdefault("TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND", "1")
    print(f"Running pytest on: {test_file}")
    sys.stdout.flush()
    result = subprocess.run(
        base_cmd + [test_file], cwd=workspace_dir, env=file_env, check=False
    )
    if result.returncode not in (0, 5):
      print(
          f"Pytest failed on {test_file} with exit code {result.returncode}.",
          file=sys.stderr,
      )
      overall_exit_code = result.returncode

  if overall_exit_code != 0:
    print(f"Pytest failed with exit code {overall_exit_code}.", file=sys.stderr)
  else:
    print("All pytest tests passed successfully.")
  return overall_exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description="Run Bazel test targets via pytest."
  )
  parser.add_argument("--mode", choices=["full", "diff"], required=True)
  parser.add_argument("--targets-file", type=pathlib.Path, default=None)
  parser.add_argument("--bazel-config", type=str, default=None)
  parser.add_argument("--device", choices=["cpu", "tpu"], default=None)
  parser.add_argument("--workspace-dir", type=pathlib.Path, default=None)
  parser.add_argument("--extra-pytest-flags", type=str, default="")
  parser.add_argument("--pytest-binary", type=str, default=None)
  parser.add_argument("--dry-run", action="store_true")
  return parser.parse_args(argv)


def _resolve_workspace_dir(
    workspace_arg: pathlib.Path | None,
) -> pathlib.Path:
  """Resolves the workspace directory from CLI argument, env var, or cwd."""
  if workspace_arg:
    return workspace_arg.resolve()
  env_workspace = os.environ.get("GITHUB_WORKSPACE")
  if env_workspace:
    return pathlib.Path(env_workspace).resolve()
  return pathlib.Path.cwd().resolve()


def _load_diff_targets(
    targets_file: pathlib.Path | None,
) -> tuple[list[str] | None, int]:
  """Loads target labels for diff mode, returning (targets, exit_code)."""
  if not targets_file:
    print("Error: --targets-file is required in diff mode.", file=sys.stderr)
    return None, 1
  if not targets_file.is_file():
    print(
        f"Warning: Targets file '{targets_file}' does not exist.",
        file=sys.stderr,
    )
    return None, 0
  lines = targets_file.read_text(encoding="utf-8").splitlines()
  targets = [line.strip() for line in lines if line.strip()]
  if not targets:
    print("Impacted targets file is empty. Skipping pytest entirely!")
    return None, 0
  return targets, 0


def main(argv: Sequence[str] | None = None) -> int:
  """Main entry point."""
  args = parse_args(argv)
  workspace_dir = _resolve_workspace_dir(args.workspace_dir)
  device = infer_device(args.bazel_config, args.device)
  tpu_gen = infer_tpu_generation(args.bazel_config) if device == "tpu" else None
  positive_tag = f"presubmit-{tpu_gen}" if tpu_gen else None
  tag_excludes = get_tag_excludes_regex(device, tpu_gen=tpu_gen)

  targets: list[str] = []
  if args.mode == "diff":
    loaded_targets, exit_code = _load_diff_targets(args.targets_file)
    if loaded_targets is None:
      return exit_code
    targets = loaded_targets

  query = build_bazel_query(
      args.mode, targets, tag_excludes, positive_tag=positive_tag
  )
  if not query:
    print(
        "No relevant test targets found from query. Skipping pytest entirely!"
    )
    return 0

  test_files = filter_test_files(
      run_bazel_query(query, workspace_dir),
      device=device,
      workspace_dir=workspace_dir,
      check_exists=True,
  )
  if args.dry_run:
    print(f"Dry-run mode ({device}): Resolved {len(test_files)} test file(s):")
    for test_file in test_files:
      print(test_file)
    return 0
  return run_pytest(
      test_files,
      workspace_dir,
      device,
      args.extra_pytest_flags,
      args.pytest_binary,
  )


if __name__ == "__main__":
  sys.exit(main())
