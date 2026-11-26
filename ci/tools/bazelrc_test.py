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

"""Unit tests for `.bazelrc` sandboxing and `--config=no_rbe` invariants."""

from collections.abc import Mapping, Sequence
import dataclasses
import pathlib
import shlex
import sys
from typing import Final

import pytest

_BAZELRC_PATH: Final[pathlib.Path] = (
    pathlib.Path(__file__).absolute().parents[2] / ".bazelrc"
)
_COMMAND_INHERITANCE: Final[Mapping[str, tuple[str, ...]]] = {
    "common": ("common",),
    "build": ("common", "build"),
    "test": ("common", "build", "test"),
    "query": ("common", "query"),
}


@dataclasses.dataclass(frozen=True)
class ParsedBazelrc:
  """Parsed `.bazelrc` unconditional and config-scoped flag sections."""

  unconditional: Mapping[str, tuple[str, ...]]
  configs: Mapping[tuple[str, str], tuple[str, ...]]
  defined_configs: frozenset[str]


@dataclasses.dataclass(frozen=True)
class _PopConfig:
  """Stack marker popping an active `--config` frame after expansion."""

  name: str


def parse_bazelrc(text: str) -> ParsedBazelrc:
  """Parses `.bazelrc` text into unconditional and `(cmd, config)` flag lists."""
  unconditional: dict[str, list[str]] = {}
  configs: dict[tuple[str, str], list[str]] = {}
  defined: set[str] = set()

  for raw_line in text.splitlines():
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
      continue
    if stripped.startswith(("import ", "try-import ")):
      continue
    tokens = shlex.split(stripped, comments=True)
    if not tokens:
      continue
    prefix, flags = tokens[0], tokens[1:]
    if ":" in prefix:
      cmd, cfg = prefix.split(":", 1)
      defined.add(cfg)
      configs.setdefault((cmd, cfg), []).extend(flags)
    else:
      unconditional.setdefault(prefix, []).extend(flags)

  return ParsedBazelrc(
      unconditional={k: tuple(v) for k, v in unconditional.items()},
      configs={k: tuple(v) for k, v in configs.items()},
      defined_configs=frozenset(defined),
  )


def expand_flags(
    rc: ParsedBazelrc,
    command: str,
    cli_configs: Sequence[str] = (),
) -> tuple[str, ...]:
  """Expands `.bazelrc` flags for `command` and `cli_configs` in Bazel order."""
  inherited_cmds = _COMMAND_INHERITANCE[command]
  initial: list[str] = []
  for cmd in inherited_cmds:
    initial.extend(rc.unconditional.get(cmd, ()))
  initial.extend(f"--config={cfg}" for cfg in cli_configs)

  stack: list[str | _PopConfig] = list(reversed(initial))
  active: set[str] = set()
  expanded: list[str] = []

  while stack:
    item = stack.pop()
    if isinstance(item, _PopConfig):
      active.remove(item.name)
      continue
    if item.startswith("--config="):
      cfg = item.removeprefix("--config=")
      if cfg not in rc.defined_configs:
        raise KeyError(f"Undefined --config={cfg}")
      if cfg in active:
        raise ValueError(f"Cyclic --config={cfg}")
      active.add(cfg)
      stack.append(_PopConfig(cfg))
      cfg_flags: list[str] = []
      for cmd in inherited_cmds:
        cfg_flags.extend(rc.configs.get((cmd, cfg), ()))
      stack.extend(reversed(cfg_flags))
    else:
      expanded.append(item)

  return tuple(expanded)


def last_flag_value(flags: Sequence[str], name: str) -> str | None:
  """Returns the last effective value of `--<name>` (handling `--no<name>`)."""
  prefix = f"--{name}="
  bare = f"--{name}"
  negated = f"--no{name}"
  value: str | None = None
  for flag in flags:
    if flag.startswith(prefix):
      value = flag.removeprefix(prefix)
    elif flag == bare:
      value = "true"
    elif flag == negated:
      value = "false"
  return value


def find_unsandboxed_build_strategies(
    flags: Sequence[str],
) -> tuple[str, ...]:
  """Returns any flags that run non-`TestRunner` build actions unsandboxed."""
  violations: list[str] = []
  for flag in flags:
    if flag.startswith("--spawn_strategy="):
      strategies = tuple(flag.removeprefix("--spawn_strategy=").split(","))
      if not _is_sandboxed_strategy_list(strategies):
        violations.append(flag)
    elif flag.startswith("--strategy="):
      rest = flag.removeprefix("--strategy=")
      if "=" in rest:
        mnemonic, strat_csv = rest.split("=", 1)
        if mnemonic == "TestRunner":
          continue
        strategies = tuple(strat_csv.split(","))
      else:
        strategies = tuple(rest.split(","))
      if not _is_sandboxed_strategy_list(strategies):
        violations.append(flag)
  return tuple(violations)


def _is_sandboxed_strategy_list(strategies: Sequence[str]) -> bool:
  if "standalone" in strategies:
    return False
  if "local" in strategies:
    if "sandboxed" not in strategies:
      return False
    return strategies.index("sandboxed") < strategies.index("local")
  return True


@pytest.fixture(scope="module")
def repo_bazelrc() -> ParsedBazelrc:
  return parse_bazelrc(_BAZELRC_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("command", ("build", "test"))
def test_all_configs_expand_and_keep_build_actions_sandboxed(
    repo_bazelrc: ParsedBazelrc,
    command: str,
) -> None:
  default_flags = expand_flags(repo_bazelrc, command)
  assert find_unsandboxed_build_strategies(default_flags) == ()

  for cfg in sorted(repo_bazelrc.defined_configs):
    flags = expand_flags(repo_bazelrc, command, (cfg,))
    assert (
        find_unsandboxed_build_strategies(flags) == ()
    ), f"--config={cfg} under `{command}` leaves build actions unsandboxed"


@pytest.mark.parametrize("cfg", ("local", "testone", "bench"))
def test_local_configs_scope_standalone_to_test_runner_only(
    repo_bazelrc: ParsedBazelrc,
    cfg: str,
) -> None:
  flags = expand_flags(repo_bazelrc, "test", (cfg,))
  assert last_flag_value(flags, "spawn_strategy") is None
  assert "--strategy=TestRunner=standalone,local" in flags
  assert "--test_env=TPU_TOPOLOGY" in flags


def test_testone_after_rbe_config_keeps_remote_build_execution(
    repo_bazelrc: ParsedBazelrc,
) -> None:
  build_flags = expand_flags(repo_bazelrc, "build", ("ci_tpu_v5", "testone"))
  assert last_flag_value(build_flags, "spawn_strategy") is None
  assert (
      last_flag_value(build_flags, "remote_executor")
      == "grpcs://remotebuildexecution.googleapis.com"
  )


def test_no_rbe_is_decoupled_from_local(repo_bazelrc: ParsedBazelrc) -> None:
  flags = expand_flags(repo_bazelrc, "test", ("no_rbe",))
  assert "--strategy=TestRunner=standalone,local" not in flags
  assert "--test_env=TPU_TOPOLOGY" not in flags
  assert find_unsandboxed_build_strategies(flags) == ()


@pytest.mark.parametrize(
    "rbe_config",
    (
        "remote_caching_readonly",
        "remote_caching_readwrite",
        "remote_execution_readonly",
        "remote_execution_readwrite",
        "wheel_common",
        "wheel_nightly",
        "wheel_release",
        "ci",
        "ci_cpu",
        "ci_tpu_v5",
        "ci_rbe_presubmit",
    ),
)
@pytest.mark.parametrize("command", ("build", "test"))
def test_no_rbe_resets_rbe_and_cache_flags(
    repo_bazelrc: ParsedBazelrc,
    rbe_config: str,
    command: str,
) -> None:
  flags = expand_flags(repo_bazelrc, command, (rbe_config, "no_rbe"))
  assert find_unsandboxed_build_strategies(flags) == ()
  assert last_flag_value(flags, "remote_cache") == ""
  assert last_flag_value(flags, "remote_executor") == ""
  assert last_flag_value(flags, "remote_instance_name") == ""
  assert last_flag_value(flags, "remote_upload_local_results") == "true"
  assert last_flag_value(flags, "extra_execution_platforms") == ""
  assert last_flag_value(flags, "bes_backend") == ""
  assert last_flag_value(flags, "google_default_credentials") == "false"
  assert last_flag_value(flags, "jobs") == "auto"
  assert last_flag_value(flags, "//:_rbe_test_execution") == "False"


def test_no_rbe_composes_cleanly_before_public_remote_cache(
    repo_bazelrc: ParsedBazelrc,
) -> None:
  flags = expand_flags(
      repo_bazelrc,
      "build",
      ("wheel_common", "no_rbe", "helion_public_caching_readwrite"),
  )
  assert last_flag_value(flags, "remote_executor") == ""
  assert last_flag_value(flags, "remote_instance_name") == ""
  assert last_flag_value(flags, "extra_execution_platforms") == ""
  assert (
      last_flag_value(flags, "remote_cache")
      == "https://storage.googleapis.com/torchtpu-bazel-cache-helion/"
  )
  assert last_flag_value(flags, "remote_upload_local_results") == "true"
  assert last_flag_value(flags, "google_default_credentials") == "true"


def test_no_vestigial_executor_define(repo_bazelrc: ParsedBazelrc) -> None:
  for cfg in sorted(repo_bazelrc.defined_configs):
    flags = expand_flags(repo_bazelrc, "build", (cfg,))
    assert "--define=EXECUTOR=remote" not in flags


@pytest.mark.parametrize(
    ("bad_line", "expected"),
    (
        (
            "common:bad --spawn_strategy=standalone",
            ("--spawn_strategy=standalone",),
        ),
        (
            "common:bad --strategy=standalone",
            ("--strategy=standalone",),
        ),
        (
            "common:bad --spawn_strategy=local,sandboxed",
            ("--spawn_strategy=local,sandboxed",),
        ),
        (
            "common:bad --strategy=CppCompile=local",
            ("--strategy=CppCompile=local",),
        ),
        (
            "common:bad --spawn_strategy=worker,sandboxed,local",
            (),
        ),
        (
            "common:bad --strategy=TestRunner=standalone,local",
            (),
        ),
    ),
)
def test_unsandboxed_strategy_detector(
    bad_line: str,
    expected: tuple[str, ...],
) -> None:
  rc = parse_bazelrc(bad_line)
  flags = expand_flags(rc, "build", ("bad",))
  assert find_unsandboxed_build_strategies(flags) == expected


def test_parser_rejects_undefined_and_cyclic_configs() -> None:
  rc = parse_bazelrc("common:a --config=b\ncommon:b --config=a\n")
  with pytest.raises(KeyError, match="Undefined --config=missing"):
    expand_flags(rc, "build", ("missing",))
  with pytest.raises(ValueError, match="Cyclic --config=a"):
    expand_flags(rc, "build", ("a",))


if __name__ == "__main__":
  sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
