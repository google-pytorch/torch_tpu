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

"""Finds the root directory of the TorchTPU repository.

This works the same way in google3 (Blaze), where the repository is at
`third_party/py/torch_tpu`, and in OSS (Bazel), where it's the workspace root.
"""

import dataclasses
import enum
import os
import pathlib
from typing import Iterator

# The path of the repository in a google3 workspace or Blaze runfiles tree.
GOOGLE3_PATH = pathlib.PurePosixPath("third_party", "py", "torch_tpu")

_NON_REPO_DIR_NAMES = frozenset(
    {"site-packages", "dist-packages", "venv", ".venv"}
)


class Layout(enum.Enum):
  """The layout of the tree that contains the repository."""

  # A google3 workspace or Blaze runfiles tree.
  GOOGLE3 = "google3"
  # An OSS workspace or Bazel runfiles tree.
  OSS = "oss"


@dataclasses.dataclass(frozen=True)
class RepoRoot:
  """A TorchTPU repository root.

  Attributes:
    path: The root directory.
    layout: The layout of the tree that contains `path`.
    in_workspace: Whether `path` is in the source workspace of a `blaze run` or
      `bazel run` command rather than in runfiles.
  """

  path: pathlib.Path
  layout: Layout
  in_workspace: bool


def _candidates(include_workspace: bool) -> Iterator[RepoRoot]:
  """Yields the possible repository roots, in order of preference."""
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  if include_workspace and workspace_dir:
    # `blaze run` in google3.
    yield RepoRoot(
        path=pathlib.Path(workspace_dir, GOOGLE3_PATH),
        layout=Layout.GOOGLE3,
        in_workspace=True,
    )
    # `bazel run` in OSS.
    yield RepoRoot(
        path=pathlib.Path(workspace_dir), layout=Layout.OSS, in_workspace=True
    )

  runfiles_dir = os.environ.get("TEST_SRCDIR") or os.environ.get("RUNFILES_DIR")
  if runfiles_dir:
    # Blaze (internal Google3).
    yield RepoRoot(
        path=pathlib.Path(runfiles_dir, "google3", GOOGLE3_PATH),
        layout=Layout.GOOGLE3,
        in_workspace=False,
    )
    # Bazel with Bzlmod, which names the main repository '_main'.
    yield RepoRoot(
        path=pathlib.Path(runfiles_dir, "_main"),
        layout=Layout.OSS,
        in_workspace=False,
    )
    # Bazel with a WORKSPACE file, which names the main repository after its
    # workspace name, and an external repository after its repository name.
    yield RepoRoot(
        path=pathlib.Path(runfiles_dir, "torch_tpu"),
        layout=Layout.OSS,
        in_workspace=False,
    )


def find(marker: str, *, include_workspace: bool = False) -> RepoRoot:
  """Returns the TorchTPU repository root.

  The candidates are, in order:

  1. If `include_workspace` is true, the workspace of `blaze run` or `bazel run`
     (`BUILD_WORKSPACE_DIRECTORY`): `<workspace>/third_party/py/torch_tpu` in
     google3, then `<workspace>` in OSS.
  2. The runfiles (`TEST_SRCDIR` or `RUNFILES_DIR`): `third_party/py/torch_tpu`
     in the `google3` repository for Blaze, then `_main` and `torch_tpu` for
     Bazel.

  Args:
    marker: A path relative to the repository root that identifies it, e.g.
      "docs/markdown/public_apis.md". It must be in the runfiles of the caller's
      target, e.g. as a `data` dependency.
    include_workspace: Whether to consider the workspace of `blaze run` or
      `bazel run` before the runfiles. Only tools that write into the workspace
      should set it, so that tests stay hermetic.

  Returns:
    The first candidate that contains `marker`.

  Raises:
    FileNotFoundError: If no candidate contains `marker`.
  """
  candidates = list(_candidates(include_workspace))
  for candidate in candidates:
    if (candidate.path / marker).exists():
      return candidate
  searched = ", ".join(str(candidate.path) for candidate in candidates)
  raise FileNotFoundError(
      "Could not find torch_tpu repository root in runfiles"
      + (" or the workspace" if include_workspace else "")
      + f". Looked for {marker} in: [{searched}]."
  )


def is_ignored_dir(name: str) -> bool:
  """Returns True if a directory should be skipped during file traversal.

  In OSS Bazel builds, rules_python creates virtual environments inside the
  test runfiles directory (e.g. tests/_<target>.venv/lib/.../site-packages/).
  These directories pull in external third-party C/C++ files (such as
  lxml/etree.h) that are not part of the TorchTPU repository and must be
  skipped. Wheel tests unpack the built torch_tpu wheel into a
  <target>_unpacked_wheel/ directory; the public headers it ships are copies
  of source files that are linted at their source location.
  Hidden directories (starting with '.') are also ignored.

  Args:
    name: The directory's name.
  """
  return (
      name.endswith(".venv")
      or name.endswith("_unpacked_wheel")
      or name in _NON_REPO_DIR_NAMES
      or name.startswith(".")
  )
