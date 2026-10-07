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

"""Paths in the TorchTPU repository.

In google3, the repository is at `third_party/py/torch_tpu`.
"""

import os
import pathlib

# The public API baseline, relative to the repository root.
TT_APIS_YAML_PATH = "tests/api/tt_apis.yaml"

# The baseline of internal APIs that the docs and examples expose, relative to
# the repository root.
EXPOSED_INTERNAL_APIS_YAML_PATH = "tests/api/exposed_internal_apis.yaml"

# The generated list of public APIs, relative to the repository root.
PUBLIC_APIS_MD_PATH = "docs/markdown/public_apis.md"

# The path of the repository in a google3 workspace or Blaze runfiles tree,
# which source file paths of Python objects in google3 contain.
G3_REPO_PATH = "third_party/py/torch_tpu"

# The package of the API scanners and their tests, relative to the repository
# root.
SCANNER_PACKAGE = "tests/api"

# The file that identifies the repository root: this module, relative to the
# repository root. It's in the workspace, and in the runfiles of every target
# that uses it.
_KEY_FILE = "tests/repo_paths.py"


def _first_containing_key_file(
    candidates: list[pathlib.Path], location: str
) -> str:
  """Returns the first candidate that contains `_KEY_FILE`.

  Args:
    candidates: The possible repository roots, in order of preference.
    location: Where the candidates are, for the error message.

  Raises:
    FileNotFoundError: If no candidate contains `_KEY_FILE`.
  """
  for candidate in candidates:
    if (candidate / _KEY_FILE).exists():
      return str(candidate)
  searched = ", ".join(str(candidate) for candidate in candidates)
  raise FileNotFoundError(
      f"Could not find torch_tpu repository root in {location}. Looked for"
      f" {_KEY_FILE} in: [{searched}]."
  )


def get_repo_root_in_runfiles() -> str:
  """Returns the TorchTPU repository root in the runfiles.

  The candidate is `third_party/py/torch_tpu` in the `google3` repository of the
  runfiles (`TEST_SRCDIR` or `RUNFILES_DIR`).

  Raises:
    FileNotFoundError: If no candidate is the repository root.
  """
  runfiles_dir = os.environ.get("TEST_SRCDIR") or os.environ.get("RUNFILES_DIR")
  candidates = []
  if runfiles_dir:
    candidates = [pathlib.Path(runfiles_dir, "google3", G3_REPO_PATH)]
  return _first_containing_key_file(candidates, "runfiles")


def get_repo_root_in_workspace() -> str:
  """Returns the TorchTPU repository root in the workspace of `blaze run`.

  Only tools that write into the source workspace should use this, so that
  tests stay hermetic. The candidate is `<workspace>/third_party/py/torch_tpu`
  in the workspace of `blaze run` (`BUILD_WORKSPACE_DIRECTORY`).

  Raises:
    FileNotFoundError: If `BUILD_WORKSPACE_DIRECTORY` isn't set, i.e. the
      caller doesn't run with `blaze run`, or no candidate is the repository
      root.
  """
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
  if not workspace_dir:
    raise FileNotFoundError(
        "BUILD_WORKSPACE_DIRECTORY is not set. Run this with `blaze run`,"
        " which sets it to the workspace to update."
    )
  candidates = [pathlib.Path(workspace_dir, G3_REPO_PATH)]
  return _first_containing_key_file(candidates, "the workspace")
