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

"""Test utilities for TorchTPU tests, including locating and loading goldens data."""

from collections.abc import Sequence
import dataclasses
import gzip
import importlib.resources
import os
import pathlib
import plistlib
import typing
from typing import Any, IO

from absl import flags
from tests import goldens_pin

# Apparent name of the Bazel external repository that holds the golden files in
# OSS builds. Must match the `goldens_repo(name = ...)` in oss/MODULE.bazel.
GOLDENS_REPO_NAME = "torch_tpu_goldens"


@dataclasses.dataclass(frozen=True)
class LoadedGolden:
  """A golden file's name and its decoded plistlib pytree."""

  filename: str
  tree: Any


def matches_prefix(filename: str, prefix: str) -> bool:
  """Checks if a filename matches the active prefix."""
  if prefix == "ops_test_gpu_golden":
    return (
        filename.startswith("ops_test_gpu_golden")
        and not filename.startswith("ops_test_gpu_golden_compiled")
        and not filename.startswith("ops_test_gpu_golden_grad")
    )
  return filename.startswith(prefix)


# TODO(b/576014545): Replace candidate directory searching with direct Bazel
# runfiles resolution once per-test-case golden artifacts land in OSS.
def goldens_candidate_local_dirs(
    base_dir: str | pathlib.Path | None = None,
) -> Sequence[pathlib.Path]:
  """Returns candidate local directories to search for golden files."""
  candidate_dirs: list[pathlib.Path] = []

  # 1. User-specified directory via parameter or flag.
  if base_dir:
    candidate_dirs.append(pathlib.Path(base_dir))
  elif (
      "golden_data_base_dir" in flags.FLAGS and flags.FLAGS.golden_data_base_dir
  ):
    candidate_dirs.append(pathlib.Path(flags.FLAGS.golden_data_base_dir))

  # 2. Directory of __file__ (standard source test mode).
  if __file__:
    candidate_dirs.append(pathlib.Path(__file__).parent)

  # 3. Specific runfiles directories when executing under Bazel or unpacked wheel.
  candidate_dirs.append(pathlib.Path.cwd() / "tests")
  if "TEST_SRCDIR" in os.environ:
    test_srcdir = pathlib.Path(os.environ["TEST_SRCDIR"])
    candidate_dirs.extend([
        test_srcdir / "_main" / "tests",
        test_srcdir / "torch_tpu" / "tests",
    ])
    # 4. In OSS the goldens are not checked in; Bazel fetches them from GCS as
    # the external repository `@torch_tpu_goldens` (see oss/bazel/goldens.bzl)
    # and the ops tests list the files they need in `data`. Its runfiles
    # directory is named after the repository's canonical name, whose
    # Bazel-version-dependent prefix (e.g. "+_repo_rules+") always ends in the
    # apparent name.
    candidate_dirs.extend(sorted(test_srcdir.glob(f"*{GOLDENS_REPO_NAME}")))
  return candidate_dirs


def locate_goldens_pinfile() -> pathlib.Path:
  """Locates the tests/goldens.pin configuration file."""
  package = goldens_pin.__package__
  resource = importlib.resources.files(package).joinpath("goldens.pin")
  if resource is None or not resource.is_file():
    raise FileNotFoundError("Could not locate goldens.pin")
  return pathlib.Path(str(resource))


def load_golden_files(
    prefix: str,
    candidate_dirs: Sequence[pathlib.Path] | None = None,
    pin_path: pathlib.Path | None = None,
) -> Sequence[LoadedGolden]:
  """Loads golden files into plistlib pytrees matching prefix.

  The source of truth for which golden files exist is the tests/goldens.pin
  file. Every matching file must be present in one of the candidate
  directories: internally they are checked in next to the tests, and in OSS
  Bazel fetches them from GCS at build time.

  Raises:
    ValueError: If no pinned golden matches `prefix`.
    FileNotFoundError: If a pinned golden file is not present in any of the
      candidate directories.
  """
  if pin_path is None:
    pin_path = locate_goldens_pinfile()
  pin_data = goldens_pin.parse_goldens_pin(pin_path)

  matching_names = [
      fname for fname in pin_data.goldens if matches_prefix(fname, prefix)
  ]
  if not matching_names:
    raise ValueError(
        f"No golden files found matching prefix '{prefix}' in {pin_path}"
    )

  if candidate_dirs is None:
    candidate_dirs = goldens_candidate_local_dirs()

  results: list[LoadedGolden] = []
  for fname in sorted(matching_names):
    local_file = _find_golden_file(fname, candidate_dirs)
    if local_file is None:
      searched = ", ".join(str(d) for d in candidate_dirs)
      raise FileNotFoundError(
          f"Golden file {fname} was not found in any of the candidate"
          f" directories: {searched}."
      )
    print(
        "Loading the collected input/output pairs for each op from"
        f" {local_file}",
        flush=True,
    )
    with gzip.open(local_file, "rb") as f:
      # gzip.GzipFile duck types as typing.IO[bytes], but does not inherit
      # from it, so need to cast.
      f = typing.cast(IO[bytes], f)
      plist_ptree = plistlib.load(f, fmt=plistlib.FMT_BINARY)
    results.append(LoadedGolden(fname, plist_ptree))

  return results


def _find_golden_file(
    filename: str, candidate_dirs: Sequence[pathlib.Path]
) -> pathlib.Path | None:
  """Returns the first candidate directory entry for `filename` that is a file."""
  for candidate_dir in candidate_dirs:
    candidate_path = candidate_dir / filename
    if candidate_path.is_file():
      return candidate_path
  return None


def find_per_test_case_goldens_dir(
    goldens_dir_name: str,
    candidate_dirs: Sequence[pathlib.Path] | None = None,
) -> pathlib.Path | None:
  """Returns directory containing per-test-case golden files, if found.

  Args:
    goldens_dir_name: Name of the per-test-case golden directory artifact to
      locate (e.g. "ops_test_gpu_golden_per_test_case").
    candidate_dirs: Candidate search paths. If None, defaults to
      `goldens_candidate_local_dirs()`.

  Returns:
    The directory path if found in one of candidate_dirs, or None otherwise.
  """
  if candidate_dirs is None:
    candidate_dirs = goldens_candidate_local_dirs()
  for d in candidate_dirs:
    candidate_path = d / goldens_dir_name
    if candidate_path.is_dir():
      return candidate_path
  return None


def load_single_test_case_golden(
    test_case_name: str,
    per_test_case_dir: pathlib.Path,
) -> dict[str, Any] | None:
  """Loads the plist pytree for a single test case, if present.

  Args:
    test_case_name: Name of the test case method (e.g. "test_sin").
    per_test_case_dir: Directory containing `<test_case_name>.plist` files.

  Returns:
    The deserialized dictionary mapping variant -> dtype -> samples for the
    given test case, or None if the corresponding `.plist` file is not present.
  """
  test_case_file = per_test_case_dir / f"{test_case_name}.plist"
  if not test_case_file.is_file():
    return None
  with open(test_case_file, "rb") as f:
    return plistlib.load(f, fmt=plistlib.FMT_BINARY)
