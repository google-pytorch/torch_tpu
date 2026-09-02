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
import gzip
import hashlib
import importlib.resources
import io
import os
import pathlib
import plistlib
import typing
from typing import Any, IO
import urllib.request

from absl import flags
from tests import goldens_pin


def matches_prefix(filename: str, prefix: str) -> bool:
  """Checks if a filename matches the active prefix."""
  if prefix == "ops_test_gpu_golden":
    return (
        filename.startswith("ops_test_gpu_golden")
        and not filename.startswith("ops_test_gpu_golden_compiled")
        and not filename.startswith("ops_test_gpu_golden_grad")
    )
  return filename.startswith(prefix)


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
  return candidate_dirs


def locate_goldens_pinfile() -> pathlib.Path:
  """Locates the tests/goldens.pin configuration file."""
  package = goldens_pin.__package__
  resource = importlib.resources.files(package).joinpath("goldens.pin")
  if resource is None or not resource.is_file():
    raise FileNotFoundError("Could not locate goldens.pin")
  return pathlib.Path(str(resource))


def validate_sha256(
    data: bytes,
    expected_sha256: str,
    filename: str | None = None,
    url: str | None = None,
) -> None:
  """Validates that data matches expected SHA-256 checksum."""
  actual_sha256 = hashlib.sha256(data).hexdigest()
  if actual_sha256.lower() != expected_sha256.lower():
    details = []
    if filename:
      details.append(f" for {filename}")
    if url:
      details.append(f" downloaded from {url}")
    target_info = "".join(details)
    raise ValueError(
        f"SHA-256 checksum mismatch{target_info}:"
        f" expected {expected_sha256}, got {actual_sha256}"
    )


def download_golden_file(
    filename: str,
    expected_sha256: str,
    gcs_bucket: str,
    gcs_path: str,
) -> bytes:
  """Downloads a golden file from GCS and validates its SHA-256 checksum."""
  url = f"https://storage.googleapis.com/{gcs_bucket}/{gcs_path}"
  print(f"Downloading golden file from {url}...", flush=True)
  req = urllib.request.Request(
      url, headers={"User-Agent": "TorchTPU-Golden-Downloader"}
  )
  with urllib.request.urlopen(req, timeout=60) as response:
    data = response.read()

  validate_sha256(data, expected_sha256, filename=filename, url=url)

  return data


def load_golden_files(
    prefix: str,
    candidate_dirs: Sequence[pathlib.Path] | None = None,
    pin_path: pathlib.Path | None = None,
) -> Sequence[Any]:
  """Loads golden files into plistlib pytrees matching prefix.

  The source of truth for the golden files is the tests/goldens.pin file.
  The golden files are resolved in 2 stages:
  1. Check the local filesystem in the candidate directories.
  2. If a local file is not found, download it from GCS.
  """
  if pin_path is None:
    pin_path = locate_goldens_pinfile()
  pin_data = goldens_pin.parse_goldens_pin(pin_path)
  gcs_bucket = pin_data.gcs_bucket
  goldens_map = pin_data.goldens

  matching_items = [
      (fname, finfo)
      for fname, finfo in goldens_map.items()
      if matches_prefix(fname, prefix)
  ]
  if not matching_items:
    raise ValueError(
        f"No golden files found matching prefix '{prefix}' in {pin_path}"
    )

  if candidate_dirs is None:
    candidate_dirs = goldens_candidate_local_dirs()

  missing_goldens: list[tuple[str, goldens_pin.GoldenPath]] = []
  results: list[Any] = []

  # Attempt to locate the golden file in the local candidate directories.
  for fname, finfo in sorted(matching_items):
    local_file: pathlib.Path | None = None
    for candidate_dir in candidate_dirs:
      if candidate_dir.is_dir():
        candidate_path = candidate_dir / fname
        if candidate_path.is_file():
          with open(candidate_path, "rb") as f:
            data = bytes(f.read())
          try:
            validate_sha256(data, finfo.sha256, filename=fname)
            local_file = candidate_path
          except ValueError:
            # If sha mismatches, do not throw, mark as missing.
            print(
                f"SHA-256 mismatch for {fname} in {candidate_path}",
                flush=True,
            )
          break

    if local_file is not None:
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
      results.append(plist_ptree)
    else:
      missing_goldens.append((fname, finfo))

  if missing_goldens:
    missing_files = [fname for fname, _ in missing_goldens]
    print(
        "The following golden files were missing locally and will be downloaded"
        f" from GCS: {missing_files}",
        flush=True,
    )

  # Download missing golden files from GCS.
  for fname, finfo in missing_goldens:
    data = download_golden_file(
        filename=fname,
        expected_sha256=finfo.sha256,
        gcs_bucket=gcs_bucket,
        gcs_path=finfo.gcs_path,
    )
    print(
        "Loading the collected input/output pairs for each op from GCS:"
        f" {fname}",
        flush=True,
    )
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
      # Cast required: gzip.GzipFile does not inherit from typing.IO[bytes].
      f = typing.cast(IO[bytes], f)
      plist_ptree = plistlib.load(f, fmt=plistlib.FMT_BINARY)
    results.append(plist_ptree)

  return results
