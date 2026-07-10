#!/usr/bin/env python3
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

"""Standalone within-minor PyTorch ABI check for the separated wheel.

The wheel ships one glue per supported PyTorch *minor* at its latest patch (see
WHEEL_TORCH_VERSIONS), relying on PyTorch's ABI being stable across patch
releases so the dispatch loader can serve, e.g., a torch 2.12.0 install from the
2.12.1 glue. This script proves that stability in practice: for each intermediate
patch we ship no exact glue for, it installs that patch into a throwaway venv
alongside the built wheel and confirms that ``import torch_tpu`` loads the glue
(linked with ``-z now``, so an ABI-incompatible glue fails with an
undefined-symbol error) and that the loader selected the expected same-minor
glue.

This is deliberately NOT a bazel test: bazel's wheel-test venv only provides the
default PyTorch, and having bazel drive a hand-rolled venv is exactly what we
want to avoid. Instead it is a plain entrypoint, invoked from the wheel CI
workflow (.github/workflows/build_wheel.yml) after the wheel is built.

Usage:
    python ci/wheel/check_intermediate_patch_abi.py --wheel dist/torch_tpu-*.whl

Exits 0 if every intermediate patch loads its same-minor glue, non-zero
otherwise. Requires network access to fetch the CPU PyTorch wheels.
"""

import argparse
import glob
import pathlib
import subprocess
import sys
import tempfile
import venv

# Intermediate patch release -> the same-minor glue suffix the loader must pick.
# Each entry is a patch that sits between two WHEEL_TORCH_VERSIONS minors (or an
# earlier patch of a built minor) for which we intentionally ship no exact glue.
_INTERMEDIATE_PATCHES = {
    "2.12.0": "2_12_1",
}

_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# Run inside the throwaway venv: import torch_tpu (loading + binding the glue)
# and report which glue the loader selected.
_PROBE = """
import torch  # noqa: F401  -- load libtorch first, as a real user would.
import torch_tpu  # noqa: F401  -- loads and binds the version-suffixed glue.
from torch_tpu import _versioned_so_loader
print("SELECTED_SUFFIX=" + str(_versioned_so_loader._VERSION_SUFFIX))
"""


def _resolve_wheel(pattern: str) -> str:
  matches = glob.glob(pattern)
  if not matches:
    raise SystemExit(f"No wheel matched: {pattern}")
  if len(matches) > 1:
    raise SystemExit(f"Multiple wheels matched {pattern}: {matches}")
  return matches[0]


def _check_patch(wheel: str, torch_version: str, expected_suffix: str) -> bool:
  """Returns True if the intermediate patch loads the expected same-minor glue."""
  with tempfile.TemporaryDirectory() as tmp:
    env_dir = pathlib.Path(tmp) / "venv"
    venv.create(env_dir, with_pip=True, clear=True)
    py = env_dir / "bin" / "python"

    subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet",
         f"torch=={torch_version}", "--index-url", _TORCH_CPU_INDEX],
        check=True,
    )
    # Wheel second so its deps (which do not pin torch) leave the torch we just
    # installed in place.
    subprocess.run(
        [str(py), "-m", "pip", "install", "--quiet", wheel],
        check=True,
    )

    # Run from the temp dir, not wherever this script was invoked: `python -c`
    # prepends the current directory to sys.path, so running at the repo root
    # would import the source torch_tpu (which has no built extensions) instead
    # of the wheel we just installed into the venv.
    result = subprocess.run(
        [str(py), "-c", _PROBE], capture_output=True, text=True, cwd=tmp,
    )

  if result.returncode != 0:
    print(
        f"FAIL torch {torch_version}: import torch_tpu failed (an "
        f"undefined-symbol error means the {expected_suffix.replace('_', '.')} "
        f"glue is not ABI-compatible with this patch):\n{result.stderr}",
        file=sys.stderr,
    )
    return False
  if f"SELECTED_SUFFIX={expected_suffix}" not in result.stdout:
    print(
        f"FAIL torch {torch_version}: expected the loader to select "
        f"{expected_suffix}, got:\n{result.stdout}\n{result.stderr}",
        file=sys.stderr,
    )
    return False
  print(f"OK   torch {torch_version} -> {expected_suffix} glue loads")
  return True


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--wheel", required=True,
      help="Path or glob for the built torch_tpu wheel.",
  )
  args = parser.parse_args()

  wheel = _resolve_wheel(args.wheel)
  print(f"Checking within-minor ABI against {wheel}")
  ok = True
  for torch_version, expected_suffix in _INTERMEDIATE_PATCHES.items():
    ok &= _check_patch(wheel, torch_version, expected_suffix)
  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
