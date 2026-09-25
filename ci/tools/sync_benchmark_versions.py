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

"""Keeps the benchmark registry's torch/libtpu pins on the latest stable lock.

The benchmark registry pins the PyTorch and libtpu versions every benchmark
environment installs. They should track the newest stable releases the wheel
supports, which `requirements/lock_environments.sh` resolves into the
highest-resolution lock (`requirements/requirements_<py>_latest.txt`).

Usage (from the repository root):

  # Rewrite the registry pins to match the latest lock.
  $ python3 ci/tools/sync_benchmark_versions.py

  # Only report drift; exits non-zero if the registry is out of sync.
  $ python3 ci/tools/sync_benchmark_versions.py --check
"""

import argparse
import dataclasses
import pathlib
import re
import sys
from typing import Final

_REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[2]

_REGISTRY_RELATIVE_PATH: Final[pathlib.Path] = pathlib.Path(
    "examples/benchmarks/e2e/harness/benchmark_registry.pbtxt"
)

# Registry lines of the form:
#   workload_action_inputs { key: "torch_spec" value: "torch==2.14.0" }
_SPEC_PATTERN_TEMPLATE: Final[str] = (
    r'(key:\s*"{key}"\s+value:\s*"{package}==)([^"]+)(")'
)

# Lock lines of the form `torch==2.14.0+cpu \`.
_LOCK_PIN_TEMPLATE: Final[str] = r"^{package}==([^\s\\]+)"


@dataclasses.dataclass(frozen=True)
class Pins:
  """Stable package versions the benchmarks should install."""

  torch: str
  libtpu: str


# (registry action input key, package name, Pins attribute)
_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("torch_spec", "torch", "torch"),
    ("libtpu_spec", "libtpu", "libtpu"),
)


def find_latest_lock(repo_root: pathlib.Path) -> pathlib.Path:
  """Returns the single highest-resolution lock under requirements/."""
  locks = sorted((repo_root / "requirements").glob("requirements_*_latest.txt"))
  if len(locks) != 1:
    raise ValueError(
        "expected exactly one requirements/requirements_*_latest.txt, found"
        f" {[str(p) for p in locks]}"
    )
  return locks[0]


def find_registry(repo_root: pathlib.Path) -> pathlib.Path:
  """Returns the benchmark registry path.

  The registry sits under `examples/` at the repository root, or one level up
  when the build files live in a subdirectory of the source tree.
  """
  for base in (repo_root, repo_root.parent):
    candidate = base / _REGISTRY_RELATIVE_PATH
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(f"cannot find {_REGISTRY_RELATIVE_PATH}")


def read_pins(lock_text: str) -> Pins:
  """Extracts the torch and libtpu versions pinned in a lock file."""
  versions = {}
  for _, package, attribute in _SPECS:
    match = re.search(
        _LOCK_PIN_TEMPLATE.format(package=re.escape(package)),
        lock_text,
        flags=re.MULTILINE,
    )
    if match is None:
      raise ValueError(f"no {package}== pin found in the lock")
    # Drop the local version label, e.g. the "+cpu" of "2.14.0+cpu".
    versions[attribute] = match.group(1).split("+", 1)[0]
  return Pins(**versions)


def sync_registry(registry_text: str, pins: Pins) -> str:
  """Returns the registry text with every `==` pin set to `pins`."""
  for key, package, attribute in _SPECS:
    pattern = _SPEC_PATTERN_TEMPLATE.format(
        key=re.escape(key), package=re.escape(package)
    )
    version = getattr(pins, attribute)
    registry_text = re.sub(
        pattern,
        lambda m, v=version: f"{m.group(1)}{v}{m.group(3)}",
        registry_text,
    )
  return registry_text


def find_drift(registry_text: str, pins: Pins) -> list[str]:
  """Lists `==` pins in the registry that differ from `pins`."""
  drift = []
  for key, package, attribute in _SPECS:
    pattern = _SPEC_PATTERN_TEMPLATE.format(
        key=re.escape(key), package=re.escape(package)
    )
    expected = getattr(pins, attribute)
    stale = sorted(
        {m.group(2) for m in re.finditer(pattern, registry_text)} - {expected}
    )
    drift.extend(
        f"{key}: {package}=={version} (expected {package}=={expected})"
        for version in stale
    )
  return drift


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
      "--check",
      action="store_true",
      help="report drift and exit non-zero instead of rewriting the registry",
  )
  args = parser.parse_args(argv)

  lock = find_latest_lock(_REPO_ROOT)
  registry = find_registry(_REPO_ROOT)
  pins = read_pins(lock.read_text())
  registry_text = registry.read_text()

  if args.check:
    drift = find_drift(registry_text, pins)
    if drift:
      print(
          f"::error::{registry.name} is out of sync with {lock.name}:",
          file=sys.stderr,
      )
      for line in drift:
        print(f"  {line}", file=sys.stderr)
      print(
          "Run `python3 ci/tools/sync_benchmark_versions.py` to update it.",
          file=sys.stderr,
      )
      return 1
    return 0

  synced = sync_registry(registry_text, pins)
  if synced != registry_text:
    registry.write_text(synced)
    print(
        f"Updated {registry.name} to torch=={pins.torch},"
        f" libtpu=={pins.libtpu}."
    )
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
