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

"""Nox configuration for torch_tpu automation environments.

Nox manages isolated execution environments (backed here by `uv`), so
contributors do not need to manually install tools like `clang-format` via
OS-specific package managers. Run `nox -l` to list the available sessions.

Details on configuration can be found at: https://nox.thea.codes/
"""

from collections.abc import Mapping, Sequence
import pathlib
import re
from typing import Final

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint"]

# The index that must satisfy the `torch` dependency for each variant. `None`
# means the default index (PyPI), whose Linux wheels bundle CUDA support.
_TORCH_INDEXES: Final[Mapping[str, str | None]] = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": None,
    "nightly": "https://download.pytorch.org/whl/nightly/cpu",
}


@nox.session
def lint(session: nox.Session) -> None:
  """Run clang-format check on target branch modifications."""
  session.install("clang-format")
  session.run("ci/tools/clang_format.sh", "lint", external=True)


@nox.session
def format(session: nox.Session) -> None:
  """Apply clang-format fixes to modified files automatically."""
  session.install("clang-format")
  session.run("ci/tools/clang_format.sh", "format", external=True)


def _newest_supported_torch() -> str:
  """Return the newest PyTorch version the wheel ships a glue for.

  Parsed out of the `WHEEL_TORCH_VERSIONS` source of truth in
  bazel/pytorch_versions.bzl so the smoke tests track it automatically.
  """
  text = pathlib.Path("bazel/pytorch_versions.bzl").read_text()
  default = re.search(
      r'^DEFAULT_TORCH_VERSION = "([0-9.]+)"', text, re.MULTILINE
  )
  extras_block = re.search(
      r"^EXTRA_PYTORCH_VERSIONS = \[(.*?)\]", text, re.MULTILINE | re.DOTALL
  )
  if default is None or extras_block is None:
    raise RuntimeError(
        "Could not parse torch versions from bazel/pytorch_versions.bzl"
    )
  versions = [default.group(1), *re.findall(r'"([0-9.]+)"', extras_block.group(1))]
  return max(versions, key=lambda v: tuple(map(int, v.split("."))))


def _dist_wheel(session: nox.Session) -> pathlib.Path:
  wheels = sorted(pathlib.Path("dist").glob("torch_tpu-*.whl"))
  if len(wheels) != 1:
    session.error(
        "Expected exactly one torch_tpu wheel in dist/, found:"
        f" {[str(wheel) for wheel in wheels]}"
    )
  return wheels[0]


@nox.session
@nox.parametrize(
    "device, torch_variant",
    [
        ("cpu", "cpu"),
        ("cpu", "cuda"),
        ("cpu", "nightly"),
        ("tpu", "nightly"),
    ],
    ids=["cpu-cpu", "cpu-cuda", "cpu-nightly", "tpu-nightly"],
)
def smoke_test(session: nox.Session, device: str, torch_variant: str) -> None:
  """Install a pre-built wheel with a specific torch variant and smoke test it.

  The wheel from `dist/`, the torch variant, and the test dependencies are
  all installed in a single `uv` invocation so the resolver sees the full
  dependency set together, then the autoload smoke test for the given device
  is run against the resulting environment.
  """
  install_args = [str(_dist_wheel(session)), "pytest>=9"]
  if (index := _TORCH_INDEXES[torch_variant]) is not None:
    # Unlike pip, uv gives --extra-index-url priority over the default index
    # and resolves each package from the first index that carries it, so
    # torch is guaranteed to come from the variant index while every other
    # dependency falls back to PyPI.
    install_args += ["--extra-index-url", index]
  if torch_variant == "nightly":
    install_args.append("--prerelease=allow")
  else:
    # An unconstrained resolve may pick a torch newer than any the wheel
    # ships a glue for, which fails to load; pin to the newest supported one.
    install_args.append(f"torch=={_newest_supported_torch()}")
  session.install(*install_args)
  # Mirrors the env of the corresponding bazel target in tests/BUILD: the
  # xla_cpu backend only registers when explicitly allowed.
  env = {"TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND": "1"} if device == "cpu" else {}
  session.run(
      "pytest", f"tests/autoload_{device}_test.py", *session.posargs, env=env
  )
