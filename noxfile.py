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
def format(session: nox.Session) -> None:  # pylint: disable=redefined-builtin
  """Apply clang-format fixes to modified files automatically."""
  session.install("clang-format")
  session.run("ci/tools/clang_format.sh", "format", external=True)


# Use venv_backend="none" to allow running the script directly in the host.
# The script relies exclusively on Python standard libraries, so creating
# a virtual env adds unnecessary overhead with no benefits.
@nox.session(venv_backend="none")
def refresh_compile_commands(session: nox.Session) -> None:
  """Refresh compile_commands.json for clangd."""
  session.run(
      "./setup_clangd.py",
      *session.posargs,
      external=True,
  )


def _load_pytorch_versions() -> Mapping[str, object]:
  """Load the torch-version source of truth from bazel/pytorch_versions.bzl.

  That file is deliberately data-only (plain assignments, no `load()`)
  precisely so that it can be parsed as Python as well as Starlark.

  Returns:
    A dictionary equivalent to globals() executed in bazel/pytorch_version.bzl
  """
  namespace: dict[str, object] = {}
  bzl = pathlib.Path(__file__).parent / "bazel" / "pytorch_versions.bzl"
  exec(bzl.read_text(), namespace)  # pylint: disable=exec-used
  return namespace


_PYTORCH_VERSIONS: Final[Mapping[str, object]] = _load_pytorch_versions()
_WHEEL_TORCH_VERSIONS: Final[Sequence[str]] = _PYTORCH_VERSIONS[
    "WHEEL_TORCH_VERSIONS"
]
_INTERMEDIATE_PATCH_VERSIONS: Final[Sequence[str]] = _PYTORCH_VERSIONS[
    "INTERMEDIATE_PATCH_VERSIONS"
]


def _newest_supported_torch() -> str:
  """Return the newest PyTorch version the wheel ships a glue for."""
  return max(_WHEEL_TORCH_VERSIONS, key=lambda v: tuple(map(int, v.split("."))))


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

  Args:
    session: the nox session the checks run in.
    device: the accelerator the smoke test targets (`cpu` or `tpu`).
    torch_variant: the torch build to install, a `_TORCH_INDEXES` key.
  """
  install_args: list[str] = []
  if (index := _TORCH_INDEXES[torch_variant]) is not None:
    # Unlike pip, uv gives --extra-index-url priority over the default index
    # and resolves each package from the first index that carries it, so
    # torch is guaranteed to come from the variant index while every other
    # dependency falls back to PyPI.
    install_args += ["--extra-index-url", index]
  if torch_variant == "nightly":
    install_args.append("--prerelease=allow")
  else:
    # Pin so the session is deterministic and exercises the newest glue as an
    # exact match. Releases without an exact glue are covered by the
    # intermediate_patch_abi sessions, and newer-than-any-glue by the nightly
    # variant (the dispatch loader serves both from the nearest older glue).
    install_args.append(f"torch=={_newest_supported_torch()}")
  _install_and_run_smoke_tests(session, device, *install_args)


@nox.session
@nox.parametrize("torch_version", _INTERMEDIATE_PATCH_VERSIONS)
def intermediate_patch_abi(session: nox.Session, torch_version: str) -> None:
  """Run the smoke tests against an intermediate PyTorch release.

  The wheel ships a glue only for the PyTorch releases that broke ABI
  compatibility with the previous glue; the dispatch loader serves every other
  release from the newest older glue (e.g. a torch 2.12.0 install from the
  2.11.0 glue). This session proves that compatibility in practice: it
  installs the intermediate release alongside the built wheel and runs the
  CPU smoke tests against it. The glue is linked with `-z now`, so an
  ABI-incompatible glue fails the autoload import outright with an
  undefined-symbol error -- which means the release broke ABI and needs a
  glue of its own in WHEEL_TORCH_VERSIONS.

  Args:
    session: the nox session the checks run in.
    torch_version: the intermediate PyTorch release to install and test.
  """
  _install_and_run_smoke_tests(
      session,
      "cpu",
      f"torch=={torch_version}",
      "--extra-index-url",
      _TORCH_INDEXES["cpu"],
  )


def _install_and_run_smoke_tests(
    session: nox.Session, device: str, *install_args: str
) -> None:
  """Install the dist/ wheel plus install_args, then run the autoload tests."""
  session.install(str(_dist_wheel(session)), "pytest>=9", *install_args)
  # Mirrors the env of the corresponding bazel target in tests/BUILD: the
  # xla_cpu backend only registers when explicitly allowed.
  env = {"TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND": "1"} if device == "cpu" else {}
  session.run(
      "pytest", f"tests/autoload_{device}_test.py", *session.posargs, env=env
  )
