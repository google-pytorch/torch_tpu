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

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint"]


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
