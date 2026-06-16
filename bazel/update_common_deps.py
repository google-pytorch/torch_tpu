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

"""Updates the common_deps.bzl file with dynamic dependencies."""

import operator
import os
import pathlib
import subprocess
from typing import Final

COPYRIGHT_HEADER: Final[str] = """\
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
"""

WARNING_HEADER: Final[str] = '''\
"""
DO NOT EDIT: This file is auto-generated.
To update, run:
  blaze run -c opt //bazel:update_common_deps

This file contains the list of PyTorch-agnostic XLA/MLIR dependencies
that are used to build the common PyTorch/XLA glue libraries.
"""\
'''

BAZEL_EXECUTABLE: Final[str] = "bazel"
BAZEL_QUERY: Final[str] = """\
let direct_deps = deps(//torch_tpu/...:all, 1) except //torch_tpu/...:all in
visible(
    //torch_tpu/common:pywrap_torch_tpu_deps,
    attr("testonly", 0, filter("^@", kind("cc_library", $direct_deps)))
)\
"""


def is_pytorch_dep(target: str) -> bool:
  return (
      target.startswith("@local_torch")
      or target.startswith("@pypi_torch")
      or target.startswith("@pypi//torch")
  )


def main():

  print("Running bazel query...")
  output = subprocess.check_output(
      [BAZEL_EXECUTABLE, "query", BAZEL_QUERY],
      cwd=os.environ.get("BUILD_WORKSPACE_DIRECTORY"),
      text=True,
  )

  targets = list(
      filter(None, map(operator.methodcaller("strip"), output.splitlines()))
  )
  targets = [
      t
      for t in targets
      if not t.startswith("//bazel_rules/") and not is_pytorch_dep(t)
  ]
  targets.sort()

  out_file = pathlib.Path(__file__).parent / "common_deps.bzl"
  with open(out_file, "w") as f:
    f.write(COPYRIGHT_HEADER)
    f.write(WARNING_HEADER + "\n\n")
    f.write("COMMON_DEPS = [\n")
    for t in targets:
      f.write(f'    "{t}",\n')
    f.write("]\n")


if __name__ == "__main__":
  main()
