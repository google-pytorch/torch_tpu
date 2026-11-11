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

r"""Checks that a documented direct run of an example resolves its imports.

Example docs tell users to run scripts directly from the example's directory,
e.g. `python3 ./flux_test.py`. Bazel tests of those scripts can pass even when
such runs are broken, because Bazel puts every examples/ package on the Python
import path. This runs one documented command block from the example's
directory with an import path that has the test's dependencies (torch_tpu,
torch, absl, ...) but no examples/ directory, as for a user who installed the
torch_tpu wheel.

To mark a command block in a markdown file, put an HTML comment (which GitHub
doesn't render) with the name of the doc test target on the line before the
block's opening fence:

    <!-- doc-test: flux_doc_test -->

    ```bash
    python3 ./flux_test.py
    ```

The block runs with `bash -e` from the directory of the markdown file, with
`python3` and `python` replaced by a shim that only imports the given script:
it runs the script's top level, as `python3 <script>.py` would, but not its
`if __name__ == "__main__":` part. This checks the imports without running the
example, and needs no TPU. The shim supports only `python3 <script>.py [args]`.

Usage: run_doc_commands.py <markdown file> <doc test name>
"""

from collections.abc import Sequence, Set
import os
import re
import runpy
import shlex
import subprocess
import sys
import tempfile

# A marker line, optional blank lines, then a fenced code block.
_MARKED_BLOCK_RE = re.compile(
    r"^<!-- doc-test: (?P<id>[\w-]+) -->[ \t]*\n(?:[ \t]*\n)*"
    r"```[^\n]*\n(?P<commands>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

# The test's import path, which the shim appends to PYTHONPATH.
_IMPORT_PATH_ENV = "TORCH_TPU_DOC_TEST_IMPORT_PATH"
_SHIM_FLAG = "--import_script"

# Runs this file with _SHIM_FLAG. `-S` keeps `site` from adding the excluded
# directories back from the test virtualenv's .pth files.
_PYTHON_SHIM = """#!/bin/bash
export PYTHONPATH="${{PYTHONPATH:+$PYTHONPATH:}}${env}"
exec {python} -S {runner} {flag} "$@"
"""


def _import_script(args: list[str]) -> int:
  """Runs the top level of `args[0]` as `python3 <args>` would."""
  if not args or not args[0].endswith(".py"):
    print(
        "doc-test supports only `python3 <script>.py [args]`, got:"
        f" python3 {shlex.join(args)}",
        file=sys.stderr,
    )
    return 1
  sys.argv = args
  sys.path[0] = os.path.dirname(os.path.abspath(args[0]))
  runpy.run_path(args[0], run_name="__doc_test__")
  print(f"Imported {args[0]}.")
  return 0


def _forms(path: str) -> Set[str]:
  return {os.path.abspath(path), os.path.realpath(path)}


def _import_path(excluded_tree: str) -> Sequence[str]:
  """Returns this test's import path without the excluded directory tree.

  The test's own virtualenv (`site-packages`), which Bazel may put under the
  test's package, is kept.
  """
  trees = _forms(excluded_tree)
  return [
      entry
      for entry in sys.path
      if entry
      and not any(
          "site-packages" not in form.split(os.sep)
          and (form == t or form.startswith(t + os.sep))
          for form in _forms(entry)
          for t in trees
      )
  ]


def main(argv: Sequence[str]) -> int:
  if len(argv) > 1 and argv[1] == _SHIM_FLAG:
    return _import_script(argv[2:])
  if len(argv) != 3:
    print(__doc__, file=sys.stderr)
    return 2
  doc, block_id = argv[1:]

  with open(doc) as f:
    blocks = [
        m["commands"]
        for m in _MARKED_BLOCK_RE.finditer(f.read())
        if m["id"] == block_id
    ]
  if len(blocks) != 1:
    if blocks:
      print(
          f"More than one block marked `<!-- doc-test: {block_id} -->` in"
          f" {doc}.",
          file=sys.stderr,
      )
    else:
      print(
          f"No block marked `<!-- doc-test: {block_id} -->` in {doc}.",
          file=sys.stderr,
      )
    return 1
  commands = blocks[0]

  # The test runs from the repo root in runfiles. Drop the top-level directory
  # holding the doc (e.g. examples/) from the import path.
  import_path = _import_path(
      os.path.join(os.getcwd(), os.path.normpath(doc).split(os.sep)[0])
  )
  print("Import path:", *import_path, sep="\n  ")
  env = dict(os.environ)
  env.pop("PYTHONPATH", None)
  # The shim replaces sys.path[0], which Python only sets without this.
  env.pop("PYTHONSAFEPATH", None)
  env[_IMPORT_PATH_ENV] = os.pathsep.join(import_path)

  with tempfile.TemporaryDirectory() as shim_dir:
    for name in ("python3", "python"):
      shim = os.path.join(shim_dir, name)
      with open(shim, "w") as f:
        f.write(
            _PYTHON_SHIM.format(
                env=_IMPORT_PATH_ENV,
                python=shlex.quote(sys.executable),
                runner=shlex.quote(os.path.abspath(__file__)),
                flag=_SHIM_FLAG,
            )
        )
      os.chmod(shim, 0o755)
    env["PATH"] = shim_dir + os.pathsep + env.get("PATH", "")

    print(f"==== Running block {block_id} of {doc}:\n{commands}", flush=True)
    result = subprocess.run(
        ["bash", "-e", "-c", commands],
        cwd=os.path.dirname(os.path.abspath(doc)),
        env=env,
        check=False,
    )
  if result.returncode != 0:
    print(f"FAILED: block {block_id} of {doc}", file=sys.stderr)
  return result.returncode


if __name__ == "__main__":
  sys.exit(main(sys.argv))
