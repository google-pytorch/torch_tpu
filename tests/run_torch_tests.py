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

"""Runs an unmodified upstream PyTorch test module against the TPU backend.

The module named by TORCH_TPU_INTERNAL_TORCH_TEST_MODULE — "test_torch" or
a subdirectory path like "nn/test_pooling", resolved from the
pytorch/pytorch checkout whose test/ directory is given by
TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR — is executed as ``__main__``, exactly
like ``python test/test_torch.py`` (or ``python test/nn/test_pooling.py``)
upstream. The module's own
``instantiate_device_type_tests(...)`` call materializes device-specific
test classes; the ``tpu`` variants are created through PyTorch's
``PrivateUse1TestBase`` because torch_tpu registers itself as the
PrivateUse1 backend. ``PYTORCH_TESTING_DEVICE_ONLY_FOR=tpu`` (set by the
Bazel target) restricts instantiation to the TPU.

TorchTPU targets the newest PyTorch it can: the capability configuration in
``tests/torch_test_configs.py`` uses ``PrivateUse1TestBase.set_test_configs``
(introduced in torch 2.13) so that variants the backend cannot survive are
never instantiated. On older torch wheels this runner exits early with a
notice instead of failing, so the suite is inert until the repo's torch pin
catches up.

Bazel integration implemented here on top of the upstream unittest runner:
  - ``--test_filter=<pattern>`` is translated into unittest's ``-k``.
  - Bazel test sharding (TEST_TOTAL_SHARDS/TEST_SHARD_INDEX) is implemented
    by deterministically partitioning test method names in the loader, since
    plain unittest does not speak Bazel's sharding protocol.
  - Verbose output is always on: a backend C++ CHECK failure aborts the
    process, and only per-test output lines make the culprit identifiable.
  - unittest's "no tests ran" exit (code 5) is success: capability
    exclusions may legitimately remove every variant a module would
    instantiate for this backend.
"""

import hashlib
import os
import pathlib
import re
import runpy
import sys
import unittest

_MODULE_ENV_VAR = "TORCH_TPU_INTERNAL_TORCH_TEST_MODULE"
_TEST_DIR_ENV_VAR = "TORCH_TPU_INTERNAL_PYTORCH_TEST_DIR"


def _setup_xml_report():
  """Routes upstream --save-xml output into Bazel's XML_OUTPUT_FILE.

  Bazel exposes XML_OUTPUT_FILE and uploads it via BES, which is what makes
  per-test results visible in ResultStore/Build Explorer instead of one
  opaque pass/fail per shard. Upstream run_tests() natively supports JUnit
  XML through --save-xml (backed by unittest-xml-reporting), writing one
  file per test class under a directory; Bazel wants a single file, so the
  returned finalizer merges them.
  """
  xml_output = os.environ.get("XML_OUTPUT_FILE")
  if not xml_output:
    return lambda: None

  xml_dir = pathlib.Path(
      os.environ.get("TEST_TMPDIR", "/tmp"), "upstream_xml_reports"
  )
  xml_dir.mkdir(parents=True, exist_ok=True)
  sys.argv.extend(["--save-xml", str(xml_dir)])

  def merge_reports():
    import xml.etree.ElementTree as ET  # pylint: disable=g-import-not-at-top

    suites = ET.Element("testsuites")
    for report in sorted(xml_dir.rglob("*.xml")):
      try:
        root = ET.parse(report).getroot()
      except ET.ParseError:
        continue
      if root.tag == "testsuite":
        suites.append(root)
      else:  # A <testsuites> wrapper: lift its children.
        suites.extend(list(root))
    if len(suites):
      ET.ElementTree(suites).write(
          xml_output, encoding="unicode", xml_declaration=True
      )

  return merge_reports


def _acknowledge_bazel_sharding():
  """Touches TEST_SHARD_STATUS_FILE so Bazel accepts sharded runs.

  Must happen on every exit path (including the early version-guard exit),
  or Bazel fails the shard with "test runner did not advertise support".
  """
  status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
  if status_file:
    pathlib.Path(status_file).touch()


def _torch_supports_test_configs():
  """True if this torch ships PrivateUse1TestBase.set_test_configs (2.13+)."""
  from torch.testing._internal import common_device_type  # pylint: disable=g-import-not-at-top

  return hasattr(common_device_type.PrivateUse1TestBase, "set_test_configs")


def _install_loader_filters(module):
  """Installs Bazel sharding and crash-exclusion filters in the loader.

  Sharding: with shard_count = N, Bazel starts N independent processes of
  this runner, each told TEST_TOTAL_SHARDS=N and its TEST_SHARD_INDEX.
  Every process discovers the full test list, then this hook filters
  unittest's loader so a test runs iff md5(qualified test id) % N == shard
  index — a deterministic, disjoint, complete partition with no
  coordination between processes. Stdlib ``hash()`` would not work here:
  it is randomized per process, so the shards would disagree about
  ownership. Under CI's parallel_accelerator_execute.sh run_under wrapper,
  each shard process is assigned its own TPU chip, so shards run
  concurrently.

  Exclusion (backstop): ``torch_test_configs.loader_filters(module)``
  supplies (a) the dtype names excluded for this module, dropped by name
  token (``..._tpu_complex64``) so the envelope also holds for
  parametrizers that take dtype lists straight from their decorator
  arguments (``@optims``, ...), and (b) the ``_TEST_NAME_EXCLUSIONS``
  regexes over ``ClassName.test_name`` for tests ``set_test_configs``
  cannot reach. Applied identically in every shard, so the partition
  stays consistent.
  """
  total_shards = int(os.environ.get("TEST_TOTAL_SHARDS", "1"))
  shard_index = int(os.environ.get("TEST_SHARD_INDEX", "0"))

  from tests import torch_test_configs  # pylint: disable=g-import-not-at-top

  dtype_names, patterns = torch_test_configs.loader_filters(module)
  dtype_re = (
      re.compile(r"_(?:" + "|".join(dtype_names) + r")(?:_|$)")
      if dtype_names
      else None
  )
  exclude_re = (
      re.compile("|".join(f"(?:{p})" for p in patterns)) if patterns else None
  )
  if total_shards <= 1 and exclude_re is None and dtype_re is None:
    return

  original = unittest.TestLoader.getTestCaseNames

  def filtered_get_test_case_names(self, test_case_class):
    names = original(self, test_case_class)
    qualname = f"{test_case_class.__module__}.{test_case_class.__qualname__}"

    def keep(name):
      if dtype_re and dtype_re.search(name):
        return False
      if exclude_re and exclude_re.search(
          f"{test_case_class.__name__}.{name}"
      ):
        return False
      if total_shards <= 1:
        return True
      digest = hashlib.md5(f"{qualname}.{name}".encode()).hexdigest()
      return int(digest, 16) % total_shards == shard_index

    return [name for name in names if keep(name)]

  unittest.TestLoader.getTestCaseNames = filtered_get_test_case_names


def main():
  _acknowledge_bazel_sharding()

  module = os.environ.get(_MODULE_ENV_VAR)
  if not module:
    raise EnvironmentError(f"{_MODULE_ENV_VAR} is not set.")
  test_dir = os.environ.get(_TEST_DIR_ENV_VAR)
  if not test_dir:
    raise EnvironmentError(f"{_TEST_DIR_ENV_VAR} is not set.")
  module_file = pathlib.Path(test_dir, f"{module}.py")
  if not module_file.is_file():
    # Note: the bazel sandbox mounts a private /tmp, so a checkout under
    # /tmp is invisible here even if it exists on the host.
    raise FileNotFoundError(
        f"{_TEST_DIR_ENV_VAR}={test_dir} does not contain {module}.py; point"
        " it at the test/ directory of a pytorch/pytorch checkout."
    )
  # Mirror `python test/<...>/x.py`: the CPython launcher puts the script's
  # own directory on sys.path (runpy.run_path does not), and upstream
  # modules import helpers both relative to test/ and to their own
  # directory. Also make argv[0] the script path, which upstream uses to
  # name report directories.
  sys.path.insert(0, test_dir)
  if str(module_file.parent) != test_dir:
    sys.path.insert(0, str(module_file.parent))
  sys.argv[0] = str(module_file)

  # Pre-import torch and torch_tpu to initialize C++ extensions and the TPU
  # runtime before the upstream module imports anything.
  import torch  # pylint: disable=g-import-not-at-top
  import torch_tpu  # pylint: disable=g-import-not-at-top,unused-import  # noqa: F401

  # TorchTPU tracks the newest PyTorch; the capability configuration needs
  # PrivateUse1TestBase.set_test_configs (torch 2.13+). On older wheels the
  # suite is inert: exit early as success rather than failing.
  if not _torch_supports_test_configs():
    print(
        f"NOTE: torch {torch.__version__} predates"
        " PrivateUse1TestBase.set_test_configs (torch 2.13); skipping"
        f" upstream module {module}.",
        file=sys.stderr,
    )
    sys.exit(0)

  # Declare the TorchTPU capability envelope before the upstream module is
  # imported: instantiate_device_type_tests() consumes it at import time.
  from tests import torch_test_configs  # pylint: disable=g-import-not-at-top

  torch_test_configs.install(module)

  # Always run verbose: a backend C++ CHECK failure aborts the process, and
  # only per-test output lines make the culprit identifiable from the log.
  sys.argv.append("-v")

  # Translate Bazel's --test_filter (surfaced as TESTBRIDGE_TEST_ONLY) into
  # unittest's -k so `bazel test --test_filter=<pattern> <target>` works.
  test_filter = os.environ.get("TESTBRIDGE_TEST_ONLY")
  if test_filter:
    sys.argv.extend(["-k", test_filter])

  _install_loader_filters(module)
  merge_xml_reports = _setup_xml_report()

  # Execute the upstream file as a program, exactly like
  # `python test/test_torch.py`. Merely importing it and picking classes out
  # of it would not work: when the file's own
  # `instantiate_device_type_tests(TestFoo, globals())` line runs during
  # import, it (a) generates the concrete per-device classes (TestFooTPU)
  # inside that file's namespace and (b) deletes the TestFoo template and
  # its test methods, so there is no reusable template left to instantiate
  # a second time from here. Running the file with run_name="__main__"
  # also triggers its `if __name__ == "__main__": run_tests()` tail, which
  # is the upstream-supported entry point.
  try:
    runpy.run_path(str(module_file), run_name="__main__")
  except SystemExit as e:
    if e.code == 5:  # unittest: no tests ran.
      print(
          f"NOTE: no tests ran for {module}; all variants are excluded for"
          " this backend by the TorchTPU test configuration.",
          file=sys.stderr,
      )
      sys.exit(0)
    raise
  finally:
    merge_xml_reports()


if __name__ == "__main__":
  main()
