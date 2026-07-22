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

"""End-to-end test of version dispatch against a real extension module.

The unit tests call the selection functions directly; this test exercises the
whole import path instead. A real ELF extension (`dispatch_probe_module.c`,
which exports only the unversioned `PyInit_dispatch_probe`) is laid out under
version-suffixed filenames the way the multi-ABI wheel bundles its glues, the
hook is installed against that directory, and a plain `import` must yield a
live module created from the artifact matching the (faked) installed PyTorch
-- proving the `MetaPathFinder` redirect and the `ExtensionProxyLoader` name
bridge work against CPython's actual extension loading, not just in
unit-tested isolation.

The test deliberately depends on nothing but the loader module: the
`torch_tpu._internal` parent package is a stub whose `__path__` points at the
temp directory of fabricated glues, and `torch` is a fake carrying only
`__version__`. That keeps the full import machinery real while the test stays
runnable without building any real extension but the probe.
"""

from collections.abc import Sequence
import importlib
import os
import pathlib
import shutil
import sys
import types
from unittest import mock

from absl.testing import absltest
from torch_tpu import _versioned_so_loader as loader

_MODULE = "dispatch_probe"
_IMPORT_NAME = f"torch_tpu._internal.{_MODULE}"


class DispatchIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._probe_so = pathlib.Path(os.environ["DISPATCH_PROBE_SO"])
    self._glue_dir = pathlib.Path(self.create_tempdir().full_path)

    # Snapshot sys.modules for the duration of the test: everything below
    # (the stub parent package, the fake torch, the imported probe modules)
    # is rolled back on exit.
    patcher = mock.patch.dict(sys.modules)
    patcher.start()
    self.addCleanup(patcher.stop)

    # A stub parent package whose __path__ points at the fabricated glue
    # directory, so the finder's PathFinder lookup (which searches the parent
    # package's __path__) sees the versioned artifacts.
    internal_pkg = types.ModuleType("torch_tpu._internal")
    internal_pkg.__path__ = [str(self._glue_dir)]
    sys.modules["torch_tpu._internal"] = internal_pkg

    # install_hook() mutates the meta path and the loader's resolved suffix;
    # restore both so test cases stay independent.
    saved_meta_path = list(sys.meta_path)
    saved_suffix = loader._VERSION_SUFFIX

    def restore():
      sys.meta_path[:] = saved_meta_path
      loader._VERSION_SUFFIX = saved_suffix

    self.addCleanup(restore)

  def _install(self, torch_version: str, built_suffixes: Sequence[str]) -> None:
    """Lays out versioned artifacts and installs the hook for a fake torch."""
    for suffix in built_suffixes:
      shutil.copy(self._probe_so, self._glue_dir / f"{_MODULE}_{suffix}.so")
    sys.modules["torch"] = types.SimpleNamespace(__version__=torch_version)
    loader.install_hook(self._glue_dir)

  def test_import_loads_the_glue_matching_the_running_torch(self):
    # 2.12.5 postdates the 2.12.0 glue and predates the 9.99.0 one, so the
    # 2.12.0 artifact must be the one loaded.
    self._install("2.12.5", ["2_11_0", "2_12_0", "9_99_0"])

    module = importlib.import_module(_IMPORT_NAME)

    self.assertEqual(module.__name__, _IMPORT_NAME)
    self.assertTrue(
        module.__file__.endswith(f"{_MODULE}_2_12_0.so"),
        f"expected the 2_12_0 artifact, got {module.__file__}",
    )
    self.assertIs(sys.modules[_IMPORT_NAME], module)

  def test_import_falls_back_to_the_newest_glue(self):
    # A torch newer than every glue is served by the newest one.
    self._install("99.0.0", ["2_11_0", "9_99_0"])

    module = importlib.import_module(_IMPORT_NAME)

    self.assertTrue(
        module.__file__.endswith(f"{_MODULE}_9_99_0.so"),
        f"expected the 9_99_0 artifact, got {module.__file__}",
    )

  def test_module_is_reimportable_under_its_unversioned_name(self):
    self._install("2.11.3", ["2_11_0"])

    first = importlib.import_module(_IMPORT_NAME)
    second = importlib.import_module(_IMPORT_NAME)

    self.assertIs(first, second)


if __name__ == "__main__":
  absltest.main()
