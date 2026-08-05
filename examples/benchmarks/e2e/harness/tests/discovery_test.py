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

"""Tests for harness/discovery.py."""

import importlib
import pathlib
import sys
import tempfile
import textwrap

from absl.testing import absltest
from examples.benchmarks.e2e.harness import discovery as discovery_lib
from examples.benchmarks.e2e.harness import registry as registry_lib
from tests import seed_test_utils


class DiscoveryTest(seed_test_utils.RepeatableTest):

  def _make_pkg(self, modules: dict[str, str]):
    """Makes a throwaway python package for testing import side-effects."""
    tmp = tempfile.mkdtemp()
    pkg_name = f"fake_models_{pathlib.Path(tmp).name}"
    pkg_dir = pathlib.Path(tmp) / pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    for mod_name, body in modules.items():
      (pkg_dir / f"{mod_name}.py").write_text(textwrap.dedent(body))
    sys.path.insert(0, tmp)
    self.addCleanup(lambda: sys.path.remove(tmp))

    pkg = importlib.import_module(pkg_name)
    self.addCleanup(
        lambda: [
            sys.modules.pop(m, None)
            for m in list(sys.modules)
            if m.startswith(pkg_name)
        ]
    )
    return pkg

  def test_imports_all_submodules(self):
    pkg = self._make_pkg({"a": "LOADED='a'", "b": "LOADED='b'"})
    self.assertEqual(discovery_lib.import_submodules(pkg), [])
    self.assertIn(f"{pkg.__name__}.a", sys.modules)
    self.assertIn(f"{pkg.__name__}.b", sys.modules)

  def test_broken_module_surfaced_not_swallowed(self):
    pkg = self._make_pkg(
        {"good": "LOADED=True", "broken": "raise RuntimeError('boom')"}
    )
    failures = discovery_lib.import_submodules(pkg)
    self.assertLen(failures, 1)
    self.assertIsInstance(failures[0], discovery_lib.ImportFailure)
    self.assertIn("broken", failures[0].module)
    self.assertIn("boom", str(failures[0]))

  def test_one_broken_module_does_not_block_others(self):
    pkg = self._make_pkg(
        {"broken": "raise RuntimeError('boom')", "good": "LOADED=True"}
    )
    discovery_lib.import_submodules(pkg)
    self.assertIn(f"{pkg.__name__}.good", sys.modules)

  def test_registration_happens_via_import(self):
    saved = dict(registry_lib.REGISTRY)
    registry_lib.REGISTRY.clear()
    try:
      pkg = self._make_pkg({
          "m": (
              """
              from examples.benchmarks.e2e.harness import registry as registry_lib
              from examples.benchmarks.e2e.harness import step_lib

              @registry_lib.register_benchmark(
                  stepper=step_lib.StepperType.FORWARD
              )
              def discovered_probe():
                return ("m", (), {}, None)
              """
          ),
      })
      self.assertEqual(discovery_lib.import_submodules(pkg), [])
      self.assertIn("discovered_probe", registry_lib.REGISTRY)
    finally:
      registry_lib.REGISTRY.clear()
      registry_lib.REGISTRY.update(saved)


if __name__ == "__main__":
  absltest.main()
