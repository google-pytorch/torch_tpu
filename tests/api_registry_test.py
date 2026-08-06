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

"""Unit tests for PEP-562 lazy loading and @annotations.experimental class decorators."""

import warnings

from absl.testing import absltest
import torch_tpu._loader

# Register torch.tpu device module for unit test environment
torch_tpu._loader._init_device("tpu")


class ApiRegistryTest(absltest.TestCase):

  def test_tpu_profiler_config_export_and_warning_once(self):
    # Test lazy module access resolves class without error
    import torch.tpu.profiler  # pylint: disable=g-import-not-at-top

    cls = torch.tpu.profiler.TpuProfilerConfig
    self.assertIsNotNone(cls)
    self.assertEqual(getattr(cls, "__tt_api_stage__", None), "Experimental")

    # Test single UserWarning on first instantiation
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      cfg1 = torch.tpu.profiler.TpuProfilerConfig()
      self.assertEqual(len(w), 1)
      self.assertTrue(issubclass(w[0].category, UserWarning))
      self.assertIn("TpuProfilerConfig", str(w[0].message))
      self.assertIn("experimental", str(w[0].message))
      self.assertIsNotNone(cfg1)

    # Test second instantiation emits zero warnings
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      cfg2 = torch.tpu.profiler.TpuProfilerConfig()
      self.assertEqual(len(w), 0)
      self.assertIsNotNone(cfg2)


if __name__ == "__main__":
  absltest.main()
