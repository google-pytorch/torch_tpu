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

"""Unit tests for public TorchTPU API exposure and registration."""

import sys
import warnings

from absl.testing import absltest
import torch
import torch_tpu
from torch_tpu import _loader  # build_cleaner: keep
from torch_tpu._internal.profiler._impl import start_server as internal_start_server
from torch_tpu._internal.profiler._impl import stop_server as internal_stop_server
from torch_tpu._internal.profiler.profiler_config import (
    TpuProfilerConfig as InternalTpuProfilerConfig,
)
from torch_tpu._internal.utils import annotations
from tests import seed_test_utils


class ApiRegistryTest(seed_test_utils.RepeatableTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    _loader._init_device("tpu")

  def test_tpu_profiler_config_accessibility(self):
    """Verifies TpuProfilerConfig is accessible via torch.tpu.profiler and

    torch_tpu.profiler.
    """
    self.assertTrue(hasattr(torch, "tpu"))
    self.assertTrue(hasattr(torch.tpu, "profiler"))
    self.assertTrue(hasattr(torch.tpu.profiler, "TpuProfilerConfig"))
    self.assertTrue(hasattr(torch_tpu.profiler, "TpuProfilerConfig"))
    self.assertIs(sys.modules.get("torch.tpu.profiler"), torch_tpu.profiler)

  def test_tpu_profiler_config_identity(self):
    """Verifies identity between public export and internal class definition."""
    self.assertIs(
        torch.tpu.profiler.TpuProfilerConfig, InternalTpuProfilerConfig
    )
    self.assertIs(
        torch_tpu.profiler.TpuProfilerConfig, InternalTpuProfilerConfig
    )

  def test_invalid_attribute_raises(self):
    """Verifies accessing non-existent attributes on proxy module raises

    AttributeError.
    """
    with self.assertRaises(AttributeError):
      _ = getattr(torch_tpu.profiler, "non_existent_attribute")

  def test_tpu_profiler_config_metadata(self):
    """Verifies lifecycle stage metadata attached to TpuProfilerConfig."""
    cls = torch.tpu.profiler.TpuProfilerConfig
    self.assertEqual(
        getattr(cls, annotations.TT_API_STAGE, None),
        annotations.Stage.EXPERIMENTAL.value,
    )
    self.assertEqual(
        getattr(cls, annotations.TT_API_STAGE_REASON, None),
        "torch.tpu.profiler.TpuProfilerConfig is experimental and subject to"
        " change.",
    )

  def test_tpu_profiler_config_instantiation_warning(self):
    """Verifies instantiating TpuProfilerConfig triggers UserWarning."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      torch.tpu.profiler.TpuProfilerConfig()
      self.assertNotEmpty(w)
      experimental_warnings = [
          item for item in w if issubclass(item.category, UserWarning)
      ]
      self.assertLen(experimental_warnings, 1)
      self.assertIn(
          "TpuProfilerConfig is experimental",
          str(experimental_warnings[0].message),
      )

  def test_profiler_module_dir(self):
    """Verifies dir(torch.tpu.profiler) returns public symbol list."""
    self.assertEqual(dir(torch.tpu.profiler), ["TpuProfilerConfig"])
    self.assertEqual(dir(torch_tpu.profiler), ["TpuProfilerConfig"])

  def test_server_accessibility(self):
    """Verifies start_server and stop_server are accessible via public modules."""
    self.assertTrue(hasattr(torch, "tpu"))
    self.assertTrue(hasattr(torch.tpu, "server"))
    self.assertTrue(hasattr(torch.tpu.server, "start_server"))
    self.assertTrue(hasattr(torch.tpu.server, "stop_server"))
    self.assertTrue(hasattr(torch_tpu.server, "start_server"))
    self.assertTrue(hasattr(torch_tpu.server, "stop_server"))
    self.assertIs(sys.modules.get("torch.tpu.server"), torch_tpu.server)

  def test_server_identity(self):
    """Verifies identity between public export and internal function definitions."""
    self.assertIs(torch.tpu.server.start_server, internal_start_server)
    self.assertIs(torch.tpu.server.stop_server, internal_stop_server)
    self.assertIs(torch_tpu.server.start_server, internal_start_server)
    self.assertIs(torch_tpu.server.stop_server, internal_stop_server)

  def test_server_invalid_attribute_raises(self):
    """Verifies accessing non-existent attributes raises AttributeError."""
    with self.assertRaises(AttributeError):
      _ = getattr(torch_tpu.server, "non_existent_attribute")

  def test_server_metadata(self):
    """Verifies lifecycle stage metadata attached to start_server and stop_server."""
    for fn, name in [
        (torch.tpu.server.start_server, "start_server"),
        (torch.tpu.server.stop_server, "stop_server"),
    ]:
      self.assertEqual(
          getattr(fn, annotations.TT_API_STAGE, None),
          annotations.Stage.EXPERIMENTAL.value,
      )
      self.assertEqual(
          getattr(fn, annotations.TT_API_STAGE_REASON, None),
          f"torch.tpu.server.{name} is experimental and subject to change.",
      )

  def test_server_warning(self):
    """Verifies calling server APIs triggers UserWarning."""
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      with self.assertRaises(ValueError):
        torch.tpu.server.stop_server()
      self.assertNotEmpty(w)
      experimental_warnings = [
          item for item in w if issubclass(item.category, UserWarning)
      ]
      self.assertLen(experimental_warnings, 1)
      self.assertIn(
          "stop_server is experimental",
          str(experimental_warnings[0].message),
      )

  def test_server_module_dir(self):
    """Verifies dir(torch.tpu.server) returns public symbol list."""
    self.assertEqual(
        sorted(dir(torch.tpu.server)), ["start_server", "stop_server"]
    )
    self.assertEqual(
        sorted(dir(torch_tpu.server)), ["start_server", "stop_server"]
    )


if __name__ == "__main__":
  absltest.main()
