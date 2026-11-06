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

"""Tests to verify that the traceback flag doesn't override the context manager."""

import os

from absl.testing import absltest
import torch
import torch.export
from torch_tpu._internal.export import export as torch_tpu_export
from tests import seed_test_utils


class SimpleModule(torch.nn.Module):

  def forward(self, x):
    return x + 1.0


class TracebackPrecedenceTest(seed_test_utils.RepeatableTest):

  # --torch_tpu_internal_mlir_tracebacks=true is set in the BUILD file, as the
  # flag is memoized on first use. The flag is only meant for Google internal
  # use, so this test isn't run in OSS.

  def test_traceback_enabled_by_flag(self):
    sample_input = (torch.tensor([0.1, 0.2, 0.3]),)
    exported = torch.export.export(SimpleModule(), args=sample_input)

    exported_mlir = torch_tpu_export.exported_to_mlir(exported)

    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Expect python filenames since tracebacks are enabled by the flag.
    current_filename = os.path.basename(__file__)
    self.assertIn(current_filename, mlir_text)

  def test_traceback_disabled_by_context_manager_even_if_flag_enabled(self):
    sample_input = (torch.tensor([0.1, 0.2, 0.3]),)
    exported = torch.export.export(SimpleModule(), args=sample_input)

    # Context manager overrides the flag to False
    with torch_tpu_export.enable_tracebacks(False):
      exported_mlir = torch_tpu_export.exported_to_mlir(exported)

    mlir_text = exported_mlir.serialize_text(enable_debug_info=True)

    # Expect no python filenames if tracebacks are disabled by the context
    # manager.
    current_filename = os.path.basename(__file__)
    self.assertNotIn(current_filename, mlir_text)
    self.assertNotRegex(mlir_text, r"loc.*\.py")


if __name__ == "__main__":
  absltest.main()
