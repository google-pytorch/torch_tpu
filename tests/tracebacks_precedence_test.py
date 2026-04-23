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

os.environ["TORCH_TPU_INTERNAL_ALLOW_XLA_BACKEND"] = "1"

# pylint: disable=g-import-not-at-top
from absl.testing import absltest
import torch
import torch.export
import torch_tpu  # pylint: disable=unused-import
from torch_tpu._internal.export import export as torch_tpu_export


class SimpleModule(torch.nn.Module):

  def forward(self, x):
    return x + 1.0


class TracebackPrecedenceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    torch.device("xla")

  def test_traceback_disabled_by_context_manager_even_if_flag_enabled(self):
    # This test assumes the flag --torch_tpu_internal_mlir_tracebacks=true
    # is set in the BUILD file via test target args.

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
