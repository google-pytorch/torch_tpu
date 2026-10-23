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

"""Test error handling for dynamic execution through torch.compile()."""

from unittest import mock

from absl.testing import absltest
import torch
from torch_tpu._internal.compile import _backend
from tests import seed_test_utils


class DynamicErrorHandlingTest(seed_test_utils.RepeatableTest):

  def test_mlir_lowering_failure_raises_not_implemented_error(self):
    def simple(x):
      return x + 1.0

    with mock.patch.object(
        _backend.compiler.StaticCompiler,
        "__call__",
        side_effect=RuntimeError("MLIR lowering failed"),
    ):
      compiled = torch.compile(simple, backend="tpu")
      device = torch.accelerator.current_accelerator()
      t = torch.ones(4, device=device)
      torch._dynamo.mark_dynamic(t, 0, min=2, max=8)
      with self.assertRaises(
          (NotImplementedError, torch._dynamo.exc.BackendCompilerFailed)
      ) as ctx:
        compiled(t)
      self.assertIn(
          "torch.compile(..., dynamic=False, ...)", str(ctx.exception)
      )
      self.assertIn("MLIR lowering failed", str(ctx.exception))


if __name__ == "__main__":
  absltest.main()
