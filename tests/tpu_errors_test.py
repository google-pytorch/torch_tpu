# Copyright 2025 Google LLC
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

"""Tests error handling on TPU only."""

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import torch
from tests import error_testing as et


def setUpModule():
  """Called by absltest after flags are parsed and before tests are run."""
  et.set_up_module()


def _get_aminmax_outputs(
    op: Any, device: str, dtype: torch.dtype
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
  """Creates the output tensors for `op` of `dtype`, on `device`.

  Args:
    op: a torch function. Should be one of: amax, amin, or aminmax.
    device: the device of the output tensors
    dtype: the dtype of the output tensors

  Returns:
    The out object to be used as a kwarg when calling `op`.
  """

  assert op in (torch.amax, torch.amin, torch.aminmax)

  def scalar_tensor() -> torch.Tensor:
    return torch.tensor(0, device=device, dtype=dtype)

  return (
      scalar_tensor()
      if op != torch.aminmax
      else (scalar_tensor(), scalar_tensor())
  )


def _make_lu_unpack_outputs(
    p: tuple[int, ...], l: tuple[int, ...], u: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
  """Creates a 3-tuple of tensors for `lu_unpack()` op."""
  return (
      torch.empty(p, device=et.device()),
      torch.empty(l, device=et.device()),
      torch.empty(u, device=et.device()),
  )


class TpuOnlyErrorTest(et.TpuOnlyErrorTestBase, parameterized.TestCase):
  """Tests error messages on TPU."""

  def test_ones_unsupported_dtype(self):
    with et.assert_raises_message(
        NotImplementedError,
        # PyTorch implements ones() in terms of empty() automatically, so we
        # don't have a good way to know that the op in question is "ones"
        # instead of "empty".
        tpu="""empty(): TorchTPU does not yet support dtype int1""",
        message_reviewed_by="wan",
    ):
      # This succeeds on CPU but is unimplemented on TPU.
      torch.ones(2, 3, device=et.device(), dtype=torch.int1)

  def test_scaled_mm_invalid_shapes(self):
    """Tests that scaled_mm fails if matrix sizes are not divisible by 16."""
    device = "tpu"
    # Generate in F32 and cast to FP8 to avoid randn failure on TPU!
    mat1 = torch.randn(15, 16, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    mat2 = torch.randn(16, 32, dtype=torch.float32, device=device).to(
        torch.float8_e4m3fn
    )
    scale_a = torch.tensor([1.0], dtype=torch.float32, device=device)
    scale_b = torch.tensor([1.0], dtype=torch.float32, device=device)

    with et.assert_raises_message(
        RuntimeError,
        tpu="""scaled_mm(): expected matrices sizes to be divisible by 16, got shapes [15, 16] and [16, 32]""",
    ):
      torch._scaled_mm(mat1, mat2, scale_a, scale_b)


if __name__ == "__main__":
  absltest.main()
