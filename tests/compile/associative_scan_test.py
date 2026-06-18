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

"""Tests lowering torch.associative_scan to the native TPU scan emitter."""

import os

from absl.testing import absltest
import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch_tpu._internal import compile as compile_lib
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import utils


def _add(a, b):
  return a + b


def _mul(a, b):
  return a * b


def _max(a, b):
  return torch.maximum(a, b)


def _min(a, b):
  return torch.minimum(a, b)


def _logaddexp(a, b):
  return torch.log(torch.exp(a) + torch.exp(b))


class AssociativeScanTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    # Clear dynamo's compile cache so each test recompiles through its own
    # backend; otherwise a cross-test cache hit leaves _compiled_executables
    # empty and the mlir_text assertions flake.
    torch._dynamo.reset()
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"

  def _compile_and_run(self, combine_fn, xs_cpu, dim=0, reverse=False):
    """Compiles associative_scan on TPU and returns (cpu_result, backend)."""

    def func(xs):
      # combine_mode="pointwise" emits the associative_scan higher-order op,
      # which the TorchTPU FX interpreter lowers to chlo.ScanOp.
      return associative_scan(
          combine_fn, xs, dim=dim, reverse=reverse, combine_mode="pointwise"
      )

    xs_tpu = xs_cpu.to(torch.device("tpu"))
    tpu_backend = compile_lib.TpuBackend(debug=True)
    compiled = torch.compile(func, backend=tpu_backend, dynamic=False)
    result = _backend.to_device(compiled(xs_tpu), "cpu")
    return result, tpu_backend

  def test_cumsum_1d(self):
    xs = torch.arange(1, 9, dtype=torch.float32)
    result, backend = self._compile_and_run(_add, xs)
    utils.assert_close(result, torch.cumsum(xs, dim=0))
    # Confirm it routed to the native scan op, not a while loop.
    self.assertIn("chlo.scan", backend._compiled_executables[0].mlir_text)

  def test_cumprod_1d(self):
    xs = torch.arange(1, 9, dtype=torch.float32)
    result, _ = self._compile_and_run(_mul, xs)
    utils.assert_close(result, torch.cumprod(xs, dim=0), rtol=1e-3, atol=1e-3)

  def test_cummax_1d(self):
    xs = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    result, _ = self._compile_and_run(_max, xs)
    utils.assert_close(result, torch.cummax(xs, dim=0).values)

  def test_cumsum_2d_dim1(self):
    xs = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    result, _ = self._compile_and_run(_add, xs, dim=1)
    utils.assert_close(result, torch.cumsum(xs, dim=1))

  def test_cumsum_reverse(self):
    xs = torch.arange(1, 9, dtype=torch.float32)
    result, _ = self._compile_and_run(_add, xs, reverse=True)
    expected = torch.flip(torch.cumsum(torch.flip(xs, [0]), dim=0), [0])
    utils.assert_close(result, expected)

  def test_cummin_1d(self):
    # All-positive inputs: out[0] must equal xs[0] > 0, so a spurious init=0
    # folded as the scan prefix (min(0, xs[0]) = 0) would be caught. Confirms
    # the lowering seeds the carry with the first element, not a fake identity.
    xs = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    result, _ = self._compile_and_run(_min, xs)
    utils.assert_close(result, torch.cummin(xs, dim=0).values)

  def test_cummax_negative_1d(self):
    # All-negative inputs: out[0] must equal xs[0] < 0, so a spurious init=0
    # folded as the scan prefix (max(0, xs[0]) = 0) would be caught. Confirms
    # the carry is seeded with the first element, not a zero/identity constant.
    xs = torch.tensor([-3.0, -1.0, -4.0, -1.0, -5.0, -9.0, -2.0, -6.0])
    result, _ = self._compile_and_run(_max, xs)
    utils.assert_close(result, torch.cummax(xs, dim=0).values)

  def test_logaddexp_custom_combiner_1d(self):
    # logaddexp has no zero/finite identity, so it exercises the first-element
    # seeding path: scanning the rest with a fabricated init would fold it into
    # every output. associative_scan(logaddexp) is exactly logcumsumexp.
    xs = torch.tensor([0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 3.0, -2.0])
    result, backend = self._compile_and_run(_logaddexp, xs)
    utils.assert_close(
        result, torch.logcumsumexp(xs, dim=0), rtol=1e-3, atol=1e-3
    )
    # A custom combiner must still route to the native scan op.
    self.assertIn("chlo.scan", backend._compiled_executables[0].mlir_text)

  def test_logaddexp_custom_combiner_2d_dim1(self):
    xs = torch.tensor(
        [[0.5, -1.0, 2.0, 0.0], [1.5, -0.5, 3.0, -2.0], [0.1, 0.2, 0.3, 0.4]]
    )
    result, _ = self._compile_and_run(_logaddexp, xs, dim=1)
    utils.assert_close(
        result, torch.logcumsumexp(xs, dim=1), rtol=1e-3, atol=1e-3
    )

  def test_cumsum_bf16_1d(self):
    xs = torch.arange(1, 9, dtype=torch.bfloat16)
    result, _ = self._compile_and_run(_add, xs)
    expected = torch.cumsum(xs.to(torch.float32), dim=0).to(torch.bfloat16)
    utils.assert_close(result, expected, rtol=1e-2, atol=1e-1)


if __name__ == "__main__":
  absltest.main()
