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

"""Exact signed integer matrix multiplication tests."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch_tpu  # pylint: disable=unused-import  # noqa: F401
from tests import seed_test_utils


class IntMmTest(seed_test_utils.RepeatableTest):
  def test_registered(self):
    for name in ("aten::_int_mm", "aten::_int_mm.out"):
      self.assertTrue(
        torch._C._dispatch_has_kernel_for_dispatch_key(name, "PrivateUse1")
      )

  @parameterized.parameters(
    (1, 1, 1),
    (1, 7, 9),
    (5, 1, 3),
    (3, 5, 1),
    (3, 5, 7),
    (16, 257, 65),
    (128, 128, 128),
    (0, 7, 5),
    (4, 0, 5),
    (4, 7, 0),
    (0, 0, 0),
  )
  def test_exact(self, m, k, n):
    a = torch.randint(-128, 128, (m, k), dtype=torch.int8)
    b = torch.randint(-128, 128, (k, n), dtype=torch.int8)
    expected = (a.to(torch.int64) @ b.to(torch.int64)).to(torch.int32)
    x, y = a.to("tpu"), b.to("tpu")
    out = torch.empty((m, n), dtype=torch.int32, device="tpu")
    for result in (
      torch._int_mm(x, y),
      torch.ops.aten._int_mm.out(x, y, out=out),
    ):
      self.assertEqual(result.dtype, torch.int32)
      self.assertEqual(result.device.type, "tpu")
      self.assertEqual(result.shape, (m, n))
      self.assertTrue(torch.equal(result.cpu(), expected))

  def test_no_float_roundtrip(self):
    a = torch.full((1, 1041), 127, dtype=torch.int8, device="tpu")
    b = torch.full((1041, 1), 127, dtype=torch.int8, device="tpu")
    self.assertEqual(torch._int_mm(a, b).cpu().item(), 16790289)

  def test_transposed_inputs(self):
    a = torch.randint(-128, 128, (7, 3), dtype=torch.int8)
    b = torch.randint(-128, 128, (5, 7), dtype=torch.int8)
    expected = (a.t().to(torch.int64) @ b.t().to(torch.int64)).to(torch.int32)
    self.assertTrue(
      torch.equal(
        torch._int_mm(a.to("tpu").t(), b.to("tpu").t()).cpu(), expected
      )
    )

  def test_out_contract(self):
    a = torch.ones((3, 5), dtype=torch.int8, device="tpu")
    b = torch.ones((5, 7), dtype=torch.int8, device="tpu")
    out = torch.empty((3, 7), dtype=torch.int32, device="tpu")
    self.assertIs(torch.ops.aten._int_mm.out(a, b, out=out), out)
    for bad in (
      torch.empty((0,), dtype=torch.int32, device="tpu"),
      torch.empty((3, 8), dtype=torch.int32, device="tpu"),
      torch.empty((3, 7), dtype=torch.float32, device="tpu"),
      torch.empty((7, 3), dtype=torch.int32, device="tpu").t(),
      torch.empty((3, 7), dtype=torch.int32),
    ):
      with self.assertRaises(RuntimeError):
        torch.ops.aten._int_mm.out(a, b, out=bad)

  def test_invalid_inputs_and_regular_mm(self):
    a = torch.ones((3, 5), dtype=torch.int8, device="tpu")
    b = torch.ones((5, 7), dtype=torch.int8, device="tpu")
    for x, y in (
      (a, b.cpu()),
      (a.cpu(), b),
      (a.to(torch.int32), b),
      (a, b.to(torch.float32)),
      (a.reshape(15), b),
      (a, b.t()),
    ):
      with self.assertRaises(RuntimeError):
        torch._int_mm(x, y)
    with self.assertRaises((RuntimeError, NotImplementedError)):
      torch.mm(a, b)

  def test_compiled(self):
    a = torch.full((1, 1041), 127, dtype=torch.int8, device="tpu")
    b = torch.full((1041, 1), 127, dtype=torch.int8, device="tpu")
    fn = torch.compile(torch._int_mm, backend="tpu", fullgraph=True)
    self.assertEqual(fn(a, b).cpu().item(), 16790289)


if __name__ == "__main__":
  absltest.main()
