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

"""Test demonstrating dynamic execution through torch.compile()."""

from unittest import mock
from absl.testing import absltest
import torch
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import utils


class CompileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.device = torch.accelerator.current_accelerator()

  def call_and_compare(
      self, func, test_inputs, mark_dynamic_tests_info=None
  ) -> dict[str, int]:
    if mark_dynamic_tests_info is None:
      mark_dynamic_tests_info = [() for _ in test_inputs]

    backend = _backend.TpuBackend(debug=True, dynamism=True)

    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dc:
      compiled = torch.compile(func, backend=backend)

      for inputs, per_case_mark_info in zip(
          test_inputs, mark_dynamic_tests_info
      ):
        expected = func(*inputs)
        device_inputs = [input.to(self.device) for input in inputs]
        for input_tensor, input_mark_info in zip(
            device_inputs, per_case_mark_info
        ):
          mark_dynamic, dim_infos = input_mark_info
          if mark_dynamic:
            for dim_info in dim_infos:
              dim, min_val, max_val = dim_info
              torch._dynamo.mark_dynamic(
                  input_tensor, dim, min=min_val, max=max_val
              )
        actual = compiled(*device_inputs).to("cpu")
        utils.assert_close(actual, expected)
      metrics = {"bounded_compile_events": mock_dc.call_count}
    return metrics

  def test_add_simple(self):
    def simple(x):
      return x + 3.0

    test_inputs = [
        (torch.arange(3).reshape(1, 3),),  # static, [1, 3]
        (torch.arange(4).reshape(1, 4),),  # bounded compile, [1, 4(4<=b<=8)]
        (torch.arange(6).reshape(1, 6),),  # no compile, [1, 6]
    ]
    metrics = self.call_and_compare(simple, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_add_multiple_dynamic_inputs(self):
    def simple_add(x, y):
      return x + y

    test_inputs = [
        (
            torch.arange(2).reshape(2, 1),
            torch.arange(2).reshape(2, 1),
        ),  # static, [2, 1]
        (
            torch.arange(4).reshape(4, 1),
            torch.arange(4).reshape(4, 1),
        ),  # bounded compile, [4(4<=b<=8), 1] + [4(4<=b<=8), 1]
        (
            torch.arange(6).reshape(6, 1),
            torch.arange(6).reshape(6, 1),
        ),  # no compile, [6, 1]
    ]
    metrics = self.call_and_compare(simple_add, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_mul(self):
    def simple_mul(x):
      a = 0.3 * x
      return a

    test_inputs = [
        (torch.arange(2).reshape(1, 2),),  # static, [1, 2]
        (torch.arange(4).reshape(1, 4),),  # bounded compile, [1, 4(4<=b<=8)]
        (torch.arange(6).reshape(1, 6),),  # no compile, [1, 6]
    ]
    metrics = self.call_and_compare(simple_mul, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_cat_dim0(self):
    def simple_cat(x):
      return torch.cat([x, x], dim=0)

    test_inputs = [
        (torch.arange(8).reshape(8, 1),),
        (torch.arange(16).reshape(16, 1),),
        (torch.arange(4).reshape(4, 1),),
    ]
    mark_dynamic_tests_info = [
        (
            (True, [(0, 2, 32)]),
        ),  # bounded compile, [8(2<=b<=32), 1] + [8(2<=b<=32), 1] -> [16, 1]
        ((False, []),),  # no compile, [16, 1] + [16, 1] -> [32, 1]
        ((False, []),),  # no compile, [4, 1] + [4, 1] -> [8, 1]
    ]
    metrics = self.call_and_compare(
        simple_cat, test_inputs, mark_dynamic_tests_info
    )
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_cat_dim1(self):
    def simple_cat(x, y):
      a = torch.cat([x, y], dim=1)
      return a

    input_0 = torch.arange(1).reshape(1, 1)
    input_1 = torch.arange(2).reshape(1, 2)
    input_2 = torch.arange(6).reshape(1, 6)
    input_3 = torch.arange(4).reshape(1, 4)

    test_inputs = [
        [input_1, input_0],  # static, [1, 2] + [1, 1] -> [1, 3]
        [
            input_2,
            input_0,
        ],  # bounded compile, [1, 6(6<=b<=12)] + [1, 1] -> [1, 7]
        [
            input_3,
            input_0,
        ],  # bounded compile, [1, 4(4<=b<=8)] + [1, 1] -> [1, 5]
    ]
    metrics = self.call_and_compare(simple_cat, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 2)

  def test_squeeze(self):
    def simple_squeeze(x):
      return x.squeeze(0)

    test_inputs = [
        (torch.arange(2).reshape(1, 2),),  # static, [1, 2]
        (torch.arange(4).reshape(1, 4),),  # bounded compile, [1, 4 (4<=b<=8)]
        (torch.arange(6).reshape(1, 6),),  # no compile, [1, 6]
    ]
    metrics = self.call_and_compare(simple_squeeze, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_unsqueeze(self):
    def simple_unsqueeze(x):
      return x.unsqueeze(0)

    test_inputs = [
        (torch.arange(2),),  # static, [2]
        (torch.arange(4),),  # bounded compile, [4 (4<=b<=8)]
        (torch.arange(8),),  # no compile, [8]
    ]
    metrics = self.call_and_compare(simple_unsqueeze, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_reshape(self):
    def simple_reshape(x):
      return x.reshape(x.shape[0] * x.shape[1])

    test_inputs = [
        (torch.arange(2 * 4).reshape(2, 4),),  # static, [2, 4]
        (
            torch.arange(4 * 4).reshape(4, 4),
        ),  # bounded compile, [4(4<=b<=8), 4]
        (torch.arange(8 * 4).reshape(8, 4),),  # no compile, [8, 4]
    ]
    metrics = self.call_and_compare(simple_reshape, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_arange(self):
    def arange(x):
      a = torch.arange(0, x.shape[0], device=x.device)
      return a

    test_inputs = [
        [torch.arange(2)],  # static, [2]
        [torch.arange(4)],  # bounded compile, [4 (4<=b<=8)]
        [torch.arange(8)],  # no compile, [8]
    ]
    metrics = self.call_and_compare(arange, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_dynamic_reduction(self):
    def reduction_func(x):
      return x.mean(-1, keepdim=True)

    test_inputs = [
        (torch.ones(1, 2, 4),),  # static
        (torch.ones(1, 8, 4),),  # bounded compile (1, 8<=16, 4)
        (torch.ones(1, 4, 4),),  # bounded compile, (1, 4<=8, 4)
        (torch.ones(1, 6, 4),),  # no compile
        (torch.ones(1, 12, 4),),  # no compile
    ]
    metrics = self.call_and_compare(reduction_func, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 2)

  def test_matmul(self):
    def simple_matmul(x, y):
      return torch.matmul(x, y)

    test_inputs = [
        (
            torch.arange(24).reshape(8, 3).to(torch.float32),
            torch.arange(3).reshape(3, 1).to(torch.float32),
        ),
        (
            torch.arange(48).reshape(16, 3).to(torch.float32),
            torch.arange(3).reshape(3, 1).to(torch.float32),
        ),
        (
            torch.arange(48).reshape(16, 3).to(torch.float32),
            torch.arange(3).reshape(3, 1).to(torch.float32),
        ),
    ]
    mark_dynamic_tests_info = [
        (
            (True, [(0, 2, 32)]),
            (False, []),
        ),  # bounded compile, [8 (2<=b<=32), 3] x [3, 1] -> [8, 1]
        (
            (True, [(0, 2, 32)]),
            (False, []),
        ),  # no compile, [16, 3] x [3, 1] -> [16, 1]
        (
            (True, [(0, 2, 32)]),
            (False, []),
        ),  # no compile, [16, 3] x [3, 1] -> [16, 1]
    ]
    metrics = self.call_and_compare(
        simple_matmul, test_inputs, mark_dynamic_tests_info
    )
    self.assertEqual(metrics["bounded_compile_events"], 1)


if __name__ == "__main__":
  absltest.main()
