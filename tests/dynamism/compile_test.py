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
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class CompileTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def call_and_compare(
      self, func, test_inputs, mark_dynamic_tests_info=None
  ) -> dict[str, int]:
    if mark_dynamic_tests_info is None:
      mark_dynamic_tests_info = [() for _ in test_inputs]

    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dc:
      compiled = torch.compile(
          func, backend="tpu", options={"bounded_dynamism": True}
      )

      for inputs, per_case_mark_info in zip(
          test_inputs, mark_dynamic_tests_info
      ):
        inputs = [
            input.to(torch.int32)
            if isinstance(input, torch.Tensor) and input.dtype == torch.int64
            else input
            for input in inputs
        ]
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
        (torch.arange(6),),  # no compile, [6]
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
        (torch.arange(6 * 4).reshape(6, 4),),  # no compile, [6, 4]
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

  def test_ones_multiple_dynamic_dims(self):
    def ones_func(x, y):
      return torch.ones(x.shape[0], y.shape[0], device=x.device)

    test_inputs = [
        [torch.arange(2), torch.arange(3)],  # static
        [torch.arange(4), torch.arange(5)],  # bounded compile
        [torch.arange(8), torch.arange(6)],  # no compile
    ]
    metrics = self.call_and_compare(ones_func, test_inputs, None)
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

  def test_bmm(self):
    def simple_bmm(x, y):
      return torch.bmm(x, y)

    test_inputs = [
        (
            torch.arange(48).reshape(2, 8, 3).to(torch.float32),
            torch.arange(6).reshape(2, 3, 1).to(torch.float32),
        ),
        (
            torch.arange(96).reshape(4, 8, 3).to(torch.float32),
            torch.arange(12).reshape(4, 3, 1).to(torch.float32),
        ),
        (
            torch.arange(144).reshape(6, 8, 3).to(torch.float32),
            torch.arange(18).reshape(6, 3, 1).to(torch.float32),
        ),
    ]
    mark_dynamic_tests_info = [
        (
            (True, [(0, 2, 8)]),
            (True, [(0, 2, 8)]),
        ),  # bounded compile, [2 (2<=b<=8), 8, 3] x [2 (2<=b<=8), 3, 1] -> [2, 8, 1]
        (
            (True, [(0, 2, 8)]),
            (True, [(0, 2, 8)]),
        ),  # no compile, [4, 8, 3] x [4, 3, 1] -> [4, 8, 1]
        (
            (True, [(0, 2, 8)]),
            (True, [(0, 2, 8)]),
        ),  # no compile, [6, 8, 3] x [6, 3, 1] -> [6, 8, 1]
    ]
    metrics = self.call_and_compare(
        simple_bmm, test_inputs, mark_dynamic_tests_info
    )
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_bmm_dynamic_m(self):
    def simple_bmm(x, y):
      return torch.bmm(x, y)

    test_inputs = [
        (
            torch.arange(48).reshape(2, 8, 3).to(torch.float32),
            torch.arange(6).reshape(2, 3, 1).to(torch.float32),
        ),
        (
            torch.arange(96).reshape(2, 16, 3).to(torch.float32),
            torch.arange(6).reshape(2, 3, 1).to(torch.float32),
        ),
        (
            torch.arange(144).reshape(2, 24, 3).to(torch.float32),
            torch.arange(6).reshape(2, 3, 1).to(torch.float32),
        ),
    ]
    mark_dynamic_tests_info = [
        (
            (True, [(1, 2, 32)]),
            (False, []),
        ),  # bounded compile, [2, 8 (2<=m<=32), 3] x [2, 3, 1] -> [2, 8, 1]
        (
            (True, [(1, 2, 32)]),
            (False, []),
        ),  # no compile, [2, 16, 3] x [2, 3, 1] -> [2, 16, 1]
        (
            (True, [(1, 2, 32)]),
            (False, []),
        ),  # no compile, [2, 24, 3] x [2, 3, 1] -> [2, 24, 1]
    ]
    metrics = self.call_and_compare(
        simple_bmm, test_inputs, mark_dynamic_tests_info
    )
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_bmm_dynamic_k(self):
    def simple_bmm(x, y):
      return torch.bmm(x, y)

    test_inputs = [
        (
            torch.arange(48).reshape(2, 8, 3).to(torch.float32),
            torch.arange(6).reshape(2, 3, 1).to(torch.float32),
        ),
        (
            torch.arange(64).reshape(2, 8, 4).to(torch.float32),
            torch.arange(8).reshape(2, 4, 1).to(torch.float32),
        ),
        (
            torch.arange(96).reshape(2, 8, 6).to(torch.float32),
            torch.arange(12).reshape(2, 6, 1).to(torch.float32),
        ),
    ]
    mark_dynamic_tests_info = [
        (
            (True, [(2, 2, 8)]),
            (True, [(1, 2, 8)]),
        ),  # bounded compile, [2, 8, 3 (2<=k<=8)] x [2, 3 (2<=k<=8), 1] -> [2, 8, 1]
        (
            (True, [(2, 2, 8)]),
            (True, [(1, 2, 8)]),
        ),  # no compile, [2, 8, 4] x [2, 4, 1] -> [2, 8, 1]
        (
            (True, [(2, 2, 8)]),
            (True, [(1, 2, 8)]),
        ),  # no compile, [2, 8, 6] x [2, 6, 1] -> [2, 8, 1]
    ]
    metrics = self.call_and_compare(
        simple_bmm, test_inputs, mark_dynamic_tests_info
    )
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_bmm_dynamic_n(self):
    def simple_bmm(x, y):
      return torch.bmm(x, y)

    test_inputs = [
        (
            torch.arange(48).reshape(2, 8, 3).to(torch.float32),
            torch.arange(12).reshape(2, 3, 2).to(torch.float32),
        ),
        (
            torch.arange(48).reshape(2, 8, 3).to(torch.float32),
            torch.arange(24).reshape(2, 3, 4).to(torch.float32),
        ),
        (
            torch.arange(48).reshape(2, 8, 3).to(torch.float32),
            torch.arange(48).reshape(2, 3, 8).to(torch.float32),
        ),
    ]
    mark_dynamic_tests_info = [
        (
            (False, []),
            (True, [(2, 2, 16)]),
        ),  # bounded compile, [2, 8, 3] x [2, 3, 2 (2<=n<=16)] -> [2, 8, 2]
        (
            (False, []),
            (True, [(2, 2, 16)]),
        ),  # no compile, [2, 8, 3] x [2, 3, 4] -> [2, 8, 4]
        (
            (False, []),
            (True, [(2, 2, 16)]),
        ),  # no compile, [2, 8, 3] x [2, 3, 8] -> [2, 8, 8]
    ]
    metrics = self.call_and_compare(
        simple_bmm, test_inputs, mark_dynamic_tests_info
    )
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_arange_with_expression(self):
    def arange_expr(x):
      a = torch.arange(0, x.shape[0] + 1, device=x.device)
      return a

    test_inputs = [
        [torch.arange(2)],  # static, [2]
        [torch.arange(4)],  # bounded compile, [4 (4<=b<=8)]
        [torch.arange(8)],  # no compile, [8]
    ]
    metrics = self.call_and_compare(arange_expr, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_arange_with_chained_expression(self):
    def arange_expr(x):
      n = (x.shape[0] * 3 + 5) // 2
      a = torch.arange(0, n, device=x.device)
      return a

    test_inputs = [
        [torch.arange(2)],  # static, [2]
        [torch.arange(4)],  # bounded compile, [4 (4<=b<=8)]
        [torch.arange(8)],  # no compile, [8]
    ]
    metrics = self.call_and_compare(arange_expr, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_arange_with_sym_sum(self):
    def arange_expr(x):
      n = torch.sym_sum([x.shape[0], 1])
      a = torch.arange(0, n, device=x.device)
      return a

    test_inputs = [
        [torch.arange(2)],  # static, [2]
        [torch.arange(4)],  # bounded compile, [4 (4<=b<=8)]
        [torch.arange(8)],  # no compile, [8]
    ]
    metrics = self.call_and_compare(arange_expr, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_arange_constant_length(self):
    def arange_func(x):
      # start is dynamic (x.shape[0]), end is dynamic (x.shape[0] + 5)
      # length is constant (5)
      a = torch.arange(x.shape[0], x.shape[0] + 5, device=x.device)
      return a

    test_inputs = [
        [torch.arange(2)],  # static, [2]
        [torch.arange(4)],  # bounded compile, [4 (4<=b<=8)]
        [torch.arange(8)],  # no compile, [8]
    ]
    metrics = self.call_and_compare(arange_func, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_arange_dynamic_length(self):
    def arange_func(x, y):
      a = torch.arange(x.shape[0], x.shape[0] + y.shape[0] * 2, device=x.device)
      return a

    test_inputs = [
        [torch.arange(2), torch.arange(3)],  # static, [2]
        [torch.arange(4), torch.arange(5)],  # bounded compile, [4 (4<=b<=8)]
        [torch.arange(8), torch.arange(6)],  # no compile, [8]
    ]
    metrics = self.call_and_compare(arange_func, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_arange_dynamic_step(self):
    def arange_func(x):
      a = torch.arange(
          x.shape[0],
          x.shape[0] + 100,
          step=x.shape[0] // 2,
          device=x.device,
      )
      return a

    test_inputs = [
        [torch.arange(2)],  # static, [2]
        [torch.arange(4)],  # bounded compile, [4 (4<=b<=8)]
        [torch.arange(8)],  # no compile, [8]
    ]
    metrics = self.call_and_compare(arange_func, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

  def test_duplicate_symint_in_placeholder(self):
    def f(x):
      torch._check(x.shape[0] == x.shape[1])
      return x * 2

    compiled_f = torch.compile(
        f, backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.ones((2, 2), device="tpu")
    torch._dynamo.mark_dynamic(x, 0, min=2, max=8)
    torch._dynamo.mark_dynamic(x, 1, min=2, max=8)

    res = compiled_f(x)

    expected = torch.full((2, 2), 2.0, device="cpu")
    utils.assert_close(res.cpu(), expected)

  def test_derived_dimension_placeholder(self):
    class Model(torch.nn.Module):

      # FX graph:
      # (arg0_1: "Sym(s77)", arg1_1: "f32[s77, 3]", arg2_1: "f32[2*s77, 5]")
      def forward(self, x, y):
        if x.size(0) * 2 == y.size(0):
          return x.sum() + y.sum()
        return x.sum() - y.sum()

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.ones(4, 3, device="tpu")
    y1 = torch.ones(8, 5, device="tpu")
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)
    torch._dynamo.mark_dynamic(y1, 0, min=4, max=32)

    out1 = compiled(x1, y1)
    expected = x1.to("cpu").sum() + y1.to("cpu").sum()
    utils.assert_close(out1.to("cpu"), expected)

  def test_static_tensor_with_has_symbolic_sizes_strides_set(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        torch._check(y.shape[0] == 4)
        scale = y.shape[0]
        return x * scale

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.ones((2, 2), dtype=torch.float32, device="tpu")
    y = torch.ones((4, 2), dtype=torch.float32, device="tpu")
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.maybe_mark_dynamic(y, 0)

    out = compiled(x, y)
    expected = x.to("cpu") * 4
    utils.assert_close(out, expected)

  def test_concat_attention(self):
    def decode_step(full_attention_cache, new_token_key):
      new_full = torch.cat([full_attention_cache, new_token_key], dim=-2)
      return new_full

    compiled = torch.compile(
        decode_step, backend="tpu", options={"bounded_dynamism": True}
    )

    prefill_cache = torch.randn(
        2, 1, 8, 16, device=self.device, dtype=torch.bfloat16
    )
    # Marking the prefill sequence dimension dynamic (s0)
    torch._dynamo.mark_dynamic(prefill_cache, 2, min=4, max=16)
    new_token_key = torch.randn(
        2, 1, 1, 16, device=self.device, dtype=torch.bfloat16
    )

    expected_full = decode_step(prefill_cache, new_token_key)
    actual_full = compiled(prefill_cache, new_token_key)
    utils.assert_close(actual_full, expected_full)

  def test_sliced_view_and_dynamic_input(self):
    def fn(dynamic_x, sliced_y):
      return dynamic_x + 1, sliced_y + 1

    compiled_fn = torch.compile(
        fn, backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.randn(4, 8, device=self.device)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=10)

    y_base = torch.randn(2, 1, 8, 16, device=self.device)
    y_sliced = y_base[:, :, -3:, :]

    actual_dynamic, actual_view = compiled_fn(x, y_sliced)
    expected_dynamic, expected_view = fn(x.to("cpu"), y_sliced.to("cpu"))

    utils.assert_close(actual_dynamic, expected_dynamic)
    utils.assert_close(actual_view, expected_view)

  def test_constant_symint_argument(self):
    class Model(torch.nn.Module):

      def forward(self, x, scale: int):
        torch._check(scale == 4)
        return x * scale, scale * 2

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.ones(2, 2, dtype=torch.float32, device="tpu")
    torch._dynamo.mark_dynamic(x, 0, min=2, max=8)

    # Instruct Dynamo to treat int value 4 as dynamic:
    with torch.compiler.config.patch({"dynamic_sources": ".*scale.*"}):
      out, scale_out = compiled(x, 4)
    expected = x.to("cpu") * 4
    utils.assert_close(out, expected)
    self.assertEqual(scale_out, 8)

  def test_add_simple_64bit(self):
    def simple(x):
      return x + 3.0

    compiled = torch.compile(
        simple, backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.ones(2, 3, device="tpu", dtype=torch.int64)
    torch._dynamo.mark_dynamic(x, 0, min=1, max=4)

    out = compiled(x)
    expected = torch.full((2, 3), 4.0, device="cpu")
    utils.assert_close(out.cpu(), expected)


class SymIntArithmeticTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_arange_plus_symint(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[1]
        return torch.arange(1, device=x.device) + s0

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(1, 1024, device="tpu")
    torch._dynamo.mark_dynamic(x1, 1, min=1, max=2048)

    out1 = compiled(x1)
    utils.assert_close(out1, torch.tensor([1024], device="tpu"))

  def test_arange_bounds(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[1]
        return torch.arange(s0, s0 + 1, device=x.device)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(1, 1024, device="tpu")
    torch._dynamo.mark_dynamic(x1, 1, min=1, max=2048)

    out1 = compiled(x1)
    utils.assert_close(out1, torch.tensor([1024], device="tpu"))

  def test_symint_in_output(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.shape[1] * 2 + 1

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.zeros(1, 4, device="tpu")
    torch._dynamo.mark_dynamic(x1, 1, min=1, max=16)

    out1 = compiled(x1)
    self.assertEqual(out1, 9)


class DynamicReshapeTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_squeeze(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.squeeze(0)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(20, dtype=torch.float32, device="tpu").reshape(1, 10, 2)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=20)

    out1 = compiled(x1)
    expected = (
        torch.arange(20, dtype=torch.float32, device="tpu")
        .reshape(1, 10, 2)
        .squeeze(0)
    )
    utils.assert_close(out1, expected)

  def test_unsqueeze(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.unsqueeze(0).unsqueeze(-1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(20, dtype=torch.float32, device="tpu").reshape(5, 4)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=20)

    out1 = compiled(x1)
    expected = (
        torch.arange(20, dtype=torch.float32, device="tpu")
        .reshape(5, 4)
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    utils.assert_close(out1, expected)

  def test_transpose(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.transpose(0, 1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(30, dtype=torch.float32, device="tpu").reshape(2, 5, 3)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=20)

    out1 = compiled(x1)
    expected = (
        torch.arange(30, dtype=torch.float32, device="tpu")
        .reshape(2, 5, 3)
        .transpose(0, 1)
    )
    utils.assert_close(out1, expected)

  def test_permute(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.permute(2, 0, 1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(30, dtype=torch.float32, device="tpu").reshape(2, 5, 3)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=20)

    out1 = compiled(x1)
    expected = (
        torch.arange(30, dtype=torch.float32, device="tpu")
        .reshape(2, 5, 3)
        .permute(2, 0, 1)
    )
    utils.assert_close(out1, expected)

  def test_flatten(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.flatten()

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(30, dtype=torch.float32, device="tpu").reshape(2, 5, 3)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=20)

    out1 = compiled(x1)
    expected = (
        torch.arange(30, dtype=torch.float32, device="tpu")
        .reshape(2, 5, 3)
        .flatten()
    )
    utils.assert_close(out1, expected)

  def test_collapse_reshape(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(-1, 3)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(30, dtype=torch.float32, device="tpu").reshape(2, 5, 3)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=20)

    out1 = compiled(x1)
    expected = (
        torch.arange(30, dtype=torch.float32, device="tpu")
        .reshape(2, 5, 3)
        .reshape(-1, 3)
    )
    utils.assert_close(out1, expected)

  def test_expand_reshape_unambiguous(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(2, 3, -1, 1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(30, dtype=torch.float32, device="tpu").reshape(6, 5)
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=20)

    out1 = compiled(x1)
    expected = (
        torch.arange(30, dtype=torch.float32, device="tpu")
        .reshape(6, 5)
        .reshape(2, 3, -1, 1)
    )
    utils.assert_close(out1, expected)

  def test_transpose_like_view(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.view(-1, 6, 1, 1, 5)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(300, dtype=torch.float32, device="tpu").reshape(
        1, 10, 1, 6, 5
    )
    torch._dynamo.mark_dynamic(x1, 1, min=2, max=20)

    out1 = compiled(x1)
    expected = (
        torch.arange(300, dtype=torch.float32, device="tpu")
        .reshape(1, 10, 1, 6, 5)
        .view(-1, 6, 1, 1, 5)
    )
    utils.assert_close(out1, expected)

  def test_expand_reshape_ambiguous(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(2, -1, 3)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    # Dynamic dim 0 (size 10) expands into two non-one dims (2, -1=5)
    x1 = torch.arange(30, dtype=torch.float32, device="tpu").reshape(10, 3)
    torch._dynamo.mark_dynamic(x1, 0)

    out1 = compiled(x1)
    expected = (
        torch.arange(30, dtype=torch.float32, device="tpu")
        .reshape(10, 3)
        .reshape(2, -1, 3)
    )
    utils.assert_close(out1, expected)

  def test_same_size_reshape_ambiguous(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(3, -1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    # Same rank reshape [2, 6] -> [3, 4] with dynamic dim 1
    x1 = torch.arange(12, dtype=torch.float32, device="tpu").reshape(2, 6)
    torch._dynamo.mark_dynamic(x1, 1)

    out1 = compiled(x1)
    expected = (
        torch.arange(12, dtype=torch.float32, device="tpu")
        .reshape(2, 6)
        .reshape(3, -1)
    )
    utils.assert_close(out1, expected)

  def test_unflatten_dynamic_input_trailing_static_dim(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.view(x.shape[0], 4, 2)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    # Dynamic dim 0 with trailing dims [8] unflattened to [4, 2]
    x1 = torch.arange(32, dtype=torch.float32, device="tpu").reshape(4, 8)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=16)

    out1 = compiled(x1)
    expected1 = (
        torch.arange(32, dtype=torch.float32, device="tpu")
        .reshape(4, 8)
        .view(4, 4, 2)
    )
    utils.assert_close(out1, expected1)

    # Verify dynamism with a different size at runtime
    x2 = torch.arange(48, dtype=torch.float32, device="tpu").reshape(6, 8)
    out2 = compiled(x2)
    expected2 = (
        torch.arange(48, dtype=torch.float32, device="tpu")
        .reshape(6, 8)
        .view(6, 4, 2)
    )
    utils.assert_close(out2, expected2)

  def test_view_copy_dynamic_input(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return torch.ops.aten.view_copy.default(x, [x.shape[0], 5, 2])

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(20, dtype=torch.float32, device="tpu").reshape(2, 10)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=10)

    out1 = compiled(x1)
    expected = (
        torch.arange(20, dtype=torch.float32, device="tpu")
        .reshape(2, 10)
        .view(2, 5, 2)
    )
    utils.assert_close(out1, expected)

  def test_multi_dynamic_reshape_unambiguous(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(x.shape[0], 5, x.shape[1], 1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(60, dtype=torch.float32, device="tpu").reshape(4, 3, 5)
    torch._dynamo.mark_dynamic(x1, 0)
    torch._dynamo.mark_dynamic(x1, 1)

    out1 = compiled(x1)
    expected = (
        torch.arange(60, dtype=torch.float32, device="tpu")
        .reshape(4, 3, 5)
        .reshape(4, 5, 3, 1)
    )
    utils.assert_close(out1, expected)

  def test_multi_dynamic_collapse_reshape(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(-1, 5)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(60, dtype=torch.float32, device="tpu").reshape(4, 3, 5)
    torch._dynamo.mark_dynamic(x1, 0)
    torch._dynamo.mark_dynamic(x1, 1)

    out1 = compiled(x1)
    expected = (
        torch.arange(60, dtype=torch.float32, device="tpu")
        .reshape(4, 3, 5)
        .reshape(-1, 5)
    )
    utils.assert_close(out1, expected)


class DynamicBroadcastTest(seed_test_utils.RepeatableTest):

  def test_expand_dynamic_input_unambiguous(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.expand(-1, 3, 5)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(20, dtype=torch.float32, device="tpu").reshape(4, 1, 5)
    torch._dynamo.mark_dynamic(x1, 0)

    out1 = compiled(x1)
    expected = (
        torch.arange(20, dtype=torch.float32, device="cpu")
        .reshape(4, 1, 5)
        .expand(4, 3, 5)
    )
    utils.assert_close(out1, expected)

  def test_expand_dynamic_output_dim(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        return x.expand(y.shape[0], 5)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(5, dtype=torch.float32, device="tpu").reshape(1, 5)
    y1 = torch.arange(20, dtype=torch.float32, device="tpu").reshape(4, 5)
    torch._dynamo.mark_dynamic(y1, 0, min=2, max=10)

    out1 = compiled(x1, y1)
    expected = x1.to("cpu").expand(4, 5)
    utils.assert_close(out1, expected)

  def test_expand_multi_dynamic(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        return x.expand(x.shape[0], y.shape[1])

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(4, dtype=torch.float32, device="tpu").reshape(4, 1)
    y1 = torch.arange(12, dtype=torch.float32, device="tpu").reshape(4, 3)
    torch._dynamo.mark_dynamic(x1, 0, min=2, max=10)
    torch._dynamo.mark_dynamic(y1, 0, min=2, max=10)
    torch._dynamo.mark_dynamic(y1, 1, min=2, max=10)

    out1 = compiled(x1, y1)
    expected = x1.to("cpu").expand(4, 3)
    utils.assert_close(out1, expected)

  def test_broadcast_to_dynamic_output_dim(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        z = torch.broadcast_to(x, (y.shape[0], 5))
        return z * 2

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(5, dtype=torch.float32, device="tpu").reshape(1, 5)
    y1 = torch.arange(20, dtype=torch.float32, device="tpu").reshape(4, 5)
    torch._dynamo.mark_dynamic(y1, 0, min=2, max=10)

    out1 = compiled(x1, y1)
    expected = torch.broadcast_to(x1.to("cpu"), (4, 5)) * 2
    utils.assert_close(out1, expected)

  def test_broadcast_to_dynamic_output_returned(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        return torch.broadcast_to(x, (y.shape[0], 5))

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(5, dtype=torch.float32, device="tpu").reshape(1, 5)
    y1 = torch.arange(20, dtype=torch.float32, device="tpu").reshape(4, 5)
    torch._dynamo.mark_dynamic(y1, 0, min=2, max=10)

    out1 = compiled(x1, y1)
    expected = torch.broadcast_to(x1.to("cpu"), (4, 5))
    utils.assert_close(out1, expected)

  def test_expand_method_sequence_arg(self):
    class Model(torch.nn.Module):

      def forward(self, x, y):
        return x.expand([y.shape[0], 5])

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x1 = torch.arange(5, dtype=torch.float32, device="tpu").reshape(1, 5)
    y1 = torch.arange(20, dtype=torch.float32, device="tpu").reshape(4, 5)
    torch._dynamo.mark_dynamic(y1, 0, min=2, max=10)

    out1 = compiled(x1, y1)
    expected = x1.to("cpu").expand([4, 5])
    utils.assert_close(out1, expected)

  def test_expand_before_reshape_shared_symexpr(self):
    """Tests graph topo order when expand appears before reshape with shared symexpr."""

    class Model(torch.nn.Module):

      def forward(self, x, y):
        expanded = x.expand([y.shape[1] * y.shape[0], 5])
        reshaped = y.reshape([1, y.shape[0] * y.shape[1]])
        return expanded.sum() + reshaped.sum()

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.ones((1, 5), dtype=torch.float32, device="tpu")
    y = torch.ones((2, 3), dtype=torch.float32, device="tpu")
    torch._dynamo.mark_dynamic(y, 1, min=2, max=10)

    out = compiled(x, y)
    expected = (
        x.to("cpu").expand([6, 5]).sum() + y.to("cpu").reshape([1, 6]).sum()
    )
    utils.assert_close(out, expected)


class DynamicSliceTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def test_slice_static_dim_on_dynamic_tensor(self):
    class Model(torch.nn.Module):

      def forward(self, grid):
        return grid[:, 1] * grid[:, 2]

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    grid = torch.tensor(
        [[1, 24, 48], [1, 24, 48]], dtype=torch.int32, device=self.device
    )
    torch._dynamo.mark_dynamic(grid, 0, min=1, max=32)

    out = compiled(grid)
    expected = grid[:, 1] * grid[:, 2]
    utils.assert_close(out, expected)

  def test_slice_dynamic_tensor_with_symbolic_expression_shape(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x[: x.shape[0] - 1, :] * 2.0

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.randn(6, 4, dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(x, 0)

    out = compiled(x)
    expected = x[: x.shape[0] - 1, :] * 2.0
    utils.assert_close(out, expected)

  def test_slice_dynamic_dim_to_static_size(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x[:2, :]

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.randn(5, 4, dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(x, 0, min=3, max=32)

    out = compiled(x)
    expected = x[:2, :]
    utils.assert_close(out, expected)

  def test_select_negative_dim(self):
    class Model(torch.nn.Module):

      def forward(self, grid):
        return grid.select(-1, 1) * grid.select(-1, 2)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    grid = torch.tensor(
        [[1, 24, 48], [1, 24, 48]], dtype=torch.int32, device=self.device
    )
    torch._dynamo.mark_dynamic(grid, 0, min=1, max=32)

    out = compiled(grid)
    expected = grid.select(-1, 1) * grid.select(-1, 2)
    utils.assert_close(out, expected)

  def test_select_dynamic_dim(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.select(0, 2)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.randn(6, 8, dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(x, 0, min=4, max=16)

    out = compiled(x)
    expected = x.select(0, 2)
    utils.assert_close(out, expected)

  def test_select_negative_index(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.select(0, -1)

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    for dim_size in [6, 10]:
      x = torch.randn(dim_size, 8, dtype=torch.float32, device=self.device)
      torch._dynamo.mark_dynamic(x, 0, min=4, max=16)

      out = compiled(x)
      expected = x.select(0, -1)
      utils.assert_close(out, expected)

  def test_dynamic_slice_negative_start(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x[:, -4:]

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.randn(2, 8, dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(x, 1, min=6, max=16)

    out = compiled(x)
    expected = x[:, -4:]
    utils.assert_close(out, expected)

  def test_dynamic_slice_kv_cache_sliding_window(self):
    sliding_window = 16

    class Model(torch.nn.Module):

      def forward(self, cache, new_k):
        full_k = torch.cat([cache, new_k], dim=-2)
        return full_k[:, :, -sliding_window + 1 :, :]

    compiled = torch.compile(
        Model(), backend="tpu", options={"bounded_dynamism": True}
    )

    cache = torch.randn(1, 1, 4, 16, dtype=torch.float32, device=self.device)
    new_k = torch.randn(1, 1, 1, 16, dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(cache, 2, min=2, max=13)

    out = compiled(cache, new_k)
    expected = torch.cat([cache, new_k], dim=-2)[:, :, -sliding_window + 1 :, :]
    utils.assert_close(out, expected)

  def test_dynamic_slice_static_buffer(self):
    max_len = 64

    class Model(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.register_buffer("static_table", torch.randn(4, 1, max_len))

      def forward(self, x):
        seq_len = x.shape[-1]
        sliced_table = self.static_table[:, :, :seq_len]
        return x + sliced_table

    model = Model().to(self.device)
    compiled = torch.compile(
        model, backend="tpu", options={"bounded_dynamism": True}
    )

    for seq_len in [8, 16]:
      x = torch.randn(1, 4, 1, seq_len, dtype=torch.float32, device=self.device)
      torch._dynamo.mark_dynamic(x, 3, min=4, max=32)

      out = compiled(x)
      expected = model(x)
      utils.assert_close(out, expected)


class DynamicDropoutTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def test_dropout_dynamic_shape(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return torch.nn.functional.dropout(x, p=0.4, training=True)

    model = Model().to(self.device)
    compiled = torch.compile(
        model, backend="tpu", options={"bounded_dynamism": True}
    )

    for dim_size in [4, 8, 12]:
      x = torch.randn(dim_size, 16, dtype=torch.float32, device=self.device)
      torch._dynamo.mark_dynamic(x, 0, min=2, max=32)

      out = compiled(x)
      self.assertEqual(out.shape, (dim_size, 16))
      scale = 1.0 / (1.0 - 0.4)
      out_cpu = out.cpu()
      x_cpu = x.cpu()
      mask = out_cpu != 0
      utils.assert_close(out_cpu[mask], (x_cpu * scale)[mask])

  def test_native_dropout_dynamic_shape(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return torch.ops.aten.native_dropout(x, 0.5, True)

    model = Model().to(self.device)
    compiled = torch.compile(
        model, backend="tpu", options={"bounded_dynamism": True}
    )

    for dim_size in [6, 10]:
      x = torch.randn(dim_size, 8, dtype=torch.float32, device=self.device)
      torch._dynamo.mark_dynamic(x, 0, min=2, max=32)

      out, mask = compiled(x)
      self.assertEqual(out.shape, (dim_size, 8))
      self.assertEqual(mask.shape, (dim_size, 8))
      self.assertEqual(mask.dtype, torch.bool)

  def test_native_dropout_backward_dynamic_shape(self):
    class Model(torch.nn.Module):

      def forward(self, grad_output, mask):
        return torch.ops.aten.native_dropout_backward(grad_output, mask, 2.0)

    model = Model().to(self.device)
    compiled = torch.compile(
        model, backend="tpu", options={"bounded_dynamism": True}
    )

    for dim_size in [4, 12]:
      grad = torch.randn(dim_size, 16, dtype=torch.float32, device=self.device)
      mask = torch.randint(
          0, 2, (dim_size, 16), dtype=torch.bool, device=self.device
      )
      torch._dynamo.mark_dynamic(grad, 0, min=2, max=32)
      torch._dynamo.mark_dynamic(mask, 0, min=2, max=32)

      out = compiled(grad, mask)
      expected = model(grad, mask)
      utils.assert_close(out, expected)

  def test_dropout_train_backward_dynamic_shape(self):
    class Model(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.3)

      def forward(self, x):
        return self.dropout(x)

    model = Model().to(self.device).train()
    compiled = torch.compile(
        model, backend="tpu", options={"bounded_dynamism": True}
    )

    for dim_size in [4, 8, 12]:
      x = torch.randn(
          dim_size,
          16,
          dtype=torch.float32,
          device=self.device,
          requires_grad=True,
      )
      torch._dynamo.mark_dynamic(x, 0, min=2, max=32)

      out = compiled(x)
      self.assertEqual(out.shape, (dim_size, 16))
      loss = out.sum()
      loss.backward()
      self.assertIsNotNone(x.grad)
      self.assertEqual(x.grad.shape, (dim_size, 16))

  def test_dropout_dynamic_shape_dtypes(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return torch.nn.functional.dropout(x, p=0.25, training=True)

    model = Model().to(self.device)
    compiled = torch.compile(
        model, backend="tpu", options={"bounded_dynamism": True}
    )

    for dtype in [torch.bfloat16, torch.float16, torch.float64]:
      x = torch.randn(6, 12, dtype=dtype, device=self.device)
      torch._dynamo.mark_dynamic(x, 0, min=2, max=32)

      out = compiled(x)
      self.assertEqual(out.shape, (6, 12))
      self.assertEqual(out.dtype, dtype)

  def test_dropout_dynamic_shape_eval_and_zero_p(self):
    class Model(torch.nn.Module):

      def forward(self, x, p, train):
        return torch.ops.aten.native_dropout(x, p, train)

    model = Model().to(self.device)
    compiled = torch.compile(
        model, backend="tpu", options={"bounded_dynamism": True}
    )

    x = torch.randn(4, 16, dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(x, 0, min=2, max=32)

    # eval mode
    out, mask = compiled(x, 0.5, False)
    self.assertEqual(out.shape, (4, 16))
    utils.assert_close(out, x)
    self.assertTrue(mask.all().item())

    # p = 0.0
    out, mask = compiled(x, 0.0, True)
    self.assertEqual(out.shape, (4, 16))
    utils.assert_close(out, x)
    self.assertTrue(mask.all().item())

    # p = 1.0
    out, mask = compiled(x, 1.0, True)
    self.assertEqual(out.shape, (4, 16))
    utils.assert_close(out, torch.zeros_like(x))
    self.assertFalse(mask.any().item())

  def test_dropout_multiple_dynamic_dims(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return torch.nn.functional.dropout(x, p=0.3, training=True)

    model = Model().to(self.device)
    compiled = torch.compile(
        model, backend="tpu", options={"bounded_dynamism": True}
    )

    for b, s in [(2, 8), (4, 16), (3, 12)]:
      x = torch.randn(b, s, 32, dtype=torch.float32, device=self.device)
      torch._dynamo.mark_dynamic(x, 0, min=1, max=8)
      torch._dynamo.mark_dynamic(x, 1, min=2, max=32)

      out = compiled(x)
      self.assertEqual(out.shape, (b, s, 32))
      scale = 1.0 / (1.0 - 0.3)
      out_cpu = out.cpu()
      x_cpu = x.cpu()
      mask = out_cpu != 0
      utils.assert_close(out_cpu[mask], (x_cpu * scale)[mask])


class DynamicErrorHandlingTest(seed_test_utils.RepeatableTest):

  def test_mlir_lowering_failure_raises_not_implemented_error(self):
    def simple(x):
      return x + 1.0

    with mock.patch.object(
        _backend.compiler.StaticCompiler,
        "__call__",
        side_effect=RuntimeError("MLIR lowering failed"),
    ):
      compiled = torch.compile(
          simple, backend="tpu", options={"bounded_dynamism": True}
      )
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


class SymMaxMinTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def test_sym_max_and_min(self):
    def fn(x):
      max_val = max(x.shape[0] - 5, 0)
      min_val = min(10, x.shape[0] + 2)
      return torch.arange(8, device=x.device) + max_val + min_val

    compiled = torch.compile(
        fn, backend="tpu", options={"bounded_dynamism": True}
    )

    t = torch.ones(8, device=self.device)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=16)

    out = compiled(t)
    expected = fn(t)
    utils.assert_close(out, expected)


class IndexPutDynamicBooleanMaskTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    if not torch.accelerator.is_available():
      self.skipTest("TPU accelerator not available in this test environment.")
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  def test_index_put_boolean_mask_dynamic(self):
    # Regression test for t5gemma-2 pattern: index_put_ with a 2D boolean mask
    # on a 3D tensor with dynamic sequence length under bounded dynamism.
    def fn(embeddings, mask, eoi_embedding):
      embeddings = embeddings.clone()
      embeddings[mask] = eoi_embedding
      return embeddings

    compiled = torch.compile(
        fn, backend="tpu", options={"bounded_dynamism": True}
    )

    batch_size = 2
    seq_len = 6
    hidden_dim = 8
    embeddings = torch.randn(
        batch_size, seq_len, hidden_dim, device=self.device
    )
    mask = torch.tensor(
        [
            [True, False, True, False, False, True],
            [False, True, False, True, False, False],
        ],
        dtype=torch.bool,
        device=self.device,
    )
    eoi_embedding = torch.randn(hidden_dim, device=self.device)

    torch._dynamo.mark_dynamic(embeddings, 1, min=2, max=16)
    torch._dynamo.mark_dynamic(mask, 1, min=2, max=16)

    out = compiled(embeddings, mask, eoi_embedding)
    expected = fn(embeddings, mask, eoi_embedding)
    utils.assert_close(out, expected)


if __name__ == "__main__":
  absltest.main()
