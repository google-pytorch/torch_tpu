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

from collections.abc import Callable, Sequence
from typing import Any
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._dynamo.backends.common import aot_autograd
from torch.utils._sympy.value_ranges import ValueRanges
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend
from torch_tpu._internal.compile.dynamic import sym_utils
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


def _uses_pad_subgraph(dtype: torch.dtype) -> bool:
  """Whether dynamic inputs of `dtype` are padded by the external pad subgraph."""
  return not sym_utils.can_use_pad_real_to_static(torch.empty(0, dtype=dtype))


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
      compiled = torch.compile(func, backend="tpu", dynamic=None)

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

  def test_boxed_call_does_not_mutate_caller_list(self):
    def simple(x):
      return x + 3.0

    executables = []
    orig_call = _backend.dynamic_compiler.DynamicCompiler.__call__

    def capture_executable(dc_self, *args, **kwargs):
      exe = orig_call(dc_self, *args, **kwargs)
      executables.append(exe)
      return exe

    with mock.patch.object(
        _backend.dynamic_compiler.DynamicCompiler,
        "__call__",
        autospec=True,
        side_effect=capture_executable,
    ):
      compiled = torch.compile(simple, backend="tpu")
      x = torch.arange(4, dtype=torch.float32, device=self.device).reshape(1, 4)
      torch._dynamo.mark_dynamic(x, 1, min=2, max=8)
      utils.assert_close(compiled(x).to("cpu"), simple(x).to("cpu"))

    self.assertLen(executables, 1)
    dyn_exec = executables[0]
    self.assertTrue(dyn_exec._boxed_call)
    boxed_inputs = [4, x]
    (out,) = dyn_exec(boxed_inputs)
    self.assertLen(boxed_inputs, 2)
    self.assertEqual(boxed_inputs[0], 4)
    self.assertIs(boxed_inputs[1], x)
    utils.assert_close(out.to("cpu"), simple(x).to("cpu"))

  @parameterized.named_parameters(
      ("scan_dim", 0),
      ("non_scan_dim", 1),
  )
  def test_cumsum_dynamic(self, dim: int):
    def scan_fn(x):
      return torch.cumsum(x, dim=dim)

    test_inputs = [
        (torch.randn(8, 17),),
        (torch.randn(4, 17),),
    ]
    mark_dynamic_tests_info = [
        ((True, [(0, 2, 16)]),),
        ((True, [(0, 2, 16)]),),
    ]
    metrics = self.call_and_compare(
        scan_fn, test_inputs, mark_dynamic_tests_info
    )
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
        ],  # bounded compile, [1, 6(2<=b<=127)] + [1, 1] -> [1, 7]
        [
            input_3,
            input_0,
        ],  # no compile, [1, 4] + [1, 1] -> [1, 5]
    ]
    metrics = self.call_and_compare(simple_cat, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

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
        (torch.ones(1, 8, 4),),  # bounded compile (1, 2<=b<=128, 4)
        (torch.ones(1, 4, 4),),  # no compile
        (torch.ones(1, 6, 4),),  # no compile
        (torch.ones(1, 12, 4),),  # no compile
    ]
    metrics = self.call_and_compare(reduction_func, test_inputs, None)
    self.assertEqual(metrics["bounded_compile_events"], 1)

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

    compiled_f = torch.compile(f, backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(decode_step, backend="tpu")

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

  @parameterized.named_parameters(
      ("no_mask", False),
      ("padding_mask", True),
  )
  def test_sdpa_enable_gqa_decode_dynamic_kv_len(self, use_mask):
    # GQA decode step: the KV cache length is bounded-dynamic and SDPA is
    # called with enable_gqa=True, so the grouped (5D) lowering sees a dynamic
    # key sequence dimension.
    def decode_step(q, cache_k, cache_v, new_k, new_v):
      k = torch.cat([cache_k, new_k], dim=-2)
      v = torch.cat([cache_v, new_v], dim=-2)
      mask = None
      if use_mask:
        # Mask out the first key position; the mask has the dynamic length.
        mask = (torch.arange(k.shape[-2], device=k.device) >= 1).view(
            1, 1, 1, -1
        )
      return torch.nn.functional.scaled_dot_product_attention(
          q, k, v, attn_mask=mask, is_causal=False, enable_gqa=True
      )

    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dc:
      compiled = torch.compile(decode_step, backend="tpu")
      for cache_len in (8, 13):
        q = torch.randn(1, 8, 1, 32)
        cache_k = torch.randn(1, 2, cache_len, 32)
        cache_v = torch.randn(1, 2, cache_len, 32)
        new_k = torch.randn(1, 2, 1, 32)
        new_v = torch.randn(1, 2, 1, 32)
        expected = decode_step(q, cache_k, cache_v, new_k, new_v)

        inputs = [
            t.to(self.device) for t in (q, cache_k, cache_v, new_k, new_v)
        ]
        torch._dynamo.mark_dynamic(inputs[1], 2, min=4, max=64)
        torch._dynamo.mark_dynamic(inputs[2], 2, min=4, max=64)
        actual = compiled(*inputs)
        utils.assert_close(actual.to("cpu"), expected, rtol=2e-2, atol=2e-2)
    self.assertGreaterEqual(mock_dc.call_count, 1)

  def test_sdpa_enable_gqa_causal_dynamic_seq_len(self):
    # GQA prefill with is_causal=True: the causal mask must be built for a
    # bounded-dynamic query/key sequence length.
    def prefill(q, k, v):
      return torch.nn.functional.scaled_dot_product_attention(
          q, k, v, is_causal=True, enable_gqa=True
      )

    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dc:
      compiled = torch.compile(prefill, backend="tpu")
      for seq_len in (9, 14):
        q = torch.randn(1, 8, seq_len, 32)
        k = torch.randn(1, 2, seq_len, 32)
        v = torch.randn(1, 2, seq_len, 32)
        expected = prefill(q, k, v)

        inputs = [t.to(self.device) for t in (q, k, v)]
        for t in inputs:
          torch._dynamo.mark_dynamic(t, 2, min=4, max=64)
        actual = compiled(*inputs)
        utils.assert_close(actual.to("cpu"), expected, rtol=2e-2, atol=2e-2)
    self.assertGreaterEqual(mock_dc.call_count, 1)

  def test_sliced_view_and_dynamic_input(self):
    def fn(dynamic_x, sliced_y):
      return dynamic_x + 1, sliced_y + 1

    compiled_fn = torch.compile(fn, backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(simple, backend="tpu")

    x = torch.ones(2, 3, device="tpu", dtype=torch.int64)
    torch._dynamo.mark_dynamic(x, 0, min=1, max=4)

    out = compiled(x)
    expected = torch.full((2, 3), 4.0, device="cpu")
    utils.assert_close(out.cpu(), expected)

  def test_64bit_pad_subgraph_caches_and_preserves_runtime_shape(self):
    def fn(x):
      return x * 2 + 1

    executables = []
    orig_dc_call = _backend.dynamic_compiler.DynamicCompiler.__call__

    def capture_executable(dc_self, *args, **kwargs):
      exe = orig_dc_call(dc_self, *args, **kwargs)
      executables.append(exe)
      return exe

    ttc = _backend.dynamic_compiler.tpu_torch_compile
    x3 = torch.arange(12, device=self.device, dtype=torch.int64).reshape(3, 4)
    x6 = torch.arange(24, device=self.device, dtype=torch.int64).reshape(6, 4)
    torch._dynamo.mark_dynamic(x3, 0, min=2, max=8)
    torch._dynamo.mark_dynamic(x6, 0, min=2, max=8)

    with mock.patch.object(
        _backend.dynamic_compiler.DynamicCompiler,
        "__call__",
        autospec=True,
        side_effect=capture_executable,
    ):
      compiled = torch.compile(fn, backend="tpu", fullgraph=True)
      out3 = compiled(x3)
      out6 = compiled(x6)
      utils.assert_close(out3.cpu(), fn(x3.cpu()))
      utils.assert_close(out6.cpu(), fn(x6.cpu()))

      stats_before = torch.tpu._get_cache_stats()
      out3_repeat = compiled(x3)
      stats_after = torch.tpu._get_cache_stats()
      utils.assert_close(out3_repeat.cpu(), fn(x3.cpu()))

    self.assertLen(executables, 1)
    self.assertEqual(
        stats_after.num_compilations - stats_before.num_compilations, 0
    )
    self.assertGreaterEqual(
        stats_after.num_cache_hits - stats_before.num_cache_hits, 1
    )

    dyn_exec = executables[0]
    (padded_x3,) = dyn_exec._compile_and_execute_pad_subgraph([x3], 0)
    self.assertEqual(padded_x3.shape, (3, 4))
    self.assertTrue(ttc.is_device_shape_dynamic(padded_x3))
    self.assertTrue(
        ttc.has_matching_dynamic_device_bounds(padded_x3, ([0], [8]))
    )
    utils.assert_close(padded_x3.cpu(), x3.cpu())

    (reused_x3,) = dyn_exec._compile_and_execute_pad_subgraph([padded_x3], 0)
    self.assertIs(reused_x3, padded_x3)

    # A dynamic 64-bit buffer produced under a smaller bound (<=4) is re-padded
    # — never reused raw — to the exact bound of a multi-bucket executable
    # (<=128 / <=256), and its values stay bit-exact.
    @torch.compile(backend="tpu", fullgraph=True)
    def produce_small(x):
      return x + 10

    x_small = torch.arange(32, device=self.device, dtype=torch.int64).reshape(
        2, 16
    )
    torch._dynamo.mark_dynamic(x_small, 0, min=2, max=4)
    y_small = produce_small(x_small)
    self.assertTrue(ttc.is_device_shape_dynamic(y_small))
    self.assertTrue(ttc.has_matching_dynamic_device_bounds(y_small, ([0], [4])))
    self.assertFalse(
        ttc.has_matching_dynamic_device_bounds(y_small, ([0], [128]))
    )

    mb_executables = []

    def capture_mb_executable(dc_self, *args, **kwargs):
      exe = orig_dc_call(dc_self, *args, **kwargs)
      mb_executables.append(exe)
      return exe

    torch._dynamo.mark_dynamic(y_small, 0, min=2, max=256)
    with mock.patch.object(
        _backend.dynamic_compiler.DynamicCompiler,
        "__call__",
        autospec=True,
        side_effect=capture_mb_executable,
    ):
      compiled_mb = torch.compile(
          fn,
          backend="tpu",
          fullgraph=True,
          options={"num_dynamic_buckets": 2, "dynamic_bucket_size": 128},
      )
      out_mb = compiled_mb(y_small)
    utils.assert_close(out_mb.cpu(), fn(y_small.cpu()))

    mb_exec = mb_executables[0]
    (b0_padded,) = mb_exec._compile_and_execute_pad_subgraph([y_small], 0)
    self.assertIsNot(b0_padded, y_small)
    self.assertTrue(
        ttc.has_matching_dynamic_device_bounds(b0_padded, ([0], [128]))
    )
    utils.assert_close(b0_padded.cpu(), y_small.cpu())

    (b0_reused,) = mb_exec._compile_and_execute_pad_subgraph([b0_padded], 0)
    self.assertIs(b0_reused, b0_padded)

    (b1_repadded,) = mb_exec._compile_and_execute_pad_subgraph([b0_padded], 1)
    self.assertIsNot(b1_repadded, b0_padded)
    self.assertTrue(
        ttc.has_matching_dynamic_device_bounds(b1_repadded, ([0], [256]))
    )
    utils.assert_close(b1_repadded.cpu(), y_small.cpu())

  def test_mixed_int64_and_bfloat16_per_tensor_pad_real_to_static(self):
    vocab_size, hidden_dim, num_heads, head_dim = 128, 64, 4, 16
    embed_w = (
        torch.randn(
            vocab_size, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )
    wq = (
        torch.randn(
            hidden_dim, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )
    wk = (
        torch.randn(
            hidden_dim, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )
    wv = (
        torch.randn(
            hidden_dim, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )

    def attn_step(input_ids, k_cache, v_cache):
      h = torch.nn.functional.embedding(input_ids, embed_w)
      b, s, _ = h.shape
      q = (h @ wq).view(b, s, num_heads, head_dim).transpose(1, 2)
      kn = (h @ wk).view(b, s, num_heads, head_dim).transpose(1, 2)
      vn = (h @ wv).view(b, s, num_heads, head_dim).transpose(1, 2)
      k_full = torch.cat([k_cache, kn], dim=2)
      v_full = torch.cat([v_cache, vn], dim=2)
      return torch.nn.functional.scaled_dot_product_attention(
          q, k_full, v_full, is_causal=False
      )

    def make_inputs(seq_len, kv_len):
      ids = torch.randint(
          0, vocab_size, (2, seq_len), dtype=torch.int64, device=self.device
      )
      kc = torch.randn(
          2,
          num_heads,
          kv_len,
          head_dim,
          dtype=torch.bfloat16,
          device=self.device,
      )
      vc = torch.randn(
          2,
          num_heads,
          kv_len,
          head_dim,
          dtype=torch.bfloat16,
          device=self.device,
      )
      torch._dynamo.mark_dynamic(ids, 1, min=2, max=32)
      torch._dynamo.mark_dynamic(kc, 2, min=8, max=64)
      torch._dynamo.mark_dynamic(vc, 2, min=8, max=64)
      return ids, kc, vc

    ttc = _backend.dynamic_compiler.tpu_torch_compile
    orig_dc_call = _backend.dynamic_compiler.DynamicCompiler.__call__
    orig_get_pad = ttc.get_dynamic_pad_module
    executables = []
    pad_call_infos = []
    debugs = []

    def capture_dc(dc_self, *args, **kwargs):
      exe = orig_dc_call(dc_self, *args, **kwargs)
      executables.append(exe)
      return exe

    def wrap_get_pad(tensor_info, bounds_list, *args, **kwargs):
      pad_call_infos.append(list(tensor_info))
      return orig_get_pad(tensor_info, bounds_list, *args, **kwargs)

    compiled = torch.compile(
        attn_step,
        backend="tpu",
        fullgraph=True,
        options={"debug_callback": debugs.append},
    )

    in_7 = make_inputs(7, 24)
    in_19 = make_inputs(19, 40)
    in_7_repeat = make_inputs(7, 48)

    with (
        mock.patch.object(
            _backend.dynamic_compiler.DynamicCompiler,
            "__call__",
            autospec=True,
            side_effect=capture_dc,
        ),
        mock.patch.object(
            ttc, "get_dynamic_pad_module", side_effect=wrap_get_pad
        ),
    ):
      out_7 = compiled(*in_7)
      out_19 = compiled(*in_19)
      stats_before = torch.tpu._get_cache_stats()
      out_7_repeat = compiled(*in_7_repeat)
      stats_after = torch.tpu._get_cache_stats()

    utils.assert_close(
        out_7.float().cpu(),
        attn_step(*in_7).float().cpu(),
        rtol=2e-2,
        atol=2e-2,
    )
    utils.assert_close(
        out_19.float().cpu(),
        attn_step(*in_19).float().cpu(),
        rtol=2e-2,
        atol=2e-2,
    )
    utils.assert_close(
        out_7_repeat.float().cpu(),
        attn_step(*in_7_repeat).float().cpu(),
        rtol=2e-2,
        atol=2e-2,
    )

    self.assertLen(executables, 1)
    # Inputs: int64 ids and two bfloat16 caches; only the dtypes that bypass
    # PadRealToStatic reach the pad subgraph.
    expected_pad_dtypes = [
        d
        for d in (torch.int64, torch.bfloat16, torch.bfloat16)
        if _uses_pad_subgraph(d)
    ]
    self.assertEqual(
        [[info.dtype for info in info_list] for info_list in pad_call_infos],
        [expected_pad_dtypes] * 3 if expected_pad_dtypes else [],
    )

    # Repeating S=7 (even with a new KV=48) reuses the main executable from
    # _model_executables and the cached pad module: no new compilation.
    self.assertEqual(
        stats_after.num_compilations - stats_before.num_compilations, 0
    )
    self.assertGreaterEqual(
        stats_after.num_cache_hits - stats_before.num_cache_hits, 1
    )

    mlir_text = debugs[0].stablehlo_forward_text[0]
    self.assertEqual(mlir_text.count("func.func @main"), 1)
    self.assertEqual(
        mlir_text.count("@PadRealToStatic"), 3 - len(expected_pad_dtypes)
    )
    self.assertIn("stablehlo.set_dimension_size", mlir_text)

  def _run_mixed_edge_fn(
      self,
  ) -> tuple[
      Callable[..., torch.Tensor],
      list[tuple[Any, tuple[Any, ...]]],
      list[tuple[tuple[Any, ...], torch.Tensor]],
  ]:
    """Compiles and runs a mixed int64/bfloat16 graph across multiple buckets.

    The graph takes a standalone SymInt, a dynamic bfloat16 view, int64 ids and
    multi-bucket bounds (`opt_bias=None` and `scale` are constant-folded by
    Dynamo and never reach the backend; the packing plan for such inputs is
    covered by test_prepare_input_packing_plan_mixed_inputs).

    Returns:
      The eager function, the recorded `(executable, unpacked args)` of every
      `DynamicTpuCompiledExecutable.__call__`, and `(args, output)` pairs for
      the bucket-0 and bucket-1 calls.
    """
    embed_w = (
        torch.randn(64, 16, dtype=torch.bfloat16, device=self.device) * 0.1
    )

    def mixed_edge_fn(
        opt_bias, sym_offset, kv_transposed_view, scale, input_ids
    ):
      tok_emb = torch.nn.functional.embedding(input_ids, embed_w)
      pooled = kv_transposed_view.mean(dim=1, keepdim=True)
      out = tok_emb * scale + pooled + sym_offset * 0.01
      return out if opt_bias is None else out + opt_bias

    compiled_edge = torch.compile(
        mixed_edge_fn,
        backend="tpu",
        fullgraph=True,
        options={"num_dynamic_buckets": 4},
    )

    def make_edge_args(sym_off, seq_len, kv_len):
      kv_base = torch.randn(
          2, 16, kv_len, dtype=torch.bfloat16, device=self.device
      )
      kv_view = kv_base.transpose(1, 2)
      torch._dynamo.mark_dynamic(kv_view, 1, min=4, max=384)
      ids = torch.randint(
          0, 64, (2, seq_len), dtype=torch.int64, device=self.device
      )
      torch._dynamo.mark_dynamic(ids, 1, min=4, max=384)
      return None, sym_off, kv_view, 0.5, ids

    captured_exe_calls = []
    orig_exe_call = (
        _backend.dynamic_compiler.DynamicTpuCompiledExecutable.__call__
    )

    def record_exe_call(exe_self, *exe_args):
      unpacked = (
          tuple(exe_args[0])
          if len(exe_args) == 1 and isinstance(exe_args[0], (list, tuple))
          else tuple(exe_args)
      )
      captured_exe_calls.append((exe_self, unpacked))
      return orig_exe_call(exe_self, *exe_args)

    # Two distinct sym_off values trigger Dynamo automatic int symbolization.
    args_w0 = make_edge_args(2, 8, 32)
    args_b0 = make_edge_args(3, 12, 40)
    args_b1 = make_edge_args(5, 150, 200)
    with mock.patch.object(
        _backend.dynamic_compiler.DynamicTpuCompiledExecutable,
        "__call__",
        record_exe_call,
    ):
      _ = compiled_edge(*args_w0)
      out_b0 = compiled_edge(*args_b0)
      out_b1 = compiled_edge(*args_b1)
    return (
        mixed_edge_fn,
        captured_exe_calls,
        [(args_b0, out_b0), (args_b1, out_b1)],
    )

  def test_mixed_int64_and_bfloat16_multibucket_packing(self):
    """Packs a SymInt, a dynamic bfloat16 view and int64 ids across buckets."""
    mixed_edge_fn, _, runs = self._run_mixed_edge_fn()
    for args, out in runs:
      utils.assert_close(
          out.float().cpu(),
          mixed_edge_fn(*args).float().cpu(),
          rtol=2e-2,
          atol=2e-2,
      )

  def test_mixed_int64_and_bfloat16_rejects_cpu_dynamic_input(self):
    """CPU-tensor validation still covers <8-byte dynamic inputs outside the pad lane."""
    _, captured_exe_calls, _ = self._run_mixed_edge_fn()
    exe, captured_args = captured_exe_calls[-1]
    expected_pad_dtypes = [
        d for d in (torch.bfloat16, torch.int64) if _uses_pad_subgraph(d)
    ]
    self.assertLen(exe._pad_input_indices, len(expected_pad_dtypes))  # pylint: disable=protected-access
    self.assertLen(exe._dynamic_input_indices, 2)  # pylint: disable=protected-access
    bf16_dyn_idx = next(
        idx
        for idx in exe._dynamic_input_indices  # pylint: disable=protected-access
        if captured_args[idx].dtype == torch.bfloat16
    )
    bad_args = list(captured_args)
    bad_args[bf16_dyn_idx] = torch.randn(
        2, 16, 200, dtype=torch.bfloat16, device="cpu"
    ).transpose(1, 2)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Internal exe check.
        ValueError,
        "CPU tensors are not supported in dynamic shapes compilation",
    ):
      exe(*bad_args)

  def test_mixed_int64_and_bfloat16_autograd(self):
    """AOTAutograd forward + backward with requires_grad=True on bfloat16 + dynamic int64."""

    def train_step(input_ids, hidden_act, proj_w):
      emb = torch.nn.functional.embedding(input_ids, proj_w)
      return ((emb + hidden_act) ** 2).sum()

    compiled_train = torch.compile(train_step, backend="tpu", fullgraph=True)
    proj_w = torch.randn(
        64, 16, dtype=torch.bfloat16, device=self.device, requires_grad=True
    )
    for seq_len in (6, 14):
      ids = torch.randint(
          0, 64, (2, seq_len), dtype=torch.int64, device=self.device
      )
      act = torch.randn(
          2,
          seq_len,
          16,
          dtype=torch.bfloat16,
          device=self.device,
          requires_grad=True,
      )
      torch._dynamo.mark_dynamic(ids, 1, min=2, max=32)
      torch._dynamo.mark_dynamic(act, 1, min=2, max=32)

      proj_w.grad = None
      loss = compiled_train(ids, act, proj_w)
      loss.backward()
      got_act_grad = act.grad.float().cpu()
      got_w_grad = proj_w.grad.float().cpu()

      ref_w = proj_w.detach().clone().float().cpu().requires_grad_(True)
      ref_act = act.detach().clone().float().cpu().requires_grad_(True)
      ref_loss = train_step(ids.cpu(), ref_act, ref_w)
      ref_loss.backward()

      utils.assert_close(loss.float().cpu(), ref_loss, rtol=2e-2, atol=2e-2)
      utils.assert_close(got_act_grad, ref_act.grad, rtol=3e-2, atol=3e-2)
      utils.assert_close(got_w_grad, ref_w.grad, rtol=3e-2, atol=3e-2)

  def test_prepare_input_packing_plan_mixed_inputs(self):
    """Checks the per-dtype pad/static split, bounds alignment and guard-freeness."""
    seen = {}

    def fw_compiler(graph_module, example_inputs):
      traced = list(example_inputs)
      sym = next(x for x in traced if isinstance(x, torch.SymInt))
      ids_fake = next(
          x
          for x in traced
          if isinstance(x, torch.Tensor) and x.dtype == torch.int64
      )
      kv_fake = next(
          x
          for x in traced
          if isinstance(x, torch.Tensor) and x.dtype == torch.float32
      )
      c128_fake = next(
          x
          for x in traced
          if isinstance(x, torch.Tensor) and x.dtype == torch.complex128
      )
      # Synthetic argument list: None and the Python float never reach the
      # backend from Dynamo, but the packing plan must still skip them.
      synthetic_inputs = [
          None,
          sym,
          kv_fake,
          0.5,
          ids_fake,
          torch.ones(3),
          c128_fake,
      ]
      symbols = set(sym.node.expr.free_symbols)
      for tensor in (ids_fake, kv_fake, c128_fake):
        for d in tensor.shape:
          if isinstance(d, torch.SymInt):
            symbols |= d.node.expr.free_symbols

      def make_spec(upper):
        bounds = _backend.dynamic_compiler.SymbolBounds(
            {s: ValueRanges(2, upper) for s in symbols}
        )
        bounded_inputs, input_bounds = (
            _backend.dynamic_compiler._create_bounded_model_inputs(  # pylint: disable=protected-access
                synthetic_inputs, bounds
            )
        )
        return _backend.dynamic_compiler.BucketCompilationSpec(
            symbol_bounds=bounds,
            graph_module=graph_module,
            bounded_model_inputs=bounded_inputs,
            model_input_bounds=input_bounds,
        )

      specs = [make_spec(64), make_spec(128)]
      shape_env = sym.node.shape_env
      guards_before = len(shape_env.guards)
      exe = _backend.dynamic_compiler.DynamicTpuCompiledExecutable(
          static_compiler=_backend.compiler.StaticCompiler(),
          example_inputs=synthetic_inputs,
          bucket_specs=specs,
          output_sym_shapes=[],
          is_fwd=True,
          view_arg_indices=set(),
      )
      seen["guards_added"] = len(shape_env.guards) - guards_before
      seen["exe"] = exe
      seen["specs"] = specs
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(ids, kv, c128, n):
      return ids.sum() + kv.sum() + c128.abs().sum() + n

    f(
        torch.ones((2, 6), dtype=torch.int64),
        torch.ones((3, 10, 8)),
        torch.ones((4, 5), dtype=torch.complex128),
        3,
    )

    exe, specs = seen["exe"], seen["specs"]
    # Building the plan (including the zero-element check) must not add guards.
    self.assertEqual(seen["guards_added"], 0)

    # Model input positions skip the None: sym -> 0, kv -> 1, 0.5 -> 2,
    # ids -> 3, static tensor -> 4, c128 -> 5. Tensors in argument order as
    # (arg index, model input position, dtype, rank, dynamic); dynamic tensors
    # whose dtype bypasses PadRealToStatic go through the pad subgraph.
    tensor_entries = [
        (2, 1, torch.float32, 3, True),
        (4, 3, torch.int64, 2, True),
        (5, 4, torch.float32, 1, False),
        (6, 5, torch.complex128, 2, True),
    ]
    pad_entries = [
        e for e in tensor_entries if e[4] and _uses_pad_subgraph(e[2])
    ]
    pad_arg_indices = [i for i, *_ in pad_entries]
    self.assertEqual(exe._model_inputs_template_length, 6)  # pylint: disable=protected-access
    self.assertEqual(exe._dynamic_scalar_map, [(1, 0)])  # pylint: disable=protected-access
    self.assertEqual(exe._static_scalar_map, [(3, 2)])  # pylint: disable=protected-access
    self.assertEqual(
        exe._static_tensor_map,  # pylint: disable=protected-access
        [(i, t) for i, t, *_ in tensor_entries if i not in pad_arg_indices],
    )
    self.assertEqual(exe._pad_input_indices, pad_arg_indices)  # pylint: disable=protected-access
    self.assertEqual(
        exe._pad_tensor_map,  # pylint: disable=protected-access
        [(pad_idx, t) for pad_idx, (_, t, *_) in enumerate(pad_entries)],
    )
    self.assertEqual(exe._dynamic_input_indices, [2, 4, 6])  # pylint: disable=protected-access
    self.assertEqual(exe._dynamic_target_indices, {1, 3, 5})  # pylint: disable=protected-access

    # model_input_bounds has one entry per argument except two for the SymInt
    # (value + runtime size placeholder), so every tensor after it sits at
    # argument index + 1 (int64 at 5 and complex128 at 7).
    for spec, pad_bounds in zip(specs, exe._bucket_pad_tensor_bounds):  # pylint: disable=protected-access
      self.assertLen(pad_bounds, len(pad_entries))
      for bounds, (arg_idx, *_) in zip(pad_bounds, pad_entries):
        self.assertIs(bounds, spec.model_input_bounds[arg_idx + 1])
    for pad_idx, (_, _, _, rank, _) in enumerate(pad_entries):
      self.assertEqual(
          [
              list(b[pad_idx].upper_bounds)
              for b in exe._bucket_pad_tensor_bounds  # pylint: disable=protected-access
          ],
          [[64] * rank, [128] * rank],
      )
      self.assertEqual(
          list(exe._bucket_pad_tensor_bounds[0][pad_idx].dynamic_dims),  # pylint: disable=protected-access
          list(range(rank)),
      )

  def test_64bit_input_with_two_dynamic_dims(self):
    def fn(x):
      return x * 2 + 1

    ttc = _backend.dynamic_compiler.tpu_torch_compile
    orig_get_pad = ttc.get_dynamic_pad_module
    pad_calls = []

    def wrap_get_pad(tensor_info, bounds_list, *args, **kwargs):
      pad_calls.append((
          [tuple(info.shape) for info in tensor_info],
          [(list(b.dynamic_dims), list(b.upper_bounds)) for b in bounds_list],
      ))
      return orig_get_pad(tensor_info, bounds_list, *args, **kwargs)

    def make_input(rows, cols):
      x = torch.arange(
          rows * cols, device=self.device, dtype=torch.int64
      ).reshape(rows, cols)
      torch._dynamo.mark_dynamic(x, 0, min=2, max=8)
      torch._dynamo.mark_dynamic(x, 1, min=2, max=16)
      return x

    x_a = make_input(3, 5)
    x_b = make_input(6, 11)
    with mock.patch.object(
        ttc, "get_dynamic_pad_module", side_effect=wrap_get_pad
    ):
      compiled = torch.compile(fn, backend="tpu", fullgraph=True)
      utils.assert_close(compiled(x_a).cpu(), fn(x_a.cpu()))
      utils.assert_close(compiled(x_b).cpu(), fn(x_b.cpu()))
      stats_before = torch.tpu._get_cache_stats()
      utils.assert_close(compiled(x_a).cpu(), fn(x_a.cpu()))
      stats_after = torch.tpu._get_cache_stats()

    # Each call pads the int64 input to the single bucket's bounds unless int64
    # takes in-graph PadRealToStatic, in which case no pad program runs.
    expected_pad_calls = (
        [
            ([(3, 5)], [([0, 1], [8, 16])]),
            ([(6, 11)], [([0, 1], [8, 16])]),
            ([(3, 5)], [([0, 1], [8, 16])]),
        ]
        if _uses_pad_subgraph(torch.int64)
        else []
    )
    self.assertEqual(pad_calls, expected_pad_calls)
    self.assertEqual(
        stats_after.num_compilations - stats_before.num_compilations, 0
    )

  def test_mixed_int64_and_bfloat16_fed_back_decode_loop(self):
    """Feeds compiled dynamic KV outputs back as inputs across decode steps."""
    vocab_size, hidden_dim, num_heads, head_dim = 128, 512, 4, 128
    embed_w = (
        torch.randn(
            vocab_size, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )
    wk = (
        torch.randn(
            hidden_dim, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )
    wv = (
        torch.randn(
            hidden_dim, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )

    def decode_step(next_token, k_cache, v_cache):
      h = torch.nn.functional.embedding(next_token, embed_w)
      b, s, _ = h.shape
      kn = (h @ wk).view(b, s, num_heads, head_dim).transpose(1, 2)
      vn = (h @ wv).view(b, s, num_heads, head_dim).transpose(1, 2)
      k_full = torch.cat([k_cache, kn], dim=2)
      v_full = torch.cat([v_cache, vn], dim=2)
      scores = torch.matmul(kn, k_full.transpose(-2, -1)) * head_dim**-0.5
      attn = torch.matmul(torch.softmax(scores.float(), dim=-1), v_full.float())
      return attn.to(torch.bfloat16), k_full, v_full

    ttc = _backend.dynamic_compiler.tpu_torch_compile
    orig_get_pad = ttc.get_dynamic_pad_module
    pad_dtypes = []

    def wrap_get_pad(tensor_info, bounds_list, *args, **kwargs):
      pad_dtypes.append([info.dtype for info in tensor_info])
      return orig_get_pad(tensor_info, bounds_list, *args, **kwargs)

    batch, kv_len = 2, 130
    tok = torch.randint(
        0, vocab_size, (batch, 1), dtype=torch.int64, device=self.device
    )
    torch._dynamo.mark_dynamic(tok, 0, min=2, max=8)
    k_cache = torch.randn(
        batch,
        num_heads,
        kv_len,
        head_dim,
        dtype=torch.bfloat16,
        device=self.device,
    )
    v_cache = torch.randn(
        batch,
        num_heads,
        kv_len,
        head_dim,
        dtype=torch.bfloat16,
        device=self.device,
    )
    for cache in (k_cache, v_cache):
      torch._dynamo.mark_dynamic(cache, 0, min=2, max=8)
      torch._dynamo.mark_dynamic(cache, 2, min=16, max=256)

    compiled = torch.compile(decode_step, backend="tpu", fullgraph=True)
    ref_k, ref_v = k_cache.clone(), v_cache.clone()
    k_cur, v_cur = k_cache, v_cache
    step_outputs = []
    with mock.patch.object(
        ttc, "get_dynamic_pad_module", side_effect=wrap_get_pad
    ):
      for step in range(4):
        attn, k_cur, v_cur = compiled(tok, k_cur, v_cur)
        step_outputs.append((attn, k_cur, v_cur))
        self.assertTrue(ttc.is_device_shape_dynamic(k_cur))
        self.assertEqual(
            tuple(k_cur.shape), (batch, num_heads, kv_len + step + 1, head_dim)
        )
        if step == 0:
          compilations_after_first_step = (
              torch.tpu._get_cache_stats().num_compilations
          )
      compilations_after_last_step = (
          torch.tpu._get_cache_stats().num_compilations
      )

    # Only dtypes outside the PadRealToStatic lane (the int64 token tensor
    # today) go through the pad subgraph; the fed-back bfloat16 caches are
    # consumed in-graph. Steps 2-4 compile nothing new.
    expected_pad_dtypes = (
        [torch.int64] if _uses_pad_subgraph(torch.int64) else []
    )
    self.assertEqual(
        pad_dtypes, [expected_pad_dtypes] * 4 if expected_pad_dtypes else []
    )
    self.assertEqual(
        compilations_after_last_step, compilations_after_first_step
    )

    # Eager references are computed after the loop so that their own
    # compilations do not perturb the cache statistics above.
    for attn, k_cur, v_cur in step_outputs:
      ref_attn, ref_k, ref_v = decode_step(tok, ref_k, ref_v)
      utils.assert_close(
          attn.float().cpu(), ref_attn.float().cpu(), rtol=2e-2, atol=2e-2
      )
      utils.assert_close(
          k_cur.float().cpu(), ref_k.float().cpu(), rtol=2e-2, atol=2e-2
      )
      utils.assert_close(
          v_cur.float().cpu(), ref_v.float().cpu(), rtol=2e-2, atol=2e-2
      )

  def test_mixed_int64_and_bfloat16_multibucket_dynamic_reuse(self):
    """Reuses or re-pads dynamic int64 buffers across producer-bound and bucket transitions."""
    vocab_size, hidden_dim = 256, 64
    embed_w = (
        torch.randn(
            vocab_size, hidden_dim, dtype=torch.bfloat16, device=self.device
        )
        * 0.05
    )

    @torch.compile(backend="tpu", fullgraph=True)
    def produce_small_ids(ids):
      return (ids + 3) % vocab_size

    @torch.compile(
        backend="tpu",
        fullgraph=True,
        options={"num_dynamic_buckets": 2, "dynamic_bucket_size": 128},
    )
    def produce_bucket0_ids(ids):
      return (ids + 5) % vocab_size

    def mixed_step(ids, kv):
      tok_mean = torch.nn.functional.embedding(ids, embed_w).mean(dim=(0, 1))
      kv_mean = kv.mean(dim=0)
      return tok_mean + kv_mean

    ttc = _backend.dynamic_compiler.tpu_torch_compile
    orig_get_pad = ttc.get_dynamic_pad_module
    pad_bounds_log = []

    def wrap_get_pad(tensor_info, bounds_list, *args, **kwargs):
      pad_bounds_log.append(
          [(list(b.dynamic_dims), list(b.upper_bounds)) for b in bounds_list]
      )
      return orig_get_pad(tensor_info, bounds_list, *args, **kwargs)

    # 1. Dynamic int64 output bounded to <=4.
    raw_small = torch.randint(
        0, vocab_size, (2, 16), dtype=torch.int64, device=self.device
    )
    torch._dynamo.mark_dynamic(raw_small, 0, min=2, max=4)
    ids_dyn = produce_small_ids(raw_small)
    self.assertTrue(ttc.is_device_shape_dynamic(ids_dyn))
    self.assertTrue(ttc.has_matching_dynamic_device_bounds(ids_dyn, ([0], [4])))

    # 2. Dynamic int64 output bounded to <=128 (Bucket 0 of the consumer).
    raw_b0 = torch.randint(
        0, vocab_size, (6, 16), dtype=torch.int64, device=self.device
    )
    torch._dynamo.mark_dynamic(raw_b0, 0, min=2, max=256)
    ids_b0_dyn = produce_bucket0_ids(raw_b0)
    self.assertTrue(ttc.is_device_shape_dynamic(ids_b0_dyn))
    self.assertTrue(
        ttc.has_matching_dynamic_device_bounds(ids_b0_dyn, ([0], [128]))
    )

    compiled_step = torch.compile(
        mixed_step,
        backend="tpu",
        fullgraph=True,
        options={"num_dynamic_buckets": 3, "dynamic_bucket_size": 128},
    )

    kv_32 = torch.randn(
        32, hidden_dim, dtype=torch.bfloat16, device=self.device
    )
    kv_200 = torch.randn(
        200, hidden_dim, dtype=torch.bfloat16, device=self.device
    )
    torch._dynamo.mark_dynamic(ids_dyn, 0, min=2, max=384)
    torch._dynamo.mark_dynamic(ids_b0_dyn, 0, min=2, max=384)
    torch._dynamo.mark_dynamic(kv_32, 0, min=8, max=384)
    torch._dynamo.mark_dynamic(kv_200, 0, min=8, max=384)

    with mock.patch.object(
        ttc, "get_dynamic_pad_module", side_effect=wrap_get_pad
    ):
      # The <=4 producer buffer is re-padded to the exact bounds of Bucket 0
      # (<=128) and Bucket 1 (<=256).
      out_small_b0 = compiled_step(ids_dyn, kv_32)
      out_small_b1 = compiled_step(ids_dyn, kv_200)
      pad_count_before_b0_reuse = len(pad_bounds_log)

      # Bucket-0 ids_b0_dyn (<=128) reuses its dynamic buffer in Bucket 0 (kv=32)
      # with zero pad-subgraph calls, then re-pads to <=256 in Bucket 1 (kv=200).
      out_b0_reuse = compiled_step(ids_b0_dyn, kv_32)
      pad_count_after_b0_reuse = len(pad_bounds_log)
      out_b0_to_b1 = compiled_step(ids_b0_dyn, kv_200)

    self.assertEqual(pad_count_after_b0_reuse, pad_count_before_b0_reuse)
    if _uses_pad_subgraph(torch.int64):
      self.assertEqual(
          pad_bounds_log,
          [[([0], [128])], [([0], [256])], [([0], [256])]],
      )
    else:
      self.assertEmpty(pad_bounds_log)

    for ids_in, kv_in, got in (
        (ids_dyn, kv_32, out_small_b0),
        (ids_dyn, kv_200, out_small_b1),
        (ids_b0_dyn, kv_32, out_b0_reuse),
        (ids_b0_dyn, kv_200, out_b0_to_b1),
    ):
      ref = mixed_step(ids_in, kv_in)
      utils.assert_close(
          got.float().cpu(), ref.float().cpu(), rtol=2e-2, atol=2e-2
      )

  def test_mixed_high_precision_and_float32_dynamic_inputs(self):
    """Runs float64 and complex64 dynamic inputs next to a float32 PadRealToStatic input."""
    ttc = _backend.dynamic_compiler.tpu_torch_compile
    orig_get_pad = ttc.get_dynamic_pad_module

    def mixed_fn(hp_x, f32_y):
      return hp_x + hp_x, f32_y * 2.0 + 1.0

    pad_dtypes = []

    def wrap_get_pad(tensor_info, bounds_list, *args, **kwargs):
      pad_dtypes.append([info.dtype for info in tensor_info])
      return orig_get_pad(tensor_info, bounds_list, *args, **kwargs)

    # Dynamic complex128 (C128) routing is verified at the FX-pass and
    # packing-plan boundaries; XLA:TPU does not yet support dynamic C128 device
    # buffers at runtime.
    for hp_dtype in (torch.float64, torch.complex64):
      pad_dtypes.clear()
      compiled = torch.compile(mixed_fn, backend="tpu", fullgraph=True)
      for rows in (3, 7):
        hp_cpu = torch.randn(rows, 8, dtype=hp_dtype)
        f32_cpu = torch.randn(rows, 8, dtype=torch.float32)
        hp_tpu = hp_cpu.to(self.device)
        f32_tpu = f32_cpu.to(self.device)
        torch._dynamo.mark_dynamic(hp_tpu, 0, min=2, max=16)
        torch._dynamo.mark_dynamic(f32_tpu, 0, min=2, max=16)

        with mock.patch.object(
            ttc, "get_dynamic_pad_module", side_effect=wrap_get_pad
        ):
          got_hp, got_f32 = compiled(hp_tpu, f32_tpu)

        ref_hp, ref_f32 = mixed_fn(hp_cpu, f32_cpu)
        utils.assert_close(got_hp.cpu(), ref_hp, rtol=1e-5, atol=1e-5)
        utils.assert_close(got_f32.cpu(), ref_f32, rtol=1e-5, atol=1e-5)

      # Only the high-precision input is padded by the pad program (once per
      # distinct runtime shape), and only while its dtype bypasses
      # PadRealToStatic.
      self.assertEqual(
          pad_dtypes,
          [[hp_dtype], [hp_dtype]] if _uses_pad_subgraph(hp_dtype) else [],
      )

  @absltest.skipUnless(
      not _uses_pad_subgraph(torch.int64),
      "int64 dynamic inputs still take the external pad subgraph",
  )
  def test_int64_dynamic_input_uses_in_graph_pad_real_to_static(self):
    """Once libtpu supports it, int64 inputs are padded in-graph without a pad program."""

    def fn(x):
      return x * 2 + 1

    ttc = _backend.dynamic_compiler.tpu_torch_compile
    debugs = []
    x3 = torch.arange(12, device=self.device, dtype=torch.int64).reshape(3, 4)
    x6 = torch.arange(24, device=self.device, dtype=torch.int64).reshape(6, 4)
    torch._dynamo.mark_dynamic(x3, 0, min=2, max=8)
    torch._dynamo.mark_dynamic(x6, 0, min=2, max=8)

    with mock.patch.object(
        ttc, "get_dynamic_pad_module", wraps=ttc.get_dynamic_pad_module
    ) as get_pad:
      compiled = torch.compile(
          fn,
          backend="tpu",
          fullgraph=True,
          options={"debug_callback": debugs.append},
      )
      out3 = compiled(x3)
      out6 = compiled(x6)

    utils.assert_close(out3.cpu(), fn(x3.cpu()))
    utils.assert_close(out6.cpu(), fn(x6.cpu()))
    get_pad.assert_not_called()
    mlir_text = debugs[0].stablehlo_forward_text[0]
    self.assertEqual(mlir_text.count("@PadRealToStatic"), 1)
    self.assertRegex(mlir_text, r"@PadRealToStatic\([^\n]*xi64")


class SymIntArithmeticTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_arange_plus_symint(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[1]
        return torch.arange(1, device=x.device) + s0

    compiled = torch.compile(Model(), backend="tpu")

    x1 = torch.zeros(1, 1024, device="tpu")
    torch._dynamo.mark_dynamic(x1, 1, min=1, max=2048)

    out1 = compiled(x1)
    utils.assert_close(out1, torch.tensor([1024], device="tpu"))

  def test_arange_bounds(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[1]
        return torch.arange(s0, s0 + 1, device=x.device)

    compiled = torch.compile(Model(), backend="tpu")

    x1 = torch.zeros(1, 1024, device="tpu")
    torch._dynamo.mark_dynamic(x1, 1, min=1, max=2048)

    out1 = compiled(x1)
    utils.assert_close(out1, torch.tensor([1024], device="tpu"))

  def test_symint_in_output(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.shape[1] * 2 + 1

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

    x = torch.randn(6, 4, dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(x, 0)

    out = compiled(x)
    expected = x[: x.shape[0] - 1, :] * 2.0
    utils.assert_close(out, expected)

  def test_slice_dynamic_dim_to_static_size(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x[:2, :]

    compiled = torch.compile(Model(), backend="tpu")

    x = torch.randn(5, 4, dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(x, 0, min=3, max=32)

    out = compiled(x)
    expected = x[:2, :]
    utils.assert_close(out, expected)

  def test_select_negative_dim(self):
    class Model(torch.nn.Module):

      def forward(self, grid):
        return grid.select(-1, 1) * grid.select(-1, 2)

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

    x = torch.randn(6, 8, dtype=torch.float32, device=self.device)
    torch._dynamo.mark_dynamic(x, 0, min=4, max=16)

    out = compiled(x)
    expected = x.select(0, 2)
    utils.assert_close(out, expected)

  def test_select_negative_index(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.select(0, -1)

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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

    compiled = torch.compile(Model(), backend="tpu")

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
    compiled = torch.compile(model, backend="tpu")

    for seq_len in [8, 16]:
      x = torch.randn(1, 4, 1, seq_len, dtype=torch.float32, device=self.device)
      torch._dynamo.mark_dynamic(x, 3, min=4, max=32)

      out = compiled(x)
      expected = model(x)
      utils.assert_close(out, expected)

  def test_dynamic_slice_static_buffer_bound_from_slice_clamp(self):
    # Under automatic dynamic shapes the slice clamp refines the range of
    # seq_len to [2, max_len] while the stride-contiguity check guards
    # Ne(seq_len, max_len); the upper bound max_len must still be accepted.
    max_len = 16

    class Model(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.register_buffer("static_table", torch.randn(4, 1, max_len))

      def forward(self, x):
        return x + self.static_table[:, :, : x.shape[-1]]

    model = Model().to(self.device)
    compiled = torch.compile(model, backend="tpu", dynamic=None)

    # The second call recompiles with seq_len dynamic (bounded compile).
    for seq_len in [4, 8, 12]:
      x = torch.randn(1, 4, 1, seq_len, dtype=torch.float32, device=self.device)

      out = compiled(x)
      expected = model(x)
      utils.assert_close(out, expected)


class DynamicConvolutionTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # The parameterized cases compile the same nested functions, whose code
    # objects would otherwise hit Dynamo's recompile limit.
    torch._dynamo.reset()
    tt_testing.reset_eager_state()
    self.device = torch.accelerator.current_accelerator()

  # `x + 1` leaves nonzero values past the runtime size of the bounded input,
  # which the convolution must ignore.
  @parameterized.named_parameters(
      dict(
          # Causal depthwise convolution of LFM2 and Mamba models.
          testcase_name="conv1d_causal_depthwise",
          conv_fn=lambda x, w: torch.nn.functional.conv1d(
              x + 1, w, padding=2, groups=4
          )[..., : x.shape[-1]],
          input_shape=(2, 4, 10),
          other_shapes=[(4, 1, 3)],
          dim=2,
      ),
      dict(
          # Cached decode of LFM2 and Mamba models, which convolves the
          # concatenation of the cached convolution state and the new tokens.
          testcase_name="conv1d_causal_depthwise_with_state",
          conv_fn=lambda x, state, w: torch.nn.functional.conv1d(
              torch.cat([state, x], dim=-1), w, padding=2, groups=4
          ),
          input_shape=(2, 4, 10),
          other_shapes=[(2, 4, 3), (4, 1, 3)],
          dim=2,
      ),
      dict(
          testcase_name="conv1d_batch_with_bias",
          conv_fn=lambda x, w, b: torch.nn.functional.conv1d(
              x + 1, w, b, padding=1
          ),
          input_shape=(5, 4, 10),
          other_shapes=[(6, 4, 3), (6,)],
          dim=0,
      ),
      dict(
          testcase_name="conv2d_strided_with_bias",
          conv_fn=lambda x, w, b: torch.nn.functional.conv2d(
              x + 1, w, b, stride=2, padding=1
          ),
          input_shape=(1, 3, 9, 8),
          other_shapes=[(5, 3, 3, 3), (5,)],
          dim=2,
      ),
      dict(
          testcase_name="conv_transpose1d_strided_with_bias",
          conv_fn=lambda x, w, b: torch.nn.functional.conv_transpose1d(
              x + 1, w, b, stride=2, padding=1, output_padding=1
          ),
          input_shape=(2, 3, 6),
          other_shapes=[(3, 4, 3), (4,)],
          dim=2,
      ),
  )
  def test_convolution_dynamic_input(
      self,
      conv_fn: Callable[..., torch.Tensor],
      input_shape: tuple[int, ...],
      other_shapes: Sequence[tuple[int, ...]],
      dim: int,
  ):
    compiled = torch.compile(conv_fn, backend="tpu")
    # Integer values keep the convolution exact on TPU.
    others = [
        torch.randint(-3, 4, shape, dtype=torch.float32)
        for shape in other_shapes
    ]
    for size in (input_shape[dim], input_shape[dim] - 2):
      shape = list(input_shape)
      shape[dim] = size
      x = torch.randint(-3, 4, shape, dtype=torch.float32)
      device_args = [arg.to(self.device) for arg in (x, *others)]
      # min=3 keeps the strided output sizes above 1, which PyTorch
      # specializes on.
      torch._dynamo.mark_dynamic(device_args[0], dim, min=3, max=32)

      out = compiled(*device_args)
      utils.assert_close(out.cpu(), conv_fn(x, *others))

  # Gradients of convolutions whose input has a bounded spatial or batch
  # dimension. The strided cases have windows that overhang the input, so the
  # input size can't be derived from the output size.
  @parameterized.named_parameters(
      dict(
          testcase_name="conv1d_strided_dilated",
          input_shape=(2, 3, 17),
          weight_shape=(4, 3, 3),
          stride=[2],
          padding=[1],
          dilation=[2],
          dim=2,
      ),
      dict(
          # Causal depthwise convolution of LFM2 and Mamba models.
          testcase_name="conv1d_causal_depthwise",
          input_shape=(2, 4, 10),
          weight_shape=(4, 1, 3),
          padding=[2],
          groups=4,
          dim=2,
      ),
      dict(
          testcase_name="conv1d_grouped_bounded_batch",
          input_shape=(5, 4, 10),
          weight_shape=(6, 2, 3),
          stride=[2],
          groups=2,
          dim=0,
      ),
      dict(
          testcase_name="conv1d_grad_input_only",
          input_shape=(2, 3, 11),
          weight_shape=(4, 3, 4),
          stride=[3],
          padding=[2],
          output_mask=(True, False, False),
          dim=2,
      ),
      dict(
          testcase_name="conv1d_grad_weight_and_bias_only",
          input_shape=(2, 3, 11),
          weight_shape=(4, 3, 4),
          stride=[3],
          padding=[2],
          output_mask=(False, True, True),
          dim=2,
      ),
      dict(
          testcase_name="conv1d_bfloat16",
          input_shape=(2, 3, 12),
          weight_shape=(4, 3, 3),
          stride=[2],
          padding=[1],
          dtype=torch.bfloat16,
          dim=2,
      ),
      dict(
          testcase_name="conv2d_strided",
          input_shape=(1, 3, 9, 8),
          weight_shape=(5, 3, 3, 3),
          stride=[2, 2],
          padding=[1, 1],
          dim=2,
      ),
      dict(
          testcase_name="conv_transpose1d_strided",
          input_shape=(2, 3, 6),
          weight_shape=(3, 4, 3),
          stride=[2],
          padding=[1],
          output_padding=[1],
          transposed=True,
          dim=2,
      ),
  )
  def test_convolution_backward_dynamic_input(
      self,
      input_shape: tuple[int, ...],
      weight_shape: tuple[int, ...],
      dim: int,
      stride: Sequence[int] | None = None,
      padding: Sequence[int] | None = None,
      dilation: Sequence[int] | None = None,
      output_padding: Sequence[int] | None = None,
      transposed: bool = False,
      groups: int = 1,
      output_mask: tuple[bool, bool, bool] = (True, True, True),
      dtype: torch.dtype = torch.float32,
  ):
    num_spatial_dims = len(input_shape) - 2
    stride = stride or [1] * num_spatial_dims
    padding = padding or [0] * num_spatial_dims
    dilation = dilation or [1] * num_spatial_dims
    output_padding = output_padding or [0] * num_spatial_dims
    bias_size = weight_shape[1] * groups if transposed else weight_shape[0]

    def conv_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
      return torch.ops.aten.convolution(
          x,
          w,
          None,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
      )

    def conv_backward_fn(
        grad_output: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
      return torch.ops.aten.convolution_backward(
          grad_output,
          x,
          w,
          [bias_size],
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
          list(output_mask),
      )

    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dc:
      compiled = torch.compile(conv_backward_fn, backend="tpu")
      # Integer values keep the convolution exact on TPU.
      w = torch.randint(-3, 4, weight_shape).to(dtype)
      for size in (
          input_shape[dim],
          input_shape[dim] - 2,
          input_shape[dim] + 3,
      ):
        shape = list(input_shape)
        shape[dim] = size
        x = torch.randint(-3, 4, shape).to(dtype)
        grad_output = torch.randint(-3, 4, conv_fn(x, w).shape).to(dtype)
        expected = conv_backward_fn(grad_output.float(), x.float(), w.float())

        device_args = [arg.to(self.device) for arg in (grad_output, x, w)]
        # min=3 keeps the strided output sizes above 1, which PyTorch
        # specializes on.
        torch._dynamo.mark_dynamic(device_args[0], dim, min=3, max=32)
        torch._dynamo.mark_dynamic(device_args[1], dim, min=3, max=32)
        actual = compiled(*device_args)

        self.assertLen(actual, 3)
        for act, exp in zip(actual, expected):
          if exp is None:
            self.assertIsNone(act)
            continue
          self.assertEqual(act.dtype, dtype)
          utils.assert_close(
              act.cpu().float(),
              exp,
              rtol=1e-2 if dtype == torch.bfloat16 else 0,
          )
      # All sizes run the same bounded compilation.
      self.assertEqual(mock_dc.call_count, 1)

  @parameterized.named_parameters(
      dict(
          testcase_name="conv1d_strided_dilated_with_bias",
          conv_fn=lambda x, w, b: torch.nn.functional.conv1d(
              x, w, b, stride=2, padding=1, dilation=2
          ),
          input_shape=(2, 3, 17),
          other_shapes=[(4, 3, 3), (4,)],
          dim=2,
      ),
      dict(
          # Causal depthwise convolution of LFM2 and Mamba models.
          testcase_name="conv1d_causal_depthwise",
          conv_fn=lambda x, w: torch.nn.functional.conv1d(
              x, w, padding=2, groups=4
          )[..., : x.shape[-1]],
          input_shape=(2, 4, 10),
          other_shapes=[(4, 1, 3)],
          dim=2,
      ),
      dict(
          testcase_name="conv2d_strided_with_bias_bounded_batch",
          conv_fn=lambda x, w, b: torch.nn.functional.conv2d(
              x, w, b, stride=2, padding=1
          ),
          input_shape=(7, 3, 9, 8),
          other_shapes=[(5, 3, 3, 3), (5,)],
          dim=0,
      ),
      dict(
          testcase_name="conv_transpose1d_strided_with_bias",
          conv_fn=lambda x, w, b: torch.nn.functional.conv_transpose1d(
              x, w, b, stride=2, padding=1, output_padding=1
          ),
          input_shape=(2, 3, 8),
          other_shapes=[(3, 4, 3), (4,)],
          dim=2,
      ),
  )
  def test_convolution_autograd_dynamic_input(
      self,
      conv_fn: Callable[..., torch.Tensor],
      input_shape: tuple[int, ...],
      other_shapes: Sequence[tuple[int, ...]],
      dim: int,
  ):
    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dc:
      compiled = torch.compile(conv_fn, backend="tpu")
      # Integer values keep the convolution exact on TPU.
      others = [
          torch.randint(-3, 4, shape, dtype=torch.float32)
          for shape in other_shapes
      ]
      for size in (
          input_shape[dim],
          input_shape[dim] - 2,
          input_shape[dim] + 3,
      ):
        shape = list(input_shape)
        shape[dim] = size
        x = torch.randint(-3, 4, shape, dtype=torch.float32)
        cpu_args = [arg.clone().requires_grad_() for arg in (x, *others)]
        expected = conv_fn(*cpu_args)
        grad_output = torch.randint(-3, 4, expected.shape, dtype=torch.float32)
        expected.backward(grad_output)

        device_args = [
            arg.to(self.device).requires_grad_() for arg in (x, *others)
        ]
        # min=5 keeps the strided dilated output sizes above 1, which PyTorch
        # specializes on.
        torch._dynamo.mark_dynamic(device_args[0], dim, min=5, max=32)
        out = compiled(*device_args)
        out.backward(grad_output.to(self.device))

        utils.assert_close(out.detach().cpu(), expected.detach())
        for device_arg, cpu_arg in zip(device_args, cpu_args):
          utils.assert_close(device_arg.grad.cpu(), cpu_arg.grad)
      # All sizes run the same bounded compilation, whose compiler also
      # compiles the backward graph.
      self.assertEqual(mock_dc.call_count, 1)


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
    compiled = torch.compile(model, backend="tpu")

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
    compiled = torch.compile(model, backend="tpu")

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
    compiled = torch.compile(model, backend="tpu")

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
    compiled = torch.compile(model, backend="tpu")

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
    compiled = torch.compile(model, backend="tpu")

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
    compiled = torch.compile(model, backend="tpu")

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
    compiled = torch.compile(model, backend="tpu")

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

  def test_zero_element_dynamic_input_is_rejected(self):
    def fn(x):
      return x * 2 + 1

    device = torch.accelerator.current_accelerator()
    for dtype in (torch.int64, torch.float32):
      x0 = torch.zeros((0, 6), device=device, dtype=dtype)
      # Static compilation handles zero-element inputs.
      compiled_static = torch.compile(
          fn, backend="tpu", dynamic=False, fullgraph=True
      )
      utils.assert_close(compiled_static(x0).cpu(), fn(x0.cpu()))

      # Bounded dynamism cannot bind a zero-element buffer to a bounded
      # parameter (neither via PadRealToStatic nor via the pad subgraph), so
      # the backend reports it up front instead of failing inside PJRT.
      torch._dynamo.mark_dynamic(x0, 1, min=2, max=8)
      compiled = torch.compile(fn, backend="tpu", fullgraph=True)
      with self.assertRaises(  # ASSERT_RAISES_OK=Python backend error.
          (NotImplementedError, torch._dynamo.exc.BackendCompilerFailed)
      ) as ctx:
        compiled(x0)
      self.assertIn(
          "Zero-element dynamic input tensor at position", str(ctx.exception)
      )
      self.assertIn("(shape [0, s", str(ctx.exception))
      self.assertIn(
          "torch.compile(..., dynamic=False, ...)", str(ctx.exception)
      )

  def test_zero_element_dynamic_input_check_is_guard_free(self):
    """The compile-time rejection reads static/known-zero dims only and adds no guard."""
    seen = {}

    def fw_compiler(graph_module, example_inputs):
      shape_env = next(
          x.node.shape_env
          for x in example_inputs
          if isinstance(x, torch.SymInt)
      )
      guards_before = set(shape_env.guards)
      with self.assertRaises(  # ASSERT_RAISES_OK=Python backend error.
          NotImplementedError
      ) as ctx:
        _backend.dynamic_compiler.DynamicCompiler()(
            graph_module, list(example_inputs)
        )
      seen["message"] = str(ctx.exception)
      seen["guards_before"] = guards_before
      seen["guards_after"] = set(shape_env.guards)
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler))
    def f(x):
      return x * 2 + 1

    x = torch.zeros((0, 6), dtype=torch.int64)
    # Leave upper bound unbounded so SymBoundsManager.add_bound_checks would
    # leak `s0 <= 128` onto shape_env.guards if reached before validation.
    torch._dynamo.mark_dynamic(x, 1)
    f(x)

    self.assertIn("Zero-element dynamic input tensor", seen["message"])
    self.assertEqual(seen["guards_after"], seen["guards_before"])

  def test_runtime_zero_derived_dynamic_input_is_rejected(self):
    """Rejects derived symbolic shapes that evaluate to zero at runtime."""

    def fn(x, w):
      y = x[3:]
      return (y.to(w.dtype) @ w).sum()

    device = torch.accelerator.current_accelerator()
    with torch.fx.experimental._config.patch(backed_size_oblivious=True):  # pylint: disable=protected-access
      for dtype in (torch.int64, torch.float32):
        w = torch.ones(
            (4, 2), device=device, dtype=torch.float32, requires_grad=True
        )
        x5 = torch.ones((5, 4), device=device, dtype=dtype)
        torch._dynamo.mark_dynamic(x5, 0, min=3, max=8)
        compiled = torch.compile(fn, backend="tpu", fullgraph=True)

        # Compile forward and backward at s0=5 where saved activation y is (2, 4).
        compiled(x5, w).backward()

        # At s0=3 (within [3, 8]), the saved activation (y or y.T) enters the
        # compiled backward executable with a zero-sized runtime dimension.
        w.grad = None
        x3 = torch.ones((3, 4), device=device, dtype=dtype)
        loss3 = compiled(x3, w)
        with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python backend error.
            NotImplementedError,
            r"Zero-element dynamic input tensor.*shape \[(?:0, 4|4,"
            r" 0)\].*dynamic=False",
        ):
          loss3.backward()

  def test_dynamic_compiler_options(self):
    def simple(x):
      return x + 1.0

    with mock.patch.object(
        _backend.dynamic_compiler,
        "DynamicCompiler",
        wraps=_backend.dynamic_compiler.DynamicCompiler,
    ) as mock_dyn_compiler:
      compiled = torch.compile(
          simple,
          backend="tpu",
          options={
              "num_dynamic_buckets": 2,
              "bucket_size": 64,
          },
      )
      device = torch.accelerator.current_accelerator()
      t = torch.ones(4, device=device)
      torch._dynamo.mark_dynamic(t, 0, min=2, max=128)
      out = compiled(t)
      utils.assert_close(out, simple(t))
      mock_dyn_compiler.assert_called_once()
      self.assertEqual(mock_dyn_compiler.call_args.kwargs["num_buckets"], 2)
      self.assertEqual(mock_dyn_compiler.call_args.kwargs["bucket_size"], 64)


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

    compiled = torch.compile(fn, backend="tpu")

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

    compiled = torch.compile(fn, backend="tpu")

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
