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
from torch_tpu._internal.utils import utils


class CompileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
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

  def test_rng_with_generator(self):
    backend = _backend.TpuBackend(debug=True, dynamism=True)

    @torch.compile(backend=backend, dynamic=True)
    def fn(x, gen):
      y = torch.rand(x.size(0), generator=gen, device=self.device)
      return torch.matmul(x, y.unsqueeze(0))

    x = torch.randn(16, 1, device=self.device)
    torch._dynamo.mark_dynamic(x, 0)
    gen = torch.Generator(device=self.device)

    out = fn(x, gen)
    self.assertEqual(tuple(out.shape), (16, 16))

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


class SymIntArithmeticTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_arange_plus_symint(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[1]
        return torch.arange(1, device=x.device) + s0

    tpu_backend = _backend.TpuBackend(dynamism=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

    x1 = torch.zeros(1, 1024, device="tpu")
    torch._dynamo.mark_dynamic(x1, 1, min=1, max=2048)

    out1 = compiled(x1)
    utils.assert_close(out1, torch.tensor([1024], device="tpu"))

  def test_arange_bounds(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        s0 = x.shape[1]
        return torch.arange(s0, s0 + 1, device=x.device)

    tpu_backend = _backend.TpuBackend(dynamism=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

    x1 = torch.zeros(1, 1024, device="tpu")
    torch._dynamo.mark_dynamic(x1, 1, min=1, max=2048)

    out1 = compiled(x1)
    utils.assert_close(out1, torch.tensor([1024], device="tpu"))


class DynamicReshapeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()

  def test_squeeze(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.squeeze(0)

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

  def test_multi_dynamic_reshape_unambiguous(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.reshape(x.shape[0], 5, x.shape[1], 1)

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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

    tpu_backend = _backend.TpuBackend(dynamism=True, debug=True)
    compiled = torch.compile(Model(), backend=tpu_backend)

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


if __name__ == "__main__":
  absltest.main()
