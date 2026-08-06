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

"""Directly test the PyBind11 API for compiled mode, without using Dynamo."""

import re
import textwrap
from typing import TypeAlias
from absl.testing import absltest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.device_utils import annotations
from torch_tpu._internal.utils import test_utils as utils

EagerMode: TypeAlias = execution_mode.EagerMode


@torch.compile
def global_compile_plus_one(x):
  return x + 1


def get_mock_lookup_backend():
  return absltest.mock.patch(
      'torch._dynamo.backends.registry.lookup_backend',
      wraps=torch._dynamo.backends.registry.lookup_backend,
  )


def eager_mode_defer_all():
  """Enable EagerMode.INTERNAL_DEFER_ALL.

  This prevents tensor materialization during Python execution and enables
  `build_mlir` to trace the deferred ops and generate MLIR.

  Returns:
    A context manager for setting the execution mode.
  """
  return execution_mode.set_eager_mode(EagerMode.INTERNAL_DEFER_ALL)


# TODO: add more test coverage for the direct compile API.
class CompileApiTest(absltest.TestCase):

  def test_traverse_and_compile_with_forced_layout(self):
    # Case 1: Force default layout [1, 0]. Execution should PASS.
    with eager_mode_defer_all():
      x1 = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      z1 = x1 + x1

    compile_result_pass = tpu_torch_compile.traverse_and_compile(
        [z1], [x1], argument_layouts=[[1, 0]]
    )
    self.assertIsNotNone(compile_result_pass.executable)

    input_tensor = torch.ones(2, 3, device=torch.device('tpu'))
    input_tensor.cpu()  # Force materialization

    input_layout = tpu_torch_compile.get_device_layout_if_materialized(
        input_tensor
    )
    self.assertIsNotNone(input_layout)
    self.assertEqual(input_layout[0], [1, 0])

    results_pass = tpu_torch_compile.execute(
        compile_result_pass.executable, [input_tensor]
    )
    self.assertLen(results_pass, 1)
    results_pass[0].cpu()  # Verify it passes

    # Case 2: Force non-default layout [0, 1]. Execution should FAIL with
    # default input.
    with eager_mode_defer_all():
      x2 = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      z2 = x2 + x2

    compile_result_fail = tpu_torch_compile.traverse_and_compile(
        [z2], [x2], argument_layouts=[[0, 1]]
    )
    self.assertIsNotNone(compile_result_fail.executable)

    results_fail = tpu_torch_compile.execute(
        compile_result_fail.executable, [input_tensor]
    )
    self.assertLen(results_fail, 1)

    with self.assertRaises(RuntimeError) as err:
      results_fail[0].cpu()

    self.assertIn('layout', str(err.exception).lower())

  def test_compilation_cache_with_forced_layouts(self):
    # Enable cache (should be enabled by default, but let's be sure)
    torch.tpu._set_allow_cache(True)
    torch.tpu._clear_cache()

    initial_hits = torch.tpu._get_cache_hits()
    initial_misses = torch.tpu._get_cache_misses()

    # Compile Model 1 with forced layout [1, 0]
    with eager_mode_defer_all():
      x1 = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      z1 = x1 + x1

    # First compilation: should be a MISS
    compile_result1 = tpu_torch_compile.traverse_and_compile(
        [z1], [x1], argument_layouts=[[1, 0]]
    )
    self.assertIsNotNone(compile_result1.executable)

    self.assertEqual(torch.tpu._get_cache_hits(), initial_hits)
    self.assertEqual(torch.tpu._get_cache_misses(), initial_misses + 1)

    # Compile Model 2 with same forced layout [1, 0] (same graph structure)
    with eager_mode_defer_all():
      x2 = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      z2 = x2 + x2

    # Second compilation with same layout: should be a HIT
    compile_result2 = tpu_torch_compile.traverse_and_compile(
        [z2], [x2], argument_layouts=[[1, 0]]
    )
    self.assertIsNotNone(compile_result2.executable)

    self.assertEqual(torch.tpu._get_cache_hits(), initial_hits + 1)
    self.assertEqual(torch.tpu._get_cache_misses(), initial_misses + 1)

    # Compile Model 3 with DIFFERENT forced layout [0, 1]
    with eager_mode_defer_all():
      x3 = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      z3 = x3 + x3

    # Third compilation with different layout: should be a MISS
    compile_result3 = tpu_torch_compile.traverse_and_compile(
        [z3], [x3], argument_layouts=[[0, 1]]
    )
    self.assertIsNotNone(compile_result3.executable)

    self.assertEqual(torch.tpu._get_cache_hits(), initial_hits + 1)
    self.assertEqual(torch.tpu._get_cache_misses(), initial_misses + 2)

  def test_traverse_and_compile_skip_middle_layout(self):
    with eager_mode_defer_all():
      x = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      y = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      z = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      out = (x + y) * z

    # 3 arguments: x, y, z. Skip y (middle one).
    compile_result = tpu_torch_compile.traverse_and_compile(
        [out], [x, y, z], argument_layouts=[[0, 1], [], [0, 1]]
    )
    self.assertIsNotNone(compile_result.executable)

    # Verify execution with matching layouts passes
    layout = annotations.TpuLayout(minor_to_major=[0, 1])
    with annotations.LayoutContext(layout):
      input_x = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      input_z = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))

    # y keeps default layout (usually [1, 0] for 2x3)
    input_y = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))

    results = tpu_torch_compile.execute(
        compile_result.executable, [input_x, input_y, input_z]
    )
    self.assertLen(results, 1)
    results[0].cpu()  # Verify it passes

  def test_layout_context_auto_extracted_in_static_compile(self):
    layout = annotations.TpuLayout(minor_to_major=[0, 1], tiles=[[8]])
    with annotations.LayoutContext(layout):
      x = torch.ones(16, 128, device='cpu').to(device=torch.device('tpu'))

    def fn(val):
      return val + 1.0

    gm = make_fx(fn)(x)
    compiler_instance = compiler.StaticCompiler()
    executable = compiler_instance(gm, [x])

    param_layouts = executable.parameter_layouts
    self.assertLen(param_layouts, 1)
    self.assertIsNotNone(param_layouts[0])
    minor_to_major, _, _ = param_layouts[0]
    self.assertEqual(minor_to_major, [0, 1])

  def test_build_mlir(self):
    with eager_mode_defer_all():
      x = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      y = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      z = x + y
    result_tensors = [z]
    argument_tensors = [x, y]
    mlir = tpu_torch_compile.build_mlir(result_tensors, argument_tensors)
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)
    self.assertIn(
        'func.func @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>)',
        mlir_text,
    )
    self.assertIn('stablehlo.add', mlir_text)

  def test_extra_input_to_build_mlir(self):
    with eager_mode_defer_all():
      x = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      y = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      extra = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      z = x + y
    result_tensors = [z]
    argument_tensors = [x, y, extra]
    mlir = tpu_torch_compile.build_mlir(result_tensors, argument_tensors)
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)
    self.assertIn('func @main', mlir_text)
    self.assertIn('%arg2', mlir_text)

  def test_compile_backend_defaults_to_tpu(self):
    with get_mock_lookup_backend() as mock_lookup_backend:
      x = torch.ones(10, device=torch.device('tpu'))
      torch.compile(lambda arg: arg + 1)(x)

      mock_lookup_backend.assert_called_with('tpu')

  def test_global_compile_decorator_backend_defaults_to_tpu(self):
    with get_mock_lookup_backend() as mock_lookup_backend:
      x = torch.ones(10, device=torch.device('tpu'))
      global_compile_plus_one(x)

      mock_lookup_backend.assert_called_with('tpu')

  def test_compile_explicit_backend_is_respected(self):

    with get_mock_lookup_backend() as mock_lookup_backend:
      x = torch.ones(10)
      torch.compile(lambda arg: arg + 1, backend='aot_eager')(x)

      mock_lookup_backend.assert_called_with('aot_eager')

  def test_compile_default_backend_no_recompilations(self):
    num_calls = 0

    def user_context() -> str:
      nonlocal num_calls
      num_calls += 1
      return 'user_context: ' + str(num_calls)

    torch._dynamo.register_hook_for_recompile_user_context(user_context)

    def f(x):
      return x + 1

    for _ in range(10):
      torch.compile(f)(torch.randn(1, 5, device=torch.device('tpu')))
    self.assertEqual(num_calls, 1)

  def test_execute_with_output_shapes(self):
    with eager_mode_defer_all():
      x = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      y = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      z = x + y

    mlir = tpu_torch_compile.build_mlir([z], [x, y])
    executable = tpu_torch_compile.compile_mlir(mlir)

    results = tpu_torch_compile.execute(executable, [x, y], [[10]])
    self.assertLen(results, 1)
    self.assertEqual(results[0].shape, (10,))

  def test_execute_with_smaller_output_shapes(self):
    with eager_mode_defer_all():
      x = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      y = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      z = x + y

    mlir = tpu_torch_compile.build_mlir([z], [x, y])
    executable = tpu_torch_compile.compile_mlir(mlir)

    results = tpu_torch_compile.execute(executable, [x, y], [[5]])
    self.assertLen(results, 1)
    self.assertEqual(results[0].shape, (5,))

  def test_get_or_compile_pad_module(self):
    tensor_info = [([1, 4], torch.int64)]
    bounds_list = [([1], [8])]

    compile_result = tpu_torch_compile.get_or_compile_pad_module(
        tensor_info, bounds_list, build_mlir_module=True
    )

    with self.subTest('Test MLIR Output'):
      mlir_text = tpu_torch_compile.serialize_mlir_text(compile_result.module)
      expected_mlir = textwrap.dedent("""\
          module @pad_module {
            func.func @main(%arg0: tensor<1x4xi64>) -> tensor<1x8xi64> {
              %c = stablehlo.constant dense<0> : tensor<i64>
              %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 4], interior = [0, 0] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1x8xi64>
              return %0 : tensor<1x8xi64>
            }
          }""")
      self.assertEqual(mlir_text.strip(), expected_mlir.strip())
      self.assertIsNotNone(compile_result.executable)

    with self.subTest('Test Execution Results'):
      result = tpu_torch_compile.execute(
          compile_result.executable,
          [torch.ones(1, 4, dtype=torch.int64).to(device=torch.device('tpu'))],
      )
      self.assertLen(result, 1)
      padded_tensor = result[0]
      self.assertEqual(padded_tensor.shape, (1, 8))
      self.assertEqual(padded_tensor.dtype, torch.int64)
      expected = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.int64)
      utils.assert_close(padded_tensor.cpu(), expected)

  def test_get_or_compile_slice_module(self):
    target_shapes = [[1, 4]]
    padded_shapes = [[1, 8]]
    input_scalar_types = [torch.float32]

    compile_result = tpu_torch_compile.get_or_compile_slice_module(
        target_shapes, padded_shapes, input_scalar_types, build_mlir_module=True
    )

    with self.subTest('Test MLIR Output'):
      mlir_text = tpu_torch_compile.serialize_mlir_text(compile_result.module)
      expected_mlir = textwrap.dedent("""\
          module @slice_module {
            func.func @main(%arg0: tensor<1x?xf32, #stablehlo.bounds<?, 8>>) -> tensor<1x4xf32> {
              %c = stablehlo.constant dense<8> : tensor<i32>
              %0 = stablehlo.set_dimension_size %arg0, %c, dim = 1 : (tensor<1x?xf32, #stablehlo.bounds<?, 8>>, tensor<i32>) -> tensor<1x8xf32>
              %1 = stablehlo.slice %0 [0:1, 0:4] : (tensor<1x8xf32>) -> tensor<1x4xf32>
              return %1 : tensor<1x4xf32>
            }
          }""")
      self.assertEqual(mlir_text.strip(), expected_mlir.strip())
      self.assertIsNotNone(compile_result.executable)

    with self.subTest('Test Execution Results'):
      result = tpu_torch_compile.execute(
          compile_result.executable,
          [
              torch.ones(1, 8, dtype=torch.float32).to(
                  device=torch.device('tpu')
              )
          ],
      )
      self.assertLen(result, 1)
      self.assertEqual(result[0].shape, (1, 4))
      self.assertEqual(result[0].dtype, torch.float32)

  def test_make_constant_tensor(self):
    # Arrange
    tpu_device = torch.device('tpu')
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cpu')

    # Act
    y = tpu_torch_compile.make_constant_tensor(x)
    mlir = tpu_torch_compile.build_mlir([y], [])
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)
    actual = y.cpu()

    # Assert
    # y has the same shape and dtype as x.
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(y.dtype, x.dtype)
    # y is on the TPU.
    self.assertEqual(y.device.type, tpu_device.type)

    expected_mlir = """module @tt_jit_torchtpu_internal_constant {
  func.func @main() -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
    return %cst : tensor<2x2xf32>
  }
}"""
    self.assertEqual(mlir_text.strip(), expected_mlir.strip())

    # The logical values of y should be identical to x.
    utils.assert_close(actual=actual, expected=x)

  def test_make_constant_bool_tensor(self):
    # Arrange
    tpu_device = torch.device('tpu')
    x = torch.tensor(
        [[False, True], [True, False]], dtype=torch.bool, device='cpu'
    )

    # Act
    y = tpu_torch_compile.make_constant_tensor(x)
    mlir = tpu_torch_compile.build_mlir([y], [])
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)
    actual = y.cpu()

    # Assert
    # y has the same shape and dtype as x.
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(y.dtype, x.dtype)
    # y is on the TPU.
    self.assertEqual(y.device.type, tpu_device.type)

    # XLA packs booleans into i1s, so we need an extra convert operation to
    # go from 0x00 or 0x01 to False or True.
    # Whether or not the initial constant is an i8 or ui8 depends on the
    # implementation of std::numeric_limits<char>::is_signed in C++.
    expected_mlir_unsigned = """module @tt_jit_torchtpu_internal_constant {
  func.func @main() -> tensor<2x2xi1> {
    %c = stablehlo.constant dense<[[0, 1], [1, 0]]> : tensor<2x2xui8>
    %0 = stablehlo.convert %c : (tensor<2x2xui8>) -> tensor<2x2xi1>
    return %0 : tensor<2x2xi1>
  }
}"""
    expected_mlir_signed = """module @tt_jit_torchtpu_internal_constant {
  func.func @main() -> tensor<2x2xi1> {
    %c = stablehlo.constant dense<[[0, 1], [1, 0]]> : tensor<2x2xi8>
    %0 = stablehlo.convert %c : (tensor<2x2xi8>) -> tensor<2x2xi1>
    return %0 : tensor<2x2xi1>
  }
}"""
    unsigned_matches = mlir_text.strip() == expected_mlir_unsigned.strip()
    signed_matches = mlir_text.strip() == expected_mlir_signed.strip()
    self.assertTrue(unsigned_matches or signed_matches)

    # The logical values of y should be identical to x.
    utils.assert_close(actual=actual, expected=x)

  def test_make_constant_complex_tensor(self):
    # Arrange
    tpu_device = torch.device('tpu')
    x_real = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cpu')
    x_imag = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device='cpu')
    x = torch.complex(x_real, x_imag)

    # Act
    y = tpu_torch_compile.make_constant_tensor(x)
    mlir = tpu_torch_compile.build_mlir([y], [])
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)
    actual = y.cpu()

    # Assert
    # y has the same shape and dtype as x.
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(y.dtype, x.dtype)
    # y is on the TPU.
    self.assertEqual(y.device.type, tpu_device.type)

    expected_mlir = """module @tt_jit_torchtpu_internal_constant {
  func.func @main() -> tensor<2x2xcomplex<f32>> {
    %cst = stablehlo.constant dense<[[(1.000000e+00,5.000000e+00), (2.000000e+00,6.000000e+00)], [(3.000000e+00,7.000000e+00), (4.000000e+00,8.000000e+00)]]> : tensor<2x2xcomplex<f32>>
    return %cst : tensor<2x2xcomplex<f32>>
  }
}"""
    self.assertEqual(mlir_text.strip(), expected_mlir.strip())

    # The logical values of y should be identical to x.
    utils.assert_close(actual=actual, expected=x)

  def test_assign_constant_tensor(self):
    # Arrange
    tpu_device = torch.device('tpu')
    cpu_src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cpu')
    tpu_dst = torch.empty_like(cpu_src, device=tpu_device)

    # Act
    tpu_torch_compile.assign_constant_tensor(cpu_src, tpu_dst)
    mlir = tpu_torch_compile.build_mlir([tpu_dst], [])
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)
    actual = tpu_dst.cpu()

    # Assert
    # tpu_dst has the same shape and dtype as cpu_src.
    self.assertEqual(tpu_dst.shape, (2, 2))
    self.assertEqual(tpu_dst.dtype, torch.float32)
    # tpu_dst is (still) on the TPU.
    self.assertEqual(tpu_dst.device.type, tpu_device.type)

    expected_mlir = """module @tt_jit_torchtpu_internal_constant {
  func.func @main() -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
    return %cst : tensor<2x2xf32>
  }
}"""
    self.assertEqual(mlir_text.strip(), expected_mlir.strip())

    # The logical values of tpu_dst should be identical to cpu_src.
    utils.assert_close(actual=actual, expected=cpu_src)

  def test_assign_constant_tensor_non_contiguous(self):
    # Arrange
    tpu_device = torch.device('tpu')
    cpu_src = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cpu')
    # tpu_dst is non-contiguous, but has the right shape and dtype.
    tpu_dst = torch.empty(3, 2, dtype=torch.float32, device=tpu_device)[1:3, :]
    expected_strides = tpu_dst.stride()
    expected_storage_offset = tpu_dst.storage_offset()
    expected_shape = tpu_dst.shape
    expected_dtype = tpu_dst.dtype

    # Act
    tpu_torch_compile.assign_constant_tensor(cpu_src, tpu_dst)
    mlir = tpu_torch_compile.build_mlir([tpu_dst], [])
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)
    actual = tpu_dst.cpu()

    # Assert
    # tpu_dst's layout metadata should be preserved.
    self.assertEqual(tpu_dst.shape, expected_shape)
    self.assertEqual(tpu_dst.dtype, expected_dtype)
    self.assertEqual(tpu_dst.stride(), expected_strides)
    self.assertEqual(tpu_dst.storage_offset(), expected_storage_offset)

    # tpu_dst is (still) on the TPU.
    self.assertEqual(tpu_dst.device.type, tpu_device.type)

    # The MLIR reflects that the elements of the original tensor (NaNs,
    # 0x7FC00000) which are outside of tpu_dst's view are preserved, using a
    # dynamic_update_slice and slice operation.
    expected_mlir = """module @tt_jit_compile_mlir_as_strided {
  func.func @main() -> tensor<2x2xf32> {
    %cst = stablehlo.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
    %cst_0 = stablehlo.constant dense<0x7FC00000> : tensor<3x2xf32>
    %c = stablehlo.constant dense<1> : tensor<i64>
    %c_1 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.dynamic_update_slice %cst_0, %cst, %c, %c_1 : (tensor<3x2xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<3x2xf32>
    %1 = stablehlo.slice %0 [1:3, 0:2] : (tensor<3x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}"""
    self.assertEqual(mlir_text.strip(), expected_mlir.strip())

    # The logical values of tpu_dst should be identical to cpu_src.
    utils.assert_close(actual=actual, expected=cpu_src)

  def test_dynamic_view_writeback(self):
    with eager_mode_defer_all():
      base = torch.empty(21, device='tpu')
      out_view = base[0:20]
      x = tpu_torch_compile.dynamic_placeholder(
          [20], torch.float32, ([0], [30])
      )
      torch.lgamma(x, out=out_view)

    mlir = tpu_torch_compile.build_mlir([out_view], [x])
    executable = tpu_torch_compile.compile_mlir(mlir)
    self.assertIsNotNone(executable)

  def test_optimization_barrier(self):
    with eager_mode_defer_all():
      inputs = [
          torch.ones(10, device='cpu').to(device=torch.device('tpu')),
          torch.ones(10, device='cpu').to(device=torch.device('tpu')),
      ]
      results = torch.ops.tpu.optimization_barrier(inputs)
    expected_mlir = """%0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<10xf32>, tensor<10xf32>"""

    mlir_text = tpu_torch_compile.serialize_mlir_text(
        tpu_torch_compile.build_mlir(results, inputs)
    )

    self.assertIn(expected_mlir, mlir_text)

  def test_force_strides_no_op(self):
    # Arrange: create a contiguous tensor on CPU and a copy of it on the TPU
    expected = torch.arange(24, device='cpu').view(2, 3, 4)
    x = expected.to(device='tpu')

    # Act: force the strides to be contiguous (they already are)
    y = tpu_torch_compile.force_strides(
        x, expected.stride(), expected.storage_offset()
    )

    # Assert: the tensor metadata is unchanged
    self.assertEqual(y.shape, expected.shape)
    self.assertEqual(y.dtype, expected.dtype)
    self.assertEqual(y.stride(), expected.stride())
    self.assertEqual(y.storage_offset(), expected.storage_offset())

    # Assert: the tensor values are unchanged
    actual = y.cpu()
    utils.assert_close(actual=actual, expected=expected)

  def test_force_strides_permute(self):
    # Arrange: create a permuted tensor on the CPU, and a copy of it
    # on the TPU.
    expected = torch.arange(24, device='cpu').view(2, 3, 4).permute(2, 1, 0)
    x = expected.to(device='tpu')

    # Since the expected tensor is non_overlapping and dense, the original
    # strides are preserved for it in the compiled output. We can assert this
    # by checking the output shape and values.

    # Act: force the strides to match expected
    y = tpu_torch_compile.force_strides(
        x, expected.stride(), expected.storage_offset()
    )

    # Assert: the tensor metadata is changed to match expected
    self.assertEqual(y.shape, expected.shape)
    self.assertEqual(y.dtype, expected.dtype)
    self.assertEqual(y.stride(), expected.stride())
    self.assertEqual(y.storage_offset(), expected.storage_offset())

    # Assert: the tensor values are unchanged
    actual = y.cpu()
    utils.assert_close(actual=actual, expected=expected)

  def test_force_strides_slice(self):
    # Arrange: create a sliced tensor on the CPU, and a contiguous copy of it
    # on the TPU.
    expected = torch.arange(60, device='cpu').view(3, 4, 5)[1:, ::2, :4]
    x = expected.to(device='tpu')
    self.assertTrue(x.is_contiguous())

    # Act: force the strides to match expected
    y = tpu_torch_compile.force_strides(
        x, expected.stride(), expected.storage_offset()
    )

    # Assert: the tensor metadata is changed to match expected
    self.assertEqual(y.shape, expected.shape)
    self.assertEqual(y.dtype, expected.dtype)
    self.assertEqual(y.stride(), expected.stride())
    self.assertEqual(y.storage_offset(), expected.storage_offset())

    # Assert: the tensor values are unchanged
    actual = y.cpu()
    utils.assert_close(actual=actual, expected=expected)

  def test_force_strides_broadcast(self):
    # Arrange: create a broadcasted tensor on the CPU, and a contiguous copy of
    # it on the TPU.
    expected = torch.arange(6, device='cpu').view(2, 3, 1).expand(2, 3, 4)
    x = expected.to(device='tpu')
    self.assertTrue(x.is_contiguous())

    # Act: force the strides to match expected
    y = tpu_torch_compile.force_strides(
        x, expected.stride(), expected.storage_offset()
    )

    # Assert: the tensor metadata is changed to match expected
    self.assertEqual(y.shape, expected.shape)
    self.assertEqual(y.dtype, expected.dtype)
    self.assertEqual(y.stride(), expected.stride())
    self.assertEqual(y.storage_offset(), expected.storage_offset())

    # Assert: the tensor values are unchanged
    actual = y.cpu()
    utils.assert_close(actual=actual, expected=expected)

  def test_force_strides_one_dimension_overlap(self):
    # Arrange: create a tensor on the CPU with overlapping strides on a single
    # dimension
    expected = torch.arange(13, dtype=torch.float32, device='cpu').as_strided(
        size=(2, 3, 4), stride=(6, 1, 1), storage_offset=1
    )
    # Check that the view logic is correct on the CPU.
    # expected[0] and expected[1] each contain overlapping data,
    # but expected[0] and expected[1] are disjoint from each other.
    expected_values = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
            [
                [7.0, 8.0, 9.0, 10.0],
                [8.0, 9.0, 10.0, 11.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
        ],
        dtype=torch.float32,
        device='cpu',
    )
    utils.assert_close(actual=expected, expected=expected_values)
    x = expected.to(device='tpu')
    self.assertTrue(x.is_contiguous())

    # Act: force the strides to match expected
    y = tpu_torch_compile.force_strides(
        x, expected.stride(), expected.storage_offset()
    )

    # Assert: the tensor metadata is changed to match expected
    self.assertEqual(y.shape, expected.shape)
    self.assertEqual(y.dtype, expected.dtype)
    self.assertEqual(y.stride(), expected.stride())
    self.assertEqual(y.storage_offset(), expected.storage_offset())

    # Assert: the tensor values are unchanged
    actual = y.cpu()
    utils.assert_close(actual=actual, expected=expected)

  def test_force_strides_two_dimension_overlap(self):
    # Arrange: create a tensor on the CPU with overlapping strides on multiple
    # dimensions
    expected = torch.arange(12, dtype=torch.float32, device='cpu').as_strided(
        size=(2, 3, 4), stride=(5, 1, 1), storage_offset=1
    )
    # Check that the view logic is correct on the CPU.
    # expected[0] and expected[1] each contain overlapping data, and they also
    # overlap with each other.
    expected_values = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
            [
                [6.0, 7.0, 8.0, 9.0],
                [7.0, 8.0, 9.0, 10.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
        ],
        dtype=torch.float32,
        device='cpu',
    )
    utils.assert_close(actual=expected, expected=expected_values)
    x = expected.to(device='tpu')
    self.assertTrue(x.is_contiguous())

    # Act: force the strides to match expected
    y = tpu_torch_compile.force_strides(
        x, expected.stride(), expected.storage_offset()
    )

    # Assert: the tensor metadata is changed to match expected
    self.assertEqual(y.shape, expected.shape)
    self.assertEqual(y.dtype, expected.dtype)
    self.assertEqual(y.stride(), expected.stride())
    self.assertEqual(y.storage_offset(), expected.storage_offset())

    # Assert: the tensor values are unchanged
    actual = y.cpu()
    utils.assert_close(actual=actual, expected=expected)

  def test_optimization_barrier_on_non_contiguous_tensor(self):
    class Model(torch.nn.Module):

      def forward(self, x):
        return x.t()

    model = Model()
    x = torch.ones((2, 2), device=torch.device('tpu'))
    self.assertEqual(x.shape, (2, 2))

    gm = make_fx(model)(x)

    # Mark the transpose node for recompute
    for node in gm.graph.nodes:
      if 'aten.t' in str(node.target):
        node.meta['recompute'] = (
            torch.utils.checkpoint.CheckpointPolicy.MUST_RECOMPUTE
        )

    # Compile with StaticCompiler.
    compiler_instance = compiler.StaticCompiler()
    x = torch.ones((2, 2), device=torch.device('tpu'))
    # is_fwd=False triggers backward pass compilation which invokes marking of
    # activation checkpoints.
    executable = compiler_instance(gm, [x], is_fwd=False)

    res = executable([x])

    # Assert: the output should preserve the transposed strides (1, 2)
    # instead of being contiguous (2, 1)
    self.assertEqual(res.shape, (2, 2))
    self.assertEqual(res.stride(), (1, 2))

  def test_dynamic_placeholder(self):
    max_dynamic_dim = 100
    static_dim_1 = 128
    static_dim_2 = 64
    logical_dynamic_dim = 4

    with eager_mode_defer_all():
      sizes = [max_dynamic_dim, static_dim_1]
      dtype = torch.bfloat16
      bounds = ([0], [max_dynamic_dim])
      dynamic_placeholder = tpu_torch_compile.dynamic_placeholder(
          sizes, dtype, bounds
      )
      static_placeholder = torch.ones(
          static_dim_1,
          static_dim_2,
          dtype=torch.bfloat16,
          device='cpu',
      ).to(device=torch.device('tpu'))
      res = torch.matmul(dynamic_placeholder, static_placeholder)

    result_tensors = [res]
    argument_tensors = [dynamic_placeholder, static_placeholder]
    mlir = tpu_torch_compile.build_mlir(
        result_tensors, argument_tensors, use_stablehlo_bounds=True
    )
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)

    # Compare complete MLIR (ignoring dynamic module name)
    normalized_mlir_text = re.sub(
        r'module @tt_jit_[a-zA-Z0-9_]+ {',
        'module @tt_jit_compile_api_test_test_dynamic_placeholder_mm {',
        mlir_text,
    )
    expected_mlir = textwrap.dedent("""\
        module @tt_jit_compile_api_test_test_dynamic_placeholder_mm {
          func.func @main(%arg0: tensor<?x128xbf16, #stablehlo.bounds<100, ?>>, %arg1: tensor<128x64xbf16>) -> tensor<?x64xbf16, #stablehlo.bounds<100, ?>> {
            %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<?x128xbf16, #stablehlo.bounds<100, ?>>, tensor<128x64xbf16>) -> tensor<?x64xbf16, #stablehlo.bounds<100, ?>>
            return %0 : tensor<?x64xbf16, #stablehlo.bounds<100, ?>>
          }
        }""")

    self.assertEqual(normalized_mlir_text.strip(), expected_mlir.strip())

    # Compile and execute to verify the result
    executable = tpu_torch_compile.compile_mlir(mlir)

    physical_dynamic_input = torch.randn(
        max_dynamic_dim,
        static_dim_1,
        dtype=torch.bfloat16,
        device=torch.device('tpu'),
    )
    logical_size = torch.tensor(
        logical_dynamic_dim, dtype=torch.int32, device=torch.device('tpu')
    )
    dynamic_input = torch.ops.tpu.set_dimension_logical_size(
        physical_dynamic_input, 0, logical_size
    )

    static_input = torch.randn(
        static_dim_1,
        static_dim_2,
        dtype=torch.bfloat16,
        device=torch.device('tpu'),
    )

    results = tpu_torch_compile.execute(
        executable,
        [dynamic_input, static_input],
        output_shapes=[[logical_dynamic_dim, static_dim_2]],
    )
    self.assertLen(results, 1)
    actual_res = results[0]
    self.assertEqual(actual_res.shape, (logical_dynamic_dim, static_dim_2))

    expected_dynamic_input = physical_dynamic_input[
        :logical_dynamic_dim, :
    ].cpu()
    expected_static_input = static_input.cpu()
    expected_res = torch.matmul(expected_dynamic_input, expected_static_input)

    utils.assert_close(actual_res, expected_res, rtol=1e-2, atol=1e-2)

  def test_compiled_module_output_to_eager_op_dynamic_major_dim(self):
    """Verifies passing compiled module outputs (dynamic PJRT buffers) into eager ops works."""
    max_dynamic_dim = 100
    static_dim_1 = 128
    static_dim_2 = 64
    logical_dynamic_dim = 4

    with eager_mode_defer_all():
      sizes = [max_dynamic_dim, static_dim_1]
      dtype = torch.bfloat16
      bounds = ([0], [max_dynamic_dim])
      dynamic_placeholder = tpu_torch_compile.dynamic_placeholder(
          sizes, dtype, bounds
      )
      static_placeholder = torch.ones(
          static_dim_1,
          static_dim_2,
          dtype=torch.bfloat16,
          device='cpu',
      ).to(device=torch.device('tpu'))
      res = torch.matmul(dynamic_placeholder, static_placeholder)

    result_tensors = [res]
    argument_tensors = [dynamic_placeholder, static_placeholder]
    mlir = tpu_torch_compile.build_mlir(
        result_tensors, argument_tensors, use_stablehlo_bounds=True
    )
    executable = tpu_torch_compile.compile_mlir(mlir)

    physical_dynamic_input = torch.randn(
        max_dynamic_dim,
        static_dim_1,
        dtype=torch.bfloat16,
        device=torch.device('tpu'),
    )
    logical_size = torch.tensor(
        logical_dynamic_dim, dtype=torch.int32, device=torch.device('tpu')
    )
    dynamic_input = torch.ops.tpu.set_dimension_logical_size(
        physical_dynamic_input, 0, logical_size
    )

    static_input = torch.randn(
        static_dim_1,
        static_dim_2,
        dtype=torch.bfloat16,
        device=torch.device('tpu'),
    )

    results = tpu_torch_compile.execute(
        executable,
        [dynamic_input, static_input],
        output_shapes=[[logical_dynamic_dim, static_dim_2]],
    )
    compiled_output = results[0]
    self.assertTrue(tpu_torch_compile.is_device_shape_dynamic(compiled_output))

    # Perform an eager operation on the compiled output
    eager_result = compiled_output + 1.0

    # Verify expected values
    expected_dynamic_input = physical_dynamic_input[
        :logical_dynamic_dim, :
    ].cpu()
    expected_static_input = static_input.cpu()
    expected_res = (
        torch.matmul(expected_dynamic_input, expected_static_input) + 1.0
    )

    utils.assert_close(eager_result.cpu(), expected_res, rtol=1e-2, atol=1e-2)

  def test_compiled_module_output_to_eager_op_dynamic_intermediate_dim(self):
    """Verifies 3D tensor with dynamic middle dimension passed to eager ops works."""
    batch_dim = 8
    max_dynamic_dim = 100
    static_dim_2 = 64
    logical_dynamic_dim = 4

    with eager_mode_defer_all():
      sizes = [batch_dim, max_dynamic_dim, static_dim_2]
      dtype = torch.bfloat16
      bounds = ([1], [max_dynamic_dim])
      dynamic_placeholder = tpu_torch_compile.dynamic_placeholder(
          sizes, dtype, bounds
      )
      res = dynamic_placeholder * 2.0

    result_tensors = [res]
    argument_tensors = [dynamic_placeholder]
    mlir = tpu_torch_compile.build_mlir(
        result_tensors, argument_tensors, use_stablehlo_bounds=True
    )
    executable = tpu_torch_compile.compile_mlir(mlir)

    physical_dynamic_input = torch.randn(
        batch_dim,
        max_dynamic_dim,
        static_dim_2,
        dtype=torch.bfloat16,
        device=torch.device('tpu'),
    )
    logical_size = torch.tensor(
        logical_dynamic_dim, dtype=torch.int32, device=torch.device('tpu')
    )
    dynamic_input = torch.ops.tpu.set_dimension_logical_size(
        physical_dynamic_input, 1, logical_size
    )

    results = tpu_torch_compile.execute(
        executable,
        [dynamic_input],
        output_shapes=[[batch_dim, logical_dynamic_dim, static_dim_2]],
    )
    compiled_output = results[0]
    self.assertEqual(
        compiled_output.shape, (batch_dim, logical_dynamic_dim, static_dim_2)
    )

    # Eager mode op (+ 1.0) on 3D tensor output
    eager_result = compiled_output + 1.0

    expected_input = physical_dynamic_input[:, :logical_dynamic_dim, :].cpu()
    expected_res = (expected_input * 2.0) + 1.0

    utils.assert_close(eager_result.cpu(), expected_res, rtol=1e-2, atol=1e-2)

  def test_compiled_module_output_to_eager_op_dynamic_minor_dim(self):
    """Verifies passing compiled module outputs (dynamic PJRT buffers) into eager ops works."""
    max_dynamic_dim = 256
    static_dim = 128
    logical_dynamic_dim = 4

    with eager_mode_defer_all():
      sizes = [static_dim, max_dynamic_dim]
      dtype = torch.bfloat16
      bounds = ([1], [max_dynamic_dim])
      dynamic_placeholder = tpu_torch_compile.dynamic_placeholder(
          sizes, dtype, bounds
      )
      res = dynamic_placeholder * 2.0

    result_tensors = [res]
    argument_tensors = [dynamic_placeholder]
    mlir = tpu_torch_compile.build_mlir(
        result_tensors, argument_tensors, use_stablehlo_bounds=True
    )
    executable = tpu_torch_compile.compile_mlir(mlir)

    physical_dynamic_input = torch.randn(
        static_dim,
        max_dynamic_dim,
        dtype=torch.bfloat16,
        device=torch.device('tpu'),
    )
    logical_size = torch.tensor(
        logical_dynamic_dim, dtype=torch.int32, device=torch.device('tpu')
    )
    dynamic_input = torch.ops.tpu.set_dimension_logical_size(
        physical_dynamic_input, 1, logical_size
    )

    results = tpu_torch_compile.execute(
        executable,
        [dynamic_input],
        output_shapes=[[static_dim, logical_dynamic_dim]],
    )
    compiled_output = results[0]
    self.assertEqual(compiled_output.shape, (static_dim, logical_dynamic_dim))

    # Eager mode op (+ 1.0) on 2D output with dynamic minor dimension
    eager_result = compiled_output + 1.0

    expected_input = physical_dynamic_input[:, :logical_dynamic_dim].cpu()
    expected_res = (expected_input * 2.0) + 1.0

    utils.assert_close(eager_result.cpu(), expected_res, rtol=1e-2, atol=1e-2)

  def test_eager_module_inputs_different_layouts(self):
    """Verifies eager op execution with input tensors having different layouts."""
    dim0, dim1 = 8, 64
    x = torch.randn(dim0, dim1, dtype=torch.bfloat16, device='tpu')

    # Tensor 1 created with default layout [1, 0]
    t_default = torch.randn(dim0, dim1, dtype=torch.bfloat16, device='cpu').to(
        device=torch.device('tpu')
    )

    # Tensor 2 created with non-default column-major layout [0, 1]
    layout_non_default = annotations.TpuLayout(minor_to_major=[0, 1])
    with annotations.LayoutContext(layout_non_default):
      t_non_default = torch.randn(
          dim0, dim1, dtype=torch.bfloat16, device='cpu'
      ).to(device=torch.device('tpu'))

    # Eager Op 1: operates with default layout input
    eager_result_1 = x + t_default

    # Eager Op 2: operates with non-default layout input
    eager_result_2 = x + t_non_default

    expected_res_1 = x.cpu() + t_default.cpu()
    expected_res_2 = x.cpu() + t_non_default.cpu()

    utils.assert_close(
        eager_result_1.cpu(), expected_res_1, rtol=1e-2, atol=1e-2
    )
    utils.assert_close(
        eager_result_2.cpu(), expected_res_2, rtol=1e-2, atol=1e-2
    )

  def test_get_default_layout(self):
    layout = tpu_torch_compile.get_default_layout(torch.float32, [2, 3])
    self.assertIsInstance(layout, tuple)
    self.assertLen(layout, 3)
    minor_to_major, tiles, element_size_in_bits = layout
    self.assertIsInstance(minor_to_major, list)
    self.assertIsInstance(tiles, list)
    self.assertIsInstance(element_size_in_bits, int)
    self.assertEqual(minor_to_major, [1, 0])

  def test_is_device_shape_dynamic(self):
    x_static = torch.ones(2, 3, device='tpu')
    self.assertFalse(tpu_torch_compile.is_device_shape_dynamic(x_static))

    max_dynamic_dim = 22
    static_dim_1 = 128
    logical_dynamic_dim = 5

    physical_dynamic_input = torch.randn(
        max_dynamic_dim,
        static_dim_1,
        device=torch.device('tpu'),
    )
    logical_size = torch.tensor(
        logical_dynamic_dim, dtype=torch.int32, device=torch.device('tpu')
    )
    dynamic_input = torch.ops.tpu.set_dimension_logical_size(
        physical_dynamic_input, 0, logical_size
    )
    self.assertTrue(tpu_torch_compile.is_device_shape_dynamic(dynamic_input))

  def test_get_dynamic_pad_module(self):
    tensor_info = [([1, 4], torch.int32)]
    bounds_list = [([1], [8])]

    compile_result = tpu_torch_compile.get_dynamic_pad_module(
        tensor_info, bounds_list
    )
    self.assertIsNotNone(compile_result.executable)
    self.assertIsNotNone(compile_result.module)

    mlir_text = tpu_torch_compile.serialize_mlir_text(compile_result.module)
    self.assertIn('stablehlo.set_dimension_size', mlir_text)
    self.assertIn('stablehlo.get_dimension_size', mlir_text)

    input_tensor = torch.ones(1, 4, dtype=torch.int32, device='tpu')
    result = tpu_torch_compile.execute(
        compile_result.executable, [input_tensor]
    )
    self.assertLen(result, 1)
    padded_tensor = result[0]
    self.assertEqual(padded_tensor.shape, (1, 8))
    self.assertTrue(tpu_torch_compile.is_device_shape_dynamic(padded_tensor))

    expected = torch.tensor([[1, 1, 1, 1]], dtype=torch.int32)
    utils.assert_close(padded_tensor.cpu()[:, :4], expected)

  def test_executable_layouts(self):
    with eager_mode_defer_all():
      x = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      y = torch.ones(2, 3, device='cpu').to(device=torch.device('tpu'))
      z = x + y

    mlir = tpu_torch_compile.build_mlir([z], [x, y])
    executable = tpu_torch_compile.compile_mlir(mlir)

    param_layouts = executable.get_parameter_layouts()
    output_layouts = executable.get_output_layouts()

    self.assertIsInstance(param_layouts, list)
    self.assertLen(param_layouts, 2)
    self.assertIsInstance(output_layouts, list)
    self.assertLen(output_layouts, 1)

    for layout in param_layouts + output_layouts:
      if layout is not None:
        self.assertIsInstance(layout, tuple)
        self.assertLen(layout, 3)
        minor_to_major, tiles, element_size_in_bits = layout
        self.assertIsInstance(minor_to_major, list)
        self.assertIsInstance(tiles, list)
        self.assertIsInstance(element_size_in_bits, int)
        self.assertEqual(minor_to_major, [1, 0])


if __name__ == '__main__':
  absltest.main()
