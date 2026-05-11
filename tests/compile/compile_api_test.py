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

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.utils import utils


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
  return execution_mode.eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL)


# TODO: add more test coverage for the direct compile API.
class CompileApiTest(absltest.TestCase):

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

  def test_missing_input_to_build_mlir(self):
    with eager_mode_defer_all():
      x = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      y = torch.ones(10, device='cpu').to(device=torch.device('tpu'))
      z = x + y
    result_tensors = [z]
    argument_tensors = [x]
    with self.assertRaises(RuntimeError) as err:
      tpu_torch_compile.build_mlir(result_tensors, argument_tensors)
    self.assertIn(
        'identified an argument that was not provided',
        str(err.exception),
    )

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

  def test_get_pad_module_mlir(self):
    tensor_info = [([1, 4], torch.int64)]
    bounds_list = [([1], [8])]

    mlir = tpu_torch_compile.get_pad_module_mlir(tensor_info, bounds_list)
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)

    expected_mlir = """module @pad_module {
  func.func @main(%arg0: tensor<1x4xi64>) -> (tensor<1x8xi64>, tensor<i32>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 4], interior = [0, 0] : (tensor<1x4xi64>, tensor<i64>) -> tensor<1x8xi64>
    %1 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<1x4xi64>) -> tensor<i32>
    return %0, %1 : tensor<1x8xi64>, tensor<i32>
  }
}"""
    self.assertEqual(mlir_text.strip(), expected_mlir.strip())

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
    expected_strides = tpu_dst.stride
    expected_storage_offset = tpu_dst.storage_offset
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
    self.assertEqual(tpu_dst.stride, expected_strides)
    self.assertEqual(tpu_dst.storage_offset, expected_storage_offset)

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

  def test_optimization_barrier(self):
    with eager_mode_defer_all():
      inputs = [
          torch.ones(10, device='cpu').to(device=torch.device('tpu')),
          torch.ones(10, device='cpu').to(device=torch.device('tpu')),
      ]
      results = torch.ops.torch_tpu.optimization_barrier(inputs)
    expected_mlir = """%0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<10xf32>, tensor<10xf32>"""

    mlir_text = tpu_torch_compile.serialize_mlir_text(
        tpu_torch_compile.build_mlir(results, inputs)
    )

    self.assertIn(expected_mlir, mlir_text)


if __name__ == '__main__':
  absltest.main()
