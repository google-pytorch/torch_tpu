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

"""Tests that automatic mixed precision (AMP)/torch.autocast works on TPU."""

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile


class AmpTest(absltest.TestCase):

  def test_autocast_is_available(self):
    self.assertTrue(torch.amp.is_autocast_available("tpu"))

  def test_bfloat16_is_preferred(self):
    self.assertEqual(torch.tpu._get_autocast_dtype(), torch.bfloat16)

  def test_autocast_works(self):
    # Define a simple model with autocast enabled.
    # The model uses float32 for its parameters, as normal for training.
    class AutocastModel(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10, dtype=torch.float32)
        self.linear = torch.nn.Linear(10, 10, dtype=torch.float32)
        self.loss_fn = torch.nn.MSELoss()

      @torch.autocast("tpu", dtype=torch.bfloat16)
      def forward(self, x, y, dtypes: dict[str, torch.dtype]):
        # Record the dtypes of each value.
        dtypes["x"] = x.dtype
        dtypes["y"] = y.dtype

        after_embedded = self.embedding(x)
        dtypes["after_embedded"] = after_embedded.dtype

        after_linear = self.linear(after_embedded)
        dtypes["after_linear"] = after_linear.dtype

        after_inv = after_linear * -1.0
        dtypes["after_inv"] = after_inv.dtype

        loss = self.loss_fn(after_inv, y)
        dtypes["loss"] = loss.dtype

        return loss

    # Construct the model and move it to TPU.
    model = AutocastModel().to("tpu")

    # Create dummy input data at full precision (float32) to match the model
    # parameters.
    x = torch.randint(0, 10, (10,), dtype=torch.int64, device="cpu").to("tpu")
    y = torch.randn(10, 10, dtype=torch.float32, device="cpu").to("tpu")

    actual_dtypes = {}

    # Run the forward pass to get the loss.
    # Use DeferAll mode so that we can inspect the MLIR as well.
    with execution_mode.eager_mode(execution_mode.EagerMode.INTERNAL_DEFER_ALL):
      loss = model(x, y, actual_dtypes)

    # Check that the dtypes are as expected for AMP.
    # The initial inputs are not cast.
    self.assertEqual(actual_dtypes["x"], torch.int64)
    self.assertEqual(actual_dtypes["y"], torch.float32)

    # Embedding layers have no specific policy, but the parameters are float32,
    # so the activation is also float32.
    self.assertEqual(actual_dtypes["after_embedded"], torch.float32)

    # Linear layers use the lower_precision_fp policy, which for TPU is
    # bfloat16.
    self.assertEqual(actual_dtypes["after_linear"], torch.bfloat16)

    # Multiplication has no specific policy, so it should preserve the dtype of
    # the inputs.
    self.assertEqual(actual_dtypes["after_inv"], torch.bfloat16)

    # Loss functions always use float32 for the loss.
    self.assertEqual(actual_dtypes["loss"], torch.float32)

    # Check the MLIR that the model generates.
    mlir = tpu_torch_compile.build_mlir(
        [loss], [x, y] + list(model.parameters())
    )
    mlir_text = tpu_torch_compile.serialize_mlir_text(mlir)

    # Operations that get actually performed:
    # - embedding layer uses f32 (%1 through %3)
    # - linear layer converts to bf16 (%4 through %10)
    # - multiplication preserves bf16 (%11 through %15)
    # - y is cast to bf16 (%0)
    # - loss function converts to f32 (%16 through %25)
    expected_mlir = """module @tt_jit_compile_mlir_as_strided {
  func.func @main(%arg0: tensor<10xi64>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>, %arg3: tensor<10x10xf32>, %arg4: tensor<10xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<-1.000000e+00> : tensor<f64>
    %0 = stablehlo.convert %arg3 : (tensor<10x10xf32>) -> tensor<10x10xbf16>
    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<10x10xbf16>) -> tensor<10x10xbf16>
    %2 = stablehlo.reshape %arg0 : (tensor<10xi64>) -> tensor<10x1xi64>
    %3 = "stablehlo.gather"(%arg2, %2) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 10>}> : (tensor<10x10xf32>, tensor<10x1xi64>) -> tensor<10x10xf32>
    %4 = stablehlo.convert %3 : (tensor<10x10xf32>) -> tensor<10x10xbf16>
    %5 = stablehlo.convert %arg4 : (tensor<10xf32>) -> tensor<10xbf16>
    %6 = stablehlo.dot %4, %1, precision = [DEFAULT, DEFAULT] : (tensor<10x10xbf16>, tensor<10x10xbf16>) -> tensor<10x10xbf16>
    %cst_2 = stablehlo.constant dense<1.000000e+00> : tensor<10x10xbf16>
    %7 = stablehlo.multiply %cst_2, %6 : tensor<10x10xbf16>
    %cst_3 = stablehlo.constant dense<1.000000e+00> : tensor<10xbf16>
    %8 = stablehlo.multiply %5, %cst_3 : tensor<10xbf16>
    %9 = stablehlo.broadcast_in_dim %8, dims = [1] : (tensor<10xbf16>) -> tensor<10x10xbf16>
    %10 = stablehlo.add %9, %7 : tensor<10x10xbf16>
    %11 = stablehlo.convert %cst_1 : (tensor<f64>) -> tensor<bf16>
    %12 = stablehlo.broadcast_in_dim %11, dims = [] : (tensor<bf16>) -> tensor<10x10xbf16>
    %13 = stablehlo.multiply %10, %12 : tensor<10x10xbf16>
    %14 = stablehlo.convert %13 : (tensor<10x10xbf16>) -> tensor<10x10xf32>
    %15 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<10x10xf32>
    %16 = stablehlo.multiply %arg1, %15 : tensor<10x10xf32>
    %17 = stablehlo.subtract %14, %16 : tensor<10x10xf32>
    %18 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<10x10xf32>
    %19 = stablehlo.power %17, %18 : tensor<10x10xf32>
    %cst_4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = stablehlo.reduce(%19 init: %cst_4) applies stablehlo.add across dimensions = [0, 1] : (tensor<10x10xf32>, tensor<f32>) -> tensor<f32>
    %cst_5 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
    %21 = stablehlo.divide %20, %cst_5 : tensor<f32>
    %cst_6 = stablehlo.constant dense<0x7FC00000> : tensor<10x10xf32>
    %22 = stablehlo.reshape %cst_6 : (tensor<10x10xf32>) -> tensor<100xf32>
    %23 = stablehlo.reshape %21 : (tensor<f32>) -> tensor<1xf32>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %24 = stablehlo.dynamic_update_slice %22, %23, %c : (tensor<100xf32>, tensor<1xf32>, tensor<i64>) -> tensor<100xf32>
    %25 = stablehlo.slice %24 [0:1] : (tensor<100xf32>) -> tensor<1xf32>
    %26 = stablehlo.reshape %25 : (tensor<1xf32>) -> tensor<f32>
    return %26 : tensor<f32>
  }
}"""
    self.assertEqual(mlir_text.strip(), expected_mlir.strip())


if __name__ == "__main__":
  absltest.main()
