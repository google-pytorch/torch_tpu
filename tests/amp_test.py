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

from typing import TypeAlias
from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal.compile import tpu_torch_compile
from tests import seed_test_utils

EagerMode: TypeAlias = execution_mode.EagerMode


class AmpTest(seed_test_utils.RepeatableTest):

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
    # Use FX compile mode so that we can inspect the MLIR as well.
    with execution_mode.set_eager_mode(EagerMode._INTERNAL_COMPILE_FX_GRAPH):
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
    # - embedding layer uses f32 (%2)
    # - linear layer converts to bf16 (%3 through %7)
    # - multiplication preserves bf16 (%8)
    # - loss function converts to f32 (%9 through %17)
    expected_mlir = """module @tt_jit_build_mlir_compile_mlir_as_strided {
  func.func @main(%arg0: tensor<10xi64>, %arg1: tensor<10x10xf32>, %arg2: tensor<10x10xf32>, %arg3: tensor<10x10xf32>, %arg4: tensor<10xf32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0x7FC00000> : tensor<100xf32>
    %cst_0 = stablehlo.constant dense<-1.000000e+00> : tensor<10x10xbf16>
    %c = stablehlo.constant dense<0> : tensor<i64>
    %cst_1 = stablehlo.constant dense<1.000000e+02> : tensor<f32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.convert %arg3 : (tensor<10x10xf32>) -> tensor<10x10xbf16>
    %1 = stablehlo.transpose %0, dims = [1, 0] : (tensor<10x10xbf16>) -> tensor<10x10xbf16>
    %2 = "stablehlo.gather"(%arg2, %arg0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 10>}> : (tensor<10x10xf32>, tensor<10xi64>) -> tensor<10x10xf32>
    %3 = stablehlo.convert %2 : (tensor<10x10xf32>) -> tensor<10x10xbf16>
    %4 = stablehlo.convert %arg4 : (tensor<10xf32>) -> tensor<10xbf16>
    %5 = stablehlo.dot %3, %1, precision = [DEFAULT, DEFAULT] : (tensor<10x10xbf16>, tensor<10x10xbf16>) -> tensor<10x10xbf16>
    %6 = stablehlo.broadcast_in_dim %4, dims = [1] : (tensor<10xbf16>) -> tensor<10x10xbf16>
    %7 = stablehlo.add %6, %5 : tensor<10x10xbf16>
    %8 = stablehlo.multiply %7, %cst_0 : tensor<10x10xbf16>
    %9 = stablehlo.convert %8 : (tensor<10x10xbf16>) -> tensor<10x10xf32>
    %10 = stablehlo.subtract %9, %arg1 : tensor<10x10xf32>
    %11 = stablehlo.multiply %10, %10 : tensor<10x10xf32>
    %12 = stablehlo.reduce(%11 init: %cst_2) applies stablehlo.add across dimensions = [0, 1] : (tensor<10x10xf32>, tensor<f32>) -> tensor<f32>
    %13 = stablehlo.divide %12, %cst_1 : tensor<f32>
    %14 = stablehlo.reshape %13 : (tensor<f32>) -> tensor<1xf32>
    %15 = stablehlo.dynamic_update_slice %cst, %14, %c : (tensor<100xf32>, tensor<1xf32>, tensor<i64>) -> tensor<100xf32>
    %16 = stablehlo.slice %15 [0:1] : (tensor<100xf32>) -> tensor<1xf32>
    %17 = stablehlo.reshape %16 : (tensor<1xf32>) -> tensor<f32>
    return %17 : tensor<f32>
  }
}"""
    self.assertEqual(mlir_text.strip(), expected_mlir.strip())


if __name__ == "__main__":
  absltest.main()
