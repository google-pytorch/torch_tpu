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

from absl import flags
from absl.testing import absltest
import torch  # pylint: disable=unused-import
from tests.compile import stablehlo_compile_testing as sct

_ITERATIONS = flags.DEFINE_integer(
    "iterations", 1, "Number of dependent operations in the generated chain."
)


class StableHloCompileTimeTest(sct.StableHloCompileTimeTestBase):
  """Tests the compilation time for various SHLO modules."""

  def _generate_5d_conv(self, d: int, h: int, w: int) -> str:
    """Generates a StableHLO module with a 5D convolution.

    Args:
      d: The depth of the input tensor's spatial dimensions.
      h: The height of the input tensor's spatial dimensions.
      w: The width of the input tensor's spatial dimensions.

    Returns:
      A string containing the complete StableHLO module.
    """
    padded_d, padded_h, padded_w = d + 1, h + 1, w + 1
    return f"""
      module {{
        func.func @main(%arg0: tensor<1x1x{d}x{h}x{w}xf32>, %arg1: tensor<1x1x4x4x4xf32>) -> tensor<1x1x{d}x{h}x{w}xf32> {{
          %c = stablehlo.constant dense<0> : tensor<i64>
          %0 = stablehlo.convert %c : (tensor<i64>) -> tensor<f32>
          %1 = stablehlo.pad %arg0, %0, low = [0, 0, 0, 0, 0], high = [0, 0, 1, 1, 1], interior = [0, 0, 0, 0, 0] : (tensor<1x1x{d}x{h}x{w}xf32>, tensor<f32>) -> tensor<1x1x{padded_d}x{padded_h}x{padded_w}xf32>
          %2 = stablehlo.convolution(%1, %arg1) dim_numbers = [b, f, 0, 1, 2]x[o, i, 0, 1, 2]->[b, f, 0, 1, 2], window = {{stride = [1, 1, 1], pad = [[4, 4], [4, 4], [4, 4]], rhs_dilate = [3, 3, 3]}} {{batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]}} : (tensor<1x1x{padded_d}x{padded_h}x{padded_w}xf32>, tensor<1x1x4x4x4xf32>) -> tensor<1x1x{d}x{h}x{w}xf32>
          return %2 : tensor<1x1x{d}x{h}x{w}xf32>
        }}
      }}
    """

  def _generate_dependent_chain(
      self, iterations: int, m: int, k: int, n: int
  ) -> str:
    """Generates a StableHLO module with a chain of dependent operations.

    Args:
      iterations: The number of times the core operation block is repeated.
      m: The M dimension for the initial matrix multiplication.
      k: The K dimension for the initial matrix multiplication.
      n: The N dimension for the initial matrix multiplication.

    Returns:
      A string containing the complete StableHLO module.
    """
    parts = []
    parts.append(f"""
      module {{
        func.func @main(%arg0: tensor<{m}x{k}xf32>, %arg1: tensor<{n}x{k}xf32>, %arg2: tensor<{n}xf32>) -> tensor<{m}x{n}xf32> {{
          %0 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<{n}x{k}xf32>) -> tensor<{k}x{n}xf32>
          %cst = stablehlo.constant dense<1.000000e+00> : tensor<{m}x{n}xf32>
          %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<{m}x{n}xf32>
          %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) -> tensor<{m}x{n}xf32>
    """)

    var_idx = 1
    for _ in range(iterations):
      chain_input_idx = var_idx
      var_idx += 1
      multiply_1_idx = var_idx
      parts.append(
          f"    %{multiply_1_idx} = stablehlo.multiply %{chain_input_idx}, %cst"
          f" : tensor<{m}x{n}xf32>"
      )
      var_idx += 1
      broadcast_idx = var_idx
      parts.append(
          f"    %{broadcast_idx} = stablehlo.broadcast_in_dim %arg2, dims = [1]"
          f" : (tensor<{n}xf32>) -> tensor<{m}x{n}xf32>"
      )
      var_idx += 1
      multiply_2_idx = var_idx
      parts.append(
          f"    %{multiply_2_idx} = stablehlo.multiply %{broadcast_idx}, %cst_0"
          f" : tensor<{m}x{n}xf32>"
      )
      var_idx += 1
      parts.append(
          f"    %{var_idx} = stablehlo.add %{multiply_1_idx},"
          f" %{multiply_2_idx} : tensor<{m}x{n}xf32>"
      )

    parts.append(f"    return %{var_idx} : tensor<{m}x{n}xf32>")
    parts.append(r"""
        }
      }
    """)
    return "\n".join(parts)

  def test_compile_5d_conv(self):
    """Tests compilation time for a 5D convolution across varied shapes."""

    base_d, base_h, base_w = 10, 11, 12

    def make_shlo(i: int) -> str:
      varied_w = base_w + i
      return self._generate_5d_conv(base_d, base_h, varied_w)

    self.do_test("Compilation/Conv5D", make_shlo)

  def test_compile_dependent_chain(self):
    """Tests compilation time for a chain of operations across varied shapes."""

    base_m, base_k, base_n = 2048, 1024, 3072

    def make_shlo(i: int) -> str:
      varied_m = base_m + i
      return self._generate_dependent_chain(
          _ITERATIONS.value, varied_m, base_k, base_n
      )

    self.do_test("Compilation/DepChain", make_shlo)

  def test_compile_randint_fast(self):
    """Tests compilation time for randint with dtype float32.

    If we are computing random numbers from a [0, r), where r is a number that
    fits in a uint32_t, we do the remainder in this type, which is much faster.
    """

    def make_shlo(i: int) -> str:
      # This is the SHLO for torch.randint().
      return """
module {
  func.func @main(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<3x3xf32>) {
    %c = stablehlo.constant dense<4294967295> : tensor<ui32>
    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3x3xui32>)
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui32>) -> tensor<3x3xui32>
    %1 = stablehlo.remainder %output, %0 : tensor<3x3xui32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
    %3 = stablehlo.convert %1 : (tensor<3x3xui32>) -> tensor<3x3xi64>
    %4 = stablehlo.add %2, %3 : tensor<3x3xi64>
    %5 = stablehlo.convert %4 : (tensor<3x3xi64>) -> tensor<3x3xf32>
    return %output_state, %5 : tensor<2xui64>, tensor<3x3xf32>
  }
}
"""

    self.do_test("Compilation/Randint_Float32_fast", make_shlo)

  def test_compile_randint_slow(self):
    """Tests compilation time for randint with dtype float32."""

    def make_shlo(i: int) -> str:
      # This is the SHLO for torch.randint().
      return """
module {
  func.func @main(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<3x3xf32>) {
    %c = stablehlo.constant dense<4294967296> : tensor<ui64>
    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3x3xui64>)
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<3x3xui64>
    %1 = stablehlo.remainder %output, %0 : tensor<3x3xui64>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
    %3 = stablehlo.convert %1 : (tensor<3x3xui64>) -> tensor<3x3xi64>
    %4 = stablehlo.add %2, %3 : tensor<3x3xi64>
    %5 = stablehlo.convert %4 : (tensor<3x3xi64>) -> tensor<3x3xf32>
    return %output_state, %5 : tensor<2xui64>, tensor<3x3xf32>
  }
}
"""

    self.do_test("Compilation/Randint_Float32_slow", make_shlo)

  def test_compile_randint_power_of_two(self):
    """Tests compilation time for randint with dtype float32.

    If we are computing random numbers from a range [0, r) where r is a power
    of two, we can replace the remainder operation with an `and` operation,
    which is much faster.
    """

    def make_shlo(i: int) -> str:
      # This is the SHLO for torch.randint().
      return """
module {
  func.func @main(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<3x3xf32>) {
    %c = stablehlo.constant dense<137438953472> : tensor<ui64>
    %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm =  DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<3x3xui64>)
    %c_0 = stablehlo.constant dense<0> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<3x3xui64>
    %c_1 = stablehlo.constant dense<1> : tensor<ui64>
    %1 = stablehlo.broadcast_in_dim %c_1, dims = [] : (tensor<ui64>) -> tensor<3x3xui64>
    %2 = stablehlo.subtract %0, %1 : tensor<3x3xui64>
    %3 = stablehlo.and %output, %2 : tensor<3x3xui64>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i64>) -> tensor<3x3xi64>
    %5 = stablehlo.convert %3 : (tensor<3x3xui64>) -> tensor<3x3xi64>
    %6 = stablehlo.add %4, %5 : tensor<3x3xi64>
    %7 = stablehlo.convert %6 : (tensor<3x3xi64>) -> tensor<3x3xf32>
    return %output_state, %7 : tensor<2xui64>, tensor<3x3xf32>
  }
}
"""

    self.do_test("Compilation/Randint_Float32_power_of_two", make_shlo)

  def test_compile_sort_float32(self):
    def make_shlo(i: int) -> str:
      size = 8000 * (i + 1)
      return f"""
module {{
  func.func @main(%arg0: tensor<{size}xf32>) -> (tensor<{size}xf32>, tensor<{size}xi64>) {{
    %0 = stablehlo.iota dim = 0 : tensor<{size}xi64>
    %1:2 = "stablehlo.sort"(%arg0, %0) <{{dimension = 0 : i64, is_stable = false}}> ({{
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i64>, %arg4: tensor<i64>):
      %2 = stablehlo.compare  LT, %arg1, %arg2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
    }}) : (tensor<{size}xf32>, tensor<{size}xi64>) -> (tensor<{size}xf32>, tensor<{size}xi64>)
    return %1#0, %1#1 : tensor<{size}xf32>, tensor<{size}xi64>
  }}
}}"""

    self.do_test("Compilation/Sort_Float32", make_shlo)

  def test_compile_take_float32(self):
    """Tests compilation time for take() with dtype float32."""

    def make_shlo(i: int) -> str:
      del i  # Unused.
      return """
module {
  func.func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3xi64>) -> tensor<3xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<3x3xf32>) -> tensor<9xf32>
    %c = stablehlo.constant dense<9> : tensor<3xi64>
    %1 = stablehlo.add %arg1, %c : tensor<3xi64>
    %c_0 = stablehlo.constant dense<0> : tensor<3xi64>
    %2 = stablehlo.compare  LT, %arg1, %c_0 : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi1>
    %3 = stablehlo.select %2, %1, %arg1 : tensor<3xi1>, tensor<3xi64>
    %4 = "stablehlo.gather"(%0, %3) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<9xf32>, tensor<3xi64>) -> tensor<3xf32>
    return %4 : tensor<3xf32>
  }
}
"""

    self.do_test("Compilation/Take_Float32", make_shlo)


if __name__ == "__main__":
  absltest.main()
