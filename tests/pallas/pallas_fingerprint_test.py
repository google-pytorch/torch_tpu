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

from __future__ import annotations

from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.pallas import pallas as pallas_impl
from torch_tpu._internal.pallas import tpu_torch_pallas
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils

# pylint: disable=line-too-long
_ADD_MODULE = """
module @jit_f {
  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
"""

# Same program as `_ADD_MODULE`, annotated with caller location debug info.
_ADD_MODULE_WITH_LOCS_TEMPLATE = """
#loc1 = loc("{path}":{line}:4)
module @jit_f {{
  func.func public @main(%arg0: tensor<3xf32> loc("x"), %arg1: tensor<3xf32> loc("y")) -> tensor<3xf32> {{
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32> loc(#loc1)
    return %0 : tensor<3xf32> loc(#loc1)
  }} loc(#loc1)
}} loc(#loc1)
"""

# Differs from `_ADD_MODULE` only in the op inside the function body.
_MUL_MODULE = """
module @jit_f {
  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
"""

# Mimics a Pallas kernel whose Mosaic payload lives in an opaque attribute.
_CUSTOM_CALL_MODULE_TEMPLATE = """
module @jit_f {{
  func.func public @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {{
    %0 = stablehlo.custom_call @tpu_custom_call(%arg0) {{backend_config = "{payload}"}} : (tensor<3xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }}
}}
"""
# pylint: enable=line-too-long


def _fingerprint(mlir_module_text: str) -> str:
  # The helpers are private implementation details tested directly here.
  # pylint: disable=protected-access
  return pallas_impl._fingerprint_normalized_mlir(
      pallas_impl._normalize_mlir_module(mlir_module_text)
  )
  # pylint: enable=protected-access


def fingerprint_test_add_kernel(x_ref, y_ref, o_ref):
  o_ref[...] = x_ref[...] + y_ref[...]


def fingerprint_test_mul_kernel(x_ref, y_ref, o_ref):
  o_ref[...] = x_ref[...] * y_ref[...]


def fingerprint_test_add(x: jax.Array, y: jax.Array) -> jax.Array:
  dt = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return pl.pallas_call(fingerprint_test_add_kernel, out_shape=dt)(x, y)


def fingerprint_test_mul(x: jax.Array, y: jax.Array) -> jax.Array:
  dt = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return pl.pallas_call(fingerprint_test_mul_kernel, out_shape=dt)(x, y)


class MlirFingerprintTest(seed_test_utils.RepeatableTest):
  """Tests the pure MLIR normalization and fingerprinting helpers."""

  def test_fingerprint_is_deterministic_hex_digest(self):
    fingerprint = _fingerprint(_ADD_MODULE)

    self.assertEqual(fingerprint, _fingerprint(_ADD_MODULE))
    self.assertRegex(fingerprint, r"^[0-9a-f]{32}$")

  def test_fingerprint_ignores_outer_location_info(self):
    with_locs_a = _ADD_MODULE_WITH_LOCS_TEMPLATE.format(
        path="/src/caller_a.py", line=10
    )
    with_locs_b = _ADD_MODULE_WITH_LOCS_TEMPLATE.format(
        path="/src/caller_b.py", line=99
    )

    self.assertEqual(_fingerprint(with_locs_a), _fingerprint(_ADD_MODULE))
    self.assertEqual(_fingerprint(with_locs_b), _fingerprint(_ADD_MODULE))

  def test_fingerprint_differs_for_different_ops(self):
    self.assertNotEqual(_fingerprint(_ADD_MODULE), _fingerprint(_MUL_MODULE))

  def test_fingerprint_differs_for_different_shapes(self):
    add_module_4 = _ADD_MODULE.replace("tensor<3xf32>", "tensor<4xf32>")

    self.assertNotEqual(_fingerprint(_ADD_MODULE), _fingerprint(add_module_4))

  def test_fingerprint_differs_for_different_kernel_payloads(self):
    # The embedded kernel body is opaque to normalization, so any change to it
    # (including its inner debug info) must produce a different fingerprint.
    module_a = _CUSTOM_CALL_MODULE_TEMPLATE.format(payload="kernel_a")
    module_b = _CUSTOM_CALL_MODULE_TEMPLATE.format(payload="kernel_b")

    self.assertNotEqual(_fingerprint(module_a), _fingerprint(module_b))


class PallasFingerprintTest(seed_test_utils.RepeatableTest):
  """Tests the fingerprints that real Pallas ops register on TPU."""

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.device("tpu:0")

  def _inputs(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(n, dtype=torch.float32, device=self.device)
    y = torch.full((n,), 2.0, dtype=torch.float32, device=self.device)
    return x, y

  def _fingerprints(self, op) -> list[str]:
    # The registered fingerprints are only exposed on the op's JaxCallable.
    return list(
        op._init_fn.kernel_key_to_mlir_fingerprint.values()  # pylint: disable=protected-access
    )

  def test_repeated_invocations_reuse_registered_fingerprint(self):
    op = pallas.jax_op("pallas::fp_test_repeat", fingerprint_test_add)
    x, y = self._inputs(3)

    first = op(x, y)
    second = op(x, y)

    test_utils.assert_close(first.cpu(), (x + y).cpu())
    test_utils.assert_close(second.cpu(), (x + y).cpu())
    fingerprints = self._fingerprints(op)
    self.assertLen(fingerprints, 1)
    self.assertTrue(
        tpu_torch_pallas.lookup_custom_kernel(
            "pallas::fp_test_repeat", fingerprints[0]
        )
    )

  def test_different_input_shapes_get_different_fingerprints(self):
    op = pallas.jax_op("pallas::fp_test_shapes", fingerprint_test_add)

    op(*self._inputs(8))
    op(*self._inputs(16))

    fingerprints = self._fingerprints(op)
    self.assertLen(fingerprints, 2)
    self.assertNotEqual(fingerprints[0], fingerprints[1])

  def test_different_kernels_get_different_fingerprints(self):
    add_op = pallas.jax_op("pallas::fp_test_add", fingerprint_test_add)
    mul_op = pallas.jax_op("pallas::fp_test_mul", fingerprint_test_mul)
    x, y = self._inputs(3)

    test_utils.assert_close(add_op(x, y).cpu(), (x + y).cpu())
    test_utils.assert_close(mul_op(x, y).cpu(), (x * y).cpu())

    (add_fingerprint,) = self._fingerprints(add_op)
    (mul_fingerprint,) = self._fingerprints(mul_op)
    self.assertNotEqual(add_fingerprint, mul_fingerprint)

  def test_same_kernel_registered_twice_gets_same_fingerprint(self):
    # Two ops wrapping the same kernel must share a fingerprint; it depends only
    # on the lowered program.
    op_a = pallas.jax_op("pallas::fp_test_same_a", fingerprint_test_add)
    op_b = pallas.jax_op("pallas::fp_test_same_b", fingerprint_test_add)
    x, y = self._inputs(3)

    op_a(x, y)
    op_b(x, y)

    self.assertEqual(self._fingerprints(op_a), self._fingerprints(op_b))


if __name__ == "__main__":
  absltest.main()
