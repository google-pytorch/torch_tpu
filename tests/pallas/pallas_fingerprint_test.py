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

import dataclasses
from typing import Any
from unittest import mock

from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.pallas import tpu_torch_pallas
from tests import seed_test_utils

_MOCK_MLIR_MODULE_SERIALIZED = b"mock_mlir_module_serialized"
_MOCK_MLIR_MODULE = "module {}"
_EXPECTED_FINGERPRINT = "610b806f2760cd2f9744abe7dfcab222"


@dataclasses.dataclass
class _MockExported:
  mlir_module_serialized: bytes
  out_avals: Any
  out_tree: Any
  mlir_module_text: str = _MOCK_MLIR_MODULE

  def mlir_module(self) -> str:
    return self.mlir_module_text


def fingerprint_test_kernel_body(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y


def fingerprint_test_jax(x: jax.Array, y: jax.Array) -> jax.Array:
  dt = jax.ShapeDtypeStruct(x.shape, x.dtype)
  pallas_call = pl.pallas_call(fingerprint_test_kernel_body, out_shape=dt)
  return pallas_call(x, y)


class PallasFingerprintTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.device = torch.device("tpu:0")

  def test_kernel_key_fingerprint_stability(self):
    x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, device=self.device)
    y = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, device=self.device)
    op = pallas.jax_op("pallas::test_add_1", fingerprint_test_jax)

    real_exported = op._init_fn.exported

    # Override `op._init_fn.exported` to return a fixed constant MLIR module
    # and serialized payload. Although outer module `#loc` debug info is
    # stripped during normalization, the embedded Pallas kernel MLIR inside
    # `custom_call_config.body` still contains `#loc` and source paths.
    def mock_exported(*args, **kwargs):
      lowered = real_exported(*args, **kwargs)
      return _MockExported(
          mlir_module_serialized=_MOCK_MLIR_MODULE_SERIALIZED,
          out_avals=lowered.out_avals,
          out_tree=lowered.out_tree,
          mlir_module_text=_MOCK_MLIR_MODULE,
      )

    op._init_fn.exported = mock_exported

    # Mock custom kernel registration and execution to verify the computed
    # fingerprint is passed correctly to the backend without relying on TPU
    # hardware kernel execution.
    with (
        mock.patch.object(
            tpu_torch_pallas, "register_custom_kernel"
        ) as mock_register,
        mock.patch.object(
            tpu_torch_pallas,
            "call_custom_kernel",
            return_value=[torch.zeros_like(x)],
        ) as mock_call,
    ):
      _ = op(x, y)

      mock_register.assert_called_once_with(
          "pallas::test_add_1",
          _EXPECTED_FINGERPRINT,
          serialized_mlir_module=_MOCK_MLIR_MODULE_SERIALIZED,
      )
      mock_call.assert_called_once_with(
          "pallas::test_add_1",
          _EXPECTED_FINGERPRINT,
          inputs=mock.ANY,
          output_shapes=mock.ANY,
          donate_argnums=mock.ANY,
      )


if __name__ == "__main__":
  absltest.main()
