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

"""Tests for the errors raised by the Pallas kernel verifier."""

from __future__ import annotations

from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import idl_solver
from torch_tpu._internal.pallas.verify import jaxpr_model
from torch_tpu._internal.pallas.verify import verifier
from tests import seed_test_utils


def _pallas_add_one(x: jax.Array) -> jax.Array:
  return pl.pallas_call(
      lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] + 1.0),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x)


class PallasVerifyErrorsTest(seed_test_utils.RepeatableTest):

  def test_raise_for_errors_raises_with_the_report(self):
    donating_op = pallas.jax_op(
        "pallas_verify_errors::donate_add_one",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def buggy_program(x: torch.Tensor) -> torch.Tensor:
      return donating_op(x) + x

    report = verifier.verify(
        buggy_program,
        torch.ones((8, 128)),
        mode=verifier.VerificationMode.STATIC,
    )
    with self.assertRaises(findings.VerificationError) as ctx:
      report.raise_for_errors()
    self.assertIs(ctx.exception.report, report)
    self.assertIn("use_after_donate", str(ctx.exception))

  def test_verify_rejects_non_callable_target(self):
    with self.assertRaisesRegex(TypeError, "target must be callable"):
      verifier.verify(42, (torch.ones(2),))

  def test_solver_rejects_unknown_variables(self):
    model = idl_solver.IdlPbModel()
    with self.assertRaisesRegex(ValueError, "unknown integer variable"):
      model.add_diff(0, 3, 1)
    with self.assertRaisesRegex(ValueError, "unknown boolean variable"):
      model.add_clause([idl_solver.Literal(2)])

  def test_unsupported_dma_operand_layout_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "unsupported dma operand layout"):
      jaxpr_model.decode_dma_operands((object(), object()))


if __name__ == "__main__":
  absltest.main()
