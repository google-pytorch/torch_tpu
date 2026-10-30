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

"""Tests for TorchTPU FX-graph, op-contract, and autograd verification."""

from __future__ import annotations

import functools
import operator

from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal import pallas
from torch_tpu._internal.pallas import _mock_device
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import fx_verifier
from torch_tpu._internal.pallas.verify import verifier
from tests import seed_test_utils

P = jax.sharding.PartitionSpec
Category = findings.FindingCategory
STATIC = verifier.VerificationMode.STATIC


def _add_one_kernel(x_ref, o_ref):
  o_ref[...] = x_ref[...] + 1.0


def _pallas_add_one(x: jax.Array) -> jax.Array:
  return pl.pallas_call(
      _add_one_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x)


def _pallas_binary_add(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      lambda a_ref, b_ref, o_ref: o_ref.__setitem__(
          Ellipsis, a_ref[...] + b_ref[...]
      ),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x, y)


def _pallas_scale_fwd(x: jax.Array) -> jax.Array:
  return pl.pallas_call(
      lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] * 3.0),
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
  )(x)


def _pallas_scale_bwd_correct(grad_out: jax.Array) -> jax.Array:
  return pl.pallas_call(
      lambda g_ref, o_ref: o_ref.__setitem__(Ellipsis, g_ref[...] * 3.0),
      out_shape=jax.ShapeDtypeStruct(grad_out.shape, grad_out.dtype),
  )(grad_out)


def _pallas_scale_bwd_buggy(grad_out: jax.Array) -> jax.Array:
  # Buggy backward factor: multiplies by 5.0 instead of 3.0.
  return pl.pallas_call(
      lambda g_ref, o_ref: o_ref.__setitem__(Ellipsis, g_ref[...] * 5.0),
      out_shape=jax.ShapeDtypeStruct(grad_out.shape, grad_out.dtype),
  )(grad_out)


def _pallas_two_outputs(x: jax.Array) -> tuple[jax.Array, jax.Array]:
  def kernel(x_ref, a_ref, b_ref):
    a_ref[...] = x_ref[...] * 2.0
    b_ref[...] = x_ref[...] + 1.0

  out = jax.ShapeDtypeStruct(x.shape, x.dtype)
  return pl.pallas_call(kernel, out_shape=(out, out))(x)


def _pallas_values_and_indices(x: jax.Array) -> tuple[jax.Array, jax.Array]:
  def kernel(x_ref, values_ref, indices_ref):
    values_ref[...] = x_ref[...] * 2.0
    indices_ref[...] = jnp.zeros(indices_ref.shape, jnp.int32)

  return pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct(x.shape, x.dtype),
          jax.ShapeDtypeStruct(x.shape, jnp.int32),
      ),
  )(x)


class FxVerifierTest(seed_test_utils.RepeatableTest):

  def test_safe_donated_buffer_passes_fx_verification(self):
    donating_op = pallas.jax_op(
        "pallas_verify_fx::safe_donate_add_one",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def safe_program(x: torch.Tensor) -> torch.Tensor:
      buf = x.clone()
      out = donating_op(buf)
      return out * 2.0

    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(safe_program, (x,), mode=STATIC)
    self.assertTrue(report.passed, msg=report.summary())
    self.assertEmpty(report.errors)
    self.assertIn(findings.Phase.FX_GRAPH, report.checked_phases)

  def test_direct_use_after_donate_detected_in_fx_graph(self):
    donating_op = pallas.jax_op(
        "pallas_verify_fx::direct_use_after_donate",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def buggy_program(x: torch.Tensor) -> torch.Tensor:
      y = donating_op(x)
      # `x` was donated to `donating_op` and is now invalid on TPU!
      return y + x

    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(buggy_program, (x,), mode=STATIC)
    self.assertFalse(report.passed)
    self.assertTrue(
        report.has_category(Category.USE_AFTER_DONATE), msg=report.summary()
    )

  def test_transitive_view_alias_use_after_donate_detected(self):
    donating_op = pallas.jax_op(
        "pallas_verify_fx::view_use_after_donate",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def buggy_view_program(x: torch.Tensor) -> torch.Tensor:
      flat_alias = x.view(-1)
      sliced_alias = flat_alias[:128]
      y = donating_op(x)
      # Reading `sliced_alias` reads storage invalidated when `x` was donated.
      return y.sum() + sliced_alias.sum()

    x = torch.ones((8, 128), dtype=torch.float32)
    report = fx_verifier.verify_fx_graph(buggy_view_program, (x,))
    self.assertFalse(report.passed)
    self.assertTrue(
        report.has_category(Category.USE_AFTER_DONATE), msg=report.summary()
    )

  def test_returning_donated_tensor_detected(self):
    donating_op = pallas.jax_op(
        "pallas_verify_fx::return_donated_tensor",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def buggy_return_program(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      y = donating_op(x)
      return y, x

    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(buggy_return_program, (x,), mode=STATIC)
    self.assertFalse(report.passed)
    self.assertTrue(
        report.has_category(Category.USE_AFTER_DONATE), msg=report.summary()
    )

  def test_donating_one_output_keeps_its_siblings_valid(self):
    two_outputs = pallas.jax_op(
        "pallas_verify_fx::two_outputs", _pallas_two_outputs
    )
    donating_op = pallas.jax_op(
        "pallas_verify_fx::donate_one_output",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def program(x: torch.Tensor) -> torch.Tensor:
      a, b = two_outputs(x)
      # Only `a` is donated; `b` is a separate output buffer.
      return donating_op(a) + b

    def buggy_program(x: torch.Tensor) -> torch.Tensor:
      a, _ = two_outputs(x)
      return donating_op(a) + a

    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(program, (x,), mode=STATIC)
    self.assertTrue(report.passed, msg=report.summary())
    buggy_report = verifier.verify(buggy_program, (x,), mode=STATIC)
    self.assertTrue(
        buggy_report.has_category(Category.USE_AFTER_DONATE),
        msg=buggy_report.summary(),
    )

  def test_autograd_parity_ignores_outputs_without_gradients(self):
    values_and_indices = pallas.jax_op(
        "pallas_verify_fx::values_and_indices", _pallas_values_and_indices
    )
    values_and_indices.register_autograd(
        lambda ctx, grad_values, grad_indices: grad_values * 2.0
    )
    x = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    report = fx_verifier.verify_autograd_parity(values_and_indices, (x,))
    self.assertTrue(report.passed, msg=report.summary())

  def test_slices_of_donated_tensors_are_tracked_in_given_graphs(self):
    pallas.jax_op(
        "pallas_verify_fx::donate_sliced",
        _pallas_add_one,
        donate_argnums=(0,),
    )
    # A hand-built graph that slices a tensor with `operator.getitem`, as
    # `torch.fx.symbolic_trace` does, before donating the tensor.
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.call_function(torch.relu, (x,))
    head = graph.call_function(operator.getitem, (y, slice(0, 4)))
    out = graph.call_function(
        torch.ops.pallas_verify_fx.donate_sliced.default, (y,)
    )
    graph.output((out, head))
    graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

    report = fx_verifier.verify_fx_graph(graph_module, (torch.ones((8, 128)),))
    self.assertTrue(
        report.has_category(Category.USE_AFTER_DONATE), msg=report.summary()
    )

  def test_aliased_donor_inputs_detected_in_fx_and_eager(self):
    donating_binary_op = pallas.jax_op(
        "pallas_verify_fx::aliased_donor_binary",
        _pallas_binary_add,
        donate_argnums=(0,),
    )

    def buggy_aliased_call(x: torch.Tensor) -> torch.Tensor:
      x_alias = x.reshape(8, 128)
      return donating_binary_op(x, x_alias)

    x = torch.ones((8, 128), dtype=torch.float32)
    fx_report = verifier.verify(buggy_aliased_call, (x,), mode=STATIC)
    self.assertFalse(fx_report.passed)
    self.assertTrue(
        fx_report.has_category(Category.ALIASED_DONOR_INPUTS),
        msg=fx_report.summary(),
    )

    # Calling `verify` on the op itself checks eager storage aliasing.
    eager_report = verifier.verify(
        donating_binary_op, (x, x.view(8, 128)), mode=STATIC
    )
    self.assertFalse(eager_report.passed)
    self.assertTrue(
        eager_report.has_category(Category.ALIASED_DONOR_INPUTS),
        msg=eager_report.summary(),
    )

  def test_donated_module_state_detected_in_fx_graph(self):
    donating_binary_op = pallas.jax_op(
        "pallas_verify_fx::donate_module_state",
        _pallas_binary_add,
        donate_argnums=(0,),
    )

    class Shift(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.ones((8, 128)))

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Donating `self.bias` hands the module's own buffer to the kernel.
        return donating_binary_op(self.bias, x)

    class SafeShift(Shift):

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        return donating_binary_op(self.bias.clone(), x)

    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(Shift(), (x,), mode=STATIC)
    self.assertFalse(report.passed)
    self.assertTrue(
        report.has_category(Category.DONATED_PARAMETER), msg=report.summary()
    )

    safe_report = verifier.verify(SafeShift(), (x,), mode=STATIC)
    self.assertTrue(safe_report.passed, msg=safe_report.summary())

  def test_donated_parameter_detected_in_eager_contracts(self):
    donating_op = pallas.jax_op(
        "pallas_verify_fx::donate_parameter",
        _pallas_add_one,
        donate_argnums=(0,),
    )
    weight = torch.nn.Parameter(torch.ones((8, 128)), requires_grad=False)
    report = verifier.verify(donating_op, (weight,), mode=STATIC)
    self.assertFalse(report.passed)
    self.assertTrue(
        report.has_category(Category.DONATED_PARAMETER), msg=report.summary()
    )

  def test_zero_output_jax_op_silent_skip_detected(self):
    def noop_side_effect_jax(x: jax.Array) -> None:
      del x
      return None

    zero_out_op = pallas.jax_op(
        "pallas_verify_fx::zero_output_op",
        noop_side_effect_jax,
    )
    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(zero_out_op, (x,), mode=STATIC)
    self.assertFalse(report.passed)
    self.assertTrue(
        report.has_category(Category.ZERO_OUTPUT_SILENT_SKIP),
        msg=report.summary(),
    )

  def test_partition_spec_unknown_axis_and_duplicate_mesh_devices_detected(
      self,
  ):
    devs_dup = np.array([
        _mock_device._PallasDevice(id=0),
        _mock_device._PallasDevice(id=0),
    ])
    bad_mesh = jax.sharding.Mesh(devs_dup, ("tp",))

    def distributed_ident(x: jax.Array) -> jax.Array:
      return x + 1.0

    bad_dist_op = pallas.jax_op(
        "pallas_verify_fx::bad_partition_spec_op",
        distributed_ident,
        mesh=bad_mesh,
        input_partition_specs=(P("nonexistent_axis"),),
        output_partition_specs=(P("tp"),),
    )
    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(bad_dist_op, (x,), mode=STATIC)
    self.assertFalse(report.passed)
    self.assertTrue(
        report.has_category(Category.PARTITION_SPEC_MISMATCH),
        msg=report.summary(),
    )

  def test_report_api_supports_truthiness(self):
    donating_op = pallas.jax_op(
        "pallas_verify_fx::report_api_donate",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def buggy_program(x: torch.Tensor) -> torch.Tensor:
      return donating_op(x) + x

    def safe_program(x: torch.Tensor) -> torch.Tensor:
      return donating_op(x.clone()) + x

    x = torch.ones((8, 128), dtype=torch.float32)
    # A single tensor is accepted in place of an argument tuple.
    bad = verifier.verify(buggy_program, x, mode=STATIC)
    self.assertFalse(bad)
    self.assertIn("use_after_donate", str(bad))

    good = verifier.verify(safe_program, x, mode=STATIC)
    self.assertTrue(good, msg=str(good))
    good.raise_for_errors()

  def test_autograd_parity_passes_valid_and_catches_buggy_backward(self):
    fwd_valid = pallas.jax_op(
        "pallas_verify_fx::scale_fwd_valid", _pallas_scale_fwd
    )
    bwd_valid = pallas.jax_op(
        "pallas_verify_fx::scale_bwd_valid", _pallas_scale_bwd_correct
    )
    fwd_valid.register_autograd(lambda ctx, grad: bwd_valid(grad))

    fwd_buggy = pallas.jax_op(
        "pallas_verify_fx::scale_fwd_buggy", _pallas_scale_fwd
    )
    bwd_buggy = pallas.jax_op(
        "pallas_verify_fx::scale_bwd_buggy", _pallas_scale_bwd_buggy
    )
    fwd_buggy.register_autograd(lambda ctx, grad: bwd_buggy(grad))

    fwd_bad_shape = pallas.jax_op(
        "pallas_verify_fx::scale_fwd_bad_shape", _pallas_scale_fwd
    )
    fwd_bad_shape.register_autograd(lambda ctx, grad: grad[..., :64])

    x = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    valid_report = fx_verifier.verify_autograd_parity(fwd_valid, (x,))
    self.assertTrue(valid_report.passed, msg=valid_report.summary())

    buggy_report = fx_verifier.verify_autograd_parity(fwd_buggy, (x,))
    self.assertFalse(buggy_report.passed)
    self.assertTrue(
        buggy_report.has_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=buggy_report.summary(),
    )

    bad_shape_report = fx_verifier.verify_autograd_parity(fwd_bad_shape, (x,))
    self.assertFalse(bad_shape_report.passed, msg=bad_shape_report.summary())
    self.assertTrue(
        bad_shape_report.has_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=bad_shape_report.summary(),
    )

    # Multi-chip SPMD `@pallas.jax_op(..., mesh=...)` custom autograd parity.
    mesh2 = jax.sharding.Mesh(
        np.array([_mock_device._PallasDevice(id=i) for i in range(2)]),
        ("rank",),
    )

    def dist_scale(
        global_x: jax.Array, *, factor: float, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      return jax.shard_map(
          lambda s: pl.pallas_call(
              lambda r, o: o.__setitem__(Ellipsis, r[...] * factor),
              out_shape=jax.ShapeDtypeStruct(s.shape, s.dtype),
          )(s),
          mesh=mesh,
          in_specs=(P("rank"),),
          out_specs=P("rank"),
          check_vma=False,
      )(global_x)

    dist_fwd_ok = pallas.jax_op(
        "pallas_verify_fx::dist_scale_fwd_ok",
        functools.partial(dist_scale, factor=3.0, mesh=mesh2),
        mesh=mesh2,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    dist_bwd_ok = pallas.jax_op(
        "pallas_verify_fx::dist_scale_bwd_ok",
        functools.partial(dist_scale, factor=3.0, mesh=mesh2),
        mesh=mesh2,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    dist_fwd_ok.register_autograd(lambda ctx, grad: dist_bwd_ok(grad))

    dist_fwd_bad = pallas.jax_op(
        "pallas_verify_fx::dist_scale_fwd_bad",
        functools.partial(dist_scale, factor=3.0, mesh=mesh2),
        mesh=mesh2,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    dist_bwd_bad = pallas.jax_op(
        "pallas_verify_fx::dist_scale_bwd_bad",
        functools.partial(dist_scale, factor=5.0, mesh=mesh2),
        mesh=mesh2,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    dist_fwd_bad.register_autograd(lambda ctx, grad: dist_bwd_bad(grad))

    dist_ok_rep = fx_verifier.verify_autograd_parity(dist_fwd_ok, (x,))
    self.assertTrue(dist_ok_rep.passed, msg=dist_ok_rep.summary())
    dist_bad_rep = fx_verifier.verify_autograd_parity(dist_fwd_bad, (x,))
    self.assertFalse(dist_bad_rep.passed, msg=dist_bad_rep.summary())
    self.assertTrue(
        dist_bad_rep.has_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=dist_bad_rep.summary(),
    )


if __name__ == "__main__":
  absltest.main()
