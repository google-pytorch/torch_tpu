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
import types

from absl.testing import absltest
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import torch
from torch.tpu import pallas
from torch.tpu.pallas import verifier
from torch_tpu._internal.pallas import _mock_device
from torch_tpu._internal.pallas import pallas as pallas_lib
from torch_tpu._internal.pallas.verify import fx_verifier
from torch_tpu._internal.pallas.verify import verifier as verifier_impl
from torch_tpu._internal.utils import hardware
from tests import seed_test_utils

jax_op = pallas.jax_op
P = jax.sharding.PartitionSpec
Category = verifier.FindingCategory
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
    donating_op = jax_op(
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
    self.assertEmpty(report.findings_by_severity(verifier.Severity.ERROR))
    self.assertIn(verifier.Phase.FX_GRAPH, report.checked_phases)

  def test_direct_use_after_donate_detected_in_fx_graph(self):
    donating_op = jax_op(
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
    findings = report.findings_by_category(Category.USE_AFTER_DONATE)
    self.assertNotEmpty(findings, msg=report.summary())
    self.assertEqual(findings[0].phase, verifier.Phase.FX_GRAPH)
    self.assertEqual(findings[0].severity, verifier.Severity.ERROR)
    self.assertEqual(
        findings[0].op_name, "pallas_verify_fx::direct_use_after_donate"
    )

  def test_transitive_view_alias_use_after_donate_detected(self):
    donating_op = jax_op(
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
    report = verifier.verify(buggy_view_program, (x,), mode=STATIC)
    self.assertFalse(report.passed)
    findings = report.findings_by_category(Category.USE_AFTER_DONATE)
    self.assertNotEmpty(findings, msg=report.summary())
    self.assertEqual(findings[0].phase, verifier.Phase.FX_GRAPH)

  def test_returning_donated_tensor_detected(self):
    donating_op = jax_op(
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
    findings = report.findings_by_category(Category.USE_AFTER_DONATE)
    self.assertNotEmpty(findings, msg=report.summary())
    self.assertEqual(findings[0].phase, verifier.Phase.FX_GRAPH)

  def test_donating_one_output_keeps_its_siblings_valid(self):
    two_outputs = jax_op("pallas_verify_fx::two_outputs", _pallas_two_outputs)
    donating_op = jax_op(
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
    self.assertNotEmpty(
        buggy_report.findings_by_category(Category.USE_AFTER_DONATE),
        msg=buggy_report.summary(),
    )

  def test_autograd_parity_ignores_integer_outputs_while_checking_float_grads(
      self,
  ):
    valid_op = jax_op(
        "pallas_verify_fx::values_and_indices_valid", _pallas_values_and_indices
    )
    valid_op.register_autograd(
        lambda ctx, grad_values, grad_indices: grad_values * 2.0
    )

    buggy_op = jax_op(
        "pallas_verify_fx::values_and_indices_buggy", _pallas_values_and_indices
    )
    buggy_op.register_autograd(
        lambda ctx, grad_values, grad_indices: grad_values * 5.0
    )

    x = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    valid_report = verifier.verify(valid_op, (x,))
    self.assertTrue(valid_report.passed, msg=valid_report.summary())
    self.assertIn(verifier.Phase.AUTOGRAD, valid_report.checked_phases)

    buggy_report = verifier.verify(buggy_op, (x,))
    self.assertFalse(buggy_report.passed, msg=buggy_report.summary())
    self.assertNotEmpty(
        buggy_report.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=buggy_report.summary(),
    )

  def test_slices_of_donated_tensors_are_tracked_in_given_graphs(self):
    jax_op(
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

    report = verifier.verify(graph_module, (torch.ones((8, 128)),), mode=STATIC)
    self.assertNotEmpty(
        report.findings_by_category(Category.USE_AFTER_DONATE),
        msg=report.summary(),
    )

  def test_aliased_donor_inputs_detected_in_fx_and_eager(self):
    donating_binary_op = jax_op(
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
    self.assertNotEmpty(
        fx_report.findings_by_category(Category.ALIASED_DONOR_INPUTS),
        msg=fx_report.summary(),
    )

    # Calling `verify` on the op itself checks eager storage aliasing.
    eager_report = verifier.verify(
        donating_binary_op, (x, x.view(8, 128)), mode=STATIC
    )
    self.assertFalse(eager_report.passed)
    self.assertNotEmpty(
        eager_report.findings_by_category(Category.ALIASED_DONOR_INPUTS),
        msg=eager_report.summary(),
    )

  def test_disjoint_slices_of_same_storage_detected_as_aliased_donor_inputs(
      self,
  ):
    donating_binary_op = jax_op(
        "pallas_verify_fx::disjoint_slice_donor_binary",
        _pallas_binary_add,
        donate_argnums=(0,),
    )
    # Non-overlapping views of the same backing storage share an XLA base
    # buffer on TPU and must be flagged when one view is donated.
    full = torch.ones((16, 128), dtype=torch.float32)
    report = verifier.verify(
        donating_binary_op, (full[:8], full[8:]), mode=STATIC
    )
    self.assertFalse(report.passed, msg=report.summary())
    self.assertNotEmpty(
        report.findings_by_category(Category.ALIASED_DONOR_INPUTS),
        msg=report.summary(),
    )

  def test_donated_module_state_detected_in_fx_graph(self):
    donating_binary_op = jax_op(
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
    self.assertNotEmpty(
        report.findings_by_category(Category.DONATED_PARAMETER),
        msg=report.summary(),
    )

    safe_report = verifier.verify(SafeShift(), (x,), mode=STATIC)
    self.assertTrue(safe_report.passed, msg=safe_report.summary())

  def test_donated_parameter_detected_in_eager_contracts(self):
    donating_op = jax_op(
        "pallas_verify_fx::donate_parameter",
        _pallas_add_one,
        donate_argnums=(0,),
    )
    weight = torch.nn.Parameter(torch.ones((8, 128)), requires_grad=False)
    report = verifier.verify(donating_op, (weight,), mode=STATIC)
    self.assertFalse(report.passed)
    self.assertNotEmpty(
        report.findings_by_category(Category.DONATED_PARAMETER),
        msg=report.summary(),
    )

  def test_donated_parameter_passed_as_fx_function_argument_detected(self):
    donating_op = jax_op(
        "pallas_verify_fx::donate_param_arg",
        _pallas_add_one,
        donate_argnums=(0,),
    )
    param = torch.nn.Parameter(torch.ones((8, 128), dtype=torch.float32))
    report = verifier.verify(lambda p: donating_op(p), (param,), mode=STATIC)
    self.assertFalse(report.passed, msg=report.summary())
    findings = report.findings_by_category(Category.DONATED_PARAMETER)
    self.assertNotEmpty(findings, msg=report.summary())
    self.assertEqual(findings[0].phase, verifier.Phase.FX_GRAPH)

  def test_zero_output_jax_op_silent_skip_detected(self):
    def noop_side_effect_jax(x: jax.Array) -> None:
      del x
      return None

    zero_out_op = jax_op(
        "pallas_verify_fx::zero_output_op",
        noop_side_effect_jax,
    )
    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(zero_out_op, (x,), mode=STATIC)
    self.assertFalse(report.passed)
    findings = report.findings_by_category(Category.ZERO_OUTPUT_SILENT_SKIP)
    self.assertNotEmpty(findings, msg=report.summary())
    self.assertEqual(findings[0].phase, verifier.Phase.CONTRACTS)

  def test_duplicate_mesh_device_ids_detected_in_contracts(self):
    devs_dup = np.array([
        _mock_device._PallasDevice(id=0),
        _mock_device._PallasDevice(id=0),
    ])
    dup_mesh = jax.sharding.Mesh(devs_dup, ("tp",))
    dup_mesh_op = jax_op(
        "pallas_verify_fx::dup_mesh_devices_op",
        _pallas_add_one,
        mesh=dup_mesh,
        input_partition_specs=(P("tp"),),
        output_partition_specs=(P("tp"),),
    )
    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(dup_mesh_op, (x,), mode=STATIC)
    self.assertFalse(report.passed)
    findings = report.findings_by_category(Category.PARTITION_SPEC_MISMATCH)
    self.assertLen(findings, 1, msg=report.summary())
    self.assertIn("duplicate device IDs", findings[0].message)

  def test_unknown_partition_spec_axis_detected_in_contracts(self):
    mesh2 = jax.sharding.Mesh(
        np.array([_mock_device._PallasDevice(id=i) for i in range(2)]),
        ("tp",),
    )
    bad_axis_op = jax_op(
        "pallas_verify_fx::unknown_axis_partition_spec_op",
        _pallas_add_one,
        mesh=mesh2,
        input_partition_specs=(P("nonexistent_axis"),),
        output_partition_specs=(P("tp"),),
    )
    x = torch.ones((8, 128), dtype=torch.float32)
    report = verifier.verify(bad_axis_op, (x,), mode=STATIC)
    self.assertFalse(report.passed)
    findings = report.findings_by_category(Category.PARTITION_SPEC_MISMATCH)
    self.assertLen(findings, 1, msg=report.summary())
    self.assertIn("nonexistent_axis", findings[0].message)

  def test_report_api_single_tensor_arg_and_summary(self):
    donating_op = jax_op(
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
    self.assertFalse(bad.passed)
    self.assertIn("use_after_donate", str(bad))

    good = verifier.verify(safe_program, x, mode=STATIC)
    self.assertTrue(good.passed, msg=str(good))
    good.raise_for_errors()

  def test_autograd_parity_detects_wrong_backward_scale(self):
    fwd_valid = jax_op("pallas_verify_fx::scale_fwd_valid", _pallas_scale_fwd)
    bwd_valid = jax_op(
        "pallas_verify_fx::scale_bwd_valid", _pallas_scale_bwd_correct
    )
    fwd_valid.register_autograd(lambda ctx, grad: bwd_valid(grad))

    fwd_buggy = jax_op("pallas_verify_fx::scale_fwd_buggy", _pallas_scale_fwd)
    bwd_buggy = jax_op(
        "pallas_verify_fx::scale_bwd_buggy", _pallas_scale_bwd_buggy
    )
    fwd_buggy.register_autograd(lambda ctx, grad: bwd_buggy(grad))

    x = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    valid_report = verifier.verify(fwd_valid, (x,))
    self.assertTrue(valid_report.passed, msg=valid_report.summary())

    buggy_report = verifier.verify(fwd_buggy, (x,))
    self.assertFalse(buggy_report.passed)
    self.assertNotEmpty(
        buggy_report.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=buggy_report.summary(),
    )

  def test_autograd_parity_detects_backward_gradient_shape_mismatch(self):
    fwd_bad_shape = jax_op(
        "pallas_verify_fx::scale_fwd_bad_shape", _pallas_scale_fwd
    )
    fwd_bad_shape.register_autograd(lambda ctx, grad: grad[..., :64])

    x = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    bad_shape_report = verifier.verify(fwd_bad_shape, (x,))
    self.assertFalse(bad_shape_report.passed, msg=bad_shape_report.summary())
    self.assertNotEmpty(
        bad_shape_report.findings_by_category(
            Category.AUTOGRAD_PARITY_MISMATCH
        ),
        msg=bad_shape_report.summary(),
    )

  def test_autograd_parity_verifies_spmd_sharded_jax_op(self):
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

    dist_fwd_ok = jax_op(
        "pallas_verify_fx::dist_scale_fwd_ok",
        functools.partial(dist_scale, factor=3.0, mesh=mesh2),
        mesh=mesh2,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    dist_bwd_ok = jax_op(
        "pallas_verify_fx::dist_scale_bwd_ok",
        functools.partial(dist_scale, factor=3.0, mesh=mesh2),
        mesh=mesh2,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    dist_fwd_ok.register_autograd(lambda ctx, grad: dist_bwd_ok(grad))

    dist_fwd_bad = jax_op(
        "pallas_verify_fx::dist_scale_fwd_bad",
        functools.partial(dist_scale, factor=3.0, mesh=mesh2),
        mesh=mesh2,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    dist_bwd_bad = jax_op(
        "pallas_verify_fx::dist_scale_bwd_bad",
        functools.partial(dist_scale, factor=5.0, mesh=mesh2),
        mesh=mesh2,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    dist_fwd_bad.register_autograd(lambda ctx, grad: dist_bwd_bad(grad))

    x = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    dist_ok_rep = verifier.verify(dist_fwd_ok, (x,))
    self.assertTrue(dist_ok_rep.passed, msg=dist_ok_rep.summary())
    dist_bad_rep = verifier.verify(dist_fwd_bad, (x,))
    self.assertFalse(dist_bad_rep.passed, msg=dist_bad_rep.summary())
    self.assertNotEmpty(
        dist_bad_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=dist_bad_rep.summary(),
    )

  def test_fake_kernel_mismatch_detects_missing_sharded_output_partition_spec(
      self,
  ):
    mesh4 = jax.sharding.Mesh(
        np.array([
            _mock_device._PallasDevice(device_kind="TPU v5 lite", id=i)
            for i in range(4)
        ]),
        ("tp",),
    )

    def sharded_add_one(global_x: jax.Array) -> jax.Array:
      return jax.shard_map(
          _pallas_add_one,
          mesh=mesh4,
          in_specs=(P("tp"),),
          out_specs=P("tp"),
          check_vma=False,
      )(global_x)

    # Mirrors the sharded_jax_op bug: JaxCallable omits output_partition_specs
    # (allocating global (16, 128) at runtime) while register_fake returns the
    # local shard shape (4, 128).
    jc_missing_out_spec = pallas_lib.custom_jax_kernel(
        sharded_add_one,
        "sharded_missing_out_spec",
        mesh=mesh4,
        input_partition_specs=(P("tp"),),
        warn_deprecated=False,
    )
    sharded_bad_op = torch.library.custom_op(
        "pallas_verify_fx::sharded_missing_out_spec",
        jc_missing_out_spec,
        mutates_args=(),
        schema="(Tensor global_x) -> Tensor",
    )
    sharded_bad_op.register_fake(
        lambda global_x: torch.empty(
            global_x.shape, dtype=global_x.dtype, device=global_x.device
        )
    )

    local_x = torch.ones((4, 128), dtype=torch.float32)
    bad_sharded_rep = verifier.verify(sharded_bad_op, (local_x,), mode=STATIC)
    self.assertFalse(bad_sharded_rep.passed, msg=bad_sharded_rep.summary())
    fake_findings = bad_sharded_rep.findings_by_category(
        Category.FAKE_KERNEL_MISMATCH
    )
    self.assertNotEmpty(fake_findings, msg=bad_sharded_rep.summary())
    self.assertIn("(4, 128)", fake_findings[0].message)
    self.assertIn("(16, 128)", fake_findings[0].message)
    self.assertIn("output_partition_specs", fake_findings[0].message)

    # Positive control: supplying output_partition_specs=(P("tp"),) aligns
    # JaxCallable's runtime output with register_fake.
    jc_matching_out_spec = pallas_lib.custom_jax_kernel(
        sharded_add_one,
        "sharded_matching_out_spec",
        mesh=mesh4,
        input_partition_specs=(P("tp"),),
        output_partition_specs=(P("tp"),),
        warn_deprecated=False,
    )
    sharded_ok_op = torch.library.custom_op(
        "pallas_verify_fx::sharded_matching_out_spec",
        jc_matching_out_spec,
        mutates_args=(),
        schema="(Tensor global_x) -> Tensor",
    )
    sharded_ok_op.register_fake(
        lambda global_x: torch.empty(
            global_x.shape, dtype=global_x.dtype, device=global_x.device
        )
    )

    ok_sharded_rep = verifier.verify(sharded_ok_op, (local_x,), mode=STATIC)
    self.assertTrue(ok_sharded_rep.passed, msg=ok_sharded_rep.summary())

  def test_fake_kernel_mismatch_detects_dtype_and_pytree_arity_errors(self):
    x = torch.ones((8, 128), dtype=torch.float32)

    bad_dtype_op = jax_op("pallas_verify_fx::bad_fake_dtype", _pallas_add_one)
    bad_dtype_op.register_fake(
        lambda t: t.new_empty(t.shape, dtype=torch.bfloat16)
    )
    dtype_rep = verifier.verify(bad_dtype_op, (x,), mode=STATIC)
    self.assertFalse(dtype_rep.passed, msg=dtype_rep.summary())
    self.assertNotEmpty(
        dtype_rep.findings_by_category(Category.FAKE_KERNEL_MISMATCH),
        msg=dtype_rep.summary(),
    )

    bad_tree_op = jax_op("pallas_verify_fx::bad_fake_tree", _pallas_add_one)
    bad_tree_op.register_fake(
        lambda t: (t.new_empty(t.shape), t.new_empty(t.shape))
    )
    tree_rep = verifier.verify(bad_tree_op, (x,), mode=STATIC)
    self.assertFalse(tree_rep.passed, msg=tree_rep.summary())
    self.assertNotEmpty(
        tree_rep.findings_by_category(Category.FAKE_KERNEL_MISMATCH),
        msg=tree_rep.summary(),
    )

  def test_input_partition_specs_allows_none_or_replicated_for_static_args(
      self,
  ):
    mesh2 = jax.sharding.Mesh(
        np.array([
            _mock_device._PallasDevice(device_kind="TPU v5 lite", id=i)
            for i in range(2)
        ]),
        ("rank",),
    )

    def sharded_with_statics(
        scale: float, x: jax.Array, add_one: bool
    ) -> jax.Array:
      bias = 1.0 if add_one else 0.0
      return jax.shard_map(
          lambda s: pl.pallas_call(
              lambda r, o: o.__setitem__(Ellipsis, r[...] * scale + bias),
              out_shape=jax.ShapeDtypeStruct(s.shape, s.dtype),
          )(s),
          mesh=mesh2,
          in_specs=(P("rank"),),
          out_specs=P("rank"),
          check_vma=False,
      )(x)

    valid_op = jax_op(
        "pallas_verify_fx::statics_valid",
        sharded_with_statics,
        mesh=mesh2,
        input_partition_specs=(None, P("rank"), P()),
        output_partition_specs=(P("rank"),),
    )
    x_shard = torch.ones((8, 128), dtype=torch.float32)
    valid_rep = verifier.verify(valid_op, (2.0, x_shard, True))
    self.assertTrue(valid_rep.passed, msg=valid_rep.summary())

  def test_input_partition_specs_rejects_truncated_or_sharded_static_args(self):
    mesh2 = jax.sharding.Mesh(
        np.array([
            _mock_device._PallasDevice(device_kind="TPU v5 lite", id=i)
            for i in range(2)
        ]),
        ("rank",),
    )

    def sharded_with_statics(
        scale: float, x: jax.Array, add_one: bool
    ) -> jax.Array:
      bias = 1.0 if add_one else 0.0
      return x * scale + bias

    x_shard = torch.ones((8, 128), dtype=torch.float32)

    # Passing only 1 spec for 3 positional args truncates arguments in
    # JaxCallable._get_jax_placeholders.
    truncated_op = jax_op(
        "pallas_verify_fx::statics_truncated_specs",
        sharded_with_statics,
        mesh=mesh2,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    trunc_rep = verifier.verify(truncated_op, (2.0, x_shard, True), mode=STATIC)
    self.assertFalse(trunc_rep.passed, msg=trunc_rep.summary())
    self.assertNotEmpty(
        trunc_rep.findings_by_category(Category.PARTITION_SPEC_MISMATCH),
        msg=trunc_rep.summary(),
    )

    # Sharding a static non-tensor argument across a mesh axis is invalid.
    sharded_static_op = jax_op(
        "pallas_verify_fx::statics_sharded_scalar",
        sharded_with_statics,
        mesh=mesh2,
        input_partition_specs=(P("rank"), P("rank"), None),
        output_partition_specs=(P("rank"),),
    )
    sharded_static_rep = verifier.verify(
        sharded_static_op, (2.0, x_shard, True), mode=STATIC
    )
    self.assertFalse(
        sharded_static_rep.passed, msg=sharded_static_rep.summary()
    )
    self.assertNotEmpty(
        sharded_static_rep.findings_by_category(
            Category.PARTITION_SPEC_MISMATCH
        ),
        msg=sharded_static_rep.summary(),
    )

  def test_abstract_mesh_jax_op_passes_contracts_lowering_and_fx_tracing(self):
    abstract_mesh = jax.sharding.AbstractMesh((2,), ("tp",))

    def wrapped_sharded(x: jax.Array) -> jax.Array:
      return jax.shard_map(
          _pallas_add_one,
          mesh=abstract_mesh,
          in_specs=(P("tp"),),
          out_specs=P("tp"),
          check_vma=False,
      )(x)

    x_shard = torch.ones((8, 128), dtype=torch.float32)
    valid_op = jax_op(
        "pallas_verify_fx::abstract_mesh_valid",
        wrapped_sharded,
        mesh=abstract_mesh,
        input_partition_specs=(P("tp"),),
        output_partition_specs=(P("tp"),),
    )
    valid_rep = verifier.verify(valid_op, (x_shard,))
    self.assertTrue(valid_rep.passed, msg=valid_rep.summary())
    self.assertEmpty(
        valid_rep.findings_by_severity(verifier.Severity.WARNING),
        msg=valid_rep.summary(),
    )
    self.assertIn(verifier.Phase.CONTRACTS, valid_rep.checked_phases)
    self.assertIn(verifier.Phase.LOWERING, valid_rep.checked_phases)

    # FX program tracing (`make_fx`) over an `AbstractMesh` `@jax_op` must
    # succeed on CPU without requiring physical TPU devices.
    prog_rep = verifier.verify(lambda t: valid_op(t) + 1.0, (x_shard,))
    self.assertTrue(prog_rep.passed, msg=prog_rep.summary())
    self.assertIn(verifier.Phase.FX_GRAPH, prog_rep.checked_phases)

  def test_abstract_mesh_direct_shard_map_callable_preserves_axis_binding(self):
    abstract_mesh = jax.sharding.AbstractMesh((2,), ("tp",))

    def per_shard_scale(s: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(
              Ellipsis,
              x_ref[...] * (jax.lax.axis_index("tp").astype(x_ref.dtype) + 1.0),
          ),
          out_shape=jax.ShapeDtypeStruct(s.shape, s.dtype),
      )(s)

    direct_shard_map_fn = jax.shard_map(
        per_shard_scale,
        mesh=abstract_mesh,
        in_specs=(P("tp"),),
        out_specs=P("tp"),
        check_vma=False,
    )
    direct_op = jax_op(
        "pallas_verify_fx::abstract_mesh_direct_shard_map",
        direct_shard_map_fn,
        mesh=abstract_mesh,
        input_partition_specs=(P("tp"),),
        output_partition_specs=(P("tp"),),
    )
    x_shard = torch.ones((8, 128), dtype=torch.float32)
    direct_rep = verifier.verify(direct_op, (x_shard,))
    self.assertTrue(direct_rep.passed, msg=direct_rep.summary())
    self.assertIn(verifier.Phase.LOWERING, direct_rep.checked_phases)

  def test_abstract_mesh_detects_partition_spec_and_fake_shape_errors(self):
    abstract_mesh = jax.sharding.AbstractMesh((2,), ("tp",))

    def wrapped_sharded(x: jax.Array) -> jax.Array:
      return jax.shard_map(
          _pallas_add_one,
          mesh=abstract_mesh,
          in_specs=(P("tp"),),
          out_specs=P("tp"),
          check_vma=False,
      )(x)

    x_shard = torch.ones((8, 128), dtype=torch.float32)

    bad_axis_op = jax_op(
        "pallas_verify_fx::abstract_mesh_bad_axis",
        wrapped_sharded,
        mesh=abstract_mesh,
        input_partition_specs=(P("dp"),),
        output_partition_specs=(P("tp"),),
    )
    bad_axis_rep = verifier.verify(bad_axis_op, (x_shard,), mode=STATIC)
    self.assertFalse(bad_axis_rep.passed, msg=bad_axis_rep.summary())
    self.assertNotEmpty(
        bad_axis_rep.findings_by_category(Category.PARTITION_SPEC_MISMATCH),
        msg=bad_axis_rep.summary(),
    )

    bad_fake_op = jax_op(
        "pallas_verify_fx::abstract_mesh_bad_fake",
        wrapped_sharded,
        mesh=abstract_mesh,
        input_partition_specs=(P("tp"),),
        output_partition_specs=(P("tp"),),
    )
    bad_fake_op.register_fake(
        lambda x: x.new_empty((x.shape[0] * 2, x.shape[1]))
    )
    bad_fake_rep = verifier.verify(bad_fake_op, (x_shard,), mode=STATIC)
    self.assertFalse(bad_fake_rep.passed, msg=bad_fake_rep.summary())
    self.assertNotEmpty(
        bad_fake_rep.findings_by_category(Category.FAKE_KERNEL_MISMATCH),
        msg=bad_fake_rep.summary(),
    )

  def test_single_device_lowering_overrides_and_restores_ambient_abstract_mesh(
      self,
  ):
    abstract_mesh = jax.sharding.AbstractMesh((2,), ("tp",))

    def tpu_info_fn(x: jax.Array) -> jax.Array:
      sublanes = pltpu.get_tpu_info().num_sublanes
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(
              Ellipsis, x_ref[...] + float(sublanes)
          ),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      )(x)

    tpu_info_op = jax_op("pallas_verify_fx::tpu_info_single_chip", tpu_info_fn)
    x = torch.ones((8, 128), dtype=torch.float32)
    with jax.sharding.use_abstract_mesh(abstract_mesh):
      ambient_single_rep = verifier.verify(tpu_info_op, (x,), mode=STATIC)
      self.assertEqual(jax.sharding.get_abstract_mesh(), abstract_mesh)
    self.assertTrue(ambient_single_rep.passed, msg=ambient_single_rep.summary())
    self.assertIn(verifier.Phase.LOWERING, ambient_single_rep.checked_phases)

  def test_fp8_jax_op_and_custom_autograd_pass_verification(self):
    def fp8_scale_jax(x_fp8: jax.Array, scale: jax.Array) -> jax.Array:
      def kernel(x_ref, s_ref, o_ref):
        o_ref[...] = x_ref[...].astype(jnp.float32) * s_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x_fp8.shape, jnp.float32),
      )(x_fp8, scale)

    fp8_op = jax_op("pallas_verify_fx::fp8_scale_op", fp8_scale_jax)

    def setup_context(ctx, inputs, output):
      del output
      x_fp8, _ = inputs
      ctx.save_for_backward(x_fp8)

    def backward(ctx, grad_out):
      (x_fp8,) = ctx.saved_tensors
      grad_scale = grad_out * x_fp8.to(torch.float32)
      return None, grad_scale

    fp8_op.register_autograd(backward, setup_context=setup_context)

    def program(x_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
      return fp8_op(x_fp8, scale)

    # Non-contiguous transposed FP8 input exercises stride handling before
    # uint8 byte-reinterpretation during CPU interpret execution.
    x_fp8 = (
        torch.ones((128, 8), dtype=torch.float32).to(torch.float8_e4m3fn).t()
    )
    scale = torch.full((8, 128), 2.0, dtype=torch.float32)
    rep = verifier.verify(program, (x_fp8, scale))
    self.assertTrue(rep.passed, msg=rep.summary())
    self.assertEmpty(
        rep.findings_by_severity(verifier.Severity.WARNING), msg=rep.summary()
    )
    self.assertIn(verifier.Phase.FX_GRAPH, rep.checked_phases)
    self.assertIn(verifier.Phase.AUTOGRAD, rep.checked_phases)

  def test_autograd_parity_treats_fp8_e8m0_outputs_as_non_differentiable(self):
    def act_with_e8m0_out(x: jax.Array) -> tuple[jax.Array, jax.Array]:
      out = x * 2.0
      e8m0_scale = jnp.ones(x.shape, dtype=jnp.float8_e8m0fnu)
      return out, e8m0_scale

    valid_e8m0_op = jax_op(
        "pallas_verify_fx::act_with_e8m0_out_valid", act_with_e8m0_out
    )
    valid_e8m0_op.register_autograd(
        lambda ctx, grad_out, _grad_scale: grad_out * 2.0,
        setup_context=lambda ctx, inputs, output: None,
    )

    buggy_e8m0_op = jax_op(
        "pallas_verify_fx::act_with_e8m0_out_buggy", act_with_e8m0_out
    )
    buggy_e8m0_op.register_autograd(
        lambda ctx, grad_out, _grad_scale: grad_out * 5.0,
        setup_context=lambda ctx, inputs, output: None,
    )

    x = torch.ones((8, 128), dtype=torch.float32)
    valid_rep = verifier.verify(valid_e8m0_op, (x,))
    self.assertTrue(valid_rep.passed, msg=valid_rep.summary())
    self.assertIn(verifier.Phase.AUTOGRAD, valid_rep.checked_phases)

    buggy_rep = verifier.verify(buggy_e8m0_op, (x,))
    self.assertFalse(buggy_rep.passed, msg=buggy_rep.summary())
    self.assertNotEmpty(
        buggy_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=buggy_rep.summary(),
    )

  def test_keyword_tensor_donation_and_getattr_view_aliasing(self):
    donating_op = jax_op(
        "pallas_verify_fx::kw_donate_op",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def kw_use_after_donate(*, x: torch.Tensor) -> torch.Tensor:
      out = donating_op(x)
      return out + x

    def getattr_view_returned(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
      v = x.mT.mT
      out = donating_op(x)
      return out, v

    x = torch.ones((8, 128), dtype=torch.float32)
    kw_rep = verifier.verify(
        kw_use_after_donate, (), kwargs={"x": x}, mode=STATIC
    )
    self.assertFalse(kw_rep.passed, msg=kw_rep.summary())
    self.assertNotEmpty(
        kw_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=kw_rep.summary(),
    )

    view_rep = verifier.verify(getattr_view_returned, (x,), mode=STATIC)
    self.assertFalse(view_rep.passed, msg=view_rep.summary())
    self.assertNotEmpty(
        view_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=view_rep.summary(),
    )

  def test_tpu_version_normalization_and_inference(self):
    def trillium_fn(x: jax.Array) -> jax.Array:
      if pltpu.get_tpu_info().generation != 6:
        raise ValueError("expected TPU v6e")
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] + 1.0),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      )(x)

    trillium_fn.supported_on = lambda dev: "v6" in getattr(
        dev, "device_kind", ""
    )
    doc_op = jax_op("pallas_verify_fx::trillium_supported_on_op", trillium_fn)
    x = torch.ones((8, 128), dtype=torch.float32)
    doc_rep = verifier.verify(doc_op, (x,), mode=STATIC)
    self.assertTrue(doc_rep.passed, msg=doc_rep.summary())
    self.assertEmpty(
        doc_rep.findings_by_severity(verifier.Severity.WARNING),
        msg=doc_rep.summary(),
    )

    # Wrapping `doc_op` in a plain PyTorch callable with default `None`
    # `tpu_version` must still infer `TpuVersion.V6E` per kernel.
    prog_rep = verifier.verify(lambda t: doc_op(t) + 1.0, (x,), mode=STATIC)
    self.assertTrue(prog_rep.passed, msg=prog_rep.summary())

    # Explicit override to TPU v5e must fail lowering on the TPU v6e assertion.
    v5e_rep = verifier.verify(
        doc_op,
        (x,),
        mode=STATIC,
        options=verifier.VerificationOptions(
            tpu_version=torch.tpu.TpuVersion.V5E
        ),
    )
    self.assertFalse(v5e_rep.passed, msg=v5e_rep.summary())

  def test_supported_on_intersection_and_self_access_safety(self):
    def _v6_or_v7_fn(x: jax.Array) -> jax.Array:
      return _pallas_add_one(x)

    _v6_or_v7_fn.supported_on = lambda dev: dev.device_kind in (
        "TPU v6 lite",
        "TPU7x",
    )

    def _v6_only_fn(x: jax.Array) -> jax.Array:
      assert pltpu.get_tpu_info().generation == 6
      return _pallas_add_one(x)

    _v6_only_fn.supported_on = lambda dev: dev.device_kind == "TPU v6 lite"

    op_a = jax_op("pallas_verify_fx::sup_intersect_a", _v6_or_v7_fn)
    op_b = jax_op("pallas_verify_fx::sup_intersect_b", _v6_only_fn)

    def _combined_prog(t: torch.Tensor) -> torch.Tensor:
      return op_b(op_a(t))

    self.assertEqual(
        fx_verifier.infer_tpu_version(_combined_prog),
        torch.tpu.TpuVersion.V6E,
    )
    x = torch.ones((8, 128), dtype=torch.float32)
    comb_rep = verifier.verify(_combined_prog, (x,), mode=STATIC)
    self.assertTrue(comb_rep.passed, msg=comb_rep.summary())

    # Unbound instance `supported_on(self, device)` that short-circuits `or` on
    # v5e before touching `self.enable_v7` on v7 must be discarded rather than
    # partially narrowing to v5e.
    class _InstanceDependentKernel:
      enable_v7: bool = True

      def supported_on(self, device) -> bool:
        return device.device_kind == "TPU v5 lite" or (
            device.device_kind == "TPU7x" and self.enable_v7
        )

    self.assertEqual(
        fx_verifier.infer_tpu_version(_InstanceDependentKernel),
        torch.tpu.TpuVersion.V7,
    )

  def test_negative_donate_argnums_tracks_donated_positional_and_keyword_args(
      self,
  ):
    neg_donate_op = jax_op(
        "pallas_verify_fx::neg_donate_op",
        _pallas_add_one,
        donate_argnums=(-1,),
    )
    neg_bin_op = jax_op(
        "pallas_verify_fx::neg_donate_binary_op",
        _pallas_binary_add,
        donate_argnums=(-1,),
    )
    x = torch.ones((8, 128), dtype=torch.float32)
    y = torch.ones((8, 128), dtype=torch.float32)

    def use_after_neg_donate(scale: float, t: torch.Tensor) -> torch.Tensor:
      out = neg_donate_op(t)
      return out + t * scale

    fx_rep = verifier.verify(use_after_neg_donate, (2.0, x), mode=STATIC)
    self.assertFalse(fx_rep.passed, msg=fx_rep.summary())
    self.assertNotEmpty(
        fx_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=fx_rep.summary(),
    )

    def kw_neg_donate_safe(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
      out = neg_bin_op(a, y=b)
      return out + a

    def kw_neg_donate_buggy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
      out = neg_bin_op(a, y=b)
      return out + b

    self.assertTrue(
        verifier.verify(kw_neg_donate_safe, (x, y), mode=STATIC).passed
    )
    kw_neg_rep = verifier.verify(kw_neg_donate_buggy, (x, y), mode=STATIC)
    self.assertFalse(kw_neg_rep.passed, msg=kw_neg_rep.summary())
    self.assertNotEmpty(
        kw_neg_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=kw_neg_rep.summary(),
    )

    alias_callable = types.SimpleNamespace(
        _donate_argnums_set=set(),
        input_output_aliases={-1: 0},
        _orig_params=neg_bin_op._init_fn._orig_params,
    )
    self.assertEqual(
        fx_verifier._get_donated_arg_indices(alias_callable, num_pos_args=1),
        {1},
    )

  def test_bare_partition_spec_rejected_in_op_contracts(self):
    abstract_mesh = jax.sharding.AbstractMesh((2,), ("tp",))
    bare_spec_op = jax_op(
        "pallas_verify_fx::bare_input_partition_spec",
        _pallas_add_one,
        mesh=abstract_mesh,
        input_partition_specs=P("tp"),  # pytype: disable=wrong-arg-types
        output_partition_specs=(P("tp"),),
    )
    x = torch.ones((8, 128), dtype=torch.float32)
    bare_rep = verifier.verify(bare_spec_op, (x,), mode=STATIC)
    self.assertFalse(bare_rep.passed, msg=bare_rep.summary())
    self.assertNotEmpty(
        bare_rep.findings_by_category(Category.PARTITION_SPEC_MISMATCH),
        msg=bare_rep.summary(),
    )

  def test_fx_list_containers_track_use_after_donate(self):
    donating_op = jax_op(
        "pallas_verify_fx::fx_containers_donate",
        _pallas_add_one,
        donate_argnums=(0,),
    )
    x = torch.ones((8, 128), dtype=torch.float32)

    def cat_after_donate(t: torch.Tensor) -> torch.Tensor:
      out = donating_op(t)
      return torch.cat([out, t], dim=0)

    def return_list_after_donate(t: torch.Tensor) -> list[torch.Tensor]:
      out = donating_op(t)
      return [out, t]

    cat_rep = verifier.verify(cat_after_donate, (x,), mode=STATIC)
    self.assertFalse(cat_rep.passed, msg=cat_rep.summary())
    self.assertNotEmpty(
        cat_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=cat_rep.summary(),
    )

    ret_rep = verifier.verify(return_list_after_donate, (x,), mode=STATIC)
    self.assertFalse(ret_rep.passed, msg=ret_rep.summary())
    self.assertNotEmpty(
        ret_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=ret_rep.summary(),
    )

  def test_fx_metadata_only_ops_after_donate_do_not_trigger_false_positive(
      self,
  ):
    donating_op = jax_op(
        "pallas_verify_fx::fx_metadata_only_donate",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def metadata_only_after_donate(t: torch.Tensor) -> torch.Tensor:
      out = donating_op(t)
      return out + torch.zeros_like(t)

    x = torch.ones((8, 128), dtype=torch.float32)
    meta_rep = verifier.verify(metadata_only_after_donate, (x,), mode=STATIC)
    self.assertTrue(meta_rep.passed, msg=meta_rep.summary())

  def test_fx_kwargs_bind_to_placeholders_in_insertion_order(self):
    donating_bin_op = jax_op(
        "pallas_verify_fx::fx_kw_order_donate_bin",
        _pallas_binary_add,
        donate_argnums=(0,),
    )
    x = torch.ones((8, 128), dtype=torch.float32)
    y = torch.ones((8, 128), dtype=torch.float32)

    def kw_order_safe(*, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
      out = donating_bin_op(z, a)
      return out + a

    def kw_order_buggy(*, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
      out = donating_bin_op(z, a)
      return out + z

    safe_rep = verifier.verify(
        kw_order_safe, (), kwargs={"z": x, "a": y}, mode=STATIC
    )
    self.assertTrue(safe_rep.passed, msg=safe_rep.summary())

    buggy_rep = verifier.verify(
        kw_order_buggy, (), kwargs={"z": x, "a": y}, mode=STATIC
    )
    self.assertFalse(buggy_rep.passed, msg=buggy_rep.summary())
    self.assertNotEmpty(
        buggy_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=buggy_rep.summary(),
    )

  def test_autograd_save_for_backward_of_donated_input_detected(self):
    donating_autograd_op = jax_op(
        "pallas_verify_fx::donating_saved_for_backward",
        _pallas_scale_fwd,
        donate_argnums=(0,),
    )
    donating_autograd_op.register_autograd(
        lambda ctx, grad: grad * 3.0,
        setup_context=lambda ctx, inputs, output: ctx.save_for_backward(
            inputs[0]
        ),
    )
    x_req = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    saved_donor_rep = verifier.verify(donating_autograd_op, (x_req,))
    self.assertFalse(saved_donor_rep.passed, msg=saved_donor_rep.summary())
    findings = saved_donor_rep.findings_by_category(Category.USE_AFTER_DONATE)
    self.assertNotEmpty(findings, msg=saved_donor_rep.summary())
    self.assertEqual(findings[0].phase, verifier.Phase.AUTOGRAD)

  def test_autograd_parity_detects_localized_single_element_backward_error(
      self,
  ):
    localized_bad_op = jax_op(
        "pallas_verify_fx::localized_bad_bwd", _pallas_scale_fwd
    )

    def _localized_bad_bwd(ctx, grad):
      del ctx
      bad = grad.clone() * 3.0
      bad[-1, -1] = 0.0
      return bad

    localized_bad_op.register_autograd(_localized_bad_bwd)
    x_req = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    loc_rep = verifier.verify(localized_bad_op, (x_req,))
    self.assertFalse(loc_rep.passed, msg=loc_rep.summary())
    self.assertNotEmpty(
        loc_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=loc_rep.summary(),
    )

  def test_tpu_version_inference_supports_all_torchtpu_hardware_specs(self):
    supported_tpu_names = {
        spec.name for spec in hardware.TpuSpec.specs_by_pci_device_id().values()
    }
    self.assertContainsSubset({"TPU v8i", "TPU v8t"}, supported_tpu_names)

    for expected_version in torch.tpu.TpuVersion:
      with self.subTest(tpu_version=expected_version):
        spec = hardware.TpuSpec.for_version(expected_version)
        self.assertIn(spec.name, supported_tpu_names)
        jax_kind = fx_verifier.abstract_tpu_device(expected_version).device_kind

        class _TargetTpuOp:

          @staticmethod
          def supported_on(
              device: jax.Device, expected_jax_kind: str = jax_kind
          ) -> bool:
            return getattr(device, "device_kind", "") == expected_jax_kind

          def __call__(self, x: jax.Array) -> jax.Array:
            return _pallas_add_one(x)

        target_op = _TargetTpuOp()
        self.assertIs(
            fx_verifier.infer_tpu_version(target_op), expected_version
        )

        def _closure_wrapper(
            x: jax.Array,
            _pos_default: object = target_op,
            *,
            _kw_default: object = target_op,
        ) -> jax.Array:
          del _pos_default, _kw_default
          return target_op(x)

        partial_op = functools.partial(_closure_wrapper, _kw_default=target_op)
        self.assertIs(
            fx_verifier.infer_tpu_version(partial_op), expected_version
        )

    v6e_mesh = jax.sharding.AbstractMesh(
        (2,),
        ("tp",),
        abstract_device=fx_verifier.abstract_tpu_device(
            torch.tpu.TpuVersion.V6E
        ),
    )
    v6e_mesh_op = jax_op(
        "pallas_verify_fx::v6e_abstract_mesh_op",
        _pallas_add_one,
        mesh=v6e_mesh,
        input_partition_specs=(P("tp"),),
        output_partition_specs=P("tp"),
    )
    self.assertIs(
        fx_verifier.infer_tpu_version(v6e_mesh_op),
        torch.tpu.TpuVersion.V6E,
    )

    t = torch.zeros((2, 4), dtype=torch.float32)
    self.assertEqual(
        verifier_impl._signature(t), ("tensor", (2, 4), torch.float32)
    )
    self.assertEqual(
        verifier_impl._signature([t]), (("tensor", (2, 4), torch.float32),)
    )
    self.assertEqual(verifier_impl._signature({"k": 7}), (("k", 7),))
    self.assertEqual(verifier_impl._signature({1, 2}), ("set", repr({1, 2})))

  def test_supported_on_keyerror_predicate_and_multi_op_chain_inference(self):
    def _dict_lookup_v6e_fn(x: jax.Array) -> jax.Array:
      if pltpu.get_tpu_info().generation != 6:
        raise ValueError("expected TPU v6e")
      return _pallas_add_one(x)

    # Raises `KeyError` on candidate versions other than v6e; per-version
    # `KeyError` must count as `False` rather than discarding the predicate.
    _dict_lookup_v6e_fn.supported_on = lambda d: {
        "TPU v6 lite": True,
    }[d.device_kind]

    self.assertIs(
        fx_verifier.infer_tpu_version(_dict_lookup_v6e_fn),
        torch.tpu.TpuVersion.V6E,
    )

    def _v6_or_v7_step(x: jax.Array) -> jax.Array:
      return _pallas_add_one(x)

    _v6_or_v7_step.supported_on = lambda d: d.device_kind in (
        "TPU v6 lite",
        "TPU7x",
    )
    step_a = jax_op("pallas_verify_fx::chain_step_a", _v6_or_v7_step)
    step_b = jax_op("pallas_verify_fx::chain_step_b", _v6_or_v7_step)
    step_c = jax_op("pallas_verify_fx::chain_step_c_v6e", _dict_lookup_v6e_fn)

    def _three_op_chain(t: torch.Tensor) -> torch.Tensor:
      return step_c(step_b(step_a(t)))

    x = torch.ones((8, 128), dtype=torch.float32)
    self.assertIs(
        fx_verifier.infer_tpu_version(_three_op_chain),
        torch.tpu.TpuVersion.V6E,
    )
    chain_rep = verifier.verify(_three_op_chain, (x,), mode=STATIC)
    self.assertTrue(chain_rep.passed, msg=chain_rep.summary())

    def _never_supported_fn(arr: jax.Array) -> jax.Array:
      return _pallas_add_one(arr)

    _never_supported_fn.supported_on = lambda dev: False
    self.assertIs(
        fx_verifier.infer_tpu_version(_never_supported_fn),
        torch.tpu.TpuVersion.V7,
    )

    v5p_mesh = jax.sharding.AbstractMesh(
        (2,),
        ("tp",),
        abstract_device=fx_verifier.abstract_tpu_device(
            torch.tpu.TpuVersion.V5P
        ),
    )
    sharded_arg = jax.ShapeDtypeStruct(
        (16, 128),
        jnp.float32,
        sharding=jax.sharding.NamedSharding(v5p_mesh, P("tp", None)),
    )
    self.assertIs(
        fx_verifier.infer_tpu_version(
            _pallas_add_one, example_args=(sharded_arg,)
        ),
        torch.tpu.TpuVersion.V5P,
    )

  def test_abstract_tpu_device_and_raw_jax_mesh_tpu_inference(self):
    v5p_mesh = jax.sharding.AbstractMesh(
        (2,),
        ("tp",),
        abstract_device=fx_verifier.abstract_tpu_device(
            torch.tpu.TpuVersion.V5P
        ),
    )

    def _raw_mesh_fn(x: jax.Array) -> jax.Array:
      return jax.shard_map(
          _pallas_add_one,
          mesh=v5p_mesh,
          in_specs=P("tp"),
          out_specs=P("tp"),
      )(x)

    self.assertIs(
        fx_verifier.infer_tpu_version(_raw_mesh_fn), torch.tpu.TpuVersion.V5P
    )
    self.assertIs(
        fx_verifier.infer_tpu_version(_pallas_add_one), torch.tpu.TpuVersion.V7
    )
    self.assertEqual(fx_verifier.abstract_tpu_device("v5p").num_cores, 2)
    self.assertEqual(fx_verifier.abstract_tpu_device("v6e").num_cores, 1)


if __name__ == "__main__":
  absltest.main()
