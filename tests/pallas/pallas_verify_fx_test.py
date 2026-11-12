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
import threading
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

  def _assert_autograd_ran(self, report: verifier.VerificationReport) -> None:
    self.assertTrue(report.passed, msg=report.summary())
    self.assertIn(verifier.Phase.AUTOGRAD, report.checked_phases)
    autograd_incomplete = [
        f
        for f in report.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.phase == verifier.Phase.AUTOGRAD
    ]
    self.assertEmpty(autograd_incomplete, msg=report.summary())

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
    self._assert_autograd_ran(valid_report)

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
    self._assert_autograd_ran(valid_report)

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
    self._assert_autograd_ran(dist_ok_rep)
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
    self._assert_autograd_ran(rep)
    self.assertEmpty(
        rep.findings_by_severity(verifier.Severity.WARNING), msg=rep.summary()
    )
    self.assertIn(verifier.Phase.FX_GRAPH, rep.checked_phases)

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
    self._assert_autograd_ran(valid_rep)

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

  def test_integer_tensor_signature_and_concrete_fx_propagation(self):
    t = torch.zeros((2, 4), dtype=torch.float32)
    self.assertEqual(
        verifier_impl._signature(t), ("tensor", (2, 4), torch.float32)
    )
    self.assertEqual(
        verifier_impl._signature([t]), (("tensor", (2, 4), torch.float32),)
    )
    self.assertEqual(verifier_impl._signature({"k": 7}), (("k", 7),))
    self.assertEqual(verifier_impl._signature({1, 2}), ("set", repr({1, 2})))

    i0 = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    i1 = torch.tensor([0, 1, 2, 99], dtype=torch.int32)
    self.assertNotEqual(
        verifier_impl._signature(i0), verifier_impl._signature(i1)
    )
    large0 = torch.zeros((4097,), dtype=torch.int32)
    large1 = large0.clone()
    large1[-1] = 7
    self.assertNotEqual(
        verifier_impl._signature(large0), verifier_impl._signature(large1)
    )

    def _mutating_and_pure_prog(
        x: torch.Tensor, caller_idx: torch.Tensor
    ) -> torch.Tensor:
      counts = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
      offsets = torch.cumsum(counts, dim=0, dtype=torch.int32) - 1
      uninit_buf = torch.empty((4,), dtype=torch.int32)
      uninit_buf.fill_(0)
      caller_view = caller_idx.view(-1)
      caller_view.copy_(offsets)
      return x + offsets[0] + uninit_buf[0] + caller_idx[0]

    x = torch.ones((8, 128), dtype=torch.float32)
    caller_idx = torch.tensor([99, 99, 99, 99], dtype=torch.int32)
    gm = fx_verifier.trace_to_fx_graph(_mutating_and_pure_prog, (x, caller_idx))
    phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_env = fx_verifier.match_placeholders_to_inputs(
        phs, (x, caller_idx), None
    )
    _, env = verifier_impl._collect_graph_module_envs(gm, ph_env)[0]
    # `torch.empty` is never evaluated to heap garbage, and in-place mutation of
    # `caller_view` invalidates the aliased `caller_idx` placeholder entry.
    self.assertTrue(
        all(
            not (
                isinstance(v, torch.Tensor)
                and v.shape == (4,)
                and torch.equal(v, torch.tensor([99, 99, 99, 99]))
            )
            for v in env.values()
        )
    )
    self.assertTrue(
        any(
            isinstance(v, torch.Tensor)
            and torch.equal(v, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
            for v in env.values()
        )
    )
    self.assertTrue(
        all(
            verifier_impl._fx_node_base_op_name(n) != "empty"
            for n in env
            if n.op == "call_function"
        )
    )

  def test_joint_autograd_detects_saved_activation_donated_later(self):
    donating_binary = jax_op(
        "pallas_verify_fx::joint_saved_donor",
        _pallas_binary_add,
        donate_argnums=(1,),
    )
    donating_binary.register_autograd(lambda ctx, g: (g, g))

    class _SavedThenDonated(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.ones((8, 128), dtype=torch.float32)
        )

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * self.scale
        # `h * h` saves `h` on the standard PyTorch autograd tape, and then
        # `donating_binary` donates `h` in the same forward pass!
        sq = h * h
        return donating_binary(sq, h)

    mod = _SavedThenDonated()
    x = torch.ones((8, 128), dtype=torch.float32, requires_grad=True)
    rep = verifier.verify(mod, (x,), mode=STATIC)
    self.assertFalse(rep.passed, msg=rep.summary())
    donate_findings = rep.findings_by_category(Category.USE_AFTER_DONATE)
    self.assertNotEmpty(donate_findings, msg=rep.summary())
    self.assertTrue(
        any("autograd backward" in f.message for f in donate_findings),
        msg=rep.summary(),
    )

    broken_bwd_donor = jax_op(
        "pallas_verify_fx::joint_broken_bwd_donor",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def _broken_bwd(ctx, g):
      del ctx, g
      raise RuntimeError("synthetic untraceable backward failure")

    broken_bwd_donor.register_autograd(_broken_bwd)
    inc_rep = verifier.verify(
        lambda t: broken_bwd_donor(t * 2.0), (x,), mode=STATIC
    )
    inc_findings = [
        f
        for f in inc_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.phase == verifier.Phase.FX_GRAPH
    ]
    self.assertNotEmpty(inc_findings, msg=inc_rep.summary())
    self.assertEqual(inc_findings[0].severity, verifier.Severity.WARNING)

    # A forward-only `@jax_op` without `register_autograd` must NOT emit a
    # spurious `VERIFICATION_INCOMPLETE` warning when `x.requires_grad=True`,
    # and must not mask a differentiable output's saved-activation donation bug.
    fwd_only_donor = jax_op(
        "pallas_verify_fx::joint_fwd_only_donor",
        _pallas_add_one,
        donate_argnums=(0,),
    )
    fwd_only_rep = verifier.verify(
        lambda t: fwd_only_donor(t.clone()), (x,), mode=STATIC
    )
    fwd_only_rep.raise_for_errors(fail_on_incomplete=True)

    def _mixed_loss_and_fwd_only(t: torch.Tensor):
      h = t * 2.0
      sq = h * h
      donated_res = donating_binary(sq, h)
      aux = fwd_only_donor(t.detach().clone())
      return donated_res, aux

    mixed_rep = verifier.verify(_mixed_loss_and_fwd_only, (x,), mode=STATIC)
    self.assertTrue(
        any(
            "autograd backward" in f.message
            for f in mixed_rep.findings_by_category(Category.USE_AFTER_DONATE)
        ),
        msg=mixed_rep.summary(),
    )

  def test_fx_higher_order_cond_donation_and_nested_jax_op(self):
    donating_op = jax_op(
        "pallas_verify_fx::cond_donate_branch",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def _unaligned_pallas(arr: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] + 1.0),
          out_shape=jax.ShapeDtypeStruct(arr.shape, arr.dtype),
          in_specs=[pl.BlockSpec((8, 64), lambda i: (0, i))],
          out_specs=pl.BlockSpec((8, 64), lambda i: (0, i)),
          grid=(2,),
      )(arr)

    nested_bad_lowering_op = jax_op(
        "pallas_verify_fx::cond_nested_bad_lowering",
        _unaligned_pallas,
    )
    step_op = jax_op("pallas_verify_fx::scan_step_add_one", _pallas_add_one)

    def cond_use_after_donate(
        pred: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
      y = torch.cond(
          pred,
          lambda t: donating_op(t),
          lambda t: t.clone(),
          (x,),
      )
      return y + x

    def cond_nested_bad_lowering(
        pred: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
      return torch.cond(
          pred,
          lambda t: t.clone(),
          lambda t: nested_bad_lowering_op(t),
          (x,),
      )

    def while_loop_donate_additional_input(x: torch.Tensor) -> torch.Tensor:
      from torch._higher_order_ops.while_loop import while_loop_op  # pylint: disable=g-import-not-at-top

      i0 = torch.tensor(0, dtype=torch.int32)
      acc0 = torch.zeros_like(x)
      _, out_acc = while_loop_op(
          lambda step, acc, shared: step < 2,
          lambda step, acc, shared: (step + 1, acc + donating_op(shared)),
          (i0, acc0),
          (x,),
      )
      return out_acc

    def scan_donate_additional_input(
        xs_seq: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
      from torch._higher_order_ops.scan import scan_op  # pylint: disable=g-import-not-at-top

      acc0 = torch.zeros_like(x)
      final_carry, ys = scan_op(
          lambda carry, x_slice, shared: (
              carry + step_op(x_slice),
              donating_op(shared),
          ),
          [acc0],
          [xs_seq],
          (x,),
      )
      return final_carry + ys[0]

    pred = torch.tensor(True)
    x = torch.ones((8, 128), dtype=torch.float32)
    xs_seq = torch.ones((2, 8, 128), dtype=torch.float32)
    donate_rep = verifier.verify(cond_use_after_donate, (pred, x), mode=STATIC)
    self.assertNotEmpty(
        donate_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=donate_rep.summary(),
    )

    while_rep = verifier.verify(
        while_loop_donate_additional_input, (x,), mode=STATIC
    )
    self.assertNotEmpty(
        while_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=while_rep.summary(),
    )

    scan_rep = verifier.verify(
        scan_donate_additional_input, (xs_seq, x), mode=STATIC
    )
    self.assertNotEmpty(
        scan_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=scan_rep.summary(),
    )

    lowering_rep = verifier.verify(
        cond_nested_bad_lowering, (pred, x), mode=STATIC
    )
    self.assertNotEmpty(
        lowering_rep.findings_by_category(Category.LOWERING_FAILURE),
        msg=lowering_rep.summary(),
    )

  def test_multi_output_aminmax_out_and_set_inplace_provenance(self):
    donating_op = jax_op(
        "pallas_verify_fx::multi_out_and_set_donate",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    def aminmax_disjoint_out(x: torch.Tensor) -> torch.Tensor:
      min_buf = torch.empty((8, 128), dtype=x.dtype)
      max_buf = torch.empty((8, 128), dtype=x.dtype)
      lo, hi = torch.aminmax(
          x, dim=-1, keepdim=True, out=(min_buf[:, :1], max_buf[:, :1])
      )
      del lo, hi
      out = donating_op(min_buf)
      return out + max_buf

    def safe_set_rebind(x: torch.Tensor) -> torch.Tensor:
      buf = x.clone()
      fresh = donating_op(buf)
      buf.set_(fresh)
      return buf + 1.0

    def buggy_set_shared_donate(x: torch.Tensor) -> torch.Tensor:
      alias_holder = torch.empty_like(x)
      donor = x.clone()
      alias_holder.set_(donor)
      out = donating_op(donor)
      return out + alias_holder

    x = torch.ones((8, 128), dtype=torch.float32)
    aminmax_rep = verifier.verify(aminmax_disjoint_out, (x,), mode=STATIC)
    self.assertTrue(aminmax_rep.passed, msg=aminmax_rep.summary())

    safe_set_rep = verifier.verify(safe_set_rebind, (x,), mode=STATIC)
    self.assertTrue(safe_set_rep.passed, msg=safe_set_rep.summary())

    buggy_set_rep = verifier.verify(buggy_set_shared_donate, (x,), mode=STATIC)
    self.assertNotEmpty(
        buggy_set_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=buggy_set_rep.summary(),
    )

  def test_io_aliases_exempt_from_donor_invalidation_and_catch_self_alias(self):
    io_alias_only = jax_op(
        "pallas_verify_fx::io_alias_only",
        _pallas_add_one,
        donate_argnums=(0,),
    )
    io_alias_jc = fx_verifier.extract_jax_callable(io_alias_only)
    self.assertIsNotNone(io_alias_jc)
    io_alias_jc.input_output_aliases = {0: 0}

    io_alias_binary = jax_op(
        "pallas_verify_fx::io_alias_binary",
        _pallas_binary_add,
        donate_argnums=(0, 1),
    )
    io_alias_bin_jc = fx_verifier.extract_jax_callable(io_alias_binary)
    self.assertIsNotNone(io_alias_bin_jc)
    io_alias_bin_jc.input_output_aliases = {0: 0}

    class _IoAliasParamModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones((8, 128), dtype=torch.float32))

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = io_alias_only(self.w)
        return y + self.w + x

    x = torch.ones((8, 128), dtype=torch.float32)
    ok_rep = verifier.verify(_IoAliasParamModule(), (x,), mode=STATIC)
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())

    def self_aliased(t: torch.Tensor) -> torch.Tensor:
      return io_alias_binary(t, t.view(8, 128))

    bad_rep = verifier.verify(self_aliased, (x,), mode=STATIC)
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.ALIASED_DONOR_INPUTS),
        msg=bad_rep.summary(),
    )

  def test_module_state_preserved_and_aliased_buffer_input_detected(self):
    donating_op = jax_op(
        "pallas_verify_fx::module_state_donate",
        _pallas_add_one,
        donate_argnums=(0,),
    )

    class _StatefulModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.register_buffer(
            "running_buf", torch.zeros((8, 128), dtype=torch.float32)
        )
        self.step_count = 0

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.step_count += 1
        self.running_buf.add_(1.0)
        return donating_op(x) + self.running_buf

    mod = _StatefulModule()
    x = torch.ones((8, 128), dtype=torch.float32)
    clean_rep = verifier.verify(mod, (x.clone(),), mode=STATIC)
    self.assertTrue(clean_rep.passed, msg=clean_rep.summary())
    self.assertEqual(mod.step_count, 0)
    self.assertTrue(
        torch.equal(mod.running_buf, torch.zeros((8, 128), dtype=torch.float32))
    )

    aliased_rep = verifier.verify(mod, (mod.running_buf,), mode=STATIC)
    self.assertNotEmpty(
        aliased_rep.findings_by_category(Category.DONATED_PARAMETER),
        msg=aliased_rep.summary(),
    )

  def test_partition_spec_unconstrained_and_defaulted_trailing_static_arg(self):
    mesh = jax.sharding.AbstractMesh((2,), ("tp",))
    self.assertEmpty(
        fx_verifier._validate_partition_spec_entry(
            P(P.UNCONSTRAINED, "tp"),
            2,
            mesh,
            "input 0",
            "unconstrained_spec_op",
        )
    )
    self.assertEqual(
        pallas_lib.get_global_shape((8, 128), mesh, P(P.UNCONSTRAINED, "tp")),
        (8, 256),
    )

    def _scaled_add(x: jax.Array, eps: float = 1e-5) -> jax.Array:
      return jax.shard_map(
          lambda s: _pallas_add_one(s) + eps,
          mesh=mesh,
          in_specs=(P("tp", None),),
          out_specs=P("tp", None),
          check_vma=False,
      )(x)

    sharded_op = jax_op(
        "pallas_verify_fx::unconstrained_default_static",
        _scaled_add,
        mesh=mesh,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P("tp", None),
    )

    def caller(x: torch.Tensor) -> torch.Tensor:
      return sharded_op(x)

    x = torch.ones((8, 128), dtype=torch.float32)
    rep = verifier.verify(caller, (x,), mode=STATIC)
    self.assertTrue(rep.passed, msg=rep.summary())

  def test_exported_fallback_hook_is_thread_local(self):
    def _bad_lowering_fn(x: jax.Array) -> jax.Array:
      def _bad_body(x_ref, o_ref):
        o_ref[...] = x_ref[...]
        raise ValueError("synthetic lowering failure")

      return pl.pallas_call(
          _bad_body, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x)

    bad_op = jax_op(
        "pallas_verify_fx::thread_local_bad_export", _bad_lowering_fn
    )
    jc = fx_verifier.extract_jax_callable(bad_op)
    self.assertIsNotNone(jc)
    orig_exported = jc.exported
    worker_errors: list[Exception] = []
    ph = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    with fx_verifier._eval_shape_fallback_for_jax_callables(bad_op):
      # Calling thread gets the eval_shape fallback namespace.
      fallback_res = jc.exported(ph)
      self.assertEqual(fallback_res.out_avals[0].shape, (8, 128))

      def _worker() -> None:
        try:
          jc.exported(ph)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          worker_errors.append(exc)

      t = threading.Thread(target=_worker)
      t.start()
      t.join()

    # Concurrent worker thread saw the unpatched `exported` failure, and exiting
    # the context restored `jc.exported`.
    self.assertLen(worker_errors, 1)
    self.assertIs(jc.exported, orig_exported)

  def test_hop_local_temporary_donation_and_scope_isolation(self):
    donating_op = jax_op(
        "pallas_verify_fx::hop_scoped_donate",
        _pallas_add_one,
        donate_argnums=(0,),
    )
    step_op = jax_op("pallas_verify_fx::hop_scoped_step", _pallas_add_one)

    def while_donate_local_temp(x: torch.Tensor) -> torch.Tensor:
      from torch._higher_order_ops.while_loop import while_loop_op  # pylint: disable=g-import-not-at-top

      i0 = torch.tensor(0, dtype=torch.int32)
      _, out_acc = while_loop_op(
          lambda step, acc: step < 3,
          lambda step, acc: (step + 1, acc + donating_op(acc * 1.0)),
          (i0, x.clone()),
          (),
      )
      return out_acc

    def while_replace_carry_iteratively(x: torch.Tensor) -> torch.Tensor:
      from torch._higher_order_ops.while_loop import while_loop_op  # pylint: disable=g-import-not-at-top

      i0 = torch.tensor(0, dtype=torch.int32)
      _, out_acc = while_loop_op(
          lambda step, acc: step < 3,
          lambda step, acc: (step + 1, donating_op(acc)),
          (i0, x.clone()),
          (),
      )
      return out_acc

    def scan_donate_local_temp(
        xs_seq: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
      from torch._higher_order_ops.scan import scan_op  # pylint: disable=g-import-not-at-top

      final_carry, ys = scan_op(
          lambda carry, x_slice: (
              carry + step_op(x_slice),
              donating_op(carry * 1.0),
          ),
          [x.clone()],
          [xs_seq],
          (),
      )
      return final_carry + ys[0]

    def cond_parent_child_mul_name_collision(
        pred: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
      a = x * 2.0
      y = torch.cond(
          pred,
          lambda t: donating_op(t * 3.0),
          lambda t: t + 1.0,
          (x,),
      )
      return a + y

    def cond_donate_input_without_later_use(
        pred: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
      buf = x.clone()
      y = torch.cond(
          pred,
          lambda t: donating_op(t),
          lambda t: t + 1.0,
          (buf,),
      )
      return y * 2.0

    x = torch.ones((8, 128), dtype=torch.float32)
    xs_seq = torch.ones((2, 8, 128), dtype=torch.float32)
    pred = torch.tensor(True)

    for fn, call_args in (
        (while_donate_local_temp, (x,)),
        (while_replace_carry_iteratively, (x,)),
        (scan_donate_local_temp, (xs_seq, x)),
        (cond_parent_child_mul_name_collision, (pred, x)),
        (cond_donate_input_without_later_use, (pred, x)),
    ):
      rep = verifier.verify(fn, call_args, mode=STATIC)
      self.assertTrue(rep.passed, msg=rep.summary())
      self.assertEmpty(
          rep.findings_by_category(Category.USE_AFTER_DONATE),
          msg=rep.summary(),
      )

    # Unrecognized higher-order op carrying a child `fx.GraphModule` emits an
    # `INFO` `VERIFICATION_INCOMPLETE` finding instead of mis-mapping operands.
    sub_g = torch.fx.Graph()
    sub_x = sub_g.placeholder("sub_x")
    sub_g.output(sub_x)
    sub_gm = torch.fx.GraphModule(torch.nn.Module(), sub_g)

    parent_mod = torch.nn.Module()
    parent_mod.add_module("child_sub", sub_gm)
    parent_g = torch.fx.Graph()
    ph_x = parent_g.placeholder("x")
    sub_attr = parent_g.get_attr("child_sub")
    unknown_hop = lambda gm_arg, tensor_arg: tensor_arg
    hop_out = parent_g.call_function(unknown_hop, (sub_attr, ph_x))
    parent_g.output(hop_out)
    parent_gm = torch.fx.GraphModule(parent_mod, parent_g)

    unk_rep = fx_verifier.verify_fx_graph(parent_gm, (x,))
    unk_inc = [
        f
        for f in unk_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.phase == verifier.Phase.FX_GRAPH
    ]
    self.assertNotEmpty(unk_inc, msg=unk_rep.summary())
    self.assertEqual(unk_inc[0].severity, verifier.Severity.INFO)

  def test_hop_carry_rotation_and_carry_duplication_detected(self):
    donating_op = jax_op(
        "pallas_verify_fx::hop_rot_donate",
        _pallas_add_one,
        donate_argnums=(0,),
    )
    donating_two = jax_op(
        "pallas_verify_fx::hop_dup_donate_two",
        _pallas_binary_add,
        donate_argnums=(0, 1),
    )

    def while_three_carry_rotation(x: torch.Tensor) -> torch.Tensor:
      from torch._higher_order_ops.while_loop import while_loop_op  # pylint: disable=g-import-not-at-top

      i0 = torch.tensor(0, dtype=torch.int32)
      a0 = x.clone()
      b0 = x.clone()
      c0 = x.clone()

      def _body(step, a, b, c):
        # Donates `a` and rotates stale `a` into carry slot `c`; after two
        # rotations, `a` arrives back at slot 0 and is donated a second time!
        _ = donating_op(a)
        return step + 1, b, c, a

      _, out_a, _, _ = while_loop_op(
          lambda step, a, b, c: step < 4,
          _body,
          (i0, a0, b0, c0),
          (),
      )
      return out_a

    def while_two_carry_duplication(x: torch.Tensor) -> torch.Tensor:
      from torch._higher_order_ops.while_loop import while_loop_op  # pylint: disable=g-import-not-at-top

      i0 = torch.tensor(0, dtype=torch.int32)
      a0 = x.clone()
      b0 = x.clone()

      def _dup_body(step, a, b):
        # Iteration 1 returns `(fresh, fresh)`, aliasing carry slots 0 and 1 on
        # iteration 2 when passed into 2-donor `donating_two(a, b)`.
        fresh = donating_two(a, b)
        return step + 1, fresh, fresh

      _, out_a, _ = while_loop_op(
          lambda step, a, b: step < 3,
          _dup_body,
          (i0, a0, b0),
          (),
      )
      return out_a

    x = torch.ones((8, 128), dtype=torch.float32)
    rot_rep = verifier.verify(while_three_carry_rotation, (x,), mode=STATIC)
    self.assertNotEmpty(
        rot_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=rot_rep.summary(),
    )

    dup_rep = verifier.verify(while_two_carry_duplication, (x,), mode=STATIC)
    self.assertNotEmpty(
        dup_rep.findings_by_category(Category.ALIASED_DONOR_INPUTS),
        msg=dup_rep.summary(),
    )

  def test_joint_autograd_linear_vs_nonlinear_and_backward_only_donation(self):
    donating_binary = jax_op(
        "pallas_verify_fx::joint_lin_vs_nonlin_donor",
        _pallas_binary_add,
        donate_argnums=(1,),
    )
    donating_binary.register_autograd(lambda ctx, g: (g, g))

    class _LinearNotSavedModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.ones((8, 128), dtype=torch.float32)
        )

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * self.scale
        # `h + h` has constant derivative and does NOT save `h` for backward.
        lin = h + h
        return donating_binary(lin, h)

    x = torch.ones((8, 128), dtype=torch.float32, requires_grad=True)
    lin_rep = verifier.verify(_LinearNotSavedModule(), (x,), mode=STATIC)
    self.assertTrue(lin_rep.passed, msg=lin_rep.summary())
    self.assertEmpty(
        lin_rep.findings_by_category(Category.USE_AFTER_DONATE),
        msg=lin_rep.summary(),
    )

    # Forward op has NO `donate_argnums`, while custom backward donates a saved
    # `nn.Parameter` or passes aliased views to a 2-donor backward kernel.
    bwd_param_donor = jax_op(
        "pallas_verify_fx::bwd_only_param_donor",
        _pallas_binary_add,
        donate_argnums=(1,),
    )
    bwd_aliased_donor = jax_op(
        "pallas_verify_fx::bwd_only_aliased_donor",
        _pallas_binary_add,
        donate_argnums=(0, 1),
    )

    fwd_clean_a = jax_op(
        "pallas_verify_fx::fwd_clean_bwd_param_donate", _pallas_binary_add
    )

    def _save_second(ctx, inputs, output):
      del output
      ctx.save_for_backward(inputs[1])

    def _donate_saved_param_bwd(ctx, g):
      (w_saved,) = ctx.saved_tensors
      return bwd_param_donor(g, w_saved), g

    fwd_clean_a.register_autograd(
        _donate_saved_param_bwd, setup_context=_save_second
    )

    class _BwdDonatesParamModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones((8, 128), dtype=torch.float32))

      def forward(self, t: torch.Tensor) -> torch.Tensor:
        return fwd_clean_a(t, self.w)

    bwd_param_rep = verifier.verify(_BwdDonatesParamModule(), (x,), mode=STATIC)
    self.assertNotEmpty(
        bwd_param_rep.findings_by_category(Category.DONATED_PARAMETER),
        msg=bwd_param_rep.summary(),
    )

    fwd_clean_b = jax_op(
        "pallas_verify_fx::fwd_clean_bwd_aliased_donate", _pallas_add_one
    )
    fwd_clean_b.register_autograd(
        lambda ctx, g: bwd_aliased_donor(g, g.view_as(g))
    )
    bwd_alias_rep = verifier.verify(lambda t: fwd_clean_b(t), (x,), mode=STATIC)
    self.assertNotEmpty(
        bwd_alias_rep.findings_by_category(Category.ALIASED_DONOR_INPUTS),
        msg=bwd_alias_rep.summary(),
    )

  def test_preserve_module_state_inference_tensors_and_expanded_buffers(self):
    safe_op = jax_op("pallas_verify_fx::inf_mode_mod_op", _pallas_binary_add)

    class _InferenceConstructedModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(
            torch.ones((8, 128), dtype=torch.float32), requires_grad=False
        )
        base_row = torch.ones((1, 128), dtype=torch.float32)
        self.register_buffer("expanded_buf", base_row.expand(8, 128))

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        return safe_op(x + self.expanded_buf, self.w)

    with torch.inference_mode():
      inf_mod = _InferenceConstructedModule()
    self.assertTrue(inf_mod.w.is_inference())

    x = torch.ones((8, 128), dtype=torch.float32)
    inf_rep = verifier.verify(inf_mod, (x,), mode=STATIC)
    self.assertTrue(inf_rep.passed, msg=inf_rep.summary())

    class _VersionTrackedModule(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones((8, 128), dtype=torch.float32))
        self.register_buffer(
            "scalar_buf", torch.tensor(2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "cl_buf",
            torch.ones((1, 2, 4, 4), dtype=torch.float32).to(
                memory_format=torch.channels_last
            ),
        )

      def forward(self, t: torch.Tensor) -> torch.Tensor:
        self.scalar_buf.add_(1.0)
        self.cl_buf.add_(1.0)
        return safe_op(t * self.scalar_buf + self.cl_buf.sum(), self.w)

    ver_mod = _VersionTrackedModule()
    ver_before = ver_mod.w._version
    ver_rep = verifier.verify(ver_mod, (x,), mode=STATIC)
    self.assertTrue(ver_rep.passed, msg=ver_rep.summary())
    self.assertEqual(ver_mod.w._version, ver_before)
    self.assertEqual(float(ver_mod.scalar_buf.item()), 2.0)
    self.assertTrue(
        torch.equal(
            ver_mod.cl_buf, torch.ones((1, 2, 4, 4), dtype=torch.float32)
        )
    )

  def test_concrete_int_invalidation_and_kwargs_signature_dedup(self):
    def _neg_donor_int_fn(arr: jax.Array, offsets: jax.Array) -> jax.Array:
      del offsets
      return arr

    neg_donor_int_op = jax_op(
        "pallas_verify_fx::neg_donor_int_op",
        _neg_donor_int_fn,
        donate_argnums=(-1,),
    )

    def _set_fill_and_iadd_prog(
        t: torch.Tensor,
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        idx_c: torch.Tensor,
    ) -> torch.Tensor:
      alias = torch.empty_like(idx_a)
      alias.set_(idx_a)
      alias.fill_(99)
      idx_b = operator.iadd(idx_b, 99)
      _ = neg_donor_int_op(t, offsets=idx_c)
      return t + idx_a[0] + idx_b[0] + idx_c[0]

    x3 = torch.ones((2, 8, 128), dtype=torch.float32)
    prog_args = (
        x3,
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    gm = fx_verifier.trace_to_fx_graph(_set_fill_and_iadd_prog, prog_args)
    phs = [n for n in gm.graph.nodes if n.op == "placeholder"]
    ph_env = fx_verifier.match_placeholders_to_inputs(phs, prog_args, None)
    _, env = verifier_impl._collect_graph_module_envs(gm, ph_env)[0]
    self.assertNotIn(phs[1], env)
    self.assertNotIn(phs[2], env)
    self.assertNotIn(phs[3], env)

    sym_t = torch.zeros((4,), dtype=torch.int32)
    conc_t = torch.zeros((4,), dtype=torch.int32)
    self.assertNotEqual(
        verifier_impl._signature({"offsets": sym_t}, concrete_tensor_ids=set()),
        verifier_impl._signature(
            {"offsets": conc_t}, concrete_tensor_ids={id(conc_t)}
        ),
    )

  def test_preconstructed_pallas_call_autograd_and_interior_tile_and_spmd(self):
    preconstructed_non_interp = pl.pallas_call(
        lambda g_ref, o_ref, _sem: o_ref.__setitem__(
            Ellipsis, g_ref[...] * 2.0
        ),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            scratch_shapes=(pltpu.SemaphoreType.DMA,),
        ),
    )
    params = pltpu.InterpretParams(dma_execution_mode="on_wait")
    worker_interprets: list[object] = []
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    with (
        pltpu.force_tpu_interpret_mode(params),
        fx_verifier.force_primitive_interpret_params(params),
    ):
      main_jaxpr = jax.make_jaxpr(preconstructed_non_interp)(x_spec)
      self.assertEqual(main_jaxpr.eqns[0].params["interpret"], params)

      def _check_worker() -> None:
        w_jaxpr = jax.make_jaxpr(preconstructed_non_interp)(x_spec)
        worker_interprets.append(w_jaxpr.eqns[0].params["interpret"])

      worker = threading.Thread(target=_check_worker)
      worker.start()
      worker.join()
    self.assertEqual(worker_interprets, [False])
    after_jaxpr = jax.make_jaxpr(preconstructed_non_interp)(x_spec)
    self.assertFalse(after_jaxpr.eqns[0].params["interpret"])

    preconstructed_bwd = pl.pallas_call(
        lambda g_ref, o_ref: o_ref.__setitem__(Ellipsis, g_ref[...] * 2.0),
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )

    def _bwd_wrapper(g: jax.Array) -> jax.Array:
      return preconstructed_bwd(g)

    def _fwd_scale2_fn(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] * 2.0),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      )(x)

    bwd_jax_op = jax_op("pallas_verify_fx::preconstructed_bwd", _bwd_wrapper)
    fwd_scale2 = jax_op(
        "pallas_verify_fx::fwd_scale2_preconstructed", _fwd_scale2_fn
    )
    fwd_scale2.register_autograd(lambda ctx, g: bwd_jax_op(g))
    x_small = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    pre_rep = verifier.verify(fwd_scale2, (x_small,))
    self._assert_autograd_ran(pre_rep)

    # Interior-tile zero-gradient bug on (128, 128) bfloat16: rows 0:16 and
    # 112:128 are exact, while interior rows 16:112 are zeroed out.
    bf16_op = jax_op("pallas_verify_fx::interior_tile_bug_bf16", _fwd_scale2_fn)

    def _buggy_interior_bwd(ctx, grad_out: torch.Tensor) -> torch.Tensor:
      del ctx
      g = grad_out * 2.0
      g = g.clone()
      g[16:112, :] = 0.0
      return g

    bf16_op.register_autograd(_buggy_interior_bwd)
    x_bf16 = torch.randn((128, 128), dtype=torch.bfloat16, requires_grad=True)
    bf16_rep = verifier.verify(bf16_op, (x_bf16,))
    self.assertNotEmpty(
        bf16_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=bf16_rep.summary(),
    )

    # Multi-device SPMD `all_gather` forward + `psum_scatter` (`reduce_scatter`)
    # custom backward on a 2-device mesh.
    tp_mesh = jax.sharding.AbstractMesh((2,), ("tp",))

    def _ag_fwd_fn(x: jax.Array) -> jax.Array:
      return jax.shard_map(
          lambda s: jax.lax.all_gather(s, "tp", axis=0, tiled=True),
          mesh=tp_mesh,
          in_specs=(P("tp", None),),
          out_specs=P(None, None),
          check_vma=False,
      )(x)

    def _rs_bwd_fn(g: jax.Array) -> jax.Array:
      return jax.shard_map(
          lambda s: jax.lax.psum_scatter(
              s, "tp", scatter_dimension=0, tiled=True
          ),
          mesh=tp_mesh,
          in_specs=(P(None, None),),
          out_specs=P("tp", None),
          check_vma=False,
      )(g)

    ag_fwd = jax_op(
        "pallas_verify_fx::spmd_all_gather_fwd",
        _ag_fwd_fn,
        mesh=tp_mesh,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P(None, None),
    )
    rs_bwd = jax_op(
        "pallas_verify_fx::spmd_reduce_scatter_bwd",
        _rs_bwd_fn,
        mesh=tp_mesh,
        input_partition_specs=(P(None, None),),
        output_partition_specs=P("tp", None),
    )
    ag_fwd.register_autograd(lambda ctx, g: rs_bwd(g))
    x_shard = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    spmd_rep = verifier.verify(ag_fwd, (x_shard,))
    self._assert_autograd_ran(spmd_rep)

  def test_autograd_swapped_int_before_float_and_ambient_no_grad_modes(self):
    def _pallas_indices_then_values(
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
      def kernel(x_ref, idx_ref, val_ref):
        idx_ref[...] = jnp.zeros(idx_ref.shape, jnp.int32)
        val_ref[...] = x_ref[...] * 2.0

      return pl.pallas_call(
          kernel,
          out_shape=(
              jax.ShapeDtypeStruct(x.shape, jnp.int32),
              jax.ShapeDtypeStruct(x.shape, x.dtype),
          ),
      )(x)

    swapped_ok = jax_op(
        "pallas_verify_fx::indices_then_values_ok", _pallas_indices_then_values
    )
    swapped_ok.register_autograd(lambda ctx, _g_idx, g_val: g_val * 2.0)

    swapped_bad = jax_op(
        "pallas_verify_fx::indices_then_values_bad",
        _pallas_indices_then_values,
    )
    swapped_bad.register_autograd(lambda ctx, _g_idx, g_val: g_val * 5.0)

    x = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)
    ok_rep = verifier.verify(swapped_ok, (x,))
    self._assert_autograd_ran(ok_rep)

    bad_rep = verifier.verify(swapped_bad, (x,))
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=bad_rep.summary(),
    )

    # Verify directional `jax.jvp` directly completes (`True`) rather than
    # misaligning `float0` `TypedNdArray` tangents or falling back to finite
    # differences across JAX versions (on JAX <= 0.11, `_pallas_call_jvp_rule`
    # does not yet partition `float0` `pallas_call` output tangents, so pair a
    # JAX `int32` output with a float `pallas_call` output).
    def _jax_indices_then_pallas_values(
        x: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
      val = pl.pallas_call(
          lambda x_ref, val_ref: val_ref.__setitem__(
              Ellipsis, x_ref[...] * 2.0
          ),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      )(x)
      return jnp.zeros(x.shape, jnp.int32), val

    jvp_swapped_op = jax_op(
        "pallas_verify_fx::jax_indices_then_pallas_values",
        _jax_indices_then_pallas_values,
    )
    jc = fx_verifier.extract_jax_callable(jvp_swapped_op)
    self.assertIsNotNone(jc)
    slot = fx_verifier._DiffInputSlot(
        is_kwarg=False, key=0, param_idx=0, tensor=x.detach().clone()
    )
    jvp_detected: list[verifier.VerificationFinding] = []
    jvp_ran = fx_verifier._try_directional_jax_jvp_slot(
        jc,
        slot,
        torch.ones_like(x) * 2.0,
        [x.detach().clone()],
        {},
        [torch.ones_like(x)],
        [torch.zeros_like(x), torch.ones_like(x)],
        [False, True],
        "pallas_verify_fx::jax_indices_then_pallas_values",
        jvp_detected,
        is_low_prec=False,
        rtol=1e-3,
        atol=1e-3,
    )
    self.assertTrue(jvp_ran)
    self.assertEmpty(jvp_detected)

    jvp_bad_detected: list[verifier.VerificationFinding] = []
    jvp_bad_ran = fx_verifier._try_directional_jax_jvp_slot(
        jc,
        slot,
        torch.ones_like(x) * 5.0,
        [x.detach().clone()],
        {},
        [torch.ones_like(x)],
        [torch.zeros_like(x), torch.ones_like(x)],
        [False, True],
        "pallas_verify_fx::jax_indices_then_pallas_values",
        jvp_bad_detected,
        is_low_prec=False,
        rtol=1e-3,
        atol=1e-3,
    )
    self.assertTrue(jvp_bad_ran)
    self.assertNotEmpty(jvp_bad_detected)

    # Calling `verify()` inside `torch.no_grad()` or `torch.inference_mode()`
    # must still execute `Phase.AUTOGRAD` and catch buggy backwards.
    with torch.no_grad():
      ng_ok = verifier.verify(swapped_ok, (x.detach(),))
      self._assert_autograd_ran(ng_ok)
      ng_bad = verifier.verify(swapped_bad, (x.detach(),))
      self.assertNotEmpty(
          ng_bad.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
          msg=ng_bad.summary(),
      )

    with torch.inference_mode():
      inf_x = torch.randn((8, 128), dtype=torch.float32)
      im_ok = verifier.verify(swapped_ok, (inf_x,))
      self._assert_autograd_ran(im_ok)
      im_bad = verifier.verify(swapped_bad, (inf_x,))
      self.assertNotEmpty(
          im_bad.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
          msg=im_bad.summary(),
      )

  def test_spmd_negative_collectives_ppermute_and_replicated_input_autograd(
      self,
  ):
    tp2 = jax.sharding.AbstractMesh((2,), ("tp",))
    x_shard = torch.randn((8, 128), dtype=torch.float32, requires_grad=True)

    # 1. `all_gather` forward with unreduced local-slice backward (missing
    # cross-rank `psum_scatter`) must fail `AUTOGRAD_PARITY_MISMATCH`.
    def _ag_unreduced_fn(x: jax.Array) -> jax.Array:
      return jax.shard_map(
          lambda s: jax.lax.all_gather(s, "tp", axis=0, tiled=True),
          mesh=tp2,
          in_specs=(P("tp", None),),
          out_specs=P(None, None),
          check_vma=False,
      )(x)

    ag_unreduced = jax_op(
        "pallas_verify_fx::spmd_ag_unreduced_bwd",
        _ag_unreduced_fn,
        mesh=tp2,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P(None, None),
    )
    ag_unreduced.register_autograd(lambda ctx, g: g[:8])
    unred_rep = verifier.verify(ag_unreduced, (x_shard,))
    self.assertNotEmpty(
        unred_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=unred_rep.summary(),
    )

    # 2. 4-device asymmetric ring `ppermute`: inverse permutation passes, while
    # forward-direction permutation or missing `ppermute` fails.
    ring4 = jax.sharding.AbstractMesh((4,), ("tp",))
    perm_fwd = [(i, (i + 1) % 4) for i in range(4)]
    perm_inv = [((i + 1) % 4, i) for i in range(4)]

    def _make_ppermute_fn(perm):
      def _ppermute_fn(x: jax.Array) -> jax.Array:
        return jax.shard_map(
            lambda s: jax.lax.ppermute(s, "tp", perm),
            mesh=ring4,
            in_specs=(P("tp", None),),
            out_specs=P("tp", None),
            check_vma=False,
        )(x)

      return _ppermute_fn

    pperm_bwd_good = jax_op(
        "pallas_verify_fx::pperm4_bwd_good",
        _make_ppermute_fn(perm_inv),
        mesh=ring4,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P("tp", None),
    )
    pperm_bwd_wrong_dir = jax_op(
        "pallas_verify_fx::pperm4_bwd_wrong_dir",
        _make_ppermute_fn(perm_fwd),
        mesh=ring4,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P("tp", None),
    )

    pperm_fwd_ok = jax_op(
        "pallas_verify_fx::pperm4_fwd_ok",
        _make_ppermute_fn(perm_fwd),
        mesh=ring4,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P("tp", None),
    )
    pperm_fwd_ok.register_autograd(lambda ctx, g: pperm_bwd_good(g))
    self._assert_autograd_ran(verifier.verify(pperm_fwd_ok, (x_shard,)))

    pperm_fwd_wrong = jax_op(
        "pallas_verify_fx::pperm4_fwd_wrong_dir",
        _make_ppermute_fn(perm_fwd),
        mesh=ring4,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P("tp", None),
    )
    pperm_fwd_wrong.register_autograd(lambda ctx, g: pperm_bwd_wrong_dir(g))
    wrong_dir_rep = verifier.verify(pperm_fwd_wrong, (x_shard,))
    self.assertNotEmpty(
        wrong_dir_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=wrong_dir_rep.summary(),
    )

    pperm_fwd_ident = jax_op(
        "pallas_verify_fx::pperm4_fwd_ident_bwd",
        _make_ppermute_fn(perm_fwd),
        mesh=ring4,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P("tp", None),
    )
    pperm_fwd_ident.register_autograd(lambda ctx, g: g)
    ident_rep = verifier.verify(pperm_fwd_ident, (x_shard,))
    self.assertNotEmpty(
        ident_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=ident_rep.summary(),
    )

    # 3. Replicated input `P(None, None)` -> partitioned output `P("tp", None)`:
    # `psum` backward passes, while missing `psum` fails.
    def _col_fwd(x: jax.Array) -> jax.Array:
      return jax.shard_map(
          lambda s: s
          * jnp.asarray(jax.lax.axis_index("tp") + 1, dtype=s.dtype),
          mesh=tp2,
          in_specs=(P(None, None),),
          out_specs=P("tp", None),
          check_vma=False,
      )(x)

    def _col_bwd_ok_fn(g: jax.Array) -> jax.Array:
      return jax.shard_map(
          lambda s: jax.lax.psum(
              s * jnp.asarray(jax.lax.axis_index("tp") + 1, dtype=s.dtype),
              "tp",
          ),
          mesh=tp2,
          in_specs=(P("tp", None),),
          out_specs=P(None, None),
          check_vma=False,
      )(g)

    col_bwd_ok = jax_op(
        "pallas_verify_fx::col_rep_in_bwd_ok",
        _col_bwd_ok_fn,
        mesh=tp2,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P(None, None),
    )

    def _col_bwd_missing_psum_fn(g: jax.Array) -> jax.Array:
      return jax.shard_map(
          lambda s: s
          * jnp.asarray(jax.lax.axis_index("tp") + 1, dtype=s.dtype),
          mesh=tp2,
          in_specs=(P("tp", None),),
          out_specs=P(None, None),
          check_vma=False,
      )(g)

    col_bwd_missing_psum = jax_op(
        "pallas_verify_fx::col_rep_in_bwd_missing_psum",
        _col_bwd_missing_psum_fn,
        mesh=tp2,
        input_partition_specs=(P("tp", None),),
        output_partition_specs=P(None, None),
    )

    col_op_ok = jax_op(
        "pallas_verify_fx::col_rep_in_fwd_ok",
        _col_fwd,
        mesh=tp2,
        input_partition_specs=(P(None, None),),
        output_partition_specs=P("tp", None),
    )
    col_op_ok.register_autograd(lambda ctx, g: col_bwd_ok(g))
    self._assert_autograd_ran(verifier.verify(col_op_ok, (x_shard,)))

    col_op_bad = jax_op(
        "pallas_verify_fx::col_rep_in_fwd_bad",
        _col_fwd,
        mesh=tp2,
        input_partition_specs=(P(None, None),),
        output_partition_specs=P("tp", None),
    )
    col_op_bad.register_autograd(lambda ctx, g: col_bwd_missing_psum(g))
    col_bad_rep = verifier.verify(col_op_bad, (x_shard,))
    self.assertNotEmpty(
        col_bad_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=col_bad_rep.summary(),
    )

  def test_float4_e2m1fn_x2_pack_unpack_and_export_placeholder(self):
    raw_bytes = torch.tensor(
        [[0x31, 0x25, 0x07, 0x64] * 16] * 8, dtype=torch.uint8
    )
    fp4_tensor = raw_bytes.view(torch.float4_e2m1fn_x2)
    unpacked_np, jax_dt = fx_verifier.tensor_to_numpy(fp4_tensor)
    self.assertEqual(jax_dt, jnp.float4_e2m1fn.dtype)
    self.assertEqual(unpacked_np.shape, (8, 128))
    # Low nibble 0x01 at even index 0, high nibble 0x03 at odd index 1.
    self.assertEqual(int(unpacked_np.view(np.uint8)[0, 0]), 0x01)
    self.assertEqual(int(unpacked_np.view(np.uint8)[0, 1]), 0x03)

    cpu0 = jax.devices("cpu")[0]
    jax_fp4 = jax.device_put(jnp.asarray(unpacked_np, dtype=jax_dt), cpu0)
    repacked = fx_verifier._jax_to_torch_tensor(jax_fp4)
    self.assertEqual(repacked.dtype, torch.float4_e2m1fn_x2)
    self.assertEqual(repacked.shape, (8, 64))
    self.assertTrue(torch.equal(repacked.view(torch.uint8), raw_bytes))

    scalar_fp4 = torch.tensor(0x21, dtype=torch.uint8).view(
        torch.float4_e2m1fn_x2
    )
    try:
      fx_verifier.tensor_to_numpy(scalar_fp4)
      raised_0d = False
    except ValueError:
      raised_0d = True
    self.assertTrue(raised_0d)


if __name__ == "__main__":
  absltest.main()
