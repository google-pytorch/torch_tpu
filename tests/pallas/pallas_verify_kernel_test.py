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

"""Tests for Pallas kernel Jaxpr extraction and static verification."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import types
from unittest import mock

from absl.testing import absltest
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np
import torch
from torch.tpu import pallas
from torch.tpu.pallas import verifier
from torch_tpu._internal.pallas import _mock_device
from torch_tpu._internal.pallas.verify import jaxpr_model
from tests import seed_test_utils

jax_op = pallas.jax_op
P = jax.sharding.PartitionSpec
Category = verifier.FindingCategory
STATIC = verifier.VerificationMode.STATIC
HBM_SPEC = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)


def _make_mock_tpu_mesh(num_devices: int, axis_name: str = "rank"):
  devs = np.array(
      [_mock_device._PallasDevice(id=i) for i in range(num_devices)]
  )
  return jax.sharding.Mesh(devs, (axis_name,))


def _make_tensorcore_mesh(axis_name: str = "core", num_cores: int = 2):
  mesh_cls = getattr(pltpu, "TensorCoreMesh", None)
  if mesh_cls is not None:
    return mesh_cls(axis_name=axis_name, num_cores=num_cores)
  return pltpu.create_tensorcore_mesh(axis_name=axis_name, num_cores=num_cores)


def _shard_mapped(per_shard, mesh, axis_spec=P("rank")):
  return jax.shard_map(
      per_shard,
      mesh=mesh,
      in_specs=(axis_spec,),
      out_specs=axis_spec,
      check_vma=False,
  )


def _distributed_op(name, fn, mesh):
  return jax_op(
      name,
      functools.partial(fn, mesh=mesh),
      mesh=mesh,
      input_partition_specs=(P("rank"),),
      output_partition_specs=(P("rank"),),
  )


def _single_block_call(kernel, x, scratch_shapes=(), collective_id=None):
  """Runs `kernel(x_ref, y_ref, *scratch)` on one VMEM block, no grid."""
  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[pl.BlockSpec(x.shape)],
          out_specs=pl.BlockSpec(x.shape),
          scratch_shapes=scratch_shapes,
      ),
      compiler_params=pltpu.CompilerParams(collective_id=collective_id),
  )(x)


def _make_tiled_copy(grid_size: int, rows_written: int = 8):
  def tiled_copy(x: jax.Array) -> jax.Array:
    def kernel(x_ref, o_ref):
      o_ref[0:rows_written, :] = x_ref[0:rows_written, :]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
    )(x)

  return tiled_copy


def _make_fori_copy(num_iterations: int):
  def fori_copy(x: jax.Array) -> jax.Array:
    def kernel(x_ref, o_ref):
      def body(i, carry):
        rows = pl.ds(pl.multiple_of(i * 8, 8), 8)
        o_ref[rows, :] = x_ref[rows, :]
        return carry

      lax.fori_loop(0, num_iterations, body, 0)

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x)

  return fori_copy


def _make_while_copy(num_iterations: int):
  def loop_copy(x: jax.Array) -> jax.Array:
    def kernel(x_ref, o_ref):
      def body(i):
        rows = pl.ds(pl.multiple_of(i * 8, 8), 8)
        o_ref[rows, :] = x_ref[rows, :]
        return i + 1

      lax.while_loop(lambda i: i < num_iterations, body, 0)

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x)

  return loop_copy


class KernelVerifierTest(seed_test_utils.RepeatableTest):

  # ==========================================================================
  # Static Jaxpr Unrolling, Bounds, Coverage, and Hardware Constraints
  # ==========================================================================

  def test_static_mode_does_not_query_jax_devices(self):
    """Static checks only trace, so JAX never initializes XLA backends."""
    mesh = _make_mock_tpu_mesh(4, "rank")

    def sharded_copy(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      return _shard_mapped(_make_tiled_copy(grid_size=1), mesh)(global_x)

    op = _distributed_op(
        "pallas_verify_kernel::sharded_copy", sharded_copy, mesh
    )
    xla_bridge = inspect.getmodule(jax.devices)
    assert xla_bridge is not None
    get_backend_fn = getattr(xla_bridge, "get_backend", None)
    if hasattr(get_backend_fn, "cache_clear"):
      get_backend_fn.cache_clear()
    # The verifier reports tracing errors as findings, so record calls and fail
    # if backend discovery is triggered.
    with (
        mock.patch.object(jax, "devices", wraps=jax.devices) as devices,
        mock.patch.object(
            jax, "local_devices", wraps=jax.local_devices
        ) as local_devices,
        mock.patch.object(
            xla_bridge,
            "backends",
            side_effect=AssertionError(
                "xla_bridge.backends must not run in check_kernel_static_models"
            ),
        ) as backends_spy,
        mock.patch.object(
            xla_bridge,
            "get_backend",
            side_effect=AssertionError(
                "xla_bridge.get_backend must not run in"
                " check_kernel_static_models"
            ),
        ) as get_backend_spy,
    ):
      model_rep, models = jaxpr_model.check_kernel_static_models(
          op, (torch.ones((8, 128)),)
      )
    devices.assert_not_called()
    local_devices.assert_not_called()
    backends_spy.assert_not_called()
    get_backend_spy.assert_not_called()
    self.assertTrue(model_rep.passed, msg=model_rep.summary())
    self.assertIsNotNone(models)

    with (
        mock.patch.object(jax, "devices", wraps=jax.devices) as devices2,
        mock.patch.object(
            jax, "local_devices", wraps=jax.local_devices
        ) as local_devices2,
    ):
      report = verifier.verify(op, (torch.ones((8, 128)),), mode=STATIC)
    devices2.assert_not_called()
    local_devices2.assert_not_called()
    self.assertTrue(report.passed, msg=report.summary())
    self.assertEmpty(
        report.findings_by_severity(verifier.Severity.WARNING),
        msg=report.summary(),
    )
    self.assertIn(verifier.Phase.MODEL_CHECK, report.checked_phases)

  def test_full_grid_tiled_copy_passes(self):
    x = torch.ones((32, 128), dtype=torch.float32)
    full_op = jax_op(
        "pallas_verify_kernel::full_grid_copy", _make_tiled_copy(4)
    )
    full_rep = verifier.verify(full_op, (x,), mode=STATIC)
    self.assertTrue(full_rep.passed, msg=full_rep.summary())
    self.assertEmpty(
        full_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=full_rep.summary(),
    )

  def test_short_grid_tiled_copy_reports_unwritten_output(self):
    x = torch.ones((32, 128), dtype=torch.float32)
    # A grid that is one step short never writes the last output tile.
    short_op = jax_op(
        "pallas_verify_kernel::short_grid_copy", _make_tiled_copy(3)
    )
    short_rep = verifier.verify(short_op, (x,), mode=STATIC)
    unwritten = short_rep.findings_by_category(Category.UNWRITTEN_OUTPUT)
    self.assertNotEmpty(unwritten, msg=short_rep.summary())
    self.assertEqual(unwritten[0].severity, verifier.Severity.ERROR)
    self.assertIn("24:32", unwritten[0].message)

  def test_partial_tile_write_reports_unwritten_output(self):
    x = torch.ones((32, 128), dtype=torch.float32)
    # Writing half of each output tile leaves the other half uninitialized.
    half_op = jax_op(
        "pallas_verify_kernel::half_tile_copy",
        _make_tiled_copy(4, rows_written=4),
    )
    half_rep = verifier.verify(half_op, (x,), mode=STATIC)
    unwritten = half_rep.findings_by_category(Category.UNWRITTEN_OUTPUT)
    self.assertNotEmpty(unwritten, msg=half_rep.summary())
    self.assertIn("4:32", unwritten[0].message)

  def test_aliased_vmem_pipelined_output_partial_write_detected(self):
    def aliased_partial_vmem_write(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        o_ref[0:4, :] = x_ref[0:4, :] + 1.0

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
          grid=(2,),
          input_output_aliases={0: 0},
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        aliased_partial_vmem_write, (x_spec,)
    )
    unwritten = rep.findings_by_category(Category.UNWRITTEN_OUTPUT)
    self.assertNotEmpty(unwritten, msg=rep.summary())
    self.assertIn("aliased to input", unwritten[0].message)
    self.assertIn("4:8", unwritten[0].message)

  def test_uninitialized_output_tile_read_in_reduction_grid_detected(self):
    def uninitialized_reduction(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        o_ref[...] = o_ref[...] + x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid=(2,),
          in_specs=[pl.BlockSpec((8, 128), lambda k: (k, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda k: (0, 0)),
      )(x)

    def initialized_reduction(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        @pl.when(pl.program_id(0) == 0)
        def _zero():
          o_ref[...] = jnp.zeros_like(o_ref)

        o_ref[...] = o_ref[...] + x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid=(2,),
          in_specs=[pl.BlockSpec((8, 128), lambda k: (k, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda k: (0, 0)),
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    bad_rep, _ = jaxpr_model.check_kernel_static_models(
        uninitialized_reduction, (x_spec,)
    )
    unwritten = bad_rep.findings_by_category(Category.UNWRITTEN_OUTPUT)
    self.assertNotEmpty(unwritten, msg=bad_rep.summary())
    self.assertIn("read before being written", unwritten[0].message)

    ok_rep, _ = jaxpr_model.check_kernel_static_models(
        initialized_reduction, (x_spec,)
    )
    self.assertEmpty(
        ok_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=ok_rep.summary(),
    )

  def test_fori_loop_full_coverage_passes(self):
    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    ok_rep = verifier.verify(_make_fori_copy(4), (x_spec,), mode=STATIC)
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    self.assertEmpty(
        ok_rep.findings_by_severity(verifier.Severity.WARNING),
        msg=ok_rep.summary(),
    )

  def test_fori_loop_short_coverage_reports_unwritten_output(self):
    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    short_rep = verifier.verify(_make_fori_copy(3), (x_spec,), mode=STATIC)
    self.assertNotEmpty(
        short_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=short_rep.summary(),
    )

  def test_while_loop_full_coverage_passes(self):
    x = torch.ones((32, 128), dtype=torch.float32)
    ok_op = jax_op("pallas_verify_kernel::while_copy_ok", _make_while_copy(4))
    ok_rep = verifier.verify(ok_op, (x,), mode=STATIC)
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())

  def test_while_loop_short_coverage_reports_unwritten_output(self):
    x = torch.ones((32, 128), dtype=torch.float32)
    short_op = jax_op(
        "pallas_verify_kernel::while_copy_short", _make_while_copy(3)
    )
    short_rep = verifier.verify(short_op, (x,), mode=STATIC)
    self.assertNotEmpty(
        short_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=short_rep.summary(),
    )

  def test_truncated_fori_loop_does_not_fabricate_post_loop_carry_value(self):
    def post_loop_oob_after_truncated_fori(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        final_acc = lax.fori_loop(0, 100, lambda i, acc: acc + 1, 0)
        # After 100 steps final_acc is 100; if truncation bound final_acc to the
        # 4th iteration's value (4), 4 - 4 == 0 would hide the OOB access, or
        # final_acc == 4 would fabricate a false index.
        idx = jnp.where(final_acc == 4, 99, 0)
        o_ref[...] = x_ref[pl.ds(idx * 8, 8), :]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          in_specs=[pl.BlockSpec((16, 128), lambda: (0, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda: (0, 0)),
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        post_loop_oob_after_truncated_fori,
        (x_spec,),
        max_unrolled_iterations=4,
    )
    self.assertEmpty(
        rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=rep.summary(),
    )
    self.assertNotEmpty(
        rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=rep.summary(),
    )

  def test_runtime_bounded_while_loop_does_not_fabricate_post_loop_carry(self):
    def runtime_while_carry(bound: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(bound_ref, x_ref, o_ref):
        n = bound_ref[0]
        final_i = lax.while_loop(lambda i: i < n, lambda i: i + 1, 0)
        bad_if_one = jnp.where(final_i == 1, 99, 0)
        o_ref[...] = x_ref[pl.ds(bad_if_one * 8, 8), :]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec((16, 128), lambda _b: (0, 0))],
              out_specs=pl.BlockSpec((8, 128), lambda _b: (0, 0)),
          ),
      )(bound, x)

    b_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        runtime_while_carry, (b_spec, x_spec)
    )
    self.assertEmpty(
        rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=rep.summary(),
    )
    self.assertNotEmpty(
        rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=rep.summary(),
    )

  def test_index_map_with_integer_arithmetic_passes_when_in_bounds(self):
    def copy_in_bounds(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(4,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i // 2 + 2, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
      )(x)

    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    ok_rep = verifier.verify(copy_in_bounds, (x_spec,), mode=STATIC)
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    self.assertNotIn("runtime data", ok_rep.summary())

  def test_index_map_with_integer_arithmetic_reports_oob(self):
    def copy_oob(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(4,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i // 2 + 3, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
      )(x)

    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    oob_rep = verifier.verify(copy_oob, (x_spec,), mode=STATIC)
    self.assertNotEmpty(
        oob_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=oob_rep.summary(),
    )

  def test_out_of_bounds_block_index_detected(self):
    def shifted_block_copy(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(4,),
          # Off by one: the last grid step reads block 4 of a 4-block array.
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i + 1, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
      )(x)

    block_op = jax_op(
        "pallas_verify_kernel::oob_block_index", shifted_block_copy
    )
    block_rep = verifier.verify(block_op, (torch.ones((32, 128)),), mode=STATIC)
    oob = block_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS)
    self.assertNotEmpty(oob, msg=block_rep.summary())
    self.assertIn("BlockSpec index map", oob[0].message)

  def test_out_of_bounds_dma_slice_detected(self):
    def shifted_dma_copy(x: jax.Array) -> jax.Array:
      def kernel(x_hbm_ref, o_ref, buf, sem):
        start = pl.program_id(0) * 8 + 8  # Off by one tile.
        copy = pltpu.make_async_copy(x_hbm_ref.at[pl.ds(start, 8)], buf, sem)
        copy.start()
        copy.wait()
        o_ref[...] = buf[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(2,),
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
              scratch_shapes=(
                  pltpu.VMEM((8, 128), x.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(x)

    dma_op = jax_op("pallas_verify_kernel::oob_dma_slice", shifted_dma_copy)
    dma_rep = verifier.verify(dma_op, (torch.ones((16, 128)),), mode=STATIC)
    oob = dma_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS)
    self.assertNotEmpty(oob, msg=dma_rep.summary())
    self.assertIn("16:24", oob[0].message)

  def test_negative_literal_ref_index_detected_as_oob(self):
    def negative_literal_index(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        o_ref[0, :] = x_ref[-1, :]
        o_ref[1:8, :] = x_ref[1:8, :]

      return _single_block_call(kernel, x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        negative_literal_index, (x_spec,)
    )
    oob = rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS)
    self.assertNotEmpty(oob, msg=rep.summary())
    self.assertIn("-1", oob[0].message)

  def test_chained_ref_at_subview_out_of_bounds_detected(self):
    def chained_subview_oob(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        start = pl.program_id(0) + 4
        sub = x_ref.at[pl.ds(4, 8), :]
        o_ref[...] = sub.at[pl.ds(start, 8), :][...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid=(1,),
          in_specs=[pl.BlockSpec((16, 128), lambda i: (0, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        chained_subview_oob, (x_spec,)
    )
    oob = rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS)
    self.assertNotEmpty(oob, msg=rep.summary())
    self.assertIn("4:12", oob[0].message)

  def test_input_output_aliases_with_scalar_prefetch_maps_correct_buffer(self):
    def aliased_with_prefetch(
        idx: jax.Array, x: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
      def kernel(unused_idx_ref, x_ref, unused_y_ref, o0_ref, unused_o1_ref):
        o0_ref[...] = x_ref[...] + 1.0

      return pl.pallas_call(
          kernel,
          out_shape=(
              jax.ShapeDtypeStruct(x.shape, x.dtype),
              jax.ShapeDtypeStruct(y.shape, y.dtype),
          ),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec(x.shape), pl.BlockSpec(y.shape)],
              out_specs=[pl.BlockSpec(x.shape), HBM_SPEC],
          ),
          # Output 1 is aliased to overall invar 2 (block input 1 = y).
          # Output 0 is NOT aliased, and is written by the kernel.
          input_output_aliases={2: 1},
      )(idx, x, y)

    def aliased_unwritten_non_aliased_out(
        idx: jax.Array, x: jax.Array, y: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
      def kernel(unused_idx_ref, unused_x_ref, unused_y_ref, o0_ref, o1_ref):
        del o0_ref, o1_ref

      return pl.pallas_call(
          kernel,
          out_shape=(
              jax.ShapeDtypeStruct(x.shape, x.dtype),
              jax.ShapeDtypeStruct(y.shape, y.dtype),
          ),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec(x.shape), pl.BlockSpec(y.shape)],
              out_specs=[pl.BlockSpec(x.shape), HBM_SPEC],
          ),
          input_output_aliases={2: 1},
      )(idx, x, y)

    idx_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep, models = jaxpr_model.check_kernel_static_models(
        aliased_with_prefetch, (idx_spec, x_spec, x_spec)
    )
    self.assertTrue(rep.passed, msg=rep.summary())
    self.assertIsNotNone(models)
    assert models is not None
    self.assertLen(models, 1)

    bad_rep, _ = jaxpr_model.check_kernel_static_models(
        aliased_unwritten_non_aliased_out, (idx_spec, x_spec, x_spec)
    )
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=bad_rep.summary(),
    )

  def test_padded_element_block_spec_valid_and_oob(self):
    def make_padded_element_kernel(start_offset: int):
      def kernel_fn(x: jax.Array) -> jax.Array:
        return pl.pallas_call(
            lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
            out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
            grid=(1,),
            in_specs=[
                pl.BlockSpec(
                    (pl.Element(8, padding=(2, 2)), 128),
                    lambda i: (i + start_offset, 0),
                )
            ],
            out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
        )(x)

      return kernel_fn

    # Virtual padded size along axis 0 is 4 + 2 + 2 = 8: starting at padded
    # index 0 ([0, 8)) is in bounds, while starting at 1 ([1, 9)) exceeds 8.
    x_spec = jax.ShapeDtypeStruct((4, 128), jnp.float32)
    ok_rep, _ = jaxpr_model.check_kernel_static_models(
        make_padded_element_kernel(0), (x_spec,)
    )
    self.assertEmpty(
        ok_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=ok_rep.summary(),
    )
    oob_rep, _ = jaxpr_model.check_kernel_static_models(
        make_padded_element_kernel(1), (x_spec,)
    )
    self.assertNotEmpty(
        oob_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=oob_rep.summary(),
    )

    # `Element(2, padding=(1, 2))` on a 4-row array has virtual padded size 7,
    # so padded start indices 0..5 ([0, 2)..[5, 7)) are valid on both input and
    # output BlockSpecs and collectively cover all 4 unpadded output rows,
    # whereas padded start index 6 ([6, 8)) is out of bounds.
    def make_sliding_padded_io(grid_steps: int):
      def fn(x: jax.Array) -> jax.Array:
        return pl.pallas_call(
            lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
            out_shape=jax.ShapeDtypeStruct((4, 128), x.dtype),
            grid=(grid_steps,),
            in_specs=[
                pl.BlockSpec(
                    (pl.Element(2, padding=(1, 2)), 128), lambda i: (i, 0)
                )
            ],
            out_specs=pl.BlockSpec(
                (pl.Element(2, padding=(1, 2)), 128),
                lambda i: (i, 0),
            ),
        )(x)

      return fn

    slide_ok_rep, _ = jaxpr_model.check_kernel_static_models(
        make_sliding_padded_io(6), (x_spec,)
    )
    self.assertEmpty(
        slide_ok_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=slide_ok_rep.summary(),
    )
    self.assertEmpty(
        slide_ok_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=slide_ok_rep.summary(),
    )

    slide_oob_rep, _ = jaxpr_model.check_kernel_static_models(
        make_sliding_padded_io(7), (x_spec,)
    )
    self.assertNotEmpty(
        slide_oob_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=slide_oob_rep.summary(),
    )

    # End-to-end `verifier.verify(..., mode=STATIC)` with all-Element dims and
    # high padding (`(0, 4)`) supported by Mosaic TPU lowering.
    def e2e_padded_element_fn(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid=(1,),
          in_specs=[
              pl.BlockSpec(
                  (pl.Element(8, padding=(0, 4)), pl.Element(128)),
                  lambda i: (0, 0),
              )
          ],
          out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
      )(x)

    e2e_op = jax_op(
        "pallas_verify_kernel::e2e_padded_element", e2e_padded_element_fn
    )
    e2e_rep = verifier.verify(
        e2e_op, (torch.ones((4, 128), dtype=torch.float32),), mode=STATIC
    )
    self.assertTrue(e2e_rep.passed, msg=e2e_rep.summary())

  def test_bounded_slice_exceeding_max_block_size_detected(self):
    def bounded_slice_too_large(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[0:8, :]),
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid=(1,),
          in_specs=[
              pl.BlockSpec(
                  (pl.BoundedSlice(8), 128),
                  lambda i: (pl.ds(0, 16), 0),
              )
          ],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
      )(x)

    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        bounded_slice_too_large, (x_spec,)
    )
    oob = rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS)
    self.assertNotEmpty(oob, msg=rep.summary())
    self.assertIn(
        "BoundedSlice length 16 exceeds maximum block_size 8", oob[0].message
    )

  def test_named_grid_axis_index_evaluated_in_kernel_and_block_spec(self):
    def named_grid_copy(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        row = lax.axis_index("rows")
        o_ref[...] = x_ref[...] + row.astype(x_ref.dtype)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(("rows", 2),),
          in_specs=[
              pl.BlockSpec((8, 128), lambda: (lax.axis_index("rows"), 0))
          ],
          out_specs=pl.BlockSpec((8, 128), lambda: (lax.axis_index("rows"), 0)),
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(named_grid_copy, (x_spec,))
    self.assertTrue(rep.passed, msg=rep.summary())
    self.assertEmpty(
        rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=rep.summary(),
    )

  def test_remote_device_id_outside_mesh_and_wrapped_ring(self):
    mesh = _make_mock_tpu_mesh(4, "rank")

    def make_ring_jax(*, wrap: bool):
      def ring_jax(
          global_x: jax.Array, *, mesh: jax.sharding.Mesh
      ) -> jax.Array:
        def kernel(x_ref, y_ref, send_sem, recv_sem):
          next_rank = (
              lax.rem(lax.axis_index("rank") + 1, 4)
              if wrap
              else lax.axis_index("rank") + 1
          )
          copy = pltpu.make_async_remote_copy(
              src_ref=x_ref,
              dst_ref=y_ref,
              send_sem=send_sem,
              recv_sem=recv_sem,
              device_id={"rank": next_rank},
          )
          copy.start()
          copy.wait()

        def per_shard(x_local):
          return _single_block_call(
              kernel,
              x_local,
              scratch_shapes=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
              collective_id=5,
          )

        return _shard_mapped(per_shard, mesh)(global_x)

      return ring_jax

    op = _distributed_op(
        "pallas_verify_kernel::unwrapped_ring",
        make_ring_jax(wrap=False),
        mesh,
    )
    op.register_autograd(lambda ctx, g: (g,))
    x_grad = torch.ones((8, 128), dtype=torch.float32, requires_grad=True)
    report = verifier.verify(op, (x_grad,))
    self.assertFalse(report.passed, msg=report.summary())
    self.assertNotEmpty(
        report.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=report.summary(),
    )
    self.assertNotIn(verifier.Phase.AUTOGRAD, report.checked_phases)
    bwd_skip = [
        f
        for f in report.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.phase is verifier.Phase.AUTOGRAD
    ]
    self.assertLen(bwd_skip, 1, msg=report.summary())
    self.assertIs(bwd_skip[0].severity, verifier.Severity.INFO)

    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    ok_rep, _ = jaxpr_model.check_kernel_static_models(
        functools.partial(make_ring_jax(wrap=True), mesh=mesh), (x_spec,)
    )
    self.assertEmpty(
        ok_rep.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=ok_rep.summary(),
    )

  def test_unknown_mesh_axis_key_in_device_id_detected(self):
    mesh = _make_mock_tpu_mesh(2, "rank")

    def bad_axis_key_kernel(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_signal(sem, 1, device_id={"typo_axis": 0})
        y_ref[...] = x_ref[...]

      return _shard_mapped(
          lambda x_local: _single_block_call(
              kernel, x_local, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
          ),
          mesh,
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        bad_axis_key_kernel, (x_spec,)
    )
    invalid = rep.findings_by_category(Category.INVALID_DEVICE_ID)
    self.assertNotEmpty(invalid, msg=rep.summary())
    self.assertIn("typo_axis", invalid[0].message)

  def test_joint_tuple_axis_key_and_scalar_device_id_on_2d_mesh_validated(self):
    devs = np.array(
        [_mock_device._PallasDevice(id=i) for i in range(4)]
    ).reshape(2, 2)
    mesh_2d = jax.sharding.Mesh(devs, ("x", "y"))
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)

    def joint_axis_valid(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_signal(sem, 1, device_id={("x", "y"): 3})
        y_ref[...] = x_ref[...]

      return _shard_mapped(
          lambda x_local: _single_block_call(
              kernel, x_local, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
          ),
          mesh_2d,
          axis_spec=P("x", None),
      )(x)

    def joint_axis_oob(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_signal(sem, 1, device_id={("x", "y"): 4})
        y_ref[...] = x_ref[...]

      return _shard_mapped(
          lambda x_local: _single_block_call(
              kernel, x_local, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
          ),
          mesh_2d,
          axis_spec=P("x", None),
      )(x)

    def scalar_mesh_mode_on_2d_mesh(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_signal(sem, 1, device_id=1)
        y_ref[...] = x_ref[...]

      return _shard_mapped(
          lambda x_local: _single_block_call(
              kernel, x_local, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
          ),
          mesh_2d,
          axis_spec=P("x", None),
      )(x)

    def tuple_device_id_valid_on_2d_mesh(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_signal(sem, 1, device_id=(1, 0))
        y_ref[...] = x_ref[...]

      return _shard_mapped(
          lambda x_local: _single_block_call(
              kernel, x_local, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
          ),
          mesh_2d,
          axis_spec=P("x", None),
      )(x)

    def tuple_device_id_wrong_rank_on_2d_mesh(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_signal(sem, 1, device_id=(0, 1, 0))
        y_ref[...] = x_ref[...]

      return _shard_mapped(
          lambda x_local: _single_block_call(
              kernel, x_local, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
          ),
          mesh_2d,
          axis_spec=P("x", None),
      )(x)

    ok_joint, _ = jaxpr_model.check_kernel_static_models(
        joint_axis_valid, (x_spec,)
    )
    self.assertEmpty(
        ok_joint.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=ok_joint.summary(),
    )

    bad_joint, _ = jaxpr_model.check_kernel_static_models(
        joint_axis_oob, (x_spec,)
    )
    self.assertNotEmpty(
        bad_joint.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=bad_joint.summary(),
    )

    bad_scalar, _ = jaxpr_model.check_kernel_static_models(
        scalar_mesh_mode_on_2d_mesh, (x_spec,)
    )
    self.assertNotEmpty(
        bad_scalar.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=bad_scalar.summary(),
    )

    ok_tuple, _ = jaxpr_model.check_kernel_static_models(
        tuple_device_id_valid_on_2d_mesh, (x_spec,)
    )
    self.assertEmpty(
        ok_tuple.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=ok_tuple.summary(),
    )

    bad_tuple, _ = jaxpr_model.check_kernel_static_models(
        tuple_device_id_wrong_rank_on_2d_mesh, (x_spec,)
    )
    self.assertNotEmpty(
        bad_tuple.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=bad_tuple.summary(),
    )

  def test_tpu_lowering_failure_and_hbm_vector_store_reported(self):
    def hbm_vector_store(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_hbm_ref):
        # Vector stores cannot target HBM refs; Mosaic rejects this kernel.
        y_hbm_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          out_specs=HBM_SPEC,
      )(x)

    op = jax_op("pallas_verify_kernel::hbm_store", hbm_vector_store)
    report = verifier.verify(op, (torch.ones((8, 128)),), mode=STATIC)
    lowering = report.findings_by_category(Category.LOWERING_FAILURE)
    self.assertNotEmpty(lowering, msg=report.summary())
    self.assertIn("VMEM", lowering[0].message)
    raw_report = verifier.verify(
        hbm_vector_store,
        (jax.ShapeDtypeStruct((8, 128), jnp.float32),),
        mode=STATIC,
    )
    self.assertNotEmpty(
        raw_report.findings_by_category(Category.LOWERING_FAILURE),
        msg=raw_report.summary(),
    )

  def test_bf16_matmul_without_f32_accumulator_warned(self):
    """bf16 dot without float32 accumulation is flagged."""

    def dot_without_f32_accum(a: jax.Array, b: jax.Array) -> jax.Array:
      def kernel(a_ref, b_ref, o_ref):
        # Missing `preferred_element_type=jnp.float32` with bf16 inputs.
        o_ref[...] = lax.dot(a_ref[...], b_ref[...])

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), a.dtype),
      )(a, b)

    a = torch.ones((8, 128), dtype=torch.bfloat16)

    b = torch.ones((128, 128), dtype=torch.bfloat16)
    bf16_op = jax_op(
        "pallas_verify_kernel::bf16_dot_bad_accum", dot_without_f32_accum
    )
    hw_rep = verifier.verify(bf16_op, (a, b), mode=STATIC)
    precision = hw_rep.findings_by_category(Category.LOW_PRECISION_ACCUMULATION)
    self.assertNotEmpty(precision, msg=hw_rep.summary())
    self.assertEqual(precision[0].severity, verifier.Severity.WARNING)

  def test_cross_step_bf16_ref_accumulation_warned(self):
    def cross_step_bf16_accum(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, acc_ref):
        @pl.when(pl.program_id(0) == 0)
        def _zero():
          acc_ref[...] = jnp.zeros_like(acc_ref)

        acc_ref[...] = acc_ref[...] + x_ref[...]
        o_ref[...] = acc_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(2,),
              in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
              out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
              scratch_shapes=(pltpu.VMEM((8, 128), jnp.bfloat16),),
          ),
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.bfloat16)
    rep, _ = jaxpr_model.check_kernel_static_models(
        cross_step_bf16_accum, (x_spec,)
    )
    precision = rep.findings_by_category(Category.LOW_PRECISION_ACCUMULATION)
    self.assertNotEmpty(precision, msg=rep.summary())
    self.assertIn("bfloat16", precision[0].message)

  def test_dma_operands_decode_across_jax_versions(self):
    src, dst, dst_sem, src_sem, dev = (object() for _ in range(5))
    src_tr, dst_tr, dst_sem_tr, src_sem_tr = ((1,), (2,), (3,), (4,))
    legacy = jaxpr_model.decode_dma_operands((
        src,
        src_tr,
        dst,
        dst_tr,
        dst_sem,
        dst_sem_tr,
        src_sem,
        src_sem_tr,
        dev,
    ))
    self.assertIs(legacy.src.ref, src)
    self.assertEqual(legacy.src.transforms, src_tr)
    self.assertIs(legacy.dst.ref, dst)
    self.assertEqual(legacy.dst.transforms, dst_tr)
    self.assertIs(legacy.dst_sem.ref, dst_sem)
    self.assertIs(legacy.src_sem.ref, src_sem)
    self.assertEqual(legacy.src_sem.transforms, src_sem_tr)
    self.assertIs(legacy.device_id, dev)

    transformed = types.SimpleNamespace(ref=src, transforms=src_tr)
    current = jaxpr_model.decode_dma_operands(
        (transformed, dst, dst_sem, None, None)
    )
    self.assertIs(current.src.ref, src)
    self.assertEqual(current.src.transforms, src_tr)
    self.assertIs(current.dst.ref, dst)
    self.assertEqual(current.dst.transforms, ())
    self.assertIsNone(current.src_sem)
    self.assertIsNone(current.device_id)

  def test_unsupported_dma_operand_layout_is_rejected(self):
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=internal helper check
        ValueError, "unsupported dma operand layout"
    ):
      jaxpr_model.decode_dma_operands((object(), object()))

  def test_regular_sem_in_dma_rejected_on_tpu_v5e(self):
    def regular_sem_in_dma(counts: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(counts_ref, x_hbm, y_hbm, scratch_vmem, regular_sems):
        pltpu.make_async_copy(x_hbm, scratch_vmem, regular_sems.at[0]).start()
        pl.semaphore_wait(regular_sems.at[0], counts_ref[0])
        pltpu.make_async_copy(scratch_vmem, y_hbm, regular_sems.at[1]).start()
        pl.semaphore_wait(regular_sems.at[1], 1)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[HBM_SPEC],
              out_specs=HBM_SPEC,
              scratch_shapes=(
                  pltpu.VMEM(x.shape, x.dtype),
                  pltpu.SemaphoreType.REGULAR((2,)),
              ),
          ),
      )(counts, x)

    counts = torch.tensor([1], dtype=torch.int32)
    x = torch.ones((8, 128), dtype=torch.float32)
    reg_in_dma_op = jax_op(
        "pallas_verify_kernel::regular_sem_in_dma", regular_sem_in_dma
    )
    rep = verifier.verify(
        reg_in_dma_op,
        (counts, x),
        mode=STATIC,
        options=verifier.VerificationOptions(
            tpu_version=torch.tpu.TpuVersion.V5E
        ),
    )
    self.assertFalse(rep.passed, msg=rep.summary())
    hw = rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=rep.summary())
    self.assertIn("regular semaphore", hw[0].message)

  def test_regular_sem_in_dma_passes_with_semaphore_wait_on_tpu_v7(self):
    def regular_sem_dma_with_sem_wait(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, y_hbm, scratch_vmem, regular_sem):
        pltpu.make_async_copy(x_hbm, scratch_vmem, regular_sem).start()
        pl.semaphore_wait(regular_sem, 1)
        pltpu.make_async_copy(scratch_vmem, y_hbm, regular_sem).start()
        pl.semaphore_wait(regular_sem, 1)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[HBM_SPEC],
              out_specs=HBM_SPEC,
              scratch_shapes=(
                  pltpu.VMEM(x.shape, x.dtype),
                  pltpu.SemaphoreType.REGULAR,
              ),
          ),
      )(x)

    def regular_sem_dma_with_copy_wait(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, y_vmem, regular_sem):
        copy = pltpu.make_async_copy(x_hbm, y_vmem, regular_sem)
        copy.start()
        copy.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec(x.shape, lambda: (0, 0)),
              scratch_shapes=(pltpu.SemaphoreType.REGULAR,),
          ),
      )(x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep_v7, _ = jaxpr_model.check_kernel_static_models(
        regular_sem_dma_with_sem_wait,
        (x_spec,),
        tpu_version=torch.tpu.TpuVersion.V7,
    )
    self.assertEmpty(
        rep_v7.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=rep_v7.summary(),
    )

    rep_v5e, _ = jaxpr_model.check_kernel_static_models(
        regular_sem_dma_with_sem_wait,
        (x_spec,),
        tpu_version=torch.tpu.TpuVersion.V5E,
    )
    self.assertNotEmpty(
        rep_v5e.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=rep_v5e.summary(),
    )

    rep_copy_wait = verifier.verify(
        regular_sem_dma_with_copy_wait,
        (x_spec,),
        mode=STATIC,
        options=verifier.VerificationOptions(
            tpu_version=torch.tpu.TpuVersion.V7
        ),
    )
    self.assertNotEmpty(
        rep_copy_wait.findings_by_category(Category.LOWERING_FAILURE),
        msg=rep_copy_wait.summary(),
    )

  def test_remote_dma_semaphore_rules_on_tpu_v7(self):
    mesh = jax.sharding.AbstractMesh((1,), ("x",))

    def mismatched_remote_sems(x: jax.Array) -> jax.Array:
      def per_shard(x_shard: jax.Array) -> jax.Array:
        def kernel(x_hbm, y_hbm, regular_send_sem, dma_recv_sem):
          copy = pltpu.make_async_remote_copy(
              x_hbm,
              y_hbm,
              regular_send_sem,
              dma_recv_sem,
              device_id={"x": 0},
          )
          copy.start()
          copy.wait_recv()

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_shard.shape, x_shard.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[HBM_SPEC],
                out_specs=HBM_SPEC,
                scratch_shapes=(
                    pltpu.SemaphoreType.REGULAR,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
        )(x_shard)

      return jax.shard_map(
          per_shard,
          mesh=mesh,
          in_specs=P("x"),
          out_specs=P("x"),
          check_vma=False,
      )(x)

    def matching_regular_remote_sems(x: jax.Array) -> jax.Array:
      def per_shard(x_shard: jax.Array) -> jax.Array:
        def kernel(x_hbm, y_hbm, regular_send_sem, regular_recv_sem):
          copy = pltpu.make_async_remote_copy(
              x_hbm,
              y_hbm,
              regular_send_sem,
              regular_recv_sem,
              device_id={"x": 0},
          )
          copy.start()
          pl.semaphore_wait(regular_send_sem, 1)
          pl.semaphore_wait(regular_recv_sem, 1)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_shard.shape, x_shard.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[HBM_SPEC],
                out_specs=HBM_SPEC,
                scratch_shapes=(
                    pltpu.SemaphoreType.REGULAR,
                    pltpu.SemaphoreType.REGULAR,
                ),
            ),
        )(x_shard)

      return jax.shard_map(
          per_shard,
          mesh=mesh,
          in_specs=P("x"),
          out_specs=P("x"),
          check_vma=False,
      )(x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    bad_rep = verifier.verify(
        mismatched_remote_sems,
        (x_spec,),
        mode=STATIC,
        options=verifier.VerificationOptions(
            tpu_version=torch.tpu.TpuVersion.V7
        ),
    )
    lowering = bad_rep.findings_by_category(Category.LOWERING_FAILURE)
    self.assertNotEmpty(lowering, msg=bad_rep.summary())
    self.assertIn("same type", lowering[0].message)

    ok_rep = verifier.verify(
        matching_regular_remote_sems,
        (x_spec,),
        mode=STATIC,
        options=verifier.VerificationOptions(
            tpu_version=torch.tpu.TpuVersion.V7
        ),
    )
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    self.assertEmpty(
        ok_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=ok_rep.summary(),
    )

  def test_explicit_signal_and_wait_on_dma_sem_detected(self):
    def explicit_signal_wait_on_dma_sem(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, dma_sem):
        pl.semaphore_signal(dma_sem, 1)
        pl.semaphore_wait(dma_sem, 1)
        y_ref[...] = x_ref[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.SemaphoreType.DMA,)
      )

    x = torch.ones((8, 128), dtype=torch.float32)
    dma_in_signal_op = jax_op(
        "pallas_verify_kernel::explicit_signal_wait_on_dma_sem",
        explicit_signal_wait_on_dma_sem,
    )
    rep = verifier.verify(dma_in_signal_op, (x,), mode=STATIC)
    self.assertFalse(rep.passed, msg=rep.summary())
    lowering = rep.findings_by_category(Category.LOWERING_FAILURE)
    self.assertNotEmpty(lowering, msg=rep.summary())

  def test_semaphore_wait_with_decrement_false_rejected(self):
    def peek_semaphore_wait(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_signal(sem, 1)
        pl.semaphore_wait(sem, 1, decrement=False)
        y_ref[...] = x_ref[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
      )

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep = verifier.verify(peek_semaphore_wait, (x_spec,), mode=STATIC)
    hw = rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=rep.summary())
    self.assertIn("decrement=False", hw[0].message)

  def test_dma_with_smem_buffer_rejected(self):
    def smem_dma(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, y_ref, smem_buf, regular_sem):
        pltpu.make_async_copy(x_hbm, smem_buf, regular_sem).start()
        pl.semaphore_wait(regular_sem, 1)
        y_ref[...] = jnp.zeros_like(y_ref)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec(x.shape),
              scratch_shapes=(
                  pltpu.SMEM(x.shape, x.dtype),
                  pltpu.SemaphoreType.REGULAR,
              ),
          ),
      )(x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep = verifier.verify(
        smem_dma,
        (x_spec,),
        mode=STATIC,
        options=verifier.VerificationOptions(
            tpu_version=torch.tpu.TpuVersion.V7
        ),
    )
    hw = rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=rep.summary())
    self.assertIn("SMEM", hw[0].message)

  def test_barrier_semaphore_without_collective_id_rejected(self):
    def barrier_missing_collective_id(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        sem = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(sem, 1)
        pl.semaphore_wait(sem, 1)
        o_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    bar_rep = verifier.verify(
        barrier_missing_collective_id, (x_spec,), mode=STATIC
    )
    self.assertFalse(bar_rep.passed, msg=bar_rep.summary())
    lowering = bar_rep.findings_by_category(Category.LOWERING_FAILURE)
    self.assertNotEmpty(lowering, msg=bar_rep.summary())
    self.assertIn("collective_id", lowering[0].message)

  def test_run_scoped_hbm_allocation_rejected(self):
    def run_scoped_hbm_alloc(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        pl.run_scoped(
            lambda unused_hbm: o_ref.__setitem__(Ellipsis, x_ref[...]),
            pltpu.HBM(x.shape, x.dtype),
        )

      return pl.pallas_call(
          kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    hbm_rep, _ = jaxpr_model.check_kernel_static_models(
        run_scoped_hbm_alloc, (x_spec,)
    )
    self.assertFalse(hbm_rep.passed, msg=hbm_rep.summary())
    hw = hbm_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=hbm_rep.summary())
    self.assertIn("pl.run_scoped cannot allocate HBM", hw[0].message)

  def test_mesh_lowering_context_and_flow_insensitive_hardware_checks(self):
    x_sharded_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    abs_mesh = jax.sharding.AbstractMesh((2,), ("x",))
    cpu_mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:2]), ("x",))

    # 1. `shard_map` kernel calling `pltpu.sync_copy` (which queries
    # `get_tpu_info()` during MLIR lowering) followed by an illegal synchronous
    # HBM store must NOT skip `Phase.LOWERING` and must report
    # `LOWERING_FAILURE`.
    def shard_map_sync_copy_then_illegal_hbm_store(x: jax.Array) -> jax.Array:
      def per_shard(s: jax.Array) -> jax.Array:
        def kernel(x_hbm, o_hbm):
          pltpu.sync_copy(x_hbm, o_hbm)
          o_hbm[...] = x_hbm[...] + 1.0

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(s.shape, s.dtype),
            in_specs=[HBM_SPEC],
            out_specs=HBM_SPEC,
        )(s)

      return jax.shard_map(
          per_shard,
          mesh=abs_mesh,
          in_specs=P("x"),
          out_specs=P("x"),
          check_vma=False,
      )(x)

    r1_rep = verifier.verify(
        shard_map_sync_copy_then_illegal_hbm_store,
        (x_sharded_spec,),
        mode=STATIC,
    )
    self.assertFalse(r1_rep.passed, msg=r1_rep.summary())
    self.assertNotEmpty(
        r1_rep.findings_by_category(Category.LOWERING_FAILURE),
        msg=r1_rep.summary(),
    )

    # 2. Outer `@jax.jit(in_shardings=..., out_shardings=...)` preservation in
    # both Explicit and Auto sharding modes.
    explicit_cpu_mesh = jax.sharding.Mesh(
        np.array(jax.devices("cpu")[:2]),
        ("x",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    explicit_sharded_spec = jax.ShapeDtypeStruct(
        (16, 128),
        jnp.float32,
        sharding=jax.sharding.NamedSharding(explicit_cpu_mesh, P("x")),
    )

    @functools.partial(
        jax.jit,
        in_shardings=jax.sharding.NamedSharding(explicit_cpu_mesh, P("x")),
        out_shardings=jax.sharding.NamedSharding(explicit_cpu_mesh, P("x")),
    )
    def jit_sharded_valid(x: jax.Array) -> jax.Array:
      assert jax.typeof(x).sharding.spec == P("x", None)
      return jax.shard_map(
          lambda s: _single_block_call(
              lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] + 1),
              s,
          ),
          mesh=explicit_cpu_mesh,
          in_specs=P("x"),
          out_specs=P("x"),
          check_vma=False,
      )(x)

    jit_rep = verifier.verify(
        jit_sharded_valid, (explicit_sharded_spec,), mode=STATIC
    )
    self.assertTrue(jit_rep.passed, msg=jit_rep.summary())

    # Unsharded input spec with invalid outer `@jax.jit(in_shardings=...)`
    # rank/spec must fail lowering rather than having `@jax.jit` stripped.
    @functools.partial(
        jax.jit,
        in_shardings=jax.sharding.NamedSharding(cpu_mesh, P(None, None, "x")),
    )
    def jit_bad_in_sharding(x: jax.Array) -> jax.Array:
      return _single_block_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] + 1), x
      )

    bad_jit_rep = verifier.verify(
        jit_bad_in_sharding, (x_sharded_spec,), mode=STATIC
    )
    self.assertFalse(bad_jit_rep.passed, msg=bad_jit_rep.summary())
    self.assertNotEmpty(
        bad_jit_rep.findings_by_category(Category.LOWERING_FAILURE),
        msg=bad_jit_rep.summary(),
    )

    # 3. Callable dataclass holding `.mesh = cpu_mesh`.
    @dataclasses.dataclass(frozen=True)
    class _DataclassDistributedKernel:
      mesh: jax.sharding.Mesh

      def __call__(self, x: jax.Array) -> jax.Array:
        return jax.shard_map(
            lambda s: _single_block_call(
                lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]), s
            ),
            mesh=self.mesh,
            in_specs=P("x"),
            out_specs=P("x"),
            check_vma=False,
        )(x)

    dc_rep = verifier.verify(
        _DataclassDistributedKernel(cpu_mesh), (x_sharded_spec,), mode=STATIC
    )
    self.assertTrue(dc_rep.passed, msg=dc_rep.summary())

    # 4. Flow-insensitive compile-legality checks: dead `@pl.when` v5e regular
    # DMA, barrier semaphore in `dma_start`, dynamic `decrement` in
    # `semaphore_wait`, `jax.empty_ref(pltpu.HBM(...))`, and argument-bound v5e
    # mesh target unification.
    x8 = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def dead_branch_v5e_regular_dma(
        x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      def per_shard(s: jax.Array) -> jax.Array:
        def kernel(x_hbm, o_ref, vmem_buf, reg_sem):
          @pl.when(pl.program_id(0) == 99)
          def _unreachable():
            pltpu.make_async_copy(x_hbm, vmem_buf, reg_sem).start()

          o_ref[...] = jnp.zeros_like(o_ref[...])

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(s.shape, s.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(2,),
                in_specs=[HBM_SPEC],
                out_specs=pl.BlockSpec(s.shape, lambda _i: (0, 0)),
                scratch_shapes=(
                    pltpu.VMEM(s.shape, s.dtype),
                    pltpu.SemaphoreType.REGULAR,
                ),
            ),
        )(s)

      return jax.shard_map(
          per_shard,
          mesh=mesh,
          in_specs=P("x"),
          out_specs=P("x"),
          check_vma=False,
      )(x)

    v5e_arg_mesh = jax.sharding.Mesh(
        np.array([
            _mock_device._PallasDevice(device_kind="TPU v5 lite", id=i)
            for i in range(2)
        ]),
        ("x",),
    )
    dead_v5e_rep = verifier.verify(
        dead_branch_v5e_regular_dma,
        (x_sharded_spec,),
        kwargs={"mesh": v5e_arg_mesh},
        mode=STATIC,
    )
    self.assertFalse(dead_v5e_rep.passed, msg=dead_v5e_rep.summary())
    self.assertNotEmpty(
        dead_v5e_rep.findings_by_category(
            Category.HARDWARE_CONSTRAINT_VIOLATION
        ),
        msg=dead_v5e_rep.summary(),
    )
    self.assertIn("regular semaphore", dead_v5e_rep.summary())

    def barrier_sem_in_dma(x: jax.Array) -> jax.Array:
      def per_shard(s: jax.Array) -> jax.Array:
        def kernel(x_hbm, o_ref, vmem_buf, bsem):
          pltpu.make_async_copy(x_hbm, vmem_buf, bsem).start()
          pl.semaphore_wait(bsem, 1)
          o_ref[...] = vmem_buf[...]

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(s.shape, s.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[HBM_SPEC],
                out_specs=pl.BlockSpec(s.shape, lambda: (0, 0)),
                scratch_shapes=(
                    pltpu.VMEM(s.shape, s.dtype),
                    pltpu.SemaphoreType.BARRIER,
                ),
            ),
        )(s)

      return jax.shard_map(
          per_shard,
          mesh=abs_mesh,
          in_specs=P("x"),
          out_specs=P("x"),
          check_vma=False,
      )(x)

    barrier_rep = verifier.verify(
        barrier_sem_in_dma, (x_sharded_spec,), mode=STATIC
    )
    self.assertFalse(barrier_rep.passed, msg=barrier_rep.summary())
    self.assertNotEmpty(
        barrier_rep.findings_by_category(
            Category.HARDWARE_CONSTRAINT_VIOLATION
        ),
        msg=barrier_rep.summary(),
    )
    self.assertIn("barrier semaphore", barrier_rep.summary())

    def dynamic_decrement_wait(flag: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(flag_smem, x_ref, o_ref, sem):
        pl.semaphore_signal(sem, 1)
        pl.semaphore_wait(sem, 1, decrement=(flag_smem[0] > 0))
        o_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec(x.shape, lambda _f: (0, 0))],
              out_specs=pl.BlockSpec(x.shape, lambda _f: (0, 0)),
              scratch_shapes=(pltpu.SemaphoreType.REGULAR,),
          ),
      )(flag, x)

    dyn_dec_rep = verifier.verify(
        dynamic_decrement_wait,
        (jax.ShapeDtypeStruct((1,), jnp.int32), x8),
        mode=STATIC,
    )
    self.assertFalse(dyn_dec_rep.passed, msg=dyn_dec_rep.summary())
    self.assertNotEmpty(
        dyn_dec_rep.findings_by_category(
            Category.HARDWARE_CONSTRAINT_VIOLATION
        ),
        msg=dyn_dec_rep.summary(),
    )

  def test_empty_ref_hbm_allocation_rejected(self):
    if not hasattr(jax, "empty_ref"):
      self.skipTest("jax.empty_ref is unavailable in this JAX release")
    x8 = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def hbm_empty_ref_fn(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        tmp = jax.empty_ref(
            jax.ShapeDtypeStruct(x_ref.shape, x_ref.dtype),
            memory_space=pltpu.MemorySpace.HBM,
        )
        pltpu.sync_copy(x_ref, tmp)
        pltpu.sync_copy(tmp, o_ref)

      return _single_block_call(kernel, x)

    empty_ref_rep = verifier.verify(hbm_empty_ref_fn, (x8,), mode=STATIC)
    self.assertFalse(empty_ref_rep.passed, msg=empty_ref_rep.summary())
    self.assertNotEmpty(
        empty_ref_rep.findings_by_category(
            Category.HARDWARE_CONSTRAINT_VIOLATION
        ),
        msg=empty_ref_rep.summary(),
    )
    self.assertIn("HBM", empty_ref_rep.summary())

  def test_scratch_shapes_hbm_allocation_rejected(self):
    def scratch_hbm_alloc(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, unused_hbm_scratch):
        o_ref[...] = x_ref[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.HBM(x.shape, x.dtype),)
      )

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        scratch_hbm_alloc, (x_spec,)
    )
    hw = rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=rep.summary())
    self.assertIn("scratch_shapes[0] cannot allocate HBM", hw[0].message)

  def test_negative_computed_scalar_index_oob_detected(self):
    def negative_computed_scalar_index(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        bad_row = pl.program_id(0) - 1
        o_ref[0, :] = x_ref[bad_row, :]
        o_ref[1:8, :] = x_ref[1:8, :]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(2,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
      )(x)

    neg_rep, _ = jaxpr_model.check_kernel_static_models(
        negative_computed_scalar_index,
        (jax.ShapeDtypeStruct((16, 128), jnp.float32),),
    )
    self.assertNotEmpty(
        neg_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=neg_rep.summary(),
    )

  def test_non_consecutive_output_block_revisit_detected(self):
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)

    # 1. Full-tile write-only non-consecutive revisit is well-defined and emits
    # a redundant-writeback WARNING rather than failing the report.
    def full_overwrite_revisit(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct((16, 128), x.dtype),
          grid=(4,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i % 2, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i % 2, 0)),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("arbitrary",)
          ),
      )(x)

    warn_rep, _ = jaxpr_model.check_kernel_static_models(
        full_overwrite_revisit, (x_spec,)
    )
    self.assertTrue(warn_rep.passed, msg=warn_rep.summary())
    warn_hw = warn_rep.findings_by_category(
        Category.HARDWARE_CONSTRAINT_VIOLATION
    )
    self.assertNotEmpty(warn_hw, msg=warn_rep.summary())
    self.assertEqual(warn_hw[0].severity, verifier.Severity.WARNING)
    self.assertIn("revisits output block", warn_hw[0].message)

    # 2. Partial-write non-consecutive revisit clobbers previously written rows
    # and remains an ERROR.
    def partial_write_revisit(x: jax.Array) -> jax.Array:
      def body(x_ref, o_ref):
        @pl.when(pl.program_id(0) < 2)
        def _first_half():
          o_ref[0:4, :] = x_ref[0:4, :]

        @pl.when(pl.program_id(0) >= 2)
        def _second_half():
          o_ref[4:8, :] = x_ref[4:8, :]

      return pl.pallas_call(
          body,
          out_shape=jax.ShapeDtypeStruct((16, 128), x.dtype),
          grid=(4,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i % 2, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i % 2, 0)),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("arbitrary",)
          ),
      )(x)

    err_rep, _ = jaxpr_model.check_kernel_static_models(
        partial_write_revisit, (x_spec,)
    )
    self.assertFalse(err_rep.passed, msg=err_rep.summary())
    err_hw = err_rep.findings_by_category(
        Category.HARDWARE_CONSTRAINT_VIOLATION
    )
    self.assertNotEmpty(err_hw, msg=err_rep.summary())
    self.assertEqual(err_hw[0].severity, verifier.Severity.ERROR)

    # 3. Outputs aliased to an input via `input_output_aliases` are exempt.
    def aliased_revisit(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] + 1.0),
          out_shape=jax.ShapeDtypeStruct((16, 128), x.dtype),
          grid=(4,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i % 2, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i % 2, 0)),
          input_output_aliases={0: 0},
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("arbitrary",)
          ),
      )(x)

    alias_rep, _ = jaxpr_model.check_kernel_static_models(
        aliased_revisit, (x_spec,)
    )
    self.assertEmpty(
        alias_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=alias_rep.summary(),
    )

  def test_parallel_unwritten_and_partially_unwritten_tile_clobber_detected(
      self,
  ):
    def parallel_unwritten_clobber(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        @pl.when(pl.program_id(0) == 0)
        def _write():
          o_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid=(2,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",)
          ),
      )(x)

    def parallel_partial_clobber(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        @pl.when(pl.program_id(0) == 0)
        def _write_first_half():
          o_ref[0:4, :] = x_ref[0:4, :]

        @pl.when(pl.program_id(0) == 1)
        def _write_second_half():
          o_ref[4:8, :] = x_ref[4:8, :]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid=(2,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",)
          ),
      )(x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    clobber_rep, _ = jaxpr_model.check_kernel_static_models(
        parallel_unwritten_clobber, (x_spec,)
    )
    self.assertNotEmpty(
        clobber_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=clobber_rep.summary(),
    )

    partial_rep, _ = jaxpr_model.check_kernel_static_models(
        parallel_partial_clobber, (x_spec,)
    )
    self.assertNotEmpty(
        partial_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=partial_rep.summary(),
    )

  def test_cond_partitioned_branch_writes_cover_output(self):
    def partitioned_cond_copy(flag: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(flag_ref, x_ref, o_ref):
        def write_full():
          o_ref[0:8, :] = x_ref[0:8, :]

        def write_halves():
          o_ref[0:4, :] = x_ref[0:4, :]
          o_ref[4:8, :] = x_ref[4:8, :]

        lax.cond(flag_ref[0] > 0, write_full, write_halves)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec(x.shape)],
              out_specs=pl.BlockSpec(x.shape),
          ),
      )(flag, x)

    flag_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    cond_rep, _ = jaxpr_model.check_kernel_static_models(
        partitioned_cond_copy, (flag_spec, x_spec)
    )
    self.assertEmpty(
        cond_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=cond_rep.summary(),
    )

  def test_complementary_pl_when_branches_cover_output(self):
    def complementary_when_copy(flag: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(flag_ref, x_ref, o_ref):
        @pl.when(flag_ref[0] > 0)
        def _pos():
          o_ref[...] = x_ref[...]

        @pl.when(flag_ref[0] <= 0)
        def _non_pos():
          o_ref[...] = x_ref[...] + 1.0

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec(x.shape)],
              out_specs=pl.BlockSpec(x.shape),
          ),
      )(flag, x)

    def single_when_copy(flag: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(flag_ref, x_ref, o_ref):
        @pl.when(flag_ref[0] > 0)
        def _pos():
          o_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec(x.shape)],
              out_specs=pl.BlockSpec(x.shape),
          ),
      )(flag, x)

    flag_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    ok_rep, _ = jaxpr_model.check_kernel_static_models(
        complementary_when_copy, (flag_spec, x_spec)
    )
    self.assertEmpty(
        ok_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=ok_rep.summary(),
    )

    bad_rep, _ = jaxpr_model.check_kernel_static_models(
        single_when_copy, (flag_spec, x_spec)
    )
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=bad_rep.summary(),
    )

  def test_runtime_guarded_boundary_access_emits_warning_not_error(self):
    def runtime_guarded_prefetch(limit: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(limit_ref, x_hbm, o_ref, buf, sem):
        i = pl.program_id(0)

        @pl.when(i + 1 < limit_ref[0])
        def _prefetch_next():
          copy = pltpu.make_async_copy(
              x_hbm.at[pl.ds((i + 1) * 8, 8), :], buf, sem
          )
          copy.start()
          copy.wait()

        o_ref[...] = jnp.zeros_like(o_ref)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              grid=(2,),
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec((8, 128), lambda i, unused_l: (i, 0)),
              scratch_shapes=(
                  pltpu.VMEM((8, 128), x.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(limit, x)

    limit_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        runtime_guarded_prefetch, (limit_spec, x_spec)
    )
    self.assertTrue(rep.passed, msg=rep.summary())
    oob = rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS)
    self.assertNotEmpty(oob, msg=rep.summary())
    self.assertEqual(oob[0].severity, verifier.Severity.WARNING)

  def test_vmapped_pallas_call_preserves_user_program_id_and_semantics(self):
    def single_example(idx_map: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(idx_ref, x_ref, y_ref):
        # User axis 0 has size 2; user axis 1 has size 1. Under `jax.vmap`
        # with batch size 2, `grid_mapping.grid` becomes `(2, 2, 1)` with
        # `vmapped_dims=(0,)`, while `pl.program_id(1)` must still refer to
        # user axis 1 (always 0) rather than user axis 0 (0 or 1).
        h = pl.program_id(0)
        q_blk = pl.program_id(1)
        _ = idx_ref[h, q_blk]
        y_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel,
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec((1, 8, 128), lambda h, i, _: (h, 0, 0))],
              out_specs=pl.BlockSpec((1, 8, 128), lambda h, i, _: (h, 0, 0)),
              grid=(2, 1),
          ),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "arbitrary")
          ),
      )(idx_map, x)

    idx_map = jnp.zeros((2, 1), dtype=jnp.int32)
    batched_fn = lambda xs: jax.vmap(lambda x: single_example(idx_map, x))(xs)
    xs_spec = jax.ShapeDtypeStruct((2, 2, 8, 128), jnp.float32)

    rep, models = jaxpr_model.check_kernel_static_models(batched_fn, (xs_spec,))
    self.assertTrue(rep.passed, msg=rep.summary())
    self.assertEmpty(
        rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=rep.summary(),
    )
    self.assertLen(models, 1)
    self.assertEqual(models[0].grid, (2, 2, 1))
    self.assertEqual(
        models[0].dimension_semantics, ("parallel", "parallel", "arbitrary")
    )

  def test_core_map_invalid_core_id_and_unwritten_output_detected(self):
    core_mesh = _make_tensorcore_mesh("core", num_cores=2)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def core_map_bad_target(x: jax.Array) -> jax.Array:
      @pl.kernel(
          mesh=core_mesh,
          out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
          scratch_types=[pltpu.SemaphoreType.REGULAR],
      )
      def _body(x_ref, y_ref, sem):
        del x_ref, y_ref
        pl.semaphore_signal(sem, 1, device_id={"core": 2})

      return _body(x)

    cm_rep = verifier.verify(core_map_bad_target, (x_spec,), mode=STATIC)
    self.assertNotEmpty(
        cm_rep.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=cm_rep.summary(),
    )
    self.assertNotEmpty(
        cm_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=cm_rep.summary(),
    )

    def core_map_cross_core_handshake(x: jax.Array) -> jax.Array:
      @pl.kernel(mesh=core_mesh)
      def _body():
        def _scoped(sem):
          c = lax.axis_index("core")
          pl.semaphore_signal(sem, 1, device_id={"core": 1 - c})
          pl.semaphore_wait(sem, 1)

        pl.run_scoped(_scoped, pltpu.SemaphoreType.REGULAR)

      _body()
      return x

    ok_cm_rep = verifier.verify(
        core_map_cross_core_handshake, (x_spec,), mode=STATIC
    )
    self.assertTrue(ok_cm_rep.passed, msg=ok_cm_rep.summary())
    self.assertEmpty(ok_cm_rep.findings, msg=ok_cm_rep.summary())

  def test_core_map_with_outer_empty_ref_checks_output_coverage(self):
    core_mesh = _make_tensorcore_mesh("core", num_cores=2)
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)

    def core_map_empty_ref_unwritten(x: jax.Array) -> jax.Array:
      x_ref = jax.new_ref(x)
      o_ref = jax.empty_ref(jax.ShapeDtypeStruct(x.shape, x.dtype))

      @pl.kernel(mesh=core_mesh)
      def _kernel():
        # Only core 0 writes rows 0:8; rows 8:16 are never written.
        @pl.when(lax.axis_index("core") == 0)
        def _w():
          pl.run_scoped(
              lambda vmem, sem: (
                  pltpu.make_async_copy(x_ref.at[0:8, :], vmem, sem).start(),
                  pltpu.make_async_copy(x_ref.at[0:8, :], vmem, sem).wait(),
                  pltpu.make_async_copy(vmem, o_ref.at[0:8, :], sem).start(),
                  pltpu.make_async_copy(vmem, o_ref.at[0:8, :], sem).wait(),
              ),
              pltpu.VMEM((8, 128), x.dtype),
              pltpu.SemaphoreType.DMA,
          )

      _kernel()
      return o_ref[...]

    rep, _ = jaxpr_model.check_kernel_static_models(
        core_map_empty_ref_unwritten, (x_spec,)
    )
    unwritten = rep.findings_by_category(Category.UNWRITTEN_OUTPUT)
    self.assertNotEmpty(unwritten, msg=rep.summary())
    self.assertIn("8:16", unwritten[0].message)

    def core_map_empty_ref_post_reshape_unwritten(x: jax.Array) -> jax.Array:
      x_ref = jax.new_ref(x)
      o_ref = jax.empty_ref(jax.ShapeDtypeStruct(x.shape, x.dtype))

      @pl.kernel(mesh=core_mesh)
      def _kernel():
        @pl.when(lax.axis_index("core") == 0)
        def _w():
          pl.run_scoped(
              lambda vmem, sem: (
                  pltpu.make_async_copy(x_ref.at[0:8, :], vmem, sem).start(),
                  pltpu.make_async_copy(x_ref.at[0:8, :], vmem, sem).wait(),
                  pltpu.make_async_copy(vmem, o_ref.at[0:8, :], sem).start(),
                  pltpu.make_async_copy(vmem, o_ref.at[0:8, :], sem).wait(),
              ),
              pltpu.VMEM((8, 128), x.dtype),
              pltpu.SemaphoreType.DMA,
          )

      _kernel()
      return o_ref[...].reshape(4, 4, 128)

    reshape_rep, _ = jaxpr_model.check_kernel_static_models(
        core_map_empty_ref_post_reshape_unwritten, (x_spec,)
    )
    reshape_unwritten = reshape_rep.findings_by_category(
        Category.UNWRITTEN_OUTPUT
    )
    self.assertNotEmpty(reshape_unwritten, msg=reshape_rep.summary())
    self.assertIn("8:16", reshape_unwritten[0].message)

    def core_map_empty_ref_post_reshape_complete(x: jax.Array) -> jax.Array:
      x_ref = jax.new_ref(x)
      o_ref = jax.empty_ref(jax.ShapeDtypeStruct(x.shape, x.dtype))

      @pl.kernel(mesh=core_mesh)
      def _kernel():
        c = lax.axis_index("core")
        rows = pl.ds(c * 8, 8)
        pl.run_scoped(
            lambda vmem, sem: (
                pltpu.make_async_copy(x_ref.at[rows, :], vmem, sem).start(),
                pltpu.make_async_copy(x_ref.at[rows, :], vmem, sem).wait(),
                pltpu.make_async_copy(vmem, o_ref.at[rows, :], sem).start(),
                pltpu.make_async_copy(vmem, o_ref.at[rows, :], sem).wait(),
            ),
            pltpu.VMEM((8, 128), x.dtype),
            pltpu.SemaphoreType.DMA,
        )

      _kernel()
      return o_ref[...].reshape(4, 4, 128)

    ok_reshape_rep, _ = jaxpr_model.check_kernel_static_models(
        core_map_empty_ref_post_reshape_complete, (x_spec,)
    )
    self.assertEmpty(
        ok_reshape_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=ok_reshape_rep.summary(),
    )

  def test_emit_pipeline_valid_and_unwritten_detected(self):
    def emit_pipeline_full(x: jax.Array) -> jax.Array:
      def outer_kernel(x_hbm, y_hbm):
        pltpu.emit_pipeline(
            lambda x_vmem, y_vmem: y_vmem.__setitem__(Ellipsis, x_vmem[...]),
            grid=(2,),
            in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
            out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
        )(x_hbm, y_hbm)

      return pl.pallas_call(
          outer_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=HBM_SPEC,
      )(x)

    def emit_pipeline_short(x: jax.Array) -> jax.Array:
      def outer_kernel(x_hbm, y_hbm):
        pltpu.emit_pipeline(
            lambda x_vmem, y_vmem: y_vmem.__setitem__(Ellipsis, x_vmem[...]),
            grid=(1,),
            in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
            out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
        )(x_hbm, y_hbm)

      return pl.pallas_call(
          outer_kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=HBM_SPEC,
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    ok_rep, _ = jaxpr_model.check_kernel_static_models(
        emit_pipeline_full, (x_spec,)
    )
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    self.assertEmpty(
        ok_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=ok_rep.summary(),
    )

    short_rep, _ = jaxpr_model.check_kernel_static_models(
        emit_pipeline_short, (x_spec,)
    )
    unwritten = short_rep.findings_by_category(Category.UNWRITTEN_OUTPUT)
    self.assertNotEmpty(unwritten, msg=short_rep.summary())
    self.assertIn("8:16", unwritten[0].message)

  def test_truncated_loop_alternating_carry_not_falsely_bound(self):
    def alternating_carry_fn(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        # 33 iterations flip `0 -> 1 -> 0 ... -> 1`. With
        # `max_unrolled_iterations=32`, step 32 happens to equal `init=0`,
        # whereas the true post-loop value is `1` (`1 - s == 0`, in-bounds!).
        s = lax.fori_loop(0, 33, lambda i, c: jnp.int32(1) - c, jnp.int32(0))
        o_ref[...] = x_ref[jnp.int32(1) - s, :]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((128,), x.dtype),
          in_specs=[pl.BlockSpec((1, 128), lambda i: (0, 0))],
          out_specs=pl.BlockSpec((128,), lambda i: (0,)),
          grid=(1,),
      )(x)

    x_spec = jax.ShapeDtypeStruct((1, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        alternating_carry_fn, (x_spec,), max_unrolled_iterations=32
    )
    self.assertEmpty(
        rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=rep.summary(),
    )
    self.assertNotEmpty(
        rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=rep.summary(),
    )

  def test_core_map_default_hbm_ref_catches_sync_store(self):
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def core_map_sync_hbm_store(x: jax.Array) -> jax.Array:
      del x
      o_ref = jax.empty_ref(jax.ShapeDtypeStruct((8, 128), jnp.float32))

      @pl.kernel(mesh=_make_tensorcore_mesh("core", 1))
      def _kernel():
        o_ref[...] = jnp.zeros((8, 128), jnp.float32)

      _kernel()
      return o_ref[...]

    rep = verifier.verify(core_map_sync_hbm_store, (x_spec,), mode=STATIC)
    self.assertNotEmpty(
        rep.findings_by_category(Category.LOWERING_FAILURE),
        msg=rep.summary(),
    )

  def test_revisit_mode_any_single_vs_double_buffered(self):
    revisit_cls = getattr(pl, "RevisitMode", None)
    if revisit_cls is None:
      self.skipTest("pl.RevisitMode is unavailable in this JAX release")
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)

    def make_revisit_fn(buffer_count: int):
      def fn(x: jax.Array) -> jax.Array:
        return pl.pallas_call(
            lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            in_specs=[pl.BlockSpec((8, 128), lambda i: (i % 2, 0))],
            out_specs=pl.BlockSpec(
                (8, 128),
                lambda i: (i % 2, 0),
                pipeline_mode=pl.Buffered(
                    buffer_count=buffer_count, revisit=revisit_cls.ANY
                ),
            ),
            grid=(4,),
        )(x)

      return fn

    ok_rep, _ = jaxpr_model.check_kernel_static_models(
        make_revisit_fn(1), (x_spec,)
    )
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())

    bad_rep = verifier.verify(make_revisit_fn(2), (x_spec,), mode=STATIC)
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.LOWERING_FAILURE),
        msg=bad_rep.summary(),
    )

  def test_fixed_width_shift_right_logical_and_uint32_overflow(self):
    def logical_shift_oob(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        # Unsigned 32-bit shift of `-2` (`0xFFFFFFFE >> 31`) is `+1`, which is
        # out-of-bounds on shape `(1, 128)`.
        idx = lax.shift_right_logical(jnp.int32(-2), jnp.int32(31))
        o_ref[...] = x_ref[idx, :]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((128,), x.dtype),
          in_specs=[pl.BlockSpec((1, 128), lambda i: (0, 0))],
          out_specs=pl.BlockSpec((128,), lambda i: (0,)),
          grid=(1,),
      )(x)

    def uint32_wrap_valid(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        u = jnp.uint32(0xFFFFFFFF) + jnp.uint32(1)
        o_ref[...] = x_ref[u.astype(jnp.int32), :]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((128,), x.dtype),
          in_specs=[pl.BlockSpec((1, 128), lambda i: (0, 0))],
          out_specs=pl.BlockSpec((128,), lambda i: (0,)),
          grid=(1,),
      )(x)

    x_spec = jax.ShapeDtypeStruct((1, 128), jnp.float32)
    oob_rep, _ = jaxpr_model.check_kernel_static_models(
        logical_shift_oob, (x_spec,)
    )
    self.assertNotEmpty(
        oob_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=oob_rep.summary(),
    )

    wrap_rep, _ = jaxpr_model.check_kernel_static_models(
        uint32_wrap_valid, (x_spec,)
    )
    self.assertTrue(wrap_rep.passed, msg=wrap_rep.summary())

  def test_run_scoped_collective_axes_and_done_count_model_flag(self):
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def bad_collective_run_scoped(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        pl.run_scoped(
            lambda sem: o_ref.__setitem__(Ellipsis, x_ref[...]),
            pltpu.SemaphoreType.REGULAR,
            collective_axes="x",
        )

      return pl.pallas_call(
          kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
      )(x)

    bad_rep = verifier.verify(bad_collective_run_scoped, (x_spec,), mode=STATIC)
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.LOWERING_FAILURE),
        msg=bad_rep.summary(),
    )

    def done_count_v7_fn(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_hbm, vmem, reg_sem):
        pltpu.make_async_copy(x_hbm, vmem, reg_sem).start()
        pl.semaphore_wait(reg_sem, 1)
        pltpu.make_async_copy(vmem, o_hbm, reg_sem).start()
        pl.semaphore_wait(reg_sem, 1)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=HBM_SPEC,
          scratch_shapes=[
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.SemaphoreType.REGULAR,
          ],
      )(x)

    ok_rep, models = jaxpr_model.check_kernel_static_models(
        done_count_v7_fn,
        (x_spec,),
        tpu_version=torch.tpu.TpuVersion.V7,
    )
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    self.assertTrue(models[0].uses_done_count_regular_dma)

  def test_trace_time_oob_slice_and_generic_trace_error_classification(self):
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def trace_time_oob_slice(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        # All-static slice validated by `NDIndexer` during `make_jaxpr` raises
        # `ValueError("Out of bound slice...")`.
        o_ref[...] = x_ref[pl.ds(16, 8), :]

      return _single_block_call(kernel, x)

    def trace_time_oob_end_slice(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[pl.ds(2, 8), :]

      return _single_block_call(kernel, x)

    for oob_fn in (trace_time_oob_slice, trace_time_oob_end_slice):
      oob_model_rep, oob_models = jaxpr_model.check_kernel_static_models(
          oob_fn, (x_spec,)
      )
      self.assertIsNone(oob_models)
      oob_findings = oob_model_rep.findings_by_category(
          Category.OUT_OF_BOUNDS_ACCESS
      )
      self.assertNotEmpty(oob_findings, msg=oob_model_rep.summary())
      self.assertEqual(oob_findings[0].severity, verifier.Severity.ERROR)

    oob_static_rep = verifier.verify(
        trace_time_oob_slice, (x_spec,), mode=STATIC
    )
    self.assertFalse(oob_static_rep.passed, msg=oob_static_rep.summary())
    self.assertNotEmpty(
        oob_static_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=oob_static_rep.summary(),
    )
    self.assertNotIn(verifier.Phase.LOWERING, oob_static_rep.checked_phases)

    # Non-Pallas host shape errors mentioning `"must have the same shape"` must
    # report `VERIFICATION_INCOMPLETE` (`WARNING`) rather than a false Pallas
    # bounds or device-id category.
    def host_shape_error_before_kernel(x: jax.Array) -> jax.Array:
      bad = x + jnp.ones((3, 5), dtype=x.dtype)
      return _single_block_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]), bad
      )

    host_rep, host_models = jaxpr_model.check_kernel_static_models(
        host_shape_error_before_kernel, (x_spec,)
    )
    self.assertIsNone(host_models)
    self.assertTrue(host_rep.passed, msg=host_rep.summary())
    inc = host_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE)
    self.assertLen(inc, 1, msg=host_rep.summary())
    self.assertEqual(inc[0].severity, verifier.Severity.WARNING)

  def test_stride_zero_sublane_broadcast_load_allowed_and_store_rejected(self):
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def sublane_broadcast_load(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[pl.ds(0, 8, stride=0), :]

      return _single_block_call(kernel, x)

    def stride_zero_store(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        o_ref[pl.ds(0, 8, stride=0), :] = x_ref[...]

      return _single_block_call(kernel, x)

    load_rep, _ = jaxpr_model.check_kernel_static_models(
        sublane_broadcast_load, (x_spec,)
    )
    self.assertTrue(load_rep.passed, msg=load_rep.summary())
    self.assertEmpty(
        load_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=load_rep.summary(),
    )

    store_rep, _ = jaxpr_model.check_kernel_static_models(
        stride_zero_store, (x_spec,)
    )
    self.assertFalse(store_rep.passed, msg=store_rep.summary())
    self.assertNotEmpty(
        store_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=store_rep.summary(),
    )

  def test_sparsecore_sync_any_access_unknown_primitives_and_mpmd_scratch(self):
    x16 = jax.ShapeDtypeStruct((16,), jnp.float32)
    x2x16 = jax.ShapeDtypeStruct((2, 16), jnp.float32)

    def sc_sync_any_access(x: jax.Array) -> jax.Array:
      v_mesh = plsc.VectorSubcoreMesh(
          core_axis_name="s_core",
          subcore_axis_name="subcore",
          num_cores=1,
          num_subcores=1,
      )
      x_any = jax.new_ref(x, memory_space=pl.ANY)
      o_any = jax.empty_ref(
          jax.ShapeDtypeStruct(x.shape, x.dtype), memory_space=pl.ANY
      )

      @pl.kernel(mesh=v_mesh)
      def _sc():
        o_any[...] = x_any[...]

      _sc()
      return o_any[...]

    any_rep, _ = jaxpr_model.check_kernel_static_models(
        sc_sync_any_access, (x16,)
    )
    self.assertFalse(any_rep.passed, msg=any_rep.summary())
    self.assertNotEmpty(
        any_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=any_rep.summary(),
    )
    self.assertIn("pl.ANY", any_rep.summary())

    def sc_barrier_kernel(x: jax.Array) -> jax.Array:
      num_lanes = plsc.get_sparse_core_info().num_lanes
      v_mesh = plsc.VectorSubcoreMesh(
          core_axis_name="s_core",
          subcore_axis_name="subcore",
          num_cores=1,
          num_subcores=2,
      )

      @pl.kernel(
          mesh=v_mesh,
          out_type=jax.ShapeDtypeStruct((2, num_lanes), jnp.float32),
          scratch_types=[
              pltpu.VMEM((num_lanes,), jnp.float32),
              pltpu.SemaphoreType.DMA,
          ],
      )
      def _body(x_hbm, o_hbm, local_vmem, sem):
        s = lax.axis_index("subcore")
        cin = pltpu.make_async_copy(x_hbm.at[s], local_vmem, sem)
        cin.start()
        cin.wait()
        plsc.subcore_barrier()
        cout = pltpu.make_async_copy(local_vmem, o_hbm.at[s], sem)
        cout.start()
        cout.wait()

      return _body(x)

    bar_rep, _ = jaxpr_model.check_kernel_static_models(
        sc_barrier_kernel, (x2x16,)
    )
    self.assertTrue(bar_rep.passed, msg=bar_rep.summary())
    self.assertNotEmpty(
        bar_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=bar_rep.summary(),
    )

    def sc_gather_kernel(x: jax.Array) -> jax.Array:
      num_lanes = plsc.get_sparse_core_info().num_lanes
      v_mesh = plsc.VectorSubcoreMesh(
          core_axis_name="s_core",
          subcore_axis_name="subcore",
          num_cores=1,
          num_subcores=1,
      )

      @pl.kernel(
          mesh=v_mesh,
          out_type=jax.ShapeDtypeStruct((num_lanes,), jnp.float32),
          scratch_types=[
              pltpu.VMEM((num_lanes,), jnp.float32),
              pltpu.SemaphoreType.DMA,
          ],
      )
      def _body(x_hbm, o_hbm, local_vmem, sem):
        cin = pltpu.make_async_copy(x_hbm, local_vmem, sem)
        cin.start()
        cin.wait()
        idx = jnp.arange(num_lanes, dtype=jnp.int32)
        local_vmem[...] = plsc.load_gather(local_vmem, [idx])
        cout = pltpu.make_async_copy(local_vmem, o_hbm, sem)
        cout.start()
        cout.wait()

      return _body(x)

    gather_rep, _ = jaxpr_model.check_kernel_static_models(
        sc_gather_kernel, (x16,)
    )
    self.assertTrue(gather_rep.passed, msg=gather_rep.summary())
    self.assertNotEmpty(
        gather_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=gather_rep.summary(),
    )

    # Multi-mesh SparseCore `mpmd_map` with `HBM` and `VMEM_SHARED` scratch and
    # cross-mesh `device_id` keys must not emit false scratch or device-id
    # errors.
    def sc_mpmd_handover(x: jax.Array) -> jax.Array:
      del x
      tile_size = plsc.get_sparse_core_info().num_lanes
      v_mesh = plsc.VectorSubcoreMesh(
          core_axis_name="s_core",
          subcore_axis_name="subcore",
          num_cores=1,
          num_subcores=2,
      )
      s_mesh = plsc.ScalarSubcoreMesh(axis_name="s_core", num_cores=1)

      def vector_fn(out_ref, hbm_scr, vmem_shared_scr, s_dma, tec_sem, scs_sem):
        del out_ref, vmem_shared_scr, s_dma
        core_id = lax.axis_index("s_core")
        subcore_id = lax.axis_index("subcore")
        vmem_buf = jax.empty_ref(
            jax.ShapeDtypeStruct((tile_size,), jnp.float32), pltpu.VMEM
        )
        vmem_buf[...] = jnp.ones((tile_size,), jnp.float32)
        pltpu.sync_copy(vmem_buf, hbm_scr.at[subcore_id, core_id])
        pl.semaphore_signal(
            scs_sem.at[subcore_id], 1, device_id={"s_core": core_id}
        )
        pl.semaphore_wait(tec_sem.at[subcore_id], 1)

      def scalar_fn(out_ref, hbm_scr, vmem_shared_scr, s_dma, tec_sem, scs_sem):
        del vmem_shared_scr
        core_id = lax.axis_index("s_core")
        smem = jax.empty_ref(
            jax.ShapeDtypeStruct((2, tile_size), jnp.float32), pltpu.SMEM
        )
        for i in range(2):
          pl.semaphore_wait(scs_sem.at[i], 1)
          copy = pltpu.make_async_copy(
              hbm_scr.at[i, core_id], smem.at[i], s_dma.at[i]
          )
          copy.start()
          copy.wait()
          pl.semaphore_signal(
              tec_sem.at[i], 1, device_id={"s_core": core_id, "subcore": i}
          )
        pltpu.sync_copy(smem.at[0], out_ref.at[core_id])

      return pl.kernel(
          body=[vector_fn, scalar_fn],
          mesh=[v_mesh, s_mesh],
          out_type=jax.ShapeDtypeStruct((1, tile_size), jnp.float32),
          scratch_types=[
              pltpu.HBM((2, 1, tile_size), jnp.float32),
              pltpu.VMEM_SHARED((2, tile_size), jnp.float32),
              pltpu.SemaphoreType.DMA((2,)) @ s_mesh,
              pltpu.SemaphoreType.REGULAR((2,)) @ v_mesh,
              pltpu.SemaphoreType.REGULAR((2,)) @ s_mesh,
          ],
      )()

    mpmd_rep, _ = jaxpr_model.check_kernel_static_models(
        sc_mpmd_handover, (x16,)
    )
    self.assertTrue(mpmd_rep.passed, msg=mpmd_rep.summary())
    self.assertEmpty(
        mpmd_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=mpmd_rep.summary(),
    )
    self.assertEmpty(
        mpmd_rep.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=mpmd_rep.summary(),
    )

  def test_negative_semaphore_count_rejected(self):
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def negative_signal_fn(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, sem):
        pl.semaphore_signal(sem, -1)
        o_ref[...] = x_ref[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
      )

    rep, _ = jaxpr_model.check_kernel_static_models(
        negative_signal_fn, (x_spec,)
    )
    self.assertFalse(rep.passed, msg=rep.summary())
    hw = rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=rep.summary())
    self.assertIn("-1", hw[0].message)

  def test_pl_when_predicate_ref_mutation_invalidates_complementary_coverage(
      self,
  ):
    def mutated_predicate_when(flag: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(s_ref, x_ref, o_ref):
        @pl.when(s_ref[0] <= 0)
        def _non_pos():
          o_ref[...] = x_ref[...]

        @pl.when(s_ref[0] > 0)
        def _clear_flag():
          # When `s_ref[0]` starts at 1, clearing it here causes the second
          # `s_ref[0] > 0` guard below to evaluate False, leaving `o_ref`
          # unwritten!
          s_ref[0] = jnp.int32(0)

        @pl.when(s_ref[0] > 0)
        def _pos_after_clear():
          o_ref[...] = x_ref[...] + 1.0

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[
              pl.BlockSpec(memory_space=pltpu.SMEM),
              pl.BlockSpec(x.shape, lambda: (0, 0)),
          ],
          out_specs=pl.BlockSpec(x.shape, lambda: (0, 0)),
      )(flag, x)

    flag_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        mutated_predicate_when, (flag_spec, x_spec)
    )
    self.assertFalse(rep.passed, msg=rep.summary())
    self.assertNotEmpty(
        rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=rep.summary(),
    )

  def test_inner_parallel_reduction_revisit_detected_on_non_megacore_tpu(self):
    x_spec = jax.ShapeDtypeStruct((64, 128), jnp.float32)

    def inner_parallel_reduction(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        @pl.when(pl.program_id(0) == 0)
        def _zero():
          o_ref[...] = jnp.zeros_like(o_ref)

        o_ref[...] += x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((16, 128), x.dtype),
          grid=(4, 2),
          in_specs=[pl.BlockSpec((8, 128), lambda k, i: (2 * k + i, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda k, i: (i, 0)),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("arbitrary", "parallel")
          ),
      )(x)

    def outer_parallel_reduction(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        @pl.when(pl.program_id(1) == 0)
        def _zero():
          o_ref[...] = jnp.zeros_like(o_ref)

        o_ref[...] += x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((16, 128), x.dtype),
          grid=(2, 4),
          in_specs=[pl.BlockSpec((8, 128), lambda i, k: (2 * k + i, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i, k: (i, 0)),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "arbitrary")
          ),
      )(x)

    bad_rep, _ = jaxpr_model.check_kernel_static_models(
        inner_parallel_reduction,
        (x_spec,),
        tpu_version=torch.tpu.TpuVersion.V7,
    )
    self.assertFalse(bad_rep.passed, msg=bad_rep.summary())
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=bad_rep.summary(),
    )

    ok_rep, _ = jaxpr_model.check_kernel_static_models(
        outer_parallel_reduction,
        (x_spec,),
        tpu_version=torch.tpu.TpuVersion.V7,
    )
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    self.assertEmpty(
        ok_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=ok_rep.summary(),
    )

  def test_concrete_tpu_mesh_overrides_conflicting_tpu_version_option(self):
    dev_cls = jax.sharding.AbstractDevice
    v5e_dev = (
        dev_cls("TPU v5e", 1, "tpu")
        if "platform" in inspect.signature(dev_cls).parameters
        else dev_cls("TPU v5e", 1)
    )
    v5e_abs_mesh = jax.sharding.AbstractMesh(
        (2,), ("rank",), abstract_device=v5e_dev
    )

    def v5e_regular_dma_jax(global_x: jax.Array) -> jax.Array:
      def per_shard(x_local: jax.Array) -> jax.Array:
        def kernel(x_hbm, o_ref, vmem, reg_sem):
          pltpu.make_async_copy(x_hbm, vmem, reg_sem).start()
          pl.semaphore_wait(reg_sem, 1)
          o_ref[...] = vmem[...]

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_local.shape, x_local.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[HBM_SPEC],
                out_specs=pl.BlockSpec(x_local.shape, lambda: (0, 0)),
                scratch_shapes=(
                    pltpu.VMEM(x_local.shape, x_local.dtype),
                    pltpu.SemaphoreType.REGULAR,
                ),
            ),
        )(x_local)

      return _shard_mapped(per_shard, v5e_abs_mesh)(global_x)

    v5e_op = jax_op(
        "pallas_verify_kernel::v5e_mesh_v7_option_conflict",
        v5e_regular_dma_jax,
        mesh=v5e_abs_mesh,
        input_partition_specs=(P("rank"),),
        output_partition_specs=(P("rank"),),
    )
    rep = verifier.verify(
        v5e_op,
        (torch.ones((8, 128), dtype=torch.float32),),
        mode=STATIC,
        options=verifier.VerificationOptions(
            tpu_version=torch.tpu.TpuVersion.V7
        ),
    )
    self.assertFalse(rep.passed, msg=rep.summary())
    self.assertNotEmpty(
        rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=rep.summary(),
    )
    override_notes = [
        f
        for f in rep.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.severity is verifier.Severity.INFO
        and "overrides VerificationOptions.tpu_version" in f.message
    ]
    self.assertLen(override_notes, 1, msg=rep.summary())

  def test_cross_phase_options_early_return_and_trace_error_dedup(self):
    # 1. Public `options.max_unrolled_iterations` plumbing emits
    # `VERIFICATION_INCOMPLETE` when bounded, without false `UNWRITTEN_OUTPUT`.
    x64_spec = jax.ShapeDtypeStruct((64, 128), jnp.float32)
    fori8 = _make_fori_copy(8)
    bounded_rep = verifier.verify(
        fori8,
        (x64_spec,),
        mode=STATIC,
        options=verifier.VerificationOptions(max_unrolled_iterations=4),
    )
    self.assertTrue(bounded_rep.passed, msg=bounded_rep.summary())
    self.assertNotEmpty(
        [
            f
            for f in bounded_rep.findings_by_category(
                Category.VERIFICATION_INCOMPLETE
            )
            if f.phase is verifier.Phase.MODEL_CHECK
        ],
        msg=bounded_rep.summary(),
    )
    unbounded_rep = verifier.verify(fori8, (x64_spec,), mode=STATIC)
    self.assertTrue(unbounded_rep.passed, msg=unbounded_rep.summary())
    self.assertEmpty(
        unbounded_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=unbounded_rep.summary(),
    )

    # 2. `check_autograd=False` skips `Phase.AUTOGRAD` on a buggy custom
    # backward while `check_autograd=True` catches it.
    def scale2_jax(x: jax.Array) -> jax.Array:
      return _single_block_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] * 2.0), x
      )

    buggy_bwd_op = jax_op(
        "pallas_verify_kernel::check_autograd_false_op", scale2_jax
    )
    buggy_bwd_op.register_autograd(lambda ctx, g: (g * 99.0,))
    x_grad = torch.ones((8, 128), dtype=torch.float32, requires_grad=True)
    skip_bwd_rep = verifier.verify(
        buggy_bwd_op,
        (x_grad,),
        options=verifier.VerificationOptions(check_autograd=False),
    )
    self.assertTrue(skip_bwd_rep.passed, msg=skip_bwd_rep.summary())
    self.assertNotIn(verifier.Phase.AUTOGRAD, skip_bwd_rep.checked_phases)

    run_bwd_rep = verifier.verify(buggy_bwd_op, (x_grad,))
    self.assertFalse(run_bwd_rep.passed, msg=run_bwd_rep.summary())
    self.assertNotEmpty(
        run_bwd_rep.findings_by_category(Category.AUTOGRAD_PARITY_MISMATCH),
        msg=run_bwd_rep.summary(),
    )

    # 3. `PARTITION_SPEC_MISMATCH` returns immediately after `Phase.CONTRACTS`.
    tp_mesh = jax.sharding.AbstractMesh((2,), ("tp",))
    bad_spec_op = jax_op(
        "pallas_verify_kernel::bad_spec_early_return",
        scale2_jax,
        mesh=tp_mesh,
        input_partition_specs=(P("unknown_axis"),),
        output_partition_specs=P("tp"),
    )
    bad_spec_op.register_autograd(lambda ctx, g: (g,))
    spec_rep = verifier.verify(bad_spec_op, (x_grad,))
    self.assertFalse(spec_rep.passed, msg=spec_rep.summary())
    self.assertNotEmpty(
        spec_rep.findings_by_category(Category.PARTITION_SPEC_MISMATCH),
        msg=spec_rep.summary(),
    )
    self.assertEqual(
        tuple(spec_rep.checked_phases), (verifier.Phase.CONTRACTS,)
    )

    # 4. `@jax_op` trace-time errors are deduplicated rather than reported twice
    # across `LOWERING` and `MODEL_CHECK`: classified trace errors skip
    # `LOWERING`, while unclassified trace errors drop the duplicate fallback
    # `MODEL_CHECK` warning once `LOWERING` reports `LOWERING_FAILURE`.
    def float_id_jax(global_x: jax.Array) -> jax.Array:
      def per_shard(s: jax.Array) -> jax.Array:
        def kernel(x_ref, y_ref, sem):
          pl.semaphore_signal(sem, 1, device_id={"tp": 0.5})
          y_ref[...] = x_ref[...]

        return _single_block_call(
            kernel, s, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
        )

      return jax.shard_map(
          per_shard,
          mesh=tp_mesh,
          in_specs=P("tp"),
          out_specs=P("tp"),
          check_vma=False,
      )(global_x)

    float_id_op = jax_op(
        "pallas_verify_kernel::float_device_id_dedup",
        float_id_jax,
        mesh=tp_mesh,
        input_partition_specs=(P("tp"),),
        output_partition_specs=P("tp"),
    )
    x_one = torch.ones((8, 128), dtype=torch.float32)
    classified_rep = verifier.verify(float_id_op, (x_one,), mode=STATIC)
    self.assertFalse(classified_rep.passed, msg=classified_rep.summary())
    self.assertLen(
        classified_rep.findings_by_severity(verifier.Severity.ERROR),
        1,
        msg=classified_rep.summary(),
    )
    self.assertNotIn(verifier.Phase.LOWERING, classified_rep.checked_phases)
    self.assertEmpty(
        classified_rep.findings_by_severity(verifier.Severity.WARNING),
        msg=classified_rep.summary(),
    )

    def dup_axis_key_jax(global_x: jax.Array) -> jax.Array:
      def per_shard(s: jax.Array) -> jax.Array:
        def kernel(x_ref, y_ref, sem):
          pl.semaphore_signal(sem, 1, device_id={"tp": 0, ("tp",): 0})
          y_ref[...] = x_ref[...]

        return _single_block_call(
            kernel, s, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
        )

      return jax.shard_map(
          per_shard,
          mesh=tp_mesh,
          in_specs=P("tp"),
          out_specs=P("tp"),
          check_vma=False,
      )(global_x)

    dup_key_op = jax_op(
        "pallas_verify_kernel::dup_axis_key_dedup",
        dup_axis_key_jax,
        mesh=tp_mesh,
        input_partition_specs=(P("tp"),),
        output_partition_specs=P("tp"),
    )
    dedup_rep = verifier.verify(dup_key_op, (x_one,), mode=STATIC)
    self.assertFalse(dedup_rep.passed, msg=dedup_rep.summary())
    self.assertLen(
        dedup_rep.findings_by_severity(verifier.Severity.ERROR),
        1,
        msg=dedup_rep.summary(),
    )
    self.assertEmpty(
        dedup_rep.findings_by_severity(verifier.Severity.WARNING),
        msg=dedup_rep.summary(),
    )

    # 5. Pure non-Pallas raw JAX function skips `Phase.LOWERING` and notes
    # `VERIFICATION_INCOMPLETE` (`INFO`) in `Phase.MODEL_CHECK`.
    pure_jax_rep = verifier.verify(lambda a: a + 1.0, (x64_spec,), mode=STATIC)
    self.assertTrue(pure_jax_rep.passed, msg=pure_jax_rep.summary())
    self.assertNotIn(verifier.Phase.LOWERING, pure_jax_rep.checked_phases)
    self.assertNotEmpty(
        pure_jax_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=pure_jax_rep.summary(),
    )


if __name__ == "__main__":
  absltest.main()
