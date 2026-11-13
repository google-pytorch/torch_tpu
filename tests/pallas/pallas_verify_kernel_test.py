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

"""Tests for Pallas kernel static analysis and concurrency model checking."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import threading
import types
from typing import NamedTuple
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
from torch_tpu._internal.pallas.verify import interpret_verifier
from torch_tpu._internal.pallas.verify import jaxpr_model
from torch_tpu._internal.pallas.verify import smt_checker
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


def _shared_sem_race_jax(x: jax.Array) -> jax.Array:
  """Litmus test: two concurrent DMAs that share one semaphore race."""

  def kernel(x_hbm_ref, y_ref):
    def body(buf0, buf1, shared_sem):
      c0 = pltpu.make_async_copy(x_hbm_ref.at[0:8], buf0, shared_sem)
      c1 = pltpu.make_async_copy(x_hbm_ref.at[8:16], buf1, shared_sem)
      c0.start()
      c1.start()
      # Under unordered TPU DMA execution `c1` can finish first and satisfy
      # `c0.wait()` while `c0` is still writing `buf0`.
      c0.wait()
      y_ref.at[0:8][...] = buf0[...] + 1.0
      c1.wait()
      y_ref.at[8:16][...] = buf1[...] + 1.0

    pl.run_scoped(
        body,
        pltpu.VMEM((8, 128), x.dtype),
        pltpu.VMEM((8, 128), x.dtype),
        pltpu.SemaphoreType.DMA,
    )

  return pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      in_specs=[HBM_SPEC],
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

  # ==========================================================================
  # Concurrency, Semaphore Conservation, and CP-SAT Schedule Model Checking
  # ==========================================================================

  def test_valid_double_buffered_async_dma_pipeline_passes_all_phases(self):
    """Disjoint double-buffer slots with separate semaphores have no FPs."""
    num_tiles = 4

    def double_buffered_copy(x: jax.Array) -> jax.Array:

      def kernel(x_hbm_ref, y_hbm_ref):

        def body(vmem_bufs, sems):
          def in_copy(step):
            return pltpu.make_async_copy(
                x_hbm_ref.at[pl.ds(step * 8, 8)],
                vmem_bufs.at[step % 2],
                sems.at[step % 2],
            )

          in_copy(0).start()
          for step in range(num_tiles):
            if step + 1 < num_tiles:
              # Prefetch the next tile into the other slot.
              in_copy(step + 1).start()
            in_copy(step).wait()
            slot = vmem_bufs.at[step % 2]
            slot[...] = slot[...] * 2.0 + 1.0
            out_copy = pltpu.make_async_copy(
                slot, y_hbm_ref.at[pl.ds(step * 8, 8)], sems.at[2 + step % 2]
            )
            out_copy.start()
            out_copy.wait()

        pl.run_scoped(
            body,
            pltpu.VMEM((2, 8, 128), x.dtype),
            pltpu.SemaphoreType.DMA((4,)),
        )

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=HBM_SPEC,
      )(x)

    op = jax_op(
        "pallas_verify_kernel::valid_double_buffer", double_buffered_copy
    )
    x = torch.arange(num_tiles * 8 * 128, dtype=torch.float32).reshape(-1, 128)
    report = verifier.verify(op, (x,))
    self.assertTrue(report.passed, msg=report.summary())
    self.assertContainsSubset(
        {
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
            verifier.Phase.MODEL_CHECK,
            verifier.Phase.INTERPRET,
        },
        set(report.checked_phases),
    )

  def test_unordered_dma_shared_semaphore_race_caught_with_counterexample(self):
    op = jax_op(
        "pallas_verify_kernel::buggy_shared_sem_dma", _shared_sem_race_jax
    )
    x = torch.ones((16, 128), dtype=torch.float32)
    report = verifier.verify(op, (x,), mode=STATIC)
    self.assertFalse(report.passed, msg=report.summary())
    race_findings = report.findings_by_category(
        Category.CONCURRENT_DMA_SEM_SHARING
    )
    self.assertNotEmpty(race_findings, msg=report.summary())
    trace = race_findings[0].counterexample
    self.assertNotEmpty(trace)
    self.assertEqual(
        [step.timestamp for step in trace],
        sorted(step.timestamp for step in trace),
    )
    self.assertTrue(any("dma" in str(step) for step in trace))

  def test_unthrottled_multichip_pipelined_rdma_caught_and_handshake_fixes(
      self,
  ):
    """Push-only 2-step RDMA overwrites neighbor buffer unless throttled."""
    mesh = _make_mock_tpu_mesh(4, "rank")

    def make_pipelined_ring(*, use_capacity_handshake: bool):
      def pipelined_ring_jax(
          global_x: jax.Array, *, mesh: jax.sharding.Mesh
      ) -> jax.Array:
        def pallas_body(
            x_ref, y_ref, recv_buf, send_sem, recv_sem, capacity_sem
        ):
          my_rank = lax.axis_index("rank")
          next_rank = lax.rem(my_rank + 1, 4)
          prev_rank = lax.rem(my_rank + 3, 4)

          barrier_sem = pltpu.get_barrier_semaphore()
          pl.semaphore_signal(barrier_sem, 1, device_id={"rank": prev_rank})
          pl.semaphore_signal(barrier_sem, 1, device_id={"rank": next_rank})
          pl.semaphore_wait(barrier_sem, 2)

          for step in range(2):
            copy = pltpu.make_async_remote_copy(
                src_ref=x_ref.at[step],
                dst_ref=recv_buf,
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id={"rank": next_rank},
            )
            copy.start()
            copy.wait()
            y_ref.at[step][...] = recv_buf[...] * (step + 2.0)
            if use_capacity_handshake and step == 0:
              # Tell `prev_rank` that `recv_buf` was consumed, and wait until
              # `next_rank` consumed ours before pushing into it again.
              pl.semaphore_signal(
                  capacity_sem, 1, device_id={"rank": prev_rank}
              )
              pl.semaphore_wait(capacity_sem, 1)

        def per_shard(x_local):
          return _single_block_call(
              pallas_body,
              x_local,
              scratch_shapes=(
                  pltpu.VMEM(x_local.shape[1:], x_local.dtype),
                  pltpu.SemaphoreType.DMA,
                  pltpu.SemaphoreType.DMA,
                  pltpu.SemaphoreType.REGULAR,
              ),
              collective_id=2,
          )

        return _shard_mapped(per_shard, mesh)(global_x)

      return pipelined_ring_jax

    buggy_op = _distributed_op(
        "pallas_verify_kernel::unthrottled_rdma_ring",
        make_pipelined_ring(use_capacity_handshake=False),
        mesh,
    )
    fixed_op = _distributed_op(
        "pallas_verify_kernel::throttled_rdma_ring",
        make_pipelined_ring(use_capacity_handshake=True),
        mesh,
    )
    x_local = torch.ones((2, 8, 128), dtype=torch.float32)

    buggy_rep = verifier.verify(buggy_op, (x_local,), mode=STATIC)
    self.assertFalse(buggy_rep.passed, msg=buggy_rep.summary())
    unthrottled = buggy_rep.findings_by_category(
        Category.UNTHROTTLED_REMOTE_OVERWRITE
    )
    self.assertNotEmpty(unthrottled, msg=buggy_rep.summary())
    self.assertNotEmpty(unthrottled[0].counterexample)

    fixed_rep = verifier.verify(fixed_op, (x_local,))
    self.assertTrue(fixed_rep.passed, msg=fixed_rep.summary())
    self.assertEmpty(
        fixed_rep.findings_by_severity(verifier.Severity.WARNING),
        msg=fixed_rep.summary(),
    )
    self.assertIn(verifier.Phase.INTERPRET, fixed_rep.checked_phases)

  def test_semaphore_nonzero_exit_and_deadlock_detected_statically(self):
    """Unawaited DMA -> SEMAPHORE_NONZERO_EXIT; over-wait/cycle -> DEADLOCK."""

    def leaky_sem_jax(x: jax.Array) -> jax.Array:
      def kernel(x_hbm_ref, y_ref, vmem_buf, sem):
        # Starts an async copy into `vmem_buf` but never waits for it.
        pltpu.make_async_copy(x_hbm_ref, vmem_buf, sem).start()
        y_ref[...] = jnp.zeros_like(y_ref)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec(x.shape),
              scratch_shapes=(
                  pltpu.VMEM(x.shape, x.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(x)

    def deadlocking_sem_jax(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_signal(sem, 1)
        # Waits for 2 signals when only 1 ever arrives.
        pl.semaphore_wait(sem, 2)
        y_ref[...] = x_ref[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
      )

    mesh2 = _make_mock_tpu_mesh(2, "rank")

    def cyclic_multichip_deadlock_jax(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        peer = lax.rem(lax.axis_index("rank") + 1, 2)
        # Both chips wait before signaling their peer: the semaphore counts are
        # balanced, yet the causal cycle deadlocks.
        pl.semaphore_wait(sem, 1)
        pl.semaphore_signal(sem, 1, device_id={"rank": peer})
        y_ref[...] = x_ref[...]

      def per_shard(x_local):
        return _single_block_call(
            kernel, x_local, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
        )

      return _shard_mapped(per_shard, mesh)(global_x)

    def asym_misrouted_sem_jax(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        # Both ranks signal rank 0's semaphore: rank 0 exits with +1 and
        # rank 1 waits forever.
        pl.semaphore_signal(sem, 1, device_id={"rank": 0})
        pl.semaphore_wait(sem, 1)
        y_ref[...] = x_ref[...]

      def per_shard(x_local):
        return _single_block_call(
            kernel, x_local, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
        )

      return _shard_mapped(per_shard, mesh)(global_x)

    x = torch.ones((8, 128), dtype=torch.float32)

    leak_op = jax_op("pallas_verify_kernel::leaky_sem_op", leaky_sem_jax)
    leak_rep = verifier.verify(leak_op, (x,), mode=STATIC)
    self.assertFalse(leak_rep.passed)
    self.assertNotEmpty(
        leak_rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=leak_rep.summary(),
    )

    deadlock_op = jax_op(
        "pallas_verify_kernel::deadlock_sem_op", deadlocking_sem_jax
    )
    deadlock_rep = verifier.verify(deadlock_op, (x,), mode=STATIC)
    self.assertFalse(deadlock_rep.passed)
    self.assertNotEmpty(
        deadlock_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=deadlock_rep.summary(),
    )

    cyclic_op = _distributed_op(
        "pallas_verify_kernel::cyclic_deadlock_sem_op",
        cyclic_multichip_deadlock_jax,
        mesh2,
    )
    cyclic_rep = verifier.verify(cyclic_op, (x,), mode=STATIC)
    self.assertFalse(cyclic_rep.passed)
    self.assertNotEmpty(
        cyclic_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=cyclic_rep.summary(),
    )

    # FULL mode must report both findings without hanging on the deadlocked
    # rank.
    asym_op = _distributed_op(
        "pallas_verify_kernel::asym_misrouted_sem_op",
        asym_misrouted_sem_jax,
        mesh2,
    )
    asym_rep = verifier.verify(asym_op, (x,))
    self.assertFalse(asym_rep.passed, msg=asym_rep.summary())
    self.assertNotEmpty(
        asym_rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=asym_rep.summary(),
    )
    self.assertNotEmpty(
        asym_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=asym_rep.summary(),
    )
    self.assertNotIn(verifier.Phase.INTERPRET, asym_rep.checked_phases)

  def test_asymmetric_barrier_participation_deadlock_detected(self):
    mesh2 = _make_mock_tpu_mesh(2, "rank")

    def make_barrier(*, every_rank_signals: bool):
      def barrier_jax(
          global_x: jax.Array, *, mesh: jax.sharding.Mesh
      ) -> jax.Array:
        def kernel(x_ref, y_ref):
          rank = lax.axis_index("rank")
          barrier = pltpu.get_barrier_semaphore()

          @pl.when(jnp.logical_or(every_rank_signals, rank == 0))
          def _():
            pl.semaphore_signal(barrier, 1, device_id={"rank": 1 - rank})

          pl.semaphore_wait(barrier, 1)
          y_ref[...] = x_ref[...]

        def per_shard(x_local):
          return _single_block_call(kernel, x_local, collective_id=4)

        return _shard_mapped(per_shard, mesh)(global_x)

      return barrier_jax

    x = torch.ones((8, 128), dtype=torch.float32)
    bad_op = _distributed_op(
        "pallas_verify_kernel::asymmetric_barrier",
        make_barrier(every_rank_signals=False),
        mesh2,
    )
    bad_rep = verifier.verify(bad_op, (x,), mode=STATIC)
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=bad_rep.summary(),
    )

    ok_op = _distributed_op(
        "pallas_verify_kernel::symmetric_barrier",
        make_barrier(every_rank_signals=True),
        mesh2,
    )
    ok_rep = verifier.verify(ok_op, (x,), mode=STATIC)
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())

  def test_dma_source_overwritten_before_wait_detected(self):
    def make_store(*, overwrite_before_wait: bool):
      def store_jax(x: jax.Array) -> jax.Array:
        def kernel(x_ref, y_hbm_ref, buf, sem):
          buf[...] = x_ref[...]
          copy = pltpu.make_async_copy(buf, y_hbm_ref, sem)
          copy.start()
          if overwrite_before_wait:
            # The DMA may still be reading `buf`.
            buf[...] = jnp.zeros_like(buf)
          copy.wait()
          if not overwrite_before_wait:
            buf[...] = jnp.zeros_like(buf)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(x.shape)],
                out_specs=HBM_SPEC,
                scratch_shapes=(
                    pltpu.VMEM(x.shape, x.dtype),
                    pltpu.SemaphoreType.DMA,
                ),
            ),
        )(x)

      return store_jax

    x = torch.ones((8, 128), dtype=torch.float32)
    bad_op = jax_op(
        "pallas_verify_kernel::dma_src_overwrite",
        make_store(overwrite_before_wait=True),
    )
    bad_rep = verifier.verify(bad_op, (x,), mode=STATIC)
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.DATA_RACE), msg=bad_rep.summary()
    )

    ok_op = jax_op(
        "pallas_verify_kernel::dma_src_after_wait",
        make_store(overwrite_before_wait=False),
    )
    ok_rep = verifier.verify(ok_op, (x,), mode=STATIC)
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())

  def test_parallel_grid_overlapping_write_conflict_detected(self):
    """Reduction over a grid axis marked `parallel` instead of `arbitrary`."""

    def make_grid_reduction(semantics: str):

      def grid_reduce(x: jax.Array) -> jax.Array:

        def kernel(x_ref, out_ref):
          @pl.when(pl.program_id(0) == 0)
          def _():
            out_ref[...] = jnp.zeros_like(out_ref)

          out_ref[...] += x_ref[...]

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
            grid=(x.shape[0],),
            in_specs=[pl.BlockSpec((None, 8, 128), lambda i: (i, 0, 0))],
            # Every grid iteration maps to the same output tile.
            out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(semantics,)
            ),
        )(x)

      return grid_reduce

    x = torch.ones((4, 8, 128), dtype=torch.float32)
    parallel_bug_op = jax_op(
        "pallas_verify_kernel::parallel_grid_bug",
        make_grid_reduction("parallel"),
    )
    arbitrary_ok_op = jax_op(
        "pallas_verify_kernel::arbitrary_grid_ok",
        make_grid_reduction("arbitrary"),
    )

    bug_rep = verifier.verify(parallel_bug_op, (x,), mode=STATIC)
    self.assertFalse(bug_rep.passed, msg=bug_rep.summary())
    self.assertNotEmpty(
        bug_rep.findings_by_category(Category.PARALLEL_GRID_CONFLICT),
        msg=bug_rep.summary(),
    )

    ok_rep = verifier.verify(arbitrary_ok_op, (x,))
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())

  def test_truncated_semaphore_traffic_is_not_reported_as_errors(self):
    """Correct cross-step prefetching past the unrolling bound passes."""

    def make_cross_step_prefetch(grid: int):
      def prefetch(x: jax.Array) -> jax.Array:
        def kernel(x_hbm, o_ref, buf, sem):
          i = pl.program_id(0)

          def copy(step, slot):
            return pltpu.make_async_copy(
                x_hbm.at[pl.ds(step * 8, 8)], buf.at[slot], sem.at[slot]
            )

          @pl.when(i == 0)
          def _():
            copy(0, 0).start()

          @pl.when(i + 1 < pl.num_programs(0))
          def _():
            copy(i + 1, lax.rem(i + 1, 2)).start()

          slot = lax.rem(i, 2)
          copy(i, slot).wait()
          o_ref[...] = buf[slot] + 1.0

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid,),
                in_specs=[HBM_SPEC],
                out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
                scratch_shapes=(
                    pltpu.VMEM((2, 8, 128), jnp.float32),
                    pltpu.SemaphoreType.DMA((2,)),
                ),
            ),
        )(x)

      return prefetch

    for grid in (8, 40):  # 40 exceeds the default unrolling bound.
      x_spec = jax.ShapeDtypeStruct((grid * 8, 128), jnp.float32)
      report = verifier.verify(
          make_cross_step_prefetch(grid), (x_spec,), mode=STATIC
      )
      self.assertTrue(report.passed, msg=report.summary())

  def test_bugs_are_still_found_past_the_unrolling_bound(self):
    grid = 40  # Exceeds the default unrolling bound.

    def make_per_step_copy(*, leak: bool):
      def per_step_copy(x: jax.Array) -> jax.Array:
        def kernel(x_hbm, o_ref, buf, sems):
          i = pl.program_id(0)
          copy = pltpu.make_async_copy(
              x_hbm.at[pl.ds(i * 8, 8)], buf, sems.at[i if leak else 0]
          )
          copy.start()
          if leak:
            o_ref[...] = jnp.zeros_like(o_ref)  # Never waits for `copy`.
          else:
            o_ref[...] = buf[...]  # Reads `buf` while `copy` writes it.
            copy.wait()

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(grid,),
                in_specs=[HBM_SPEC],
                out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
                scratch_shapes=(
                    pltpu.VMEM((8, 128), jnp.float32),
                    pltpu.SemaphoreType.DMA((grid,)),
                ),
            ),
        )(x)

      return per_step_copy

    x_spec = jax.ShapeDtypeStruct((grid * 8, 128), jnp.float32)
    leak_rep = verifier.verify(
        make_per_step_copy(leak=True), (x_spec,), mode=STATIC
    )
    self.assertNotEmpty(
        leak_rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=leak_rep.summary(),
    )
    race_rep = verifier.verify(
        make_per_step_copy(leak=False), (x_spec,), mode=STATIC
    )
    self.assertNotEmpty(
        race_rep.findings_by_category(Category.DATA_RACE),
        msg=race_rep.summary(),
    )

  def test_data_dependent_semaphore_operations_skip_sync_checks(self):
    def branch_dma(flag: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(flag_ref, x_hbm, o_ref, buf, sem):
        def from_a():
          pltpu.make_async_copy(x_hbm.at[0:8], buf, sem).start()

        def from_b():
          pltpu.make_async_copy(x_hbm.at[8:16], buf, sem).start()

        # Exactly one branch runs, so the single wait balances the semaphore.
        lax.cond(flag_ref[0] > 0, from_a, from_b)
        pltpu.make_async_copy(x_hbm.at[0:8], buf, sem).wait()
        o_ref[...] = buf[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec((8, 128), lambda *_: (0, 0)),
              scratch_shapes=(
                  pltpu.VMEM((8, 128), jnp.float32),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(flag, x)

    report = verifier.verify(
        branch_dma,
        (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((16, 128), jnp.float32),
        ),
        mode=STATIC,
    )
    self.assertTrue(report.passed, msg=report.summary())
    self.assertTrue(
        any(
            f.category is Category.VERIFICATION_INCOMPLETE
            for f in report.findings_by_severity(verifier.Severity.WARNING)
        ),
        msg=report.summary(),
    )

  def test_parallel_grid_iterations_have_their_own_scratch_semaphores(self):
    def make_per_step_dma(semantics: str):
      def per_step_dma(x: jax.Array) -> jax.Array:
        def kernel(x_hbm, o_ref, buf, sem):
          copy = pltpu.make_async_copy(
              x_hbm.at[pl.ds(pl.program_id(0) * 8, 8)], buf, sem
          )
          copy.start()
          copy.wait()
          o_ref[...] = buf[...] + 1.0

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(4,),
                in_specs=[HBM_SPEC],
                out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
                scratch_shapes=(
                    pltpu.VMEM((8, 128), jnp.float32),
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(semantics,)
            ),
        )(x)

      return per_step_dma

    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    for semantics in ("arbitrary", "parallel"):
      report = verifier.verify(
          make_per_step_dma(semantics), (x_spec,), mode=STATIC
      )
      self.assertTrue(report.passed, msg=f"{semantics}: {report.summary()}")

  def test_solver_timeout_is_reported_as_incomplete(self):
    op = jax_op(
        "pallas_verify_kernel::timeout_shared_sem", _shared_sem_race_jax
    )
    report = verifier.verify(
        op,
        (torch.ones((16, 128)),),
        mode=STATIC,
        options=verifier.VerificationOptions(solver_timeout_seconds=0.0),
    )
    incomplete = report.findings_by_category(Category.VERIFICATION_INCOMPLETE)
    self.assertNotEmpty(incomplete, msg=report.summary())
    self.assertEqual(incomplete[0].severity, verifier.Severity.WARNING)

  def test_torch_program_checks_each_dtype_specialization(self):
    """The same op called on f32 and bf16 inputs is checked per dtype."""

    def dtype_dependent_sem_jax(inp: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_signal(sem, 1)
        if x_ref.dtype == jnp.float32:
          pl.semaphore_wait(sem, 1)
        y_ref[...] = x_ref[...]

      return _single_block_call(
          kernel, inp, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
      )

    dtype_sem_op = jax_op(
        "pallas_verify_kernel::dtype_dependent_sem", dtype_dependent_sem_jax
    )

    def mixed_dtype_caller(x_f32: torch.Tensor) -> torch.Tensor:
      y_f32 = dtype_sem_op(x_f32)
      y_bf16 = dtype_sem_op(x_f32.to(torch.bfloat16))
      return y_f32 + y_bf16.to(torch.float32)

    report = verifier.verify(
        mixed_dtype_caller,
        (torch.ones((8, 128), dtype=torch.float32),),
        mode=STATIC,
    )
    self.assertFalse(report.passed, msg=report.summary())
    self.assertNotEmpty(
        report.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=report.summary(),
    )

  def test_runtime_dynamic_slice_dma_batch_reports_incomplete_not_false_race(
      self,
  ):
    """Runtime-indexed DMA batches report incomplete (INFO), not false races."""

    def batched_runtime_dslice_dma(
        offsets: jax.Array, x: jax.Array, out_buf: jax.Array
    ) -> jax.Array:
      def kernel(offsets_ref, x_hbm, out_hbm, _, scratch_vmem, dma_sem):
        in_copies = [
            pltpu.make_async_copy(
                x_hbm.at[pl.ds(i * 8, 8)], scratch_vmem.at[i], dma_sem
            )
            for i in range(2)
        ]
        for c in in_copies:
          c.start()
        for c in in_copies:
          c.wait()

        out_copies = [
            pltpu.make_async_copy(
                scratch_vmem.at[i],
                out_hbm.at[pl.ds(offsets_ref[i], 8)],
                dma_sem,
            )
            for i in range(2)
        ]
        for c in out_copies:
          c.start()
        for c in out_copies:
          c.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(out_buf.shape, out_buf.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[HBM_SPEC, HBM_SPEC],
              out_specs=HBM_SPEC,
              scratch_shapes=(
                  pltpu.VMEM((2, 8, 128), x.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
          input_output_aliases={2: 0},
      )(offsets, x, out_buf)

    def premature_scratch_reuse_with_runtime_dslice(
        offsets: jax.Array, x: jax.Array, out_buf: jax.Array
    ) -> jax.Array:
      def kernel(offsets_ref, x_hbm, out_hbm, _, scratch_vmem, dma_sem):
        scratch_vmem.at[0][...] = jnp.ones((8, 128), x_hbm.dtype)
        scratch_vmem.at[1][...] = jnp.ones((8, 128), x_hbm.dtype)
        c0 = pltpu.make_async_copy(
            scratch_vmem.at[0], out_hbm.at[pl.ds(offsets_ref[0], 8)], dma_sem
        )
        c1 = pltpu.make_async_copy(
            scratch_vmem.at[1], out_hbm.at[pl.ds(offsets_ref[1], 8)], dma_sem
        )
        c0.start()
        c1.start()
        # Bug: c1 can satisfy c0.wait() while c0 is still reading scratch[0].
        c0.wait()
        scratch_vmem.at[0][...] = jnp.zeros((8, 128), x_hbm.dtype)
        c1.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(out_buf.shape, out_buf.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[HBM_SPEC, HBM_SPEC],
              out_specs=HBM_SPEC,
              scratch_shapes=(
                  pltpu.VMEM((2, 8, 128), x.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
          input_output_aliases={2: 0},
      )(offsets, x, out_buf)

    def statically_disjoint_outer_axis_with_runtime_inner_slice(
        offsets: jax.Array, out_buf: jax.Array
    ) -> jax.Array:
      def kernel(offsets_ref, out_hbm, _, scratch_vmem, dma_sem):
        scratch_vmem.at[0][...] = jnp.ones((8, 128), out_hbm.dtype)
        scratch_vmem.at[1][...] = jnp.ones((8, 128), out_hbm.dtype)
        c0 = pltpu.make_async_copy(
            scratch_vmem.at[0],
            out_hbm.at[0, pl.ds(offsets_ref[0], 8)],
            dma_sem,
        )
        c1 = pltpu.make_async_copy(
            scratch_vmem.at[1],
            out_hbm.at[1, pl.ds(offsets_ref[1], 8)],
            dma_sem,
        )
        c0.start()
        c1.start()
        c0.wait()
        c1.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(out_buf.shape, out_buf.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[HBM_SPEC],
              out_specs=HBM_SPEC,
              scratch_shapes=(
                  pltpu.VMEM((2, 8, 128), out_buf.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
          input_output_aliases={1: 0},
      )(offsets, out_buf)

    offsets = torch.tensor([0, 8], dtype=torch.int32)
    x = torch.ones((16, 128), dtype=torch.float32)
    out_buf = torch.zeros((16, 128), dtype=torch.float32)

    batch_op = jax_op(
        "pallas_verify_kernel::batched_runtime_dslice_dma",
        batched_runtime_dslice_dma,
        donate_argnums=(2,),
    )
    batch_rep = verifier.verify(
        batch_op, (offsets, x, out_buf.clone()), mode=STATIC
    )
    self.assertTrue(batch_rep.passed, msg=batch_rep.summary())
    incomplete = [
        f
        for f in batch_rep.findings_by_category(
            Category.VERIFICATION_INCOMPLETE
        )
        if "runtime" in f.message and "slice" in f.message
    ]
    self.assertNotEmpty(incomplete, msg=batch_rep.summary())
    self.assertEqual(incomplete[0].severity, verifier.Severity.INFO)

    buggy_op = jax_op(
        "pallas_verify_kernel::premature_scratch_reuse_runtime_dslice",
        premature_scratch_reuse_with_runtime_dslice,
        donate_argnums=(2,),
    )
    buggy_rep = verifier.verify(
        buggy_op, (offsets, x, out_buf.clone()), mode=STATIC
    )
    self.assertFalse(buggy_rep.passed, msg=buggy_rep.summary())
    self.assertNotEmpty(
        buggy_rep.findings_by_category(Category.CONCURRENT_DMA_SEM_SHARING),
        msg=buggy_rep.summary(),
    )

    disjoint_op = jax_op(
        "pallas_verify_kernel::disjoint_outer_runtime_inner_dslice",
        statically_disjoint_outer_axis_with_runtime_inner_slice,
        donate_argnums=(1,),
    )
    out_buf_3d = torch.zeros((2, 16, 128), dtype=torch.float32)
    disjoint_rep = verifier.verify(
        disjoint_op, (offsets, out_buf_3d), mode=STATIC
    )
    self.assertTrue(disjoint_rep.passed, msg=disjoint_rep.summary())
    self.assertEmpty(
        disjoint_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=disjoint_rep.summary(),
    )

  def test_dynamic_slice_dma_byte_count_extracted(self):
    def dyn_slice_dma(idx: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(idx_ref, x_hbm, y_hbm, vmem_buf, dma_sem):
        start = idx_ref[0] * 8
        copy_in = pltpu.make_async_copy(
            x_hbm.at[pl.ds(start, 8), :], vmem_buf, dma_sem
        )
        copy_in.start()
        copy_in.wait()
        copy_out = pltpu.make_async_copy(
            vmem_buf, y_hbm.at[pl.ds(0, 8), :], dma_sem
        )
        copy_out.start()
        copy_out.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[HBM_SPEC],
              out_specs=HBM_SPEC,
              scratch_shapes=(
                  pltpu.VMEM((8, 128), x.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(idx, x)

    idx_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    _, models = jaxpr_model.check_kernel_static_models(
        dyn_slice_dma, (idx_spec, x_spec)
    )
    assert models is not None
    self.assertLen(models, 1)
    expected_bytes = 8 * 128 * 4
    self.assertEqual(models[0].dma_starts[0].byte_count, expected_bytes)
    self.assertEqual(models[0].dma_waits[0].byte_count, expected_bytes)

  def test_truncated_sync_loop_halts_epilogue_event_recording(self):
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def sync_sem_loop_epilogue(x: jax.Array) -> jax.Array:
      def kernel(x_vmem, o_vmem, vmem_scratch, sem):
        def body(_, carry):
          pl.semaphore_signal(sem, 1)
          pl.semaphore_wait(sem, 1)
          return carry

        _ = lax.fori_loop(0, 40, body, jnp.int32(0))
        # Concurrency event recording on this actor halts after the truncated
        # sync loop while coverage and OOB checks continue.
        vmem_scratch[...] = jnp.zeros_like(vmem_scratch[...])
        o_vmem[...] = x_vmem[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
          scratch_shapes=[
              pltpu.VMEM((8, 128), x.dtype),
              pltpu.SemaphoreType.REGULAR,
          ],
          grid=(1,),
      )(x)

    def sync_dma_loop_epilogue(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_hbm, vmem_scratch, dma_sem):
        def body(_, carry):
          cin = pltpu.make_async_copy(x_hbm, vmem_scratch, dma_sem)
          cin.start()
          cin.wait()
          return carry

        _ = lax.fori_loop(0, 40, body, jnp.int32(0))
        # Halting event recording prevents splicing this epilogue read/write
        # onto an un-waited loop prefix.
        vmem_scratch[...] = vmem_scratch[...] + 1.0
        cout = pltpu.make_async_copy(vmem_scratch, o_hbm, dma_sem)
        cout.start()
        cout.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=HBM_SPEC,
          scratch_shapes=[
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.SemaphoreType.DMA,
          ],
      )(x)

    for label, fn in (
        ("sem_loop", sync_sem_loop_epilogue),
        ("dma_loop", sync_dma_loop_epilogue),
    ):
      with self.subTest(kind=label):
        rep, models = smt_checker.check_kernel_concurrency_with_models(
            fn,
            (x_spec,),
            config=smt_checker.ModelCheckConfig(max_unrolled_iterations=4),
        )
        self.assertTrue(rep.passed, msg=rep.summary())
        self.assertEmpty(
            rep.findings_by_category(Category.DATA_RACE),
            msg=rep.summary(),
        )
        assert models is not None
        self.assertLen(models, 1)
        self.assertTrue(models[0].sync_truncated)
        if label == "sem_loop":
          vmem_writes = [
              ev
              for ev in models[0].memory_events
              if ev.buffer.buffer_id == "scratch_0"
          ]
          self.assertEmpty(vmem_writes)

  def test_strided_slice_and_bitcast_preserve_exact_dma_byte_count(self):
    def strided_and_bitcast_dma(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_hbm, vmem_a, vmem_b, sem):
        # Strided view `(16, 128)[::2, :]` has view_shape `(8, 128)` -> 4096B.
        copy1 = pltpu.make_async_copy(x_hbm.at[::2, :], vmem_a, sem)
        copy1.start()
        copy1.wait()
        # Bitcast `(8, 128)` float32 -> `(8, 256)` int16 -> still 4096B.
        copy2 = pltpu.make_async_copy(
            vmem_a.bitcast(jnp.int16), vmem_b.bitcast(jnp.int16), sem
        )
        copy2.start()
        copy2.wait()
        copy3 = pltpu.make_async_copy(vmem_b, o_hbm.at[:8, :], sem)
        copy3.start()
        copy3.wait()
        copy4 = pltpu.make_async_copy(vmem_b, o_hbm.at[8:, :], sem)
        copy4.start()
        copy4.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=HBM_SPEC,
          scratch_shapes=[
              pltpu.VMEM((8, 128), x.dtype),
              pltpu.VMEM((8, 128), x.dtype),
              pltpu.SemaphoreType.DMA,
          ],
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep, models = jaxpr_model.check_kernel_static_models(
        strided_and_bitcast_dma, (x_spec,)
    )
    self.assertTrue(rep.passed, msg=rep.summary())
    self.assertEqual(
        [d.byte_count for d in models[0].dma_starts], [4096, 4096, 4096, 4096]
    )

  def test_runtime_conditional_output_write_and_multi_axis_parallel_conflict(
      self,
  ):
    def one_branch_unwritten(flag: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(flag_ref, x_ref, o_ref):
        lax.cond(
            flag_ref[0] > 0,
            lambda: o_ref.__setitem__(Ellipsis, x_ref[...]),
            lambda: None,
        )

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec(x.shape, lambda _: (0, 0))],
              out_specs=pl.BlockSpec(x.shape, lambda _: (0, 0)),
          ),
      )(flag, x)

    flag = torch.tensor([1], dtype=torch.int32)
    x = torch.ones((8, 128), dtype=torch.float32)
    cond_op = jax_op(
        "pallas_verify_kernel::one_branch_unwritten", one_branch_unwritten
    )
    cond_rep = verifier.verify(cond_op, (flag, x), mode=STATIC)
    self.assertFalse(cond_rep.passed, msg=cond_rep.summary())
    self.assertNotEmpty(
        cond_rep.findings_by_category(Category.UNWRITTEN_OUTPUT),
        msg=cond_rep.summary(),
    )

    # 2D `(parallel, arbitrary)` grid of shape `(2, 40)` where the inner
    # sequential axis exceeds `max_unrolled_iterations=32` must still sample
    # both parallel coordinates and detect the cross-parallel-coordinate race.
    def multi_axis_parallel_race(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(2, 40),
          in_specs=[pl.BlockSpec(x.shape, lambda _p, _s: (0, 0))],
          out_specs=pl.BlockSpec(x.shape, lambda _p, _s: (0, 0)),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel", "arbitrary")
          ),
      )(x)

    par_rep = verifier.verify(
        multi_axis_parallel_race,
        (jax.ShapeDtypeStruct((8, 128), jnp.float32),),
        mode=STATIC,
    )
    self.assertFalse(par_rep.passed, msg=par_rep.summary())
    self.assertNotEmpty(
        par_rep.findings_by_category(Category.PARALLEL_GRID_CONFLICT),
        msg=par_rep.summary(),
    )

  def test_shared_dma_semaphore_unequal_transfer_sizes_race_caught(self):

    def unequal_shared_dma_sem(x: jax.Array) -> jax.Array:
      def kernel(x_hbm_ref, y_ref):
        def body(buf_big, buf_small, shared_sem):
          c_big = pltpu.make_async_copy(x_hbm_ref.at[0:16], buf_big, shared_sem)
          c_small = pltpu.make_async_copy(
              x_hbm_ref.at[16:24], buf_small, shared_sem
          )
          c_big.start()
          c_small.start()
          # Progressive byte-arrival from c_small + partial c_big can satisfy
          # c_big.wait() before the final bytes of buf_big arrive.
          c_big.wait()
          y_ref.at[0:16][...] = buf_big[...] + 1.0
          c_small.wait()
          y_ref.at[16:24][...] = buf_small[...] + 1.0

        pl.run_scoped(
            body,
            pltpu.VMEM((16, 128), x.dtype),
            pltpu.VMEM((8, 128), x.dtype),
            pltpu.SemaphoreType.DMA,
        )

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
      )(x)

    x_spec = jax.ShapeDtypeStruct((24, 128), jnp.float32)
    rep = verifier.verify(unequal_shared_dma_sem, (x_spec,), mode=STATIC)
    self.assertFalse(rep.passed, msg=rep.summary())
    self.assertNotEmpty(
        rep.findings_by_category(Category.CONCURRENT_DMA_SEM_SHARING),
        msg=rep.summary(),
    )

  def test_core_map_cross_core_remote_copy_race_and_barrier_deadlock(self):
    core_mesh = _make_tensorcore_mesh("core", num_cores=2)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def unsynced_cross_core_vmem_copy(x: jax.Array) -> jax.Array:
      @pl.kernel(
          mesh=core_mesh,
          out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
          scratch_types=[
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      )
      def _body(x_ref, y_ref, peer_vmem, send_sem, recv_sem):
        c = lax.axis_index("core")
        peer = 1 - c
        peer_vmem[...] = x_ref[...]
        copy = pltpu.make_async_remote_copy(
            src_ref=x_ref,
            dst_ref=peer_vmem,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id={"core": peer},
        )
        copy.start()
        # Bug: reads local `peer_vmem` without waiting for the peer's remote
        # copy into our `peer_vmem` to complete or synchronizing with the peer.
        y_ref[...] = peer_vmem[...] + 1.0
        copy.wait()

      return _body(x)

    race_rep = verifier.verify(
        unsynced_cross_core_vmem_copy, (x_spec,), mode=STATIC
    )
    self.assertFalse(race_rep.passed, msg=race_rep.summary())
    self.assertNotEmpty(
        race_rep.findings_by_category(Category.UNTHROTTLED_REMOTE_OVERWRITE),
        msg=race_rep.summary(),
    )

    def misrouted_core_barrier(x: jax.Array) -> jax.Array:
      @pl.kernel(
          mesh=core_mesh,
          out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
          compiler_params=pltpu.CompilerParams(collective_id=7),
      )
      def _body(x_ref, y_ref):
        barrier = pltpu.get_barrier_semaphore()
        # Both cores signal core 0's barrier: core 1 waits forever.
        pl.semaphore_signal(barrier, 1, device_id={"core": 0})
        pl.semaphore_wait(barrier, 1)
        y_ref[...] = x_ref[...]

      return _body(x)

    bar_rep = verifier.verify(misrouted_core_barrier, (x_spec,), mode=STATIC)
    self.assertFalse(bar_rep.passed, msg=bar_rep.summary())
    self.assertNotEmpty(
        bar_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=bar_rep.summary(),
    )
    self.assertNotEmpty(
        bar_rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=bar_rep.summary(),
    )

  def test_core_map_collective_axes_shared_vmem_race_caught(self):
    core_mesh = _make_tensorcore_mesh("core", num_cores=2)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def shared_vmem_race(x: jax.Array) -> jax.Array:
      @pl.kernel(
          mesh=core_mesh,
          out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
      )
      def _body(x_ref, y_ref):
        def _scoped(shared_buf):
          # Both cores write the same shared VMEM buffer concurrently.
          shared_buf[...] = x_ref[...] + lax.axis_index("core").astype(
              x_ref.dtype
          )
          y_ref[...] = shared_buf[...]

        pl.run_scoped(
            _scoped,
            pltpu.VMEM(x.shape, x.dtype),
            collective_axes="core",
        )

      return _body(x)

    rep = verifier.verify(shared_vmem_race, (x_spec,), mode=STATIC)
    self.assertFalse(rep.passed, msg=rep.summary())
    self.assertNotEmpty(
        rep.findings_by_category(Category.DATA_RACE),
        msg=rep.summary(),
    )

  def test_truncated_loop_causal_deadlock_and_zero_signal_deadlock_caught(self):
    grid = 40  # Exceeds default max_unrolled_iterations=32.
    x_spec = jax.ShapeDtypeStruct((grid * 8, 128), jnp.float32)

    def step0_wait_before_signal_deadlock(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, sem):
        @pl.when(pl.program_id(0) == 0)
        def _deadlock():
          pl.semaphore_wait(sem, 1)
          pl.semaphore_signal(sem, 1)

        o_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(grid,),
              in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
              out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
              scratch_shapes=(pltpu.SemaphoreType.REGULAR,),
          ),
      )(x)

    def zero_signal_wait_on_truncated_grid(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, sem):
        @pl.when(pl.program_id(0) == 0)
        def _wait():
          pl.semaphore_wait(sem, 1)

        o_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(grid,),
              in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
              out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
              scratch_shapes=(pltpu.SemaphoreType.REGULAR,),
          ),
      )(x)

    def prefix_race_with_tail_wait_on_truncated_grid(
        x: jax.Array,
    ) -> jax.Array:
      def kernel(x_hbm, o_ref, buf, dma_sem, reg_sem):
        i = pl.program_id(0)

        @pl.when(i == 0)
        def _signal():
          pl.semaphore_signal(reg_sem, 1)

        @pl.when(i == grid - 1)
        def _tail_wait():
          pl.semaphore_wait(reg_sem, 1)

        copy = pltpu.make_async_copy(x_hbm.at[pl.ds(i * 8, 8)], buf, dma_sem)
        copy.start()
        # Race on step 0: reads `buf` before `copy.wait()`.
        o_ref[...] = buf[...]
        copy.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(grid,),
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
              scratch_shapes=(
                  pltpu.VMEM((8, 128), jnp.float32),
                  pltpu.SemaphoreType.DMA,
                  pltpu.SemaphoreType.REGULAR,
              ),
          ),
      )(x)

    dl_rep = verifier.verify(
        step0_wait_before_signal_deadlock, (x_spec,), mode=STATIC
    )
    self.assertFalse(dl_rep.passed, msg=dl_rep.summary())
    self.assertNotEmpty(
        dl_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=dl_rep.summary(),
    )

    zs_rep = verifier.verify(
        zero_signal_wait_on_truncated_grid, (x_spec,), mode=STATIC
    )
    self.assertFalse(zs_rep.passed, msg=zs_rep.summary())
    self.assertNotEmpty(
        zs_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=zs_rep.summary(),
    )

    pr_rep = verifier.verify(
        prefix_race_with_tail_wait_on_truncated_grid, (x_spec,), mode=STATIC
    )
    self.assertFalse(pr_rep.passed, msg=pr_rep.summary())
    self.assertNotEmpty(
        pr_rep.findings_by_category(Category.DATA_RACE),
        msg=pr_rep.summary(),
    )
    self.assertNotEmpty(
        pr_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=pr_rep.summary(),
    )

  def test_dma_to_dma_write_after_write_race_and_candidate_cap(self):
    grid = 8
    x_spec = jax.ShapeDtypeStruct((grid * 8, 128), jnp.float32)

    def dma_waw_with_sequenced_vpu(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_ref, vpu_buf, waw_buf, vpu_sem, waw_sems):
        i = pl.program_id(0)
        # Properly awaited per-step DMA + VPU traffic on `vpu_buf` is pruned by
        # static completion bounds, while the concurrent unawaited DMA-DMA WAW
        # race on `waw_buf` reaches CP-SAT.
        vpu_copy = pltpu.make_async_copy(
            x_hbm.at[pl.ds(i * 8, 8)], vpu_buf, vpu_sem
        )
        vpu_copy.start()
        vpu_copy.wait()
        vpu_buf[...] = vpu_buf[...] + 1.0
        o_ref[...] = vpu_buf[...]

        @pl.when(i == 0)
        def _start0():
          pltpu.make_async_copy(x_hbm.at[0:8], waw_buf, waw_sems.at[0]).start()

        @pl.when(i == 1)
        def _start1_and_wait():
          c1 = pltpu.make_async_copy(x_hbm.at[8:16], waw_buf, waw_sems.at[1])
          c1.start()
          pltpu.make_async_copy(x_hbm.at[0:8], waw_buf, waw_sems.at[0]).wait()
          c1.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              grid=(grid,),
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
              scratch_shapes=(
                  pltpu.VMEM((8, 128), jnp.float32),
                  pltpu.VMEM((8, 128), jnp.float32),
                  pltpu.SemaphoreType.DMA,
                  pltpu.SemaphoreType.DMA((2,)),
              ),
          ),
      )(x)

    rep = verifier.verify(dma_waw_with_sequenced_vpu, (x_spec,), mode=STATIC)
    self.assertFalse(rep.passed, msg=rep.summary())
    self.assertNotEmpty(
        rep.findings_by_category(Category.DATA_RACE),
        msg=rep.summary(),
    )

    # Three concurrent DMAs writing `waw_buf` produce C(3, 2) = 3 unpruned WAW
    # candidate pairs, exercising the `_MAX_RACE_CANDIDATES` cap warning.
    def three_way_dma_waw(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_ref, waw_buf, sems):
        copies = [
            pltpu.make_async_copy(
                x_hbm.at[pl.ds(k * 8, 8)], waw_buf, sems.at[k]
            )
            for k in range(3)
        ]
        for c in copies:
          c.start()
        for c in copies:
          c.wait()
        o_ref[...] = waw_buf[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec((8, 128), lambda: (0, 0)),
              scratch_shapes=(
                  pltpu.VMEM((8, 128), jnp.float32),
                  pltpu.SemaphoreType.DMA((3,)),
              ),
          ),
      )(x)

    with mock.patch.object(smt_checker, "_MAX_RACE_CANDIDATES", 1):
      cap_rep = verifier.verify(
          three_way_dma_waw,
          (jax.ShapeDtypeStruct((24, 128), jnp.float32),),
          mode=STATIC,
      )
    self.assertFalse(cap_rep.passed, msg=cap_rep.summary())
    self.assertNotEmpty(
        cap_rep.findings_by_category(Category.DATA_RACE),
        msg=cap_rep.summary(),
    )
    cap_warnings = [
        f
        for f in cap_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.severity is verifier.Severity.WARNING
        and "conflicting access pairs" in f.message
    ]
    self.assertLen(cap_warnings, 1, msg=cap_rep.summary())

  def test_register_autograd_skips_backward_when_forward_deadlocks(self):
    def deadlocking_fwd_jax(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        pl.semaphore_wait(sem, 1)
        y_ref[...] = x_ref[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
      )

    no_bwd_op = jax_op(
        "pallas_verify_kernel::deadlock_no_autograd", deadlocking_fwd_jax
    )
    with_bwd_op = jax_op(
        "pallas_verify_kernel::deadlock_with_autograd", deadlocking_fwd_jax
    )
    with_bwd_op.register_autograd(lambda ctx, grad: (grad,))

    x = torch.ones((8, 128), dtype=torch.float32, requires_grad=True)
    no_bwd_rep = verifier.verify(no_bwd_op, (x,))
    self.assertNotEmpty(
        no_bwd_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=no_bwd_rep.summary(),
    )
    self.assertEmpty(
        [f for f in no_bwd_rep.findings if f.phase is verifier.Phase.AUTOGRAD],
        msg=no_bwd_rep.summary(),
    )

    with_bwd_rep = verifier.verify(with_bwd_op, (x,))
    self.assertNotEmpty(
        with_bwd_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=with_bwd_rep.summary(),
    )
    autograd_findings = [
        f for f in with_bwd_rep.findings if f.phase is verifier.Phase.AUTOGRAD
    ]
    self.assertLen(autograd_findings, 1)
    self.assertIs(
        autograd_findings[0].category, Category.VERIFICATION_INCOMPLETE
    )
    self.assertIs(autograd_findings[0].severity, verifier.Severity.INFO)
    self.assertNotIn(verifier.Phase.AUTOGRAD, with_bwd_rep.checked_phases)
    self.assertNotIn(verifier.Phase.INTERPRET, no_bwd_rep.checked_phases)
    self.assertNotIn(verifier.Phase.INTERPRET, with_bwd_rep.checked_phases)
    interp_skip = [
        f for f in with_bwd_rep.findings if f.phase is verifier.Phase.INTERPRET
    ]
    self.assertLen(interp_skip, 1, msg=with_bwd_rep.summary())
    self.assertIs(interp_skip[0].category, Category.VERIFICATION_INCOMPLETE)
    self.assertIs(interp_skip[0].severity, verifier.Severity.INFO)

  def test_parallel_grid_conflict_when_cpsat_unavailable_and_race_loop_timeout(
      self,
  ):
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)

    # Without `ortools` (`cp_model is None`), a sequential single-actor kernel
    # with no waits or race candidates emits zero `VERIFICATION_INCOMPLETE`
    # warnings, whereas an async DMA kernel emits a `WARNING` note.
    with mock.patch.object(smt_checker, "cp_model", None):
      seq_rep = verifier.verify(_make_tiled_copy(2), (x_spec,), mode=STATIC)
      dma_no_cpsat_rep = verifier.verify(
          _shared_sem_race_jax, (x_spec,), mode=STATIC
      )
    self.assertTrue(seq_rep.passed, msg=seq_rep.summary())
    self.assertEmpty(
        seq_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=seq_rep.summary(),
    )
    dma_warnings = [
        f
        for f in dma_no_cpsat_rep.findings_by_category(
            Category.VERIFICATION_INCOMPLETE
        )
        if f.severity is verifier.Severity.WARNING
    ]
    self.assertNotEmpty(dma_warnings, msg=dma_no_cpsat_rep.summary())

    def parallel_overlap(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid=(2,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (0, 0)),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",)
          ),
      )(x)

    with mock.patch.object(smt_checker, "cp_model", None):
      no_cpsat_rep = verifier.verify(parallel_overlap, (x_spec,), mode=STATIC)
    self.assertFalse(no_cpsat_rep.passed, msg=no_cpsat_rep.summary())
    self.assertNotEmpty(
        no_cpsat_rep.findings_by_category(Category.PARALLEL_GRID_CONFLICT),
        msg=no_cpsat_rep.summary(),
    )
    self.assertNotEmpty(
        no_cpsat_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=no_cpsat_rep.summary(),
    )

    zero_timeout_par_rep = verifier.verify(
        parallel_overlap,
        (x_spec,),
        mode=STATIC,
        options=verifier.VerificationOptions(solver_timeout_seconds=0.0),
    )
    self.assertFalse(
        zero_timeout_par_rep.passed, msg=zero_timeout_par_rep.summary()
    )
    self.assertNotEmpty(
        zero_timeout_par_rep.findings_by_category(
            Category.PARALLEL_GRID_CONFLICT
        ),
        msg=zero_timeout_par_rep.summary(),
    )

    # Simulate the initial deadlock feasibility solve succeeding while the
    # per-candidate race query hits the solver timeout budget.
    real_solve = smt_checker._solve
    call_count = 0

    def _timeout_on_race_query(solver, model, timeout_seconds):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return real_solve(solver, model, timeout_seconds)
      return smt_checker.cp_model.UNKNOWN

    with mock.patch.object(
        smt_checker,
        "_solve",
        autospec=True,
        side_effect=_timeout_on_race_query,
    ):
      race_timeout_rep = verifier.verify(
          _shared_sem_race_jax, (x_spec,), mode=STATIC
      )
    incomplete = [
        f
        for f in race_timeout_rep.findings_by_category(
            Category.VERIFICATION_INCOMPLETE
        )
        if f.severity is verifier.Severity.WARNING
    ]
    self.assertNotEmpty(incomplete, msg=race_timeout_rep.summary())
    self.assertTrue(
        any(
            "candidate pair(s) across" in f.message
            and "buffer bucket(s) unchecked" in f.message
            for f in incomplete
        ),
        msg=race_timeout_rep.summary(),
    )

  def test_single_dma_partial_byte_wait_race_caught(self):
    def partial_wait_dma(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, y_vmem, buf_vmem, dma_sem):
        pltpu.make_async_copy(x_hbm, buf_vmem, dma_sem).start()
        # Waiting for only half the bytes (2048B of 4096B) lets `buf_vmem[...]`
        # read while the remaining bytes are still arriving in `buf_vmem`.
        pltpu.make_async_copy(
            x_hbm.at[:4, :], buf_vmem.at[:4, :], dma_sem
        ).wait()
        y_vmem[...] = buf_vmem[...] + 1.0
        pltpu.make_async_copy(
            x_hbm.at[4:, :], buf_vmem.at[4:, :], dma_sem
        ).wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=pl.BlockSpec(x.shape),
          scratch_shapes=[
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.SemaphoreType.DMA,
          ],
      )(x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep = verifier.verify(partial_wait_dma, (x_spec,), mode=STATIC)
    self.assertFalse(rep.passed, msg=rep.summary())
    self.assertNotEmpty(
        rep.findings_by_category(Category.DATA_RACE),
        msg=rep.summary(),
    )

  def test_remote_dma_earlier_dst_wait_preserves_src_wait_send_order(self):
    core_mesh = _make_tensorcore_mesh("core", num_cores=2)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def remote_dma_with_earlier_dst_wait(x: jax.Array) -> jax.Array:
      @pl.kernel(
          mesh=core_mesh,
          out_type=jax.ShapeDtypeStruct(x.shape, x.dtype),
          scratch_types=[
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      )
      def _body(x_hbm, o_hbm, src_vmem, dst_vmem, send_sem, recv_sem):
        core_idx = lax.axis_index("core")
        copy_in = pltpu.make_async_copy(x_hbm, src_vmem, send_sem)
        copy_in.start()
        copy_in.wait()
        rdma = pltpu.make_async_remote_copy(
            src_ref=src_vmem,
            dst_ref=dst_vmem,
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id={"core": 1},
        )

        @pl.when(core_idx == 0)
        def _sender():
          # Extra local event pushes `wait_send` to a higher actor position than
          # Core 1's `wait_recv`.
          dst_vmem[...] = src_vmem[...]
          rdma.start()
          rdma.wait_send()
          # Safe overwrite of `src_vmem` strictly after `wait_send()`.
          src_vmem[...] = jnp.zeros_like(src_vmem[...])
          out_c0 = pltpu.make_async_copy(dst_vmem, o_hbm, send_sem)
          out_c0.start()
          out_c0.wait()

        @pl.when(core_idx == 1)
        def _receiver():
          rdma.wait_recv()

      return _body(x)

    rep, models = smt_checker.check_kernel_concurrency_with_models(
        remote_dma_with_earlier_dst_wait, (x_spec,)
    )
    self.assertTrue(rep.passed, msg=rep.summary())
    assert models is not None
    sched = smt_checker._collect_schedule(models[0])
    rdma_reads = [
        acc
        for acc in sched.accesses
        if acc.dma is not None and acc.dma.is_remote and not acc.is_write
    ]
    self.assertLen(rdma_reads, 1)
    self.assertLen(rdma_reads[0].end.completed_by, 2)
    sender_overwrites = [
        acc
        for acc in sched.accesses
        if acc.dma is None
        and acc.is_write
        and acc.buffer.buffer_id == rdma_reads[0].buffer.buffer_id
        and acc.actor == rdma_reads[0].actor
        and acc.start.position > rdma_reads[0].start.position
    ]
    self.assertNotEmpty(sender_overwrites)
    self.assertEqual(
        smt_checker._static_order(
            rdma_reads[0].end, sender_overwrites[0].start
        ),
        -1,
    )

  def test_same_buffer_data_race_and_shared_dma_sem_both_reported(self):
    def both_hazards_same_buffer(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, y_vmem, buf0, buf1, shared_sem, other_sem):
        # 1. Shared DMA semaphore across equal-size disjoint copies `c1`, `c0`.
        c1 = pltpu.make_async_copy(x_hbm.at[8:16], buf1, shared_sem)
        c0 = pltpu.make_async_copy(x_hbm.at[0:8], buf0, shared_sem)
        c1.start()
        c0.start()
        c0.wait()
        y_vmem.at[0:8][...] = buf0[...]
        c1.wait()
        y_vmem.at[8:16][...] = buf1[...]
        # 2. Independent un-waited DMA on `buf0` (`other_sem` waited too late)
        # causing a plain `DATA_RACE` on the exact same `buf0`.
        c2 = pltpu.make_async_copy(x_hbm.at[0:8], buf0, other_sem)
        c2.start()
        buf0[...] = jnp.zeros_like(buf0[...])
        c2.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=pl.BlockSpec(x.shape),
          scratch_shapes=[
              pltpu.VMEM((8, 128), x.dtype),
              pltpu.VMEM((8, 128), x.dtype),
              pltpu.SemaphoreType.DMA,
              pltpu.SemaphoreType.DMA,
          ],
      )(x)

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep = verifier.verify(both_hazards_same_buffer, (x_spec,), mode=STATIC)
    self.assertNotEmpty(
        rep.findings_by_category(Category.CONCURRENT_DMA_SEM_SHARING),
        msg=rep.summary(),
    )
    self.assertNotEmpty(
        rep.findings_by_category(Category.DATA_RACE),
        msg=rep.summary(),
    )

  def test_core_map_default_hbm_waw_race_and_concolic_while_deadlock(self):
    core_mesh = _make_tensorcore_mesh("core", num_cores=2)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def core_map_hbm_waw_race(x: jax.Array) -> jax.Array:
      del x
      out_ref = jax.empty_ref(jax.ShapeDtypeStruct((8, 128), jnp.float32))

      @pl.kernel(
          mesh=core_mesh,
          scratch_types=[
              pltpu.VMEM((8, 128), jnp.float32),
              pltpu.SemaphoreType.DMA,
          ],
      )
      def _kernel(vmem_src, sem):
        vmem_src[...] = jnp.ones((8, 128), jnp.float32)
        copy = pltpu.make_async_copy(vmem_src, out_ref, sem)
        copy.start()
        copy.wait()

      _kernel()
      return out_ref[...]

    waw_rep = verifier.verify(core_map_hbm_waw_race, (x_spec,), mode=STATIC)
    self.assertNotEmpty(
        waw_rep.findings_by_category(Category.DATA_RACE),
        msg=waw_rep.summary(),
    )

    def concolic_sync_while_deadlock(
        trips: jax.Array, x: jax.Array
    ) -> jax.Array:
      def kernel(trips_smem, x_ref, y_ref, sem):
        def cond_fn(i):
          return i < trips_smem[0]

        def body_fn(i):
          pl.semaphore_signal(sem, 1)
          # Iteration 2 waits for 2 signals when only 1 was posted on that step.
          pl.semaphore_wait(sem, lax.select(i == 2, 2, 1))
          return i + 1

        _ = lax.while_loop(cond_fn, body_fn, jnp.int32(0))
        y_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[pl.BlockSpec(x.shape)],
              out_specs=pl.BlockSpec(x.shape),
              scratch_shapes=[pltpu.SemaphoreType.REGULAR],
          ),
      )(trips, x)

    concolic_op = jax_op(
        "pallas_verify_kernel::concolic_sync_while_deadlock",
        concolic_sync_while_deadlock,
    )
    concolic_rep = verifier.verify(
        concolic_op,
        (
            torch.tensor([3], dtype=torch.int32),
            torch.ones((8, 128), dtype=torch.float32),
        ),
        mode=STATIC,
    )
    self.assertNotEmpty(
        concolic_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
        msg=concolic_rep.summary(),
    )
    concolic_incomplete = concolic_rep.findings_by_category(
        Category.VERIFICATION_INCOMPLETE
    )
    self.assertNotEmpty(concolic_incomplete, msg=concolic_rep.summary())
    self.assertTrue(
        any(
            "a while loop's trip count was evaluated at the provided example"
            " inputs"
            in f.message
            for f in concolic_incomplete
        ),
        msg=concolic_rep.summary(),
    )

  def test_dma_done_count_underwait_caught_on_regular_semaphore(self):
    spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def done_count_underwait(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, vmem_a, vmem_b, reg_sem):
        pltpu.make_async_copy(x_ref, vmem_a, reg_sem).start()
        pltpu.make_async_copy(x_ref, vmem_b, reg_sem).start()
        pl.semaphore_wait(reg_sem, 1)
        y_ref[...] = vmem_a[...] + vmem_b[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=pl.BlockSpec(x.shape, lambda: (0, 0)),
          scratch_shapes=(
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.SemaphoreType.REGULAR,
          ),
      )(x)

    underwait_rep = verifier.verify(
        done_count_underwait,
        (spec,),
        mode=STATIC,
        options=verifier.VerificationOptions(
            tpu_version=torch.tpu.TpuVersion.V7
        ),
    )
    self.assertFalse(underwait_rep.passed, msg=underwait_rep.summary())
    self.assertNotEmpty(
        underwait_rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=underwait_rep.summary(),
    )
    self.assertNotEmpty(
        underwait_rep.findings_by_category(Category.CONCURRENT_DMA_SEM_SHARING),
        msg=underwait_rep.summary(),
    )

  def test_truncated_multichip_rdma_relaxation_emits_warning_not_false_error(
      self,
  ):
    mesh2 = _make_mock_tpu_mesh(2, "rank")
    num_chunks = 8

    def truncated_rdma_gather(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      def kernel(x_hbm, y_hbm, recv_buf, send_sem, recv_sem):
        rank = lax.axis_index("rank")

        @pl.when(rank == 1)
        def _sender():
          def body(k, carry):
            rc = pltpu.make_async_remote_copy(
                src_ref=x_hbm.at[pl.ds(k * 8, 8), :],
                dst_ref=recv_buf.at[pl.ds(k * 8, 8), :],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id={"rank": 0},
            )
            rc.start()
            rc.wait_send()
            return carry

          _ = lax.fori_loop(0, num_chunks, body, jnp.int32(0))
          out_c1 = pltpu.make_async_copy(x_hbm, y_hbm, send_sem)
          out_c1.start()
          out_c1.wait()

        @pl.when(rank == 0)
        def _receiver():
          pltpu.make_async_remote_copy(
              src_ref=x_hbm,
              dst_ref=recv_buf,
              send_sem=send_sem,
              recv_sem=recv_sem,
              device_id={"rank": 0},
          ).wait_recv()
          recv_buf[...] = recv_buf[...] + 1.0
          out_c0 = pltpu.make_async_copy(recv_buf, y_hbm, send_sem)
          out_c0.start()
          out_c0.wait()

      def per_shard(x_local):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_local.shape, x_local.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[HBM_SPEC],
                out_specs=HBM_SPEC,
                scratch_shapes=(
                    pltpu.VMEM(x_local.shape, x_local.dtype),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
        )(x_local)

      return _shard_mapped(per_shard, mesh)(global_x)

    gather_op = _distributed_op(
        "pallas_verify_kernel::truncated_rdma_gather",
        truncated_rdma_gather,
        mesh2,
    )
    x_local = torch.ones((num_chunks * 8, 128), dtype=torch.float32)
    trunc_rep = verifier.verify(
        gather_op,
        (x_local,),
        mode=STATIC,
        options=verifier.VerificationOptions(max_unrolled_iterations=4),
    )
    self.assertTrue(trunc_rep.passed, msg=trunc_rep.summary())
    trunc_warnings = trunc_rep.findings_by_severity(verifier.Severity.WARNING)
    self.assertNotEmpty(trunc_warnings, msg=trunc_rep.summary())
    self.assertTrue(
        all(
            f.severity is verifier.Severity.WARNING
            for f in trunc_rep.findings_by_category(
                Category.UNTHROTTLED_REMOTE_OVERWRITE
            )
        ),
        msg=trunc_rep.summary(),
    )

    full_unroll_rep = verifier.verify(
        gather_op,
        (x_local,),
        mode=STATIC,
        options=verifier.VerificationOptions(max_unrolled_iterations=16),
    )
    self.assertTrue(full_unroll_rep.passed, msg=full_unroll_rep.summary())
    self.assertEmpty(
        full_unroll_rep.findings_by_category(
            Category.UNTHROTTLED_REMOTE_OVERWRITE
        ),
        msg=full_unroll_rep.summary(),
    )

  def test_shared_collective_id_barrier_conservation_enforced_per_kernel(self):
    mesh2 = _make_mock_tpu_mesh(2, "rank")

    def make_shared_barrier_chain(*, oversignal_first: bool):
      def chain_jax(
          global_x: jax.Array, *, mesh: jax.sharding.Mesh
      ) -> jax.Array:
        def k1(x_ref, y_ref):
          peer = 1 - lax.axis_index("rank")
          barrier = pltpu.get_barrier_semaphore()
          pl.semaphore_signal(
              barrier, 2 if oversignal_first else 1, device_id={"rank": peer}
          )
          pl.semaphore_wait(barrier, 1)
          y_ref[...] = x_ref[...] + 1.0

        def k2(x_ref, y_ref):
          peer = 1 - lax.axis_index("rank")
          barrier = pltpu.get_barrier_semaphore()
          pl.semaphore_signal(barrier, 1, device_id={"rank": peer})
          pl.semaphore_wait(barrier, 1)
          y_ref[...] = x_ref[...] * 2.0

        def per_shard(x_local):
          mid = _single_block_call(k1, x_local, collective_id=0)
          return _single_block_call(k2, mid, collective_id=0)

        return _shard_mapped(per_shard, mesh)(global_x)

      return chain_jax

    x_local = torch.ones((8, 128), dtype=torch.float32)
    bad_op = _distributed_op(
        "pallas_verify_kernel::shared_cid_oversignal",
        make_shared_barrier_chain(oversignal_first=True),
        mesh2,
    )
    bad_rep = verifier.verify(bad_op, (x_local,), mode=STATIC)
    self.assertFalse(bad_rep.passed, msg=bad_rep.summary())
    self.assertNotEmpty(
        bad_rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=bad_rep.summary(),
    )

    ok_op = _distributed_op(
        "pallas_verify_kernel::shared_cid_balanced",
        make_shared_barrier_chain(oversignal_first=False),
        mesh2,
    )
    ok_rep = verifier.verify(ok_op, (x_local,), mode=STATIC)
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())

  def test_discarded_dummy_scratch_outputs_on_parallel_grid(self):
    opts = verifier.VerificationOptions(tpu_version=torch.tpu.TpuVersion.V7)

    def make_parallel_scratch_out_fn(
        *,
        discard_scratch: bool = True,
        init_scratch: bool = True,
        oob_scratch_spec: bool = False,
        discard_all_outputs: bool = False,
    ):
      def single_fn(x_b: jax.Array):
        num_heads = x_b.shape[0]

        def kernel(x_ref, scratch_out_ref, out_ref):
          if init_scratch:
            scratch_out_ref[...] = jnp.zeros_like(scratch_out_ref)
          acc = scratch_out_ref[...] + x_ref[...]
          scratch_out_ref[...] = acc
          out_ref[...] = acc

        scratch_idx = (
            (lambda h, i: (1, 0)) if oob_scratch_spec else (lambda h, i: (0, 0))
        )
        scratch_out, out = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec((None, 8, 128), lambda h, i: (h, i, 0)),
                ],
                out_specs=[
                    pl.BlockSpec((8, 128), scratch_idx),
                    pl.BlockSpec((None, 8, 128), lambda h, i: (h, i, 0)),
                ],
                grid=(num_heads, 1),
            ),
            out_shape=(
                jax.ShapeDtypeStruct((8, 128), jnp.float32),
                jax.ShapeDtypeStruct(x_b.shape, x_b.dtype),
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
            ),
        )(x_b)
        if discard_all_outputs:
          del scratch_out, out
          return x_b + 1.0
        if discard_scratch:
          return out
        return scratch_out, out

      @jax.jit
      def batched_fn(x: jax.Array):
        return jax.vmap(single_fn)(x)

      return batched_fn

    for batch in (1, 2):
      x_spec = jax.ShapeDtypeStruct((batch, 2, 8, 128), jnp.float32)
      clean_rep = verifier.verify(
          make_parallel_scratch_out_fn(discard_scratch=True),
          (x_spec,),
          mode=STATIC,
          options=opts,
      )
      self.assertTrue(clean_rep.passed, msg=clean_rep.summary())
      self.assertEmpty(
          clean_rep.findings_by_severity(verifier.Severity.WARNING),
          msg=clean_rep.summary(),
      )

    x1_spec = jax.ShapeDtypeStruct((1, 2, 8, 128), jnp.float32)
    exposed_rep = verifier.verify(
        make_parallel_scratch_out_fn(discard_scratch=False),
        (x1_spec,),
        mode=STATIC,
        options=opts,
    )
    self.assertNotEmpty(
        exposed_rep.findings_by_category(Category.PARALLEL_GRID_CONFLICT),
        msg=exposed_rep.summary(),
    )

    uninit_rep = verifier.verify(
        make_parallel_scratch_out_fn(discard_scratch=True, init_scratch=False),
        (x1_spec,),
        mode=STATIC,
        options=opts,
    )
    uninit_findings = uninit_rep.findings_by_category(Category.UNWRITTEN_OUTPUT)
    self.assertNotEmpty(uninit_findings, msg=uninit_rep.summary())
    self.assertTrue(
        any(
            "is read before being written" in f.message for f in uninit_findings
        ),
        msg=uninit_rep.summary(),
    )

    oob_rep = verifier.verify(
        make_parallel_scratch_out_fn(
            discard_scratch=True, oob_scratch_spec=True
        ),
        (x1_spec,),
        mode=STATIC,
        options=opts,
    )
    self.assertNotEmpty(
        oob_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=oob_rep.summary(),
    )

    all_dead_rep = verifier.verify(
        make_parallel_scratch_out_fn(discard_all_outputs=True),
        (x1_spec,),
        mode=STATIC,
        options=opts,
    )
    self.assertNotEmpty(
        all_dead_rep.findings_by_category(Category.PARALLEL_GRID_CONFLICT),
        msg=all_dead_rep.summary(),
    )

  # ==========================================================================
  # Dynamic CPU TPU-Interpret Schedule and Poisoned-Memory Sweeps
  # ==========================================================================

  def test_raw_jax_function_runs_on_example_values_in_full_mode(self):
    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    report = verifier.verify(_make_tiled_copy(4), (x_spec,))
    self.assertTrue(report.passed, msg=report.summary())
    self.assertEmpty(
        report.findings_by_severity(verifier.Severity.WARNING),
        msg=report.summary(),
    )
    self.assertIn(verifier.Phase.INTERPRET, report.checked_phases)

  def test_interpreter_failure_is_reported_as_incomplete(self):
    """A crash of the CPU interpreter itself is not a kernel bug."""

    def crash(x):
      raise KeyError(f"interpreter state for {x.shape}")

    def crashing_fn(x: jax.Array) -> jax.Array:
      return jax.pure_callback(crash, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    report = verifier.verify(crashing_fn, (x_spec,))
    self.assertTrue(report.passed, msg=report.summary())
    incomplete = [
        f
        for f in report.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.phase is verifier.Phase.INTERPRET
    ]
    # Reported once: the other schedules would fail the same way.
    self.assertLen(incomplete, 1, msg=report.summary())
    self.assertEqual(incomplete[0].severity, verifier.Severity.WARNING)
    self.assertIn("KeyError", incomplete[0].message)

  def test_valid_distributed_ring_shift_passes_interpret_and_model_check(self):
    """4-chip shard_map + make_async_remote_copy ring shift passes."""
    mesh = _make_mock_tpu_mesh(4, "rank")

    def ring_shift_jax(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      def pallas_body(x_ref, y_ref, send_sem, recv_sem):
        my_rank = lax.axis_index("rank")
        next_rank = lax.rem(my_rank + 1, 4)
        prev_rank = lax.rem(my_rank + 3, 4)

        barrier_sem = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(barrier_sem, 1, device_id={"rank": prev_rank})
        pl.semaphore_signal(barrier_sem, 1, device_id={"rank": next_rank})
        pl.semaphore_wait(barrier_sem, 2)

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
            pallas_body,
            x_local,
            scratch_shapes=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
            collective_id=1,
        )

      return _shard_mapped(per_shard, mesh)(global_x)

    op = _distributed_op(
        "pallas_verify_kernel::valid_ring_shift", ring_shift_jax, mesh
    )
    x_local = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
    report = verifier.verify(op, (x_local,))
    self.assertTrue(report.passed, msg=report.summary())
    self.assertEmpty(
        report.findings_by_severity(verifier.Severity.WARNING),
        msg=report.summary(),
    )
    self.assertIn(verifier.Phase.INTERPRET, report.checked_phases)

  def test_dynamic_checks_skipped_where_jax_cannot_interpret_remote_ops(self):
    mesh2 = _make_mock_tpu_mesh(2, "rank")

    def barrier_jax(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      def kernel(x_ref, y_ref):
        barrier = pltpu.get_barrier_semaphore()
        peer = 1 - lax.axis_index("rank")
        pl.semaphore_signal(barrier, 1, device_id={"rank": peer})
        pl.semaphore_wait(barrier, 1)
        y_ref[...] = x_ref[...]

      def per_shard(x_local):
        return _single_block_call(kernel, x_local, collective_id=6)

      return _shard_mapped(per_shard, mesh)(global_x)

    op = _distributed_op(
        "pallas_verify_kernel::barrier_on_old_jax", barrier_jax, mesh2
    )
    with mock.patch.object(
        interpret_verifier, "MIN_JAX_FOR_REMOTE_INTERPRET", (99, 0, 0)
    ):
      report = verifier.verify(op, (torch.ones((8, 128)),))
    self.assertTrue(report.passed, msg=report.summary())
    self.assertNotIn(verifier.Phase.INTERPRET, report.checked_phases)
    skipped = [
        f
        for f in report.findings_by_severity(verifier.Severity.WARNING)
        if f.phase is verifier.Phase.INTERPRET
    ]
    self.assertLen(skipped, 1, msg=report.summary())

  def test_nans_computed_by_the_kernel_are_not_uninitialized_reads(self):
    def sqrt_of_negative(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(
              Ellipsis, jnp.sqrt(x_ref[...] - 2.0)
          ),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      )(x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    report = verifier.verify(sqrt_of_negative, (x_spec,))
    self.assertTrue(report.passed, msg=report.summary())
    self.assertEmpty(
        report.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=report.summary(),
    )

  def test_fp8_jax_op_full_interpret_passes(self):
    def fp8_dual_out_jax(
        x_e4m3: jax.Array, y_e5m2: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
      def kernel(x_ref, y_ref, bf16_out_ref, fp8_out_ref):
        acc = x_ref[...].astype(jnp.float32) + y_ref[...].astype(jnp.float32)
        bf16_out_ref[...] = acc.astype(jnp.bfloat16)
        fp8_out_ref[...] = acc.astype(jnp.float8_e4m3fn)

      return pl.pallas_call(
          kernel,
          out_shape=(
              jax.ShapeDtypeStruct(x_e4m3.shape, jnp.bfloat16),
              jax.ShapeDtypeStruct(x_e4m3.shape, jnp.float8_e4m3fn),
          ),
      )(x_e4m3, y_e5m2)

    valid_fp8_op = jax_op(
        "pallas_verify_kernel::fp8_dual_out", fp8_dual_out_jax
    )
    x_e4m3 = torch.ones((8, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
    y_e5m2 = torch.full((8, 128), 0.5, dtype=torch.float32).to(
        torch.float8_e5m2
    )
    rep = verifier.verify(valid_fp8_op, (x_e4m3, y_e5m2))
    self.assertTrue(rep.passed, msg=rep.summary())
    self.assertEmpty(
        rep.findings_by_severity(verifier.Severity.WARNING), msg=rep.summary()
    )
    self.assertEqual(
        set(rep.checked_phases),
        {
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
            verifier.Phase.MODEL_CHECK,
            verifier.Phase.INTERPRET,
        },
    )

  def test_sharded_fp8_jax_op_full_interpret_passes(self):
    mesh2 = _make_mock_tpu_mesh(2, "rank")
    x_e4m3 = torch.ones((8, 128), dtype=torch.float32).to(torch.float8_e4m3fn)

    def sharded_fp8_jax(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      def per_shard(local_x: jax.Array) -> jax.Array:
        return pl.pallas_call(
            lambda x_ref, o_ref: o_ref.__setitem__(
                Ellipsis,
                (x_ref[...].astype(jnp.float32) * 2.0).astype(
                    jnp.float8_e4m3fn
                ),
            ),
            out_shape=jax.ShapeDtypeStruct(local_x.shape, jnp.float8_e4m3fn),
        )(local_x)

      return _shard_mapped(per_shard, mesh)(global_x)

    sharded_fp8_op = _distributed_op(
        "pallas_verify_kernel::sharded_fp8", sharded_fp8_jax, mesh2
    )
    sharded_rep = verifier.verify(sharded_fp8_op, (x_e4m3,))
    self.assertTrue(sharded_rep.passed, msg=sharded_rep.summary())
    self.assertEmpty(
        sharded_rep.findings_by_severity(verifier.Severity.WARNING),
        msg=sharded_rep.summary(),
    )
    self.assertIn(verifier.Phase.INTERPRET, sharded_rep.checked_phases)

  def test_uninit_fp8_output_caught_by_nan_poisoning(self):
    x_e4m3 = torch.ones((8, 128), dtype=torch.float32).to(torch.float8_e4m3fn)

    def uninit_fp8_out_jax(x_in: jax.Array) -> jax.Array:
      def kernel(x_ref, fp8_out_ref, uninit_vmem):
        del x_ref
        fp8_out_ref[...] = uninit_vmem[...]

      return _single_block_call(
          kernel,
          x_in,
          scratch_shapes=(pltpu.VMEM(x_in.shape, jnp.float8_e4m3fn),),
      )

    buggy_fp8_op = jax_op(
        "pallas_verify_kernel::uninit_fp8_out", uninit_fp8_out_jax
    )
    interp_rep = interpret_verifier.verify_with_cpu_interpret(
        buggy_fp8_op, (x_e4m3,)
    )
    self.assertFalse(interp_rep.passed, msg=interp_rep.summary())
    self.assertNotEmpty(
        interp_rep.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=interp_rep.summary(),
    )

  def test_raw_shard_map_callable_with_axis_index_in_full_mode(self):
    def per_shard_rank_scale(local_x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        rank = lax.axis_index("tp").astype(x_ref.dtype)
        o_ref[...] = x_ref[...] * (rank + 1.0)

      return _single_block_call(kernel, local_x)

    global_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)

    cpu_mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:2]), ("tp",))
    mock_mesh = _make_mock_tpu_mesh(2, "tp")
    abstract_mesh = jax.sharding.AbstractMesh((2,), ("tp",))

    for label, mesh in (
        ("cpu_mesh", cpu_mesh),
        ("mock_mesh", mock_mesh),
        ("abstract_mesh", abstract_mesh),
    ):
      with self.subTest(mesh_kind=label):
        sharded_fn = jax.shard_map(
            per_shard_rank_scale,
            mesh=mesh,
            in_specs=(P("tp"),),
            out_specs=P("tp"),
            check_vma=False,
        )
        rep = verifier.verify(sharded_fn, (global_spec,))
        self.assertTrue(rep.passed, msg=rep.summary())
        self.assertEmpty(
            rep.findings_by_severity(verifier.Severity.WARNING),
            msg=rep.summary(),
        )
        self.assertEqual(
            set(rep.checked_phases),
            {
                verifier.Phase.LOWERING,
                verifier.Phase.MODEL_CHECK,
                verifier.Phase.INTERPRET,
            },
        )

    with jax.sharding.use_abstract_mesh(abstract_mesh):
      ambient_sharded_fn = jax.shard_map(
          per_shard_rank_scale,
          mesh=None,
          in_specs=(P("tp"),),
          out_specs=P("tp"),
          check_vma=False,
      )
      ambient_rep = verifier.verify(ambient_sharded_fn, (global_spec,))
    self.assertTrue(ambient_rep.passed, msg=ambient_rep.summary())
    self.assertEmpty(
        ambient_rep.findings_by_severity(verifier.Severity.WARNING),
        msg=ambient_rep.summary(),
    )
    self.assertIn(verifier.Phase.INTERPRET, ambient_rep.checked_phases)

  def test_raw_shard_map_rank1_uninitialized_read_detected(self):
    # Bug present only on rank 1 (`axis_index("tp") == 1`) proves CPU interpret
    # executes all shards across the mesh rather than running rank 0 alone.
    def per_shard_rank1_uninit(local_x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, uninit_vmem):
        rank = lax.axis_index("tp")
        o_ref[...] = jnp.where(rank == 1, uninit_vmem[...], x_ref[...])

      return _single_block_call(
          kernel,
          local_x,
          scratch_shapes=(pltpu.VMEM(local_x.shape, local_x.dtype),),
      )

    mock_mesh = _make_mock_tpu_mesh(2, "tp")
    global_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rank1_buggy_fn = jax.shard_map(
        per_shard_rank1_uninit,
        mesh=mock_mesh,
        in_specs=(P("tp"),),
        out_specs=P("tp"),
        check_vma=False,
    )
    buggy_rep = verifier.verify(rank1_buggy_fn, (global_spec,))
    self.assertFalse(buggy_rep.passed, msg=buggy_rep.summary())
    self.assertNotEmpty(
        buggy_rep.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=buggy_rep.summary(),
    )

  def test_raw_shard_map_exceeding_virtual_cpu_devices_reports_incomplete(self):
    def per_shard_rank_scale(local_x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        rank = lax.axis_index("tp").astype(x_ref.dtype)
        o_ref[...] = x_ref[...] * (rank + 1.0)

      return _single_block_call(kernel, local_x)

    mesh8 = _make_mock_tpu_mesh(8, "tp")
    oversized_fn = jax.shard_map(
        per_shard_rank_scale,
        mesh=mesh8,
        in_specs=(P("tp"),),
        out_specs=P("tp"),
        check_vma=False,
    )
    oversized_rep = verifier.verify(
        oversized_fn, (jax.ShapeDtypeStruct((64, 128), jnp.float32),)
    )
    self.assertTrue(oversized_rep.passed, msg=oversized_rep.summary())
    self.assertIn(verifier.Phase.MODEL_CHECK, oversized_rep.checked_phases)
    self.assertNotIn(verifier.Phase.INTERPRET, oversized_rep.checked_phases)
    self.assertNotEmpty(
        oversized_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE),
        msg=oversized_rep.summary(),
    )

  def test_parallel_grid_order_dependent_int32_schedule_divergence_detected(
      self,
  ):
    def parallel_order_dependent_int32(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        prog_id = pl.program_id(0).astype(o_ref.dtype)
        o_ref[...] = x_ref[...] + 100_000 + prog_id

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(8,),
          in_specs=[pl.BlockSpec(x.shape, lambda _i: (0, 0))],
          out_specs=pl.BlockSpec(x.shape, lambda _i: (0, 0)),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",)
          ),
      )(x)

    div_rep = interpret_verifier.verify_with_cpu_interpret(
        parallel_order_dependent_int32,
        (jnp.zeros((8, 128), dtype=jnp.int32),),
        config=verifier.InterpretSweepConfig(
            schedule_modes=(verifier.DmaScheduleMode.EAGER,),
            seeds=(0, 1, 2, 17),
        ),
    )
    self.assertFalse(div_rep.passed, msg=div_rep.summary())
    self.assertNotEmpty(
        div_rep.findings_by_category(Category.SCHEDULE_DIVERGENCE),
        msg=div_rep.summary(),
    )
    self.assertIn("integer element(s) differ", div_rep.summary())

  def test_race_and_uninitialized_vmem_read_both_reported_in_interpret(self):
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)

    def racy_plus_uninit_vmem(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_ref, buf, uninit_buf, sem):
        copy = pltpu.make_async_copy(x_hbm, buf, sem)
        copy.start()
        # Unawaited read triggers `DATA_RACE`, so `Phase.INTERPRET` skips the
        # NaN-vs-zero control diff on this schedule rather than misattributing
        # OS-thread race non-determinism to `NAN_OR_UNINIT_READ`.
        o_ref[0:4, :] = buf[0:4, :]
        copy.wait()
        o_ref[4:8, :] = uninit_buf[4:8, :]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec(x.shape, lambda: (0, 0)),
              scratch_shapes=(
                  pltpu.VMEM(x.shape, x.dtype),
                  pltpu.VMEM(x.shape, x.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(x)

    racy_rep = interpret_verifier.verify_with_cpu_interpret(
        racy_plus_uninit_vmem, (x_spec,)
    )
    self.assertFalse(racy_rep.passed, msg=racy_rep.summary())
    self.assertNotEmpty(
        racy_rep.findings_by_category(Category.DATA_RACE),
        msg=racy_rep.summary(),
    )
    self.assertEmpty(
        racy_rep.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=racy_rep.summary(),
    )

    def ordered_uninit_vmem(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_ref, buf, uninit_buf, sem):
        copy = pltpu.make_async_copy(x_hbm, buf, sem)
        copy.start()
        copy.wait()
        o_ref[0:4, :] = buf[0:4, :]
        o_ref[4:8, :] = uninit_buf[4:8, :]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec(x.shape, lambda: (0, 0)),
              scratch_shapes=(
                  pltpu.VMEM(x.shape, x.dtype),
                  pltpu.VMEM(x.shape, x.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(x)

    uninit_rep = interpret_verifier.verify_with_cpu_interpret(
        ordered_uninit_vmem, (x_spec,)
    )
    self.assertFalse(uninit_rep.passed, msg=uninit_rep.summary())
    self.assertEmpty(
        uninit_rep.findings_by_category(Category.DATA_RACE),
        msg=uninit_rep.summary(),
    )
    self.assertNotEmpty(
        uninit_rep.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=uninit_rep.summary(),
    )

  def test_uninitialized_int32_vmem_read_detected(self):
    def uninit_int32_vmem(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, scratch_i32):
        scratch_i32.at[0:4][...] = x_ref.at[0:4][...]
        o_ref[...] = scratch_i32[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.VMEM(x.shape, jnp.int32),)
      )

    i32_uninit_rep = interpret_verifier.verify_with_cpu_interpret(
        uninit_int32_vmem,
        (jnp.zeros((8, 128), dtype=jnp.int32),),
    )
    self.assertFalse(i32_uninit_rep.passed, msg=i32_uninit_rep.summary())
    self.assertNotEmpty(
        i32_uninit_rep.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=i32_uninit_rep.summary(),
    )

  def test_revisit_output_block_classified_as_hardware_constraint_in_interpret(
      self,
  ):
    def revisit_output_block(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct((16, 128), x.dtype),
          grid=(4,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i % 2, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i % 2, 0)),
      )(x)

    revisit_interp_rep = interpret_verifier.verify_with_cpu_interpret(
        revisit_output_block,
        (jax.ShapeDtypeStruct((16, 128), jnp.float32),),
    )
    self.assertFalse(
        revisit_interp_rep.passed, msg=revisit_interp_rep.summary()
    )
    self.assertNotEmpty(
        revisit_interp_rep.findings_by_category(
            Category.HARDWARE_CONSTRAINT_VIOLATION
        ),
        msg=revisit_interp_rep.summary(),
    )

    # Full `verifier.verify` keeps write-only full-tile revisits as a non-fatal
    # WARNING (consistent with STATIC mode) and passes valid `RevisitMode.ANY`
    # kernels cleanly.
    full_revisit_rep = verifier.verify(
        revisit_output_block,
        (jax.ShapeDtypeStruct((16, 128), jnp.float32),),
    )
    self.assertTrue(full_revisit_rep.passed, msg=full_revisit_rep.summary())

    revisit_cls = getattr(pl, "RevisitMode", None)
    if revisit_cls is None:
      self.skipTest("pl.RevisitMode is unavailable in this JAX release")

    def revisit_any_kernel(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct((16, 128), x.dtype),
          grid=(4,),
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i % 2, 0))],
          out_specs=pl.BlockSpec(
              (8, 128),
              lambda i: (i % 2, 0),
              pipeline_mode=pl.Buffered(1, revisit=revisit_cls.ANY),
          ),
      )(x)

    revisit_any_rep = verifier.verify(
        revisit_any_kernel,
        (jax.ShapeDtypeStruct((16, 128), jnp.float32),),
    )
    self.assertTrue(revisit_any_rep.passed, msg=revisit_any_rep.summary())

  def test_preconstructed_pallas_call_executes_under_cpu_interpret(self):
    shape = (8, 128)
    preconstructed = pl.pallas_call(
        lambda x_ref, o_ref, uninit_ref: o_ref.__setitem__(
            Ellipsis, x_ref[...] + uninit_ref[...]
        ),
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(shape)],
            out_specs=pl.BlockSpec(shape),
            scratch_shapes=(pltpu.VMEM(shape, jnp.float32),),
        ),
    )
    rep = verifier.verify(
        preconstructed, (jax.ShapeDtypeStruct(shape, jnp.float32),)
    )
    self.assertFalse(rep.passed, msg=rep.summary())
    self.assertIn(verifier.Phase.INTERPRET, rep.checked_phases)
    self.assertNotEmpty(
        rep.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=rep.summary(),
    )

  def test_runtime_semaphore_nonzero_exit_detected_in_interpret(self):
    def leaky_sem_fn(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, sem):
        pl.semaphore_signal(sem, 1)
        o_ref[...] = x_ref[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
      )

    rep = interpret_verifier.verify_with_cpu_interpret(
        leaky_sem_fn, (jax.ShapeDtypeStruct((8, 128), jnp.float32),)
    )
    self.assertFalse(rep.passed, msg=rep.summary())
    nonzero = rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT)
    self.assertNotEmpty(nonzero, msg=rep.summary())
    self.assertEqual(nonzero[0].phase, verifier.Phase.INTERPRET)

  def test_uninitialized_read_behind_comparison_and_with_nan_input_detected(
      self,
  ):
    # Uninitialized read used in a comparison produces finite float outputs
    # without NaNs, which the zero-filled control run still catches.
    def uninit_comparison_branch(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, uninit_buf):
        o_ref[...] = jnp.where(uninit_buf[...] == 0.0, x_ref[...], -x_ref[...])

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.VMEM(x.shape, x.dtype),)
      )

    spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    cmp_rep = verifier.verify(uninit_comparison_branch, (spec,))
    self.assertFalse(cmp_rep.passed, msg=cmp_rep.summary())
    self.assertNotEmpty(
        cmp_rep.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=cmp_rep.summary(),
    )

    # Setting `UninitMemoryMode.ZERO` via `VerificationOptions(interpret=...)`
    # disables uninitialized-memory poisoning.
    zero_opts = verifier.VerificationOptions(
        interpret=verifier.InterpretSweepConfig(
            uninitialized_memory=verifier.UninitMemoryMode.ZERO
        )
    )
    zero_rep = verifier.verify(
        uninit_comparison_branch, (spec,), options=zero_opts
    )
    self.assertEmpty(
        zero_rep.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=zero_rep.summary(),
    )

    # An input tensor containing a NaN in rows 0:4 does not mask an
    # uninitialized VMEM read in rows 4:8.
    def partial_uninit_with_nan_input(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, uninit_buf):
        o_ref[0:4, :] = x_ref[0:4, :]
        o_ref[4:8, :] = uninit_buf[4:8, :]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.VMEM(x.shape, x.dtype),)
      )

    nan_op = jax_op(
        "pallas_verify_kernel::partial_uninit_with_nan_input",
        partial_uninit_with_nan_input,
    )
    x_with_nan = torch.ones((8, 128), dtype=torch.float32)
    x_with_nan[0, 0] = float("nan")
    nan_in_rep = verifier.verify(nan_op, (x_with_nan,))
    self.assertFalse(nan_in_rep.passed, msg=nan_in_rep.summary())
    self.assertNotEmpty(
        nan_in_rep.findings_by_category(Category.NAN_OR_UNINIT_READ),
        msg=nan_in_rep.summary(),
    )

  def test_post_race_out_of_bounds_and_namedtuple_inputs_handled_in_interpret(
      self,
  ):
    def race_then_oob(idx: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(idx_ref, x_hbm, o_ref, buf, oob_buf, sem1, sem2):
        c1 = pltpu.make_async_copy(x_hbm.at[0:8, :], buf, sem1)
        c1.start()
        # Race: reads `buf` before `c1.wait()`.
        o_ref[...] = buf[...]
        c1.wait()
        # Runtime OOB DMA read after the race.
        bad_start = idx_ref[0] + 64
        c2 = pltpu.make_async_copy(
            x_hbm.at[pl.ds(bad_start, 8), :], oob_buf, sem2
        )
        c2.start()
        c2.wait()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec((8, 128), lambda _idx: (0, 0)),
              scratch_shapes=(
                  pltpu.VMEM((8, 128), x.dtype),
                  pltpu.VMEM((8, 128), x.dtype),
                  pltpu.SemaphoreType.DMA,
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(idx, x)

    idx_arr = jnp.zeros((1,), dtype=jnp.int32)
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    rep = interpret_verifier.verify_with_cpu_interpret(
        race_then_oob, (idx_arr, x_spec)
    )
    self.assertFalse(rep.passed, msg=rep.summary())
    self.assertNotEmpty(
        rep.findings_by_category(Category.DATA_RACE), msg=rep.summary()
    )
    self.assertNotEmpty(
        rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=rep.summary(),
    )

    if any(
        f.name == "on_race" for f in dataclasses.fields(pltpu.InterpretParams)
    ):
      # Older JAX releases lack `on_race="raise"`, recording `races_found=True`
      # and continuing until the subsequent OOB exception is raised.

      orig_make_params = interpret_verifier.make_interpret_params
      with mock.patch.object(
          interpret_verifier,
          "make_interpret_params",
          side_effect=lambda **kw: orig_make_params(
              **{**kw, "on_race": "warn"}
          ),
      ):
        warn_rep = interpret_verifier.verify_with_cpu_interpret(
            race_then_oob, (idx_arr, x_spec)
        )
      self.assertFalse(warn_rep.passed, msg=warn_rep.summary())
      self.assertNotEmpty(
          warn_rep.findings_by_category(Category.DATA_RACE),
          msg=warn_rep.summary(),
      )
      self.assertNotEmpty(
          warn_rep.findings_by_category(Category.OUT_OF_BOUNDS_ACCESS),
          msg=warn_rep.summary(),
      )

    class _KernelInputs(NamedTuple):
      offset: jax.Array
      data: jax.Array

    def namedtuple_fn(inputs: _KernelInputs) -> jax.Array:
      def kernel(off_ref, x_hbm, o_ref, buf, sem):
        copy = pltpu.make_async_copy(
            x_hbm.at[pl.ds(off_ref[0], 8), :], buf, sem
        )
        copy.start()
        copy.wait()
        o_ref[...] = buf[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), inputs.data.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec((8, 128), lambda unused_off: (0, 0)),
              scratch_shapes=(
                  pltpu.VMEM((8, 128), inputs.data.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(inputs.offset, inputs.data)

    nt_rep = verifier.verify(
        namedtuple_fn, (_KernelInputs(offset=idx_arr, data=x_spec),)
    )
    self.assertTrue(nt_rep.passed, msg=nt_rep.summary())
    self.assertIn(verifier.Phase.INTERPRET, nt_rep.checked_phases)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=internal helper check
        TypeError, "unexpected InterpretParams"
    ):
      interpret_verifier.make_interpret_params(unknown_field_xyz=1)

  def test_dma_done_count_mode_skips_dynamic_interpret_in_full_mode(self):
    def done_count_v7_fn(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_hbm, vmem, reg_sem):
        pltpu.make_async_copy(x_hbm, vmem, reg_sem).start()
        pl.semaphore_wait(reg_sem, 1)
        vmem[...] = vmem[...] * 2.0
        pltpu.make_async_copy(vmem, o_hbm, reg_sem).start()
        pl.semaphore_wait(reg_sem, 1)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[HBM_SPEC],
          out_specs=HBM_SPEC,
          scratch_shapes=(
              pltpu.VMEM(x.shape, x.dtype),
              pltpu.SemaphoreType.REGULAR,
          ),
      )(x)

    opts = verifier.VerificationOptions(tpu_version=torch.tpu.TpuVersion.V7)
    spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    raw_rep = verifier.verify(done_count_v7_fn, (spec,), options=opts)
    self.assertTrue(raw_rep.passed, msg=raw_rep.summary())
    self.assertEmpty(
        raw_rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=raw_rep.summary(),
    )
    self.assertNotIn(verifier.Phase.INTERPRET, raw_rep.checked_phases)
    raw_incomplete = [
        f
        for f in raw_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.phase is verifier.Phase.INTERPRET
    ]
    self.assertLen(raw_incomplete, 1)
    self.assertIs(raw_incomplete[0].severity, verifier.Severity.INFO)

    done_count_op = jax_op(
        "pallas_verify_kernel::done_count_v7_full", done_count_v7_fn
    )
    done_count_op.register_autograd(lambda ctx, grad: (grad * 2.0,))
    x_t = torch.ones((8, 128), dtype=torch.float32, requires_grad=True)
    op_rep = verifier.verify(done_count_op, (x_t,), options=opts)
    self.assertTrue(op_rep.passed, msg=op_rep.summary())
    self.assertNotIn(verifier.Phase.INTERPRET, op_rep.checked_phases)
    self.assertNotIn(verifier.Phase.AUTOGRAD, op_rep.checked_phases)

  def test_stall_progress_watchdog_allows_slow_progress_and_aborts_stalled_wait(
      self,
  ):
    shape = (8, 128)
    cpu0 = jax.devices("cpu")[0]
    with jax.default_device(cpu0):
      x_cpu = jnp.ones(shape, dtype=jnp.float32)
    params = interpret_verifier.make_interpret_params(
        dma_execution_mode="eager",
        detect_races=False,
        uninitialized_memory="zero",
        random_seed=0,
    )

    def multi_step_dma_fn(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_ref, buf, sem):
        i = pl.program_id(0)
        c = pltpu.make_async_copy(x_hbm.at[pl.ds(i * 8, 8), :], buf, sem)
        c.start()
        c.wait()
        o_ref[...] = buf[...] * 2.0

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((64, 128), x.dtype),
          grid=(8,),
          in_specs=[HBM_SPEC],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
          scratch_shapes=(
              pltpu.VMEM(shape, x.dtype),
              pltpu.SemaphoreType.DMA,
          ),
      )(x)

    with jax.default_device(cpu0):
      x_grid = jnp.ones((64, 128), dtype=jnp.float32)

    class _FakeClock:

      def __init__(self) -> None:
        self._lock = threading.Lock()
        self._now = 0.0

      def __call__(self) -> float:
        with self._lock:
          return self._now

      def advance(self, delta: float) -> None:
        with self._lock:
          self._now += delta

    interpret_mod = inspect.getmodule(pltpu.reset_tpu_interpret_mode_state)
    shm_cls = interpret_mod.memory.SharedMemory
    sem_cls = interpret_mod.memory.Semaphore
    orig_store = shm_cls.store_buffer_content
    slow_clock = _FakeClock()

    def _slow_store(self_shm, *s_args, **s_kwargs):
      slow_clock.advance(0.4)
      return orig_store(self_shm, *s_args, **s_kwargs)

    # Total virtual time exceeds `timeout_seconds=1.0` (>=3.2s across 8 steps),
    # while each step makes forward progress within 0.4s.
    with mock.patch.object(shm_cls, "store_buffer_content", _slow_store):
      slow_ok = interpret_verifier._interpret_run(
          multi_step_dma_fn,
          (x_grid,),
          {},
          params,
          cpu0,
          timeout_seconds=1.0,
          clock=slow_clock,
      )
    self.assertGreaterEqual(slow_clock(), 3.2)
    self.assertIsNone(slow_ok.exc)
    assert slow_ok.leaves is not None
    np.testing.assert_allclose(slow_ok.leaves[0], 2.0)

    # A genuinely deadlocked semaphore wait makes zero progress and is aborted
    # once virtual time advances past `timeout_seconds`, classified as
    # `SEMAPHORE_DEADLOCK`, without poisoning the next interpret run.
    def stalled_wait_fn(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, sem):
        pl.semaphore_wait(sem, 1)
        o_ref[...] = x_ref[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
      )

    orig_sem_wait = sem_cls.wait
    stall_clock = _FakeClock()

    def _stalled_sem_wait(self_sem, *w_args, **w_kwargs):
      stall_clock.advance(2.0)
      return orig_sem_wait(self_sem, *w_args, **w_kwargs)

    with mock.patch.object(sem_cls, "wait", _stalled_sem_wait):
      stalled = interpret_verifier._interpret_run(
          stalled_wait_fn,
          (x_cpu,),
          {},
          params,
          cpu0,
          timeout_seconds=0.05,
          clock=stall_clock,
      )
    self.assertIsNotNone(stalled.exc)
    assert stalled.exc is not None
    finding = interpret_verifier._classify_interpret_exception(
        stalled.exc, "stalled_wait_fn", "dma=eager, seed=0"
    )
    self.assertIs(finding.category, Category.SEMAPHORE_DEADLOCK)

    recovery_clock = _FakeClock()
    after_stall = interpret_verifier._interpret_run(
        multi_step_dma_fn,
        (x_grid,),
        {},
        params,
        cpu0,
        timeout_seconds=1.0,
        clock=recovery_clock,
    )
    self.assertIsNone(after_stall.exc)

  def test_complex64_fallback_and_second_seed_int32_uninit_control_in_interpret(
      self,
  ):
    shape = (8, 128)
    c64_spec = jax.ShapeDtypeStruct(shape, jnp.complex64)

    def valid_complex_fn(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, scratch_c64):
        scratch_c64[...] = x_ref[...] * jnp.complex64(1.0 + 2.0j)
        o_ref[...] = scratch_c64[...]

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.VMEM(shape, jnp.complex64),)
      )

    # Mosaic TPU rejects `complex64` VMEM vectors at lowering time, while the
    # CPU TPU interpreter falls back from NaN poisoning to zero-filled memory.
    full_c64_rep = verifier.verify(valid_complex_fn, (c64_spec,), mode=STATIC)
    self.assertNotEmpty(
        full_c64_rep.findings_by_category(Category.LOWERING_FAILURE),
        msg=full_c64_rep.summary(),
    )
    c64_rep = interpret_verifier.verify_with_cpu_interpret(
        valid_complex_fn, (c64_spec,)
    )
    self.assertTrue(c64_rep.passed, msg=c64_rep.summary())
    self.assertIn(verifier.Phase.INTERPRET, c64_rep.checked_phases)
    c64_notes = [
        f
        for f in c64_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.phase is verifier.Phase.INTERPRET
    ]
    self.assertLen(c64_notes, 1)
    self.assertIs(c64_notes[0].severity, verifier.Severity.INFO)

    def racy_complex_fn(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_ref, buf_c64, sem):
        copy = pltpu.make_async_copy(x_hbm, buf_c64, sem)
        copy.start()
        o_ref[...] = buf_c64[...]
        copy.wait()

      return pl.pallas_call(
          kernel,
          out_shape=c64_spec,
          in_specs=[HBM_SPEC],
          out_specs=pl.BlockSpec(shape, lambda: (0, 0)),
          scratch_shapes=(
              pltpu.VMEM(shape, jnp.complex64),
              pltpu.SemaphoreType.DMA,
          ),
      )(x)

    racy_c64_rep = interpret_verifier.verify_with_cpu_interpret(
        racy_complex_fn, (c64_spec,)
    )
    self.assertFalse(racy_c64_rep.passed, msg=racy_c64_rep.summary())
    self.assertNotEmpty(
        racy_c64_rep.findings_by_category(Category.DATA_RACE),
        msg=racy_c64_rep.summary(),
    )

    # Determine which of `(0, 1)` runs first on `seed=0` vs an opposite seed so
    # `seed=0` legitimately produces `iinfo(int32).max` on both poisoned and
    # zero-filled memory while `other_seed` reads uninitialized `scratch_i32`
    # first (matching `seed=0` under poisoned memory and differing only against
    # the zero-filled control run on `other_seed`).
    def _first_coord(seed: int) -> int:
      _, subkey = jax.random.split(jax.random.key(seed))
      return int(
          jax.random.permutation(subkey, jnp.arange(2, dtype=jnp.int32))[0]
      )

    first_on_0 = _first_coord(0)
    other_seed = next(s for s in range(1, 64) if _first_coord(s) != first_on_0)
    i32_max = int(np.iinfo(np.int32).max)

    def second_seed_only_uninit_i32(x: jax.Array) -> jax.Array:
      del x

      def kernel(o_ref, scratch_i32):
        prog_id = pl.program_id(0)

        @pl.when(prog_id == first_on_0)
        def _writer():
          scratch_i32[...] = jnp.full(shape, i32_max, dtype=jnp.int32)
          o_ref[...] = jnp.full(shape, i32_max, dtype=jnp.int32)

        @pl.when(prog_id != first_on_0)
        def _reader():
          o_ref[...] = scratch_i32[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((16, 128), jnp.int32),
          grid=(2,),
          out_specs=pl.BlockSpec(shape, lambda i: (i, 0)),
          scratch_shapes=(pltpu.VMEM(shape, jnp.int32),),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=("parallel",)
          ),
      )()

    seed2_rep = interpret_verifier.verify_with_cpu_interpret(
        second_seed_only_uninit_i32,
        (jnp.zeros(shape, dtype=jnp.int32),),
        config=verifier.InterpretSweepConfig(
            schedule_modes=(verifier.DmaScheduleMode.EAGER,),
            seeds=(0, other_seed),
        ),
    )
    self.assertFalse(seed2_rep.passed, msg=seed2_rep.summary())
    seed2_nan = seed2_rep.findings_by_category(Category.NAN_OR_UNINIT_READ)
    self.assertNotEmpty(seed2_nan, msg=seed2_rep.summary())
    self.assertIn(f"seed={other_seed}", seed2_nan[0].message)

  def test_raw_jax_abstract_int32_shape_dtype_struct_skips_interpret(self):
    def offset_minus_one_copy(offset: jax.Array, x: jax.Array) -> jax.Array:
      def kernel(off_ref, x_hbm, o_ref, buf, sem):
        # Valid whenever `offset[0] >= 1`; zero-filling an abstract `int32`
        # `ShapeDtypeStruct` would read `-8:0` and report a false OOB error.
        start = (off_ref[0] - 1) * 8
        copy = pltpu.make_async_copy(x_hbm.at[pl.ds(start, 8), :], buf, sem)
        copy.start()
        copy.wait()
        o_ref[...] = buf[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[HBM_SPEC],
              out_specs=pl.BlockSpec((8, 128), lambda _off: (0, 0)),
              scratch_shapes=(
                  pltpu.VMEM((8, 128), x.dtype),
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(offset, x)

    abs_rep = verifier.verify(
        offset_minus_one_copy,
        (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((16, 128), jnp.float32),
        ),
    )
    self.assertTrue(abs_rep.passed, msg=abs_rep.summary())
    self.assertNotIn(verifier.Phase.INTERPRET, abs_rep.checked_phases)
    interp_notes = [
        f
        for f in abs_rep.findings_by_category(Category.VERIFICATION_INCOMPLETE)
        if f.phase is verifier.Phase.INTERPRET
    ]
    self.assertLen(interp_notes, 1, msg=abs_rep.summary())
    self.assertIs(interp_notes[0].severity, verifier.Severity.INFO)

    concrete_rep = verifier.verify(
        offset_minus_one_copy,
        (
            jnp.array([1], dtype=jnp.int32),
            jnp.ones((16, 128), dtype=jnp.float32),
        ),
    )
    self.assertTrue(concrete_rep.passed, msg=concrete_rep.summary())
    self.assertIn(verifier.Phase.INTERPRET, concrete_rep.checked_phases)

  def test_fp4x2_int4_and_typed_prng_key_inputs_in_full_mode(self):
    # 1. Packed `torch.float4_e2m1fn_x2` input on a `@jax_op` in `FULL` mode.
    def fp4_upcast_jax(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(
              Ellipsis, x_ref[...].astype(jnp.float32)
          ),
          out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32),
      )(x)

    fp4_op = jax_op("pallas_verify_kernel::fp4x2_upcast_full", fp4_upcast_jax)
    x_fp4 = torch.zeros((64, 64), dtype=torch.uint8).view(
        torch.float4_e2m1fn_x2
    )
    fp4_rep = verifier.verify(fp4_op, (x_fp4,))
    self.assertTrue(fp4_rep.passed, msg=fp4_rep.summary())
    self.assertIn(verifier.Phase.MODEL_CHECK, fp4_rep.checked_phases)

    # 2. Concrete `jnp.int4` input and output across schedules in `FULL` mode.
    def int4_copy_jax(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      )(x)

    x_i4 = jnp.zeros((64, 128), dtype=jnp.int4)
    i4_rep = verifier.verify(int4_copy_jax, (x_i4,))
    self.assertTrue(i4_rep.passed, msg=i4_rep.summary())

    # 3. Typed `jax.random.key(0)` PRNG key input (and output) in `FULL` mode.
    def keyed_scale_jax(
        key: jax.Array, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
      bits = jax.random.bits(key, shape=(), dtype=jnp.uint32)
      scale = (bits & jnp.uint32(1)).astype(x.dtype) + jnp.float32(1.0)
      out = pl.pallas_call(
          lambda x_ref, s_ref, o_ref: o_ref.__setitem__(
              Ellipsis, x_ref[...] * s_ref[0]
          ),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          in_specs=[
              pl.BlockSpec(x.shape, lambda: (0, 0)),
              pl.BlockSpec(memory_space=pltpu.SMEM),
          ],
          out_specs=pl.BlockSpec(x.shape, lambda: (0, 0)),
      )(x, jnp.reshape(scale, (1,)))
      return out, key

    key_rep = verifier.verify(
        keyed_scale_jax,
        (jax.random.key(0), jnp.ones((8, 128), dtype=jnp.float32)),
    )
    self.assertTrue(key_rep.passed, msg=key_rep.summary())
    self.assertIn(verifier.Phase.INTERPRET, key_rep.checked_phases)

  def test_sharded_int32_offset_metadata_passes_full_interpret(self):
    mesh2 = _make_mock_tpu_mesh(2, "rank")

    def sharded_offset_dma_jax(
        global_offsets: jax.Array,
        global_x: jax.Array,
        *,
        mesh: jax.sharding.Mesh,
    ) -> jax.Array:
      def per_shard(off_local: jax.Array, x_local: jax.Array) -> jax.Array:
        def kernel(off_ref, x_hbm, o_ref, buf, sem):
          # Local `x_hbm` has 8 rows; adding a rank offset (`+1`) to sharded
          # `int32` metadata on rank 1 would read rows `8:16` out of bounds.
          copy = pltpu.make_async_copy(
              x_hbm.at[pl.ds(off_ref[0] * 8, 8), :], buf, sem
          )
          copy.start()
          copy.wait()
          o_ref[...] = buf[...] * 2.0

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(x_local.shape, x_local.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[HBM_SPEC],
                out_specs=pl.BlockSpec(x_local.shape, lambda _off: (0, 0)),
                scratch_shapes=(
                    pltpu.VMEM(x_local.shape, x_local.dtype),
                    pltpu.SemaphoreType.DMA,
                ),
            ),
        )(off_local, x_local)

      return jax.shard_map(
          per_shard,
          mesh=mesh,
          in_specs=(P("rank"), P("rank")),
          out_specs=P("rank"),
          check_vma=False,
      )(global_offsets, global_x)

    sharded_meta_op = jax_op(
        "pallas_verify_kernel::sharded_int32_offset_dma",
        functools.partial(sharded_offset_dma_jax, mesh=mesh2),
        mesh=mesh2,
        input_partition_specs=(P("rank"), P("rank")),
        output_partition_specs=P("rank"),
    )
    offsets = torch.zeros((1,), dtype=torch.int32)
    x_local = torch.ones((8, 128), dtype=torch.float32)
    rep = verifier.verify(sharded_meta_op, (offsets, x_local))
    self.assertTrue(rep.passed, msg=rep.summary())
    self.assertIn(verifier.Phase.INTERPRET, rep.checked_phases)
    self.assertEmpty(
        rep.findings_by_severity(verifier.Severity.WARNING),
        msg=rep.summary(),
    )

  def test_statically_unproven_deadlock_with_autograd_aborts_in_interpret_first(
      self,
  ):
    def wait_before_signal_jax(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref, sem):
        # Balanced (+1 / -1) wait-before-signal cycle passes conservation when
        # CP-SAT is unavailable, leaving the deadlock unproven statically.
        pl.semaphore_wait(sem, 1)
        pl.semaphore_signal(sem, 1)
        y_ref[...] = x_ref[...] * 2.0

      return _single_block_call(
          kernel, x, scratch_shapes=(pltpu.SemaphoreType.REGULAR,)
      )

    bwd_calls: list[bool] = []
    op = jax_op(
        "pallas_verify_kernel::unproven_deadlock_autograd_full",
        wait_before_signal_jax,
    )

    def _bwd(ctx, grad):
      del ctx
      bwd_calls.append(True)
      return (grad * 2.0,)

    op.register_autograd(_bwd)
    x = torch.ones((8, 128), dtype=torch.float32, requires_grad=True)
    opts = verifier.VerificationOptions(
        interpret=verifier.InterpretSweepConfig(
            schedule_modes=(verifier.DmaScheduleMode.EAGER,),
            seeds=(0,),
        )
    )
    assert interpret_verifier._interpret_run.__kwdefaults__ is not None
    with (
        mock.patch.object(smt_checker, "cp_model", None),
        mock.patch.dict(
            interpret_verifier._interpret_run.__kwdefaults__,
            {"timeout_seconds": 0.12},
        ),
    ):
      rep = verifier.verify(op, (x,), options=opts)

    self.assertFalse(rep.passed, msg=rep.summary())
    deadlocks = rep.findings_by_category(Category.SEMAPHORE_DEADLOCK)
    self.assertNotEmpty(deadlocks, msg=rep.summary())
    self.assertEqual(deadlocks[0].phase, verifier.Phase.INTERPRET)
    self.assertNotIn(verifier.Phase.AUTOGRAD, rep.checked_phases)
    self.assertEmpty(bwd_calls)

  def test_verify_full_mode_restores_global_and_thread_local_patch_state(self):
    pallas_call_mod = inspect.getmodule(pl.pallas_call)
    assert pallas_call_mod is not None
    pallas_call_p = pallas_call_mod.pallas_call_p

    mpmd_mod = getattr(jax._src.pallas, "mpmd", None)
    mpmd_map_p = getattr(mpmd_mod, "mpmd_map_p", None)

    interpret_mod = inspect.getmodule(pltpu.reset_tpu_interpret_mode_state)
    assert interpret_mod is not None
    shm_cls = interpret_mod.memory.SharedMemory
    had_alloc_in_shm_dict = "allocate_buffer" in shm_cls.__dict__

    fx_vfy = interpret_verifier.fx_verifier

    def scale_jax(x: jax.Array) -> jax.Array:
      return _single_block_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...] * 3.0), x
      )

    clean_op = jax_op("pallas_verify_kernel::patch_state_clean", scale_jax)
    clean_op.register_autograd(lambda ctx, g: (g * 3.0,))
    jc = fx_vfy.extract_jax_callable(clean_op)
    assert jc is not None
    op_def = fx_vfy.resolve_custom_op_def(clean_op)
    assert op_def is not None
    orig_exported = jc.exported
    orig_setup_ctx = getattr(op_def, "_setup_context_fn", None)

    def _assert_clean_patch_state() -> None:
      self.assertNotIn("bind", pallas_call_p.__dict__)
      if mpmd_map_p is not None:
        self.assertNotIn("bind", mpmd_map_p.__dict__)
      if not had_alloc_in_shm_dict:
        self.assertNotIn("allocate_buffer", shm_cls.__dict__)
      self.assertIs(jc.exported, orig_exported)
      self.assertIs(getattr(op_def, "_setup_context_fn", None), orig_setup_ctx)
      self.assertTrue(jax.sharding.get_abstract_mesh().empty)
      self.assertFalse(
          bool(
              getattr(fx_vfy._THREAD_PATCH_STATE, "interpret_params_stack", ())
          )
      )
      for depth_attr in (
          "watchdog_depth",
          "stub_pallas_call_depth",
          "eval_shape_fallback_depth",
      ):
        self.assertEqual(
            int(getattr(fx_vfy._THREAD_PATCH_STATE, depth_attr, 0)), 0
        )

    x = torch.ones((8, 128), dtype=torch.float32, requires_grad=True)

    ok_rep = verifier.verify(clean_op, (x,))
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    _assert_clean_patch_state()

    def raising_jax(x: jax.Array) -> jax.Array:
      def _boom(arr):
        raise RuntimeError(f"callback boom on {arr.shape}")

      return jax.pure_callback(_boom, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    err_rep = verifier.verify(
        raising_jax, (jax.ShapeDtypeStruct((8, 128), jnp.float32),)
    )
    self.assertTrue(err_rep.passed, msg=err_rep.summary())
    _assert_clean_patch_state()


if __name__ == "__main__":
  absltest.main()
