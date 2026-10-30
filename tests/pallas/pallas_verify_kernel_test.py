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

import functools
import types
from unittest import mock

from absl.testing import absltest
import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import torch
from torch.tpu import pallas
from torch.tpu.pallas import verifier
from torch_tpu._internal.pallas import _mock_device
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
    self.assertEmpty(report.findings_by_severity(verifier.Severity.ERROR))
    self.assertContainsSubset(
        {
            verifier.Phase.CONTRACTS,
            verifier.Phase.LOWERING,
            verifier.Phase.MODEL_CHECK,
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

  def test_raw_jax_function_is_verified_from_shape_dtype_structs(self):
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    report = verifier.verify(_shared_sem_race_jax, (x_spec,), mode=STATIC)
    self.assertFalse(report.passed, msg=report.summary())
    self.assertNotEmpty(
        report.findings_by_category(Category.CONCURRENT_DMA_SEM_SHARING),
        msg=report.summary(),
    )

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
    self.assertNotEmpty(
        buggy_rep.findings_by_category(Category.UNTHROTTLED_REMOTE_OVERWRITE),
        msg=buggy_rep.summary(),
    )
    self.assertNotEmpty(
        buggy_rep.findings_by_severity(verifier.Severity.ERROR)[
            0
        ].counterexample
    )

    fixed_rep = verifier.verify(fixed_op, (x_local,))
    self.assertTrue(fixed_rep.passed, msg=fixed_rep.summary())

  def test_static_mode_does_not_query_jax_devices(self):
    """Static checks only trace, so JAX never tries to open a TPU."""
    mesh = _make_mock_tpu_mesh(4, "rank")

    def sharded_copy(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      return _shard_mapped(_make_tiled_copy(grid_size=1), mesh)(global_x)

    op = _distributed_op(
        "pallas_verify_kernel::sharded_copy", sharded_copy, mesh
    )
    # The verifier reports tracing errors as findings, so record the calls
    # instead of raising from them.
    with mock.patch.object(jax, "devices", wraps=jax.devices) as devices:
      report = verifier.verify(op, (torch.ones((8, 128)),), mode=STATIC)
    devices.assert_not_called()
    self.assertTrue(report.passed, msg=report.summary())
    self.assertEmpty(
        report.findings_by_severity(verifier.Severity.WARNING),
        msg=report.summary(),
    )
    self.assertIn(verifier.Phase.MODEL_CHECK, report.checked_phases)

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

    # FULL mode must report both findings without hanging on the deadlocked rank.
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
      def kernel(_idx_ref, x_ref, _y_ref, o0_ref, _o1_ref):
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
          # Output 1 is aliased to overall invar 2 (which is block input 1 = y).
          # Output 0 is NOT aliased, and is written by kernel.
          input_output_aliases={2: 1},
      )(idx, x, y)

    idx_spec = jax.ShapeDtypeStruct((1,), jnp.int32)
    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep, models = jaxpr_model.check_kernel_static_models(
        aliased_with_prefetch, (idx_spec, x_spec, x_spec)
    )
    self.assertTrue(rep.passed, msg=rep.summary())
    self.assertIsNotNone(models)
    written_buffers = {
        ev.buffer.buffer_id for ev in models[0].memory_events if ev.is_write
    }
    self.assertIn("arg_out_0", written_buffers)

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

    # Unpadded window is 8 - (2 + 2) = 4 elements; starting at 0 in a 4-row
    # array is in bounds, while starting at 1 (1:5) is out of bounds.
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

  def test_remote_device_id_outside_mesh_detected(self):
    mesh = _make_mock_tpu_mesh(4, "rank")

    def unwrapped_ring_jax(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      def kernel(x_ref, y_ref, send_sem, recv_sem):
        # Missing `% 4`: rank 3 sends to non-existent device 4.
        next_rank = lax.axis_index("rank") + 1
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

    op = _distributed_op(
        "pallas_verify_kernel::unwrapped_ring", unwrapped_ring_jax, mesh
    )
    report = verifier.verify(op, (torch.ones((8, 128)),), mode=STATIC)
    self.assertNotEmpty(
        report.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=report.summary(),
    )

  def test_wrapped_ring_device_id_with_lax_rem_passes(self):
    mesh = _make_mock_tpu_mesh(4, "rank")

    def wrapped_ring_jax(
        global_x: jax.Array, *, mesh: jax.sharding.Mesh
    ) -> jax.Array:
      def kernel(x_ref, y_ref, send_sem, recv_sem):
        next_rank = lax.rem(lax.axis_index("rank") + 1, 4)
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

    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    rep, _ = jaxpr_model.check_kernel_static_models(
        functools.partial(wrapped_ring_jax, mesh=mesh), (x_spec,)
    )
    self.assertEmpty(
        rep.findings_by_category(Category.INVALID_DEVICE_ID),
        msg=rep.summary(),
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
    hw = report.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=report.summary())
    self.assertIn("HBM", hw[0].message)

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
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Testing internal decode_dma_operands error
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
        options=verifier.VerificationOptions(tpu_device_kind="TPU v5e"),
    )
    self.assertFalse(rep.passed, msg=rep.summary())
    hw = rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=rep.summary())
    self.assertIn("regular semaphore", hw[0].message)
    self.assertEmpty(
        rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=rep.summary(),
    )

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
      def kernel(x_hbm, y_hbm, scratch_vmem, regular_sem):
        copy = pltpu.make_async_copy(x_hbm, scratch_vmem, regular_sem)
        copy.start()
        copy.wait()
        y_hbm[...] = scratch_vmem[...]

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

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    rep_v7, _ = jaxpr_model.check_kernel_static_models(
        regular_sem_dma_with_sem_wait, (x_spec,), tpu_device_kind="TPU v7"
    )
    self.assertEmpty(
        rep_v7.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=rep_v7.summary(),
    )

    rep_v5e, _ = jaxpr_model.check_kernel_static_models(
        regular_sem_dma_with_sem_wait, (x_spec,), tpu_device_kind="TPU v5e"
    )
    self.assertNotEmpty(
        rep_v5e.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=rep_v5e.summary(),
    )

    rep_copy_wait, _ = jaxpr_model.check_kernel_static_models(
        regular_sem_dma_with_copy_wait, (x_spec,), tpu_device_kind="TPU v7"
    )
    self.assertNotEmpty(
        rep_copy_wait.findings_by_category(
            Category.HARDWARE_CONSTRAINT_VIOLATION
        ),
        msg=rep_copy_wait.summary(),
    )

  def test_remote_dma_semaphore_rules_on_tpu_v7(self):
    def mismatched_remote_sems(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, y_hbm, regular_send_sem, dma_recv_sem):
        copy = pltpu.make_async_remote_copy(
            x_hbm, y_hbm, regular_send_sem, dma_recv_sem, device_id=0
        )
        copy.start()
        copy.wait_recv()

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[HBM_SPEC],
              out_specs=HBM_SPEC,
              scratch_shapes=(
                  pltpu.SemaphoreType.REGULAR,
                  pltpu.SemaphoreType.DMA,
              ),
          ),
      )(x)

    def matching_regular_remote_sems(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, y_hbm, regular_send_sem, regular_recv_sem):
        copy = pltpu.make_async_remote_copy(
            x_hbm, y_hbm, regular_send_sem, regular_recv_sem, device_id=0
        )
        copy.start()
        pl.semaphore_wait(regular_send_sem, 1)
        pl.semaphore_wait(regular_recv_sem, 1)

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=0,
              in_specs=[HBM_SPEC],
              out_specs=HBM_SPEC,
              scratch_shapes=(
                  pltpu.SemaphoreType.REGULAR,
                  pltpu.SemaphoreType.REGULAR,
              ),
          ),
      )(x)

    x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    bad_rep, _ = jaxpr_model.check_kernel_static_models(
        mismatched_remote_sems, (x_spec,), tpu_device_kind="TPU v7"
    )
    hw = bad_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=bad_rep.summary())
    self.assertIn("mismatched semaphore types", hw[0].message)

    ok_rep = verifier.verify(
        matching_regular_remote_sems,
        (x_spec,),
        mode=STATIC,
        options=verifier.VerificationOptions(tpu_device_kind="TPU v7"),
    )
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    self.assertEmpty(
        ok_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION),
        msg=ok_rep.summary(),
    )
    self.assertEmpty(
        ok_rep.findings_by_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=ok_rep.summary(),
    )
    self.assertEmpty(
        ok_rep.findings_by_category(Category.SEMAPHORE_DEADLOCK),
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
    hw = rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=rep.summary())
    self.assertIn("dma_sem", hw[0].message)

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
    rep, _ = jaxpr_model.check_kernel_static_models(
        peek_semaphore_wait, (x_spec,)
    )
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
    rep, _ = jaxpr_model.check_kernel_static_models(
        smem_dma, (x_spec,), tpu_device_kind="TPU v7"
    )
    hw = rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=rep.summary())
    self.assertIn("SMEM", hw[0].message)

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
    self.assertEmpty(batch_rep.findings_by_severity(verifier.Severity.ERROR))
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
    bar_rep, _ = jaxpr_model.check_kernel_static_models(
        barrier_missing_collective_id, (x_spec,)
    )
    self.assertFalse(bar_rep.passed, msg=bar_rep.summary())
    hw = bar_rep.findings_by_category(Category.HARDWARE_CONSTRAINT_VIOLATION)
    self.assertNotEmpty(hw, msg=bar_rep.summary())
    self.assertIn("collective_id", hw[0].message)

  def test_run_scoped_hbm_allocation_rejected(self):
    def run_scoped_hbm_alloc(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref):
        pl.run_scoped(
            lambda _hbm: o_ref.__setitem__(Ellipsis, x_ref[...]),
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

  def test_scratch_shapes_hbm_allocation_rejected(self):
    def scratch_hbm_alloc(x: jax.Array) -> jax.Array:
      def kernel(x_ref, o_ref, _hbm_scratch):
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
    models = jaxpr_model.extract_kernel_models(
        dyn_slice_dma, (idx_spec, x_spec)
    )
    self.assertLen(models, 1)
    expected_bytes = 8 * 128 * 4
    self.assertEqual(models[0].dma_starts[0].byte_count, expected_bytes)
    self.assertEqual(models[0].dma_waits[0].byte_count, expected_bytes)

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
    def non_consecutive_revisit(x: jax.Array) -> jax.Array:
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

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    revisit_rep, _ = jaxpr_model.check_kernel_static_models(
        non_consecutive_revisit, (x_spec,)
    )
    self.assertNotEmpty(
        revisit_rep.findings_by_category(
            Category.HARDWARE_CONSTRAINT_VIOLATION
        ),
        msg=revisit_rep.summary(),
    )
    self.assertIn("revisits output block", revisit_rep.summary())

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
              out_specs=pl.BlockSpec((8, 128), lambda i, _l: (i, 0)),
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

  def test_dma_to_dma_write_after_write_race_caught_beyond_vpu_pair_volume(
      self,
  ):
    grid = 32
    x_spec = jax.ShapeDtypeStruct((grid * 8, 128), jnp.float32)

    def dma_waw_with_vpu_traffic(x: jax.Array) -> jax.Array:
      def kernel(x_hbm, o_ref, vpu_buf, waw_buf, vpu_sem, waw_sems):
        i = pl.program_id(0)
        # Sequenced per-step DMA + VPU read/write creates 32*33/2 = 528
        # cross-step VPU-DMA pairs on `vpu_buf`.
        vpu_copy = pltpu.make_async_copy(
            x_hbm.at[pl.ds(i * 8, 8)], vpu_buf, vpu_sem
        )
        vpu_copy.start()
        vpu_copy.wait()
        vpu_buf[...] = vpu_buf[...] + 1.0
        o_ref[...] = vpu_buf[...]

        # Concurrent unawaited DMA-DMA WAW race on `waw_buf` between steps 0
        # and 1 using distinct semaphores waited together on step 1.
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

    rep = verifier.verify(dma_waw_with_vpu_traffic, (x_spec,), mode=STATIC)
    self.assertFalse(rep.passed, msg=rep.summary())
    self.assertNotEmpty(
        rep.findings_by_category(Category.DATA_RACE),
        msg=rep.summary(),
    )

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

  def test_parallel_grid_conflict_when_cpsat_unavailable_and_race_loop_timeout(
      self,
  ):
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

    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
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
        smt_checker, "_solve", side_effect=_timeout_on_race_query
    ):
      race_timeout_rep = verifier.verify(
          _shared_sem_race_jax, (x_spec,), mode=STATIC
      )
    incomplete = race_timeout_rep.findings_by_category(
        Category.VERIFICATION_INCOMPLETE
    )
    self.assertNotEmpty(incomplete, msg=race_timeout_rep.summary())
    self.assertEqual(incomplete[0].severity, verifier.Severity.WARNING)
    self.assertIn(
        "candidate pair(s) across",
        incomplete[0].message,
    )
    self.assertIn(
        "buffer bucket(s) unchecked",
        incomplete[0].message,
    )


if __name__ == "__main__":
  absltest.main()
