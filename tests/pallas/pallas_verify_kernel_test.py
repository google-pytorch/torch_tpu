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

"""Tests for Pallas kernel model checking and CPU interpret verification."""

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
from torch_tpu._internal import pallas
from torch_tpu._internal.pallas import _mock_device
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import interpret_verifier
from torch_tpu._internal.pallas.verify import jaxpr_model
from torch_tpu._internal.pallas.verify import verifier
from tests import seed_test_utils

P = jax.sharding.PartitionSpec
Category = findings.FindingCategory
STATIC = verifier.VerificationMode.STATIC
HBM_SPEC = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)


def _make_mock_tpu_mesh(num_devices: int, axis_name: str = "rank"):
  devs = np.array(
      [_mock_device._PallasDevice(id=i) for i in range(num_devices)]
  )
  return jax.sharding.Mesh(devs, (axis_name,))


def _shard_mapped(per_shard, mesh):
  return jax.shard_map(
      per_shard,
      mesh=mesh,
      in_specs=(P("rank"),),
      out_specs=P("rank"),
      check_vma=False,
  )


def _distributed_op(name, fn, mesh):
  return pallas.jax_op(
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

    op = pallas.jax_op(
        "pallas_verify_kernel::valid_double_buffer", double_buffered_copy
    )
    x = torch.arange(num_tiles * 8 * 128, dtype=torch.float32).reshape(-1, 128)
    report = verifier.verify(op, (x,))
    self.assertTrue(report.passed, msg=report.summary())
    self.assertEmpty(report.errors)
    self.assertContainsSubset(
        {
            findings.Phase.CONTRACTS,
            findings.Phase.LOWERING,
            findings.Phase.MODEL_CHECK,
            findings.Phase.INTERPRET,
        },
        set(report.checked_phases),
    )

  def test_unordered_dma_shared_semaphore_race_caught_with_counterexample(self):
    op = pallas.jax_op(
        "pallas_verify_kernel::buggy_shared_sem_dma", _shared_sem_race_jax
    )
    x = torch.ones((16, 128), dtype=torch.float32)
    report = verifier.verify(op, (x,), mode=STATIC)
    self.assertFalse(report.passed, msg=report.summary())
    race_findings = [
        f
        for f in report.errors
        if f.category in (
            Category.DATA_RACE,
            Category.CONCURRENT_DMA_SEM_SHARING,
        )
    ]
    self.assertNotEmpty(race_findings, msg=report.summary())
    self.assertNotEmpty(race_findings[0].counterexample)

  def test_raw_jax_function_is_verified_from_shape_dtype_structs(self):
    x_spec = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    report = verifier.verify(_shared_sem_race_jax, (x_spec,), mode=STATIC)
    self.assertFalse(report.passed, msg=report.summary())
    self.assertTrue(
        report.has_category(Category.DATA_RACE)
        or report.has_category(Category.CONCURRENT_DMA_SEM_SHARING),
        msg=report.summary(),
    )

  def test_raw_jax_function_runs_on_example_values_in_full_mode(self):
    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    report = verifier.verify(_make_tiled_copy(4), (x_spec,))
    self.assertTrue(report.passed, msg=report.summary())
    self.assertEmpty(report.warnings, msg=report.summary())
    self.assertIn(findings.Phase.INTERPRET, report.checked_phases)

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
        if f.phase is findings.Phase.INTERPRET
    ]
    # Reported once: the other schedules would fail the same way.
    self.assertLen(incomplete, 1, msg=report.summary())
    self.assertEqual(incomplete[0].severity, findings.Severity.WARNING)
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
    self.assertEmpty(report.errors)

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
    self.assertTrue(
        buggy_rep.has_category(Category.UNTHROTTLED_REMOTE_OVERWRITE)
        or buggy_rep.has_category(Category.DATA_RACE),
        msg=buggy_rep.summary(),
    )
    self.assertNotEmpty(buggy_rep.errors[0].counterexample)

    fixed_rep = verifier.verify(fixed_op, (x_local,))
    self.assertTrue(fixed_rep.passed, msg=fixed_rep.summary())

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
        interpret_verifier, "remote_interpret_supported", return_value=False
    ):
      report = verifier.verify(op, (torch.ones((8, 128)),))
    self.assertTrue(report.passed, msg=report.summary())
    self.assertNotIn(findings.Phase.INTERPRET, report.checked_phases)
    skipped = [
        f for f in report.warnings if f.phase is findings.Phase.INTERPRET
    ]
    self.assertLen(skipped, 1, msg=report.summary())

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
    self.assertEmpty(report.warnings, msg=report.summary())
    self.assertIn(findings.Phase.MODEL_CHECK, report.checked_phases)

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

    leak_op = pallas.jax_op("pallas_verify_kernel::leaky_sem_op", leaky_sem_jax)
    leak_rep = verifier.verify(leak_op, (x,), mode=STATIC)
    self.assertFalse(leak_rep.passed)
    self.assertTrue(
        leak_rep.has_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=leak_rep.summary(),
    )

    deadlock_op = pallas.jax_op(
        "pallas_verify_kernel::deadlock_sem_op", deadlocking_sem_jax
    )
    deadlock_rep = verifier.verify(deadlock_op, (x,), mode=STATIC)
    self.assertFalse(deadlock_rep.passed)
    self.assertTrue(
        deadlock_rep.has_category(Category.SEMAPHORE_DEADLOCK),
        msg=deadlock_rep.summary(),
    )

    cyclic_op = _distributed_op(
        "pallas_verify_kernel::cyclic_deadlock_sem_op",
        cyclic_multichip_deadlock_jax,
        mesh2,
    )
    cyclic_rep = verifier.verify(cyclic_op, (x,), mode=STATIC)
    self.assertFalse(cyclic_rep.passed)
    self.assertTrue(
        cyclic_rep.has_category(Category.SEMAPHORE_DEADLOCK),
        msg=cyclic_rep.summary(),
    )

    # FULL mode must report both findings and skip CPU interpret instead of
    # hanging on the deadlocked rank.
    asym_op = _distributed_op(
        "pallas_verify_kernel::asym_misrouted_sem_op",
        asym_misrouted_sem_jax,
        mesh2,
    )
    asym_rep = verifier.verify(asym_op, (x,))
    self.assertFalse(asym_rep.passed, msg=asym_rep.summary())
    self.assertTrue(
        asym_rep.has_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=asym_rep.summary(),
    )
    self.assertTrue(
        asym_rep.has_category(Category.SEMAPHORE_DEADLOCK),
        msg=asym_rep.summary(),
    )
    self.assertNotIn(findings.Phase.INTERPRET, asym_rep.checked_phases)

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
    self.assertTrue(
        bad_rep.has_category(Category.SEMAPHORE_DEADLOCK),
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
    bad_op = pallas.jax_op(
        "pallas_verify_kernel::dma_src_overwrite",
        make_store(overwrite_before_wait=True),
    )
    bad_rep = verifier.verify(bad_op, (x,), mode=STATIC)
    self.assertTrue(
        bad_rep.has_category(Category.DATA_RACE), msg=bad_rep.summary()
    )

    ok_op = pallas.jax_op(
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
    parallel_bug_op = pallas.jax_op(
        "pallas_verify_kernel::parallel_grid_bug",
        make_grid_reduction("parallel"),
    )
    arbitrary_ok_op = pallas.jax_op(
        "pallas_verify_kernel::arbitrary_grid_ok",
        make_grid_reduction("arbitrary"),
    )

    bug_rep = verifier.verify(parallel_bug_op, (x,), mode=STATIC)
    self.assertFalse(bug_rep.passed, msg=bug_rep.summary())
    self.assertTrue(
        bug_rep.has_category(Category.PARALLEL_GRID_CONFLICT),
        msg=bug_rep.summary(),
    )

    ok_rep = verifier.verify(arbitrary_ok_op, (x,))
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())

  def test_output_tiles_not_written_detected(self):
    x = torch.ones((32, 128), dtype=torch.float32)
    full_op = pallas.jax_op(
        "pallas_verify_kernel::full_grid_copy", _make_tiled_copy(4)
    )
    full_rep = verifier.verify(full_op, (x,), mode=STATIC)
    self.assertTrue(full_rep.passed, msg=full_rep.summary())

    # A grid that is one step short never writes the last output tile.
    short_op = pallas.jax_op(
        "pallas_verify_kernel::short_grid_copy", _make_tiled_copy(3)
    )
    short_rep = verifier.verify(short_op, (x,), mode=STATIC)
    self.assertTrue(
        short_rep.has_category(Category.UNWRITTEN_OUTPUT),
        msg=short_rep.summary(),
    )
    self.assertIn("24:32", short_rep.summary())

    # Writing half of each output tile leaves the other half uninitialized.
    half_op = pallas.jax_op(
        "pallas_verify_kernel::half_tile_copy",
        _make_tiled_copy(4, rows_written=4),
    )
    half_rep = verifier.verify(half_op, (x,), mode=STATIC)
    self.assertTrue(
        half_rep.has_category(Category.UNWRITTEN_OUTPUT),
        msg=half_rep.summary(),
    )

  def test_fori_loop_writes_are_modeled(self):
    def make_fori_copy(num_iterations: int):
      def fori_copy(x: jax.Array) -> jax.Array:
        def kernel(x_ref, o_ref):
          def body(i, carry):
            rows = pl.ds(pl.multiple_of(i * 8, 8), 8)
            o_ref[rows, :] = x_ref[rows, :]
            return carry

          # Static bounds make this a `scan`.
          lax.fori_loop(0, num_iterations, body, 0)

        return pl.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
        )(x)

      return fori_copy

    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    ok_rep = verifier.verify(make_fori_copy(4), (x_spec,), mode=STATIC)
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    self.assertEmpty(ok_rep.warnings, msg=ok_rep.summary())
    short_rep = verifier.verify(make_fori_copy(3), (x_spec,), mode=STATIC)
    self.assertTrue(
        short_rep.has_category(Category.UNWRITTEN_OUTPUT),
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
    self.assertTrue(
        leak_rep.has_category(Category.SEMAPHORE_NONZERO_EXIT),
        msg=leak_rep.summary(),
    )
    race_rep = verifier.verify(
        make_per_step_copy(leak=False), (x_spec,), mode=STATIC
    )
    self.assertTrue(
        race_rep.has_category(Category.DATA_RACE)
        or race_rep.has_category(Category.CONCURRENT_DMA_SEM_SHARING),
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
            for f in report.warnings
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

  def test_index_maps_with_nested_calls_are_evaluated(self):
    def make_copy(index_map):
      def copy(x: jax.Array) -> jax.Array:
        return pl.pallas_call(
            lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
            grid=(4,),
            in_specs=[pl.BlockSpec((8, 128), index_map)],
            out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
        )(x)

      return copy

    x_spec = jax.ShapeDtypeStruct((32, 128), jnp.float32)
    # `//` is a nested `jit` of `jnp.floor_divide` in the index map's jaxpr.
    ok_rep = verifier.verify(
        make_copy(lambda i: (i // 2 + 2, 0)), (x_spec,), mode=STATIC
    )
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())
    self.assertNotIn("runtime data", ok_rep.summary())
    oob_rep = verifier.verify(
        make_copy(lambda i: (i // 2 + 3, 0)), (x_spec,), mode=STATIC
    )
    self.assertTrue(
        oob_rep.has_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=oob_rep.summary(),
    )

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
    self.assertFalse(
        report.has_category(Category.NAN_OR_UNINIT_READ), msg=report.summary()
    )

  def test_while_loop_writes_are_modeled(self):
    def make_loop_copy(num_iterations: int):
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

    x = torch.ones((32, 128), dtype=torch.float32)
    ok_op = pallas.jax_op(
        "pallas_verify_kernel::while_copy_ok", make_loop_copy(4)
    )
    ok_rep = verifier.verify(ok_op, (x,), mode=STATIC)
    self.assertTrue(ok_rep.passed, msg=ok_rep.summary())

    short_op = pallas.jax_op(
        "pallas_verify_kernel::while_copy_short", make_loop_copy(3)
    )
    short_rep = verifier.verify(short_op, (x,), mode=STATIC)
    self.assertTrue(
        short_rep.has_category(Category.UNWRITTEN_OUTPUT),
        msg=short_rep.summary(),
    )

  def test_out_of_bounds_block_index_and_dma_slice_detected(self):
    def shifted_block_copy(x: jax.Array) -> jax.Array:
      return pl.pallas_call(
          lambda x_ref, o_ref: o_ref.__setitem__(Ellipsis, x_ref[...]),
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid=(4,),
          # Off by one: the last grid step reads block 4 of a 4-block array.
          in_specs=[pl.BlockSpec((8, 128), lambda i: (i + 1, 0))],
          out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
      )(x)

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

    block_op = pallas.jax_op(
        "pallas_verify_kernel::oob_block_index", shifted_block_copy
    )
    block_rep = verifier.verify(block_op, (torch.ones((32, 128)),), mode=STATIC)
    self.assertTrue(
        block_rep.has_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=block_rep.summary(),
    )

    dma_op = pallas.jax_op(
        "pallas_verify_kernel::oob_dma_slice", shifted_dma_copy
    )
    dma_rep = verifier.verify(dma_op, (torch.ones((16, 128)),), mode=STATIC)
    self.assertTrue(
        dma_rep.has_category(Category.OUT_OF_BOUNDS_ACCESS),
        msg=dma_rep.summary(),
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
    self.assertTrue(
        report.has_category(Category.INVALID_DEVICE_ID), msg=report.summary()
    )

  def test_tpu_lowering_failure_reported(self):
    def hbm_vector_store(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_hbm_ref):
        # Vector stores cannot target HBM refs; Mosaic rejects this kernel.
        y_hbm_ref[...] = x_ref[...]

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          out_specs=HBM_SPEC,
      )(x)

    op = pallas.jax_op("pallas_verify_kernel::hbm_store", hbm_vector_store)
    report = verifier.verify(op, (torch.ones((8, 128)),), mode=STATIC)
    lowering = report.findings_by_category(Category.LOWERING_FAILURE)
    self.assertNotEmpty(lowering, msg=report.summary())
    self.assertIn("VMEM", lowering[0].message)

  def test_solver_timeout_is_reported_as_incomplete(self):
    op = pallas.jax_op(
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
    self.assertEqual(incomplete[0].severity, findings.Severity.WARNING)

  def test_uninitialized_vmem_nan_leak_and_bf16_matmul_precision(self):
    """Interpret catches uninit VMEM NaN leaks; bf16 dot results are flagged."""

    def uninit_vmem_leak(x: jax.Array) -> jax.Array:
      def kernel(x_ref, y_ref):
        def body(scratch_vmem):
          # Only writes the first half of `scratch_vmem`, then reads all of it.
          scratch_vmem.at[0:8][...] = x_ref.at[0:8][...]
          y_ref[...] = scratch_vmem[...]

        pl.run_scoped(body, pltpu.VMEM(x.shape, x.dtype))

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      )(x)

    def dot_without_f32_accum(a: jax.Array, b: jax.Array) -> jax.Array:
      def kernel(a_ref, b_ref, o_ref):
        # Missing `preferred_element_type=jnp.float32` with bf16 inputs.
        o_ref[...] = lax.dot(a_ref[...], b_ref[...])

      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), a.dtype),
      )(a, b)

    x = torch.ones((16, 128), dtype=torch.float32)
    uninit_op = pallas.jax_op(
        "pallas_verify_kernel::uninit_vmem_leak", uninit_vmem_leak
    )
    interp_rep = verifier.verify(uninit_op, (x,))
    self.assertFalse(interp_rep.passed)
    self.assertTrue(
        interp_rep.has_category(Category.NAN_OR_UNINIT_READ),
        msg=interp_rep.summary(),
    )

    a = torch.ones((8, 128), dtype=torch.bfloat16)
    b = torch.ones((128, 128), dtype=torch.bfloat16)
    bf16_op = pallas.jax_op(
        "pallas_verify_kernel::bf16_dot_bad_accum", dot_without_f32_accum
    )
    hw_rep = verifier.verify(bf16_op, (a, b), mode=STATIC)
    precision = hw_rep.findings_by_category(Category.LOW_PRECISION_ACCUMULATION)
    self.assertNotEmpty(precision, msg=hw_rep.summary())
    self.assertEqual(precision[0].severity, findings.Severity.WARNING)

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

    dtype_sem_op = pallas.jax_op(
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
    self.assertTrue(
        report.has_category(Category.SEMAPHORE_NONZERO_EXIT),
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


if __name__ == "__main__":
  absltest.main()
