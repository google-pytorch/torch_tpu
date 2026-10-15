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

"""Device addressing when a Pallas kernel's Mesh meets ordinary collectives.

A compiled module has one DeviceAssignment, installed here from a Pallas
kernel's `mesh.devices.flat`. Collectives in that same module name devices by
their index in `pjrt_client->devices()`. These tests ask whether the kernel, a
MESH-addressed DMA and six collectives all still resolve to the PyTorch rank.

Needs a part where rank order and device order differ: on single-core-per-chip
parts (v5e, v6e) rank r opens device r, the Mesh installs the default
DeviceAssignment, and nothing here can detect anything -- hence
running on TPU v7x and `_require_non_identity_device_order`.
"""

from collections.abc import Callable, Sequence
import dataclasses
import re

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal import pallas
from torch_tpu._internal.distributed import device_mesh
from torch_tpu._internal.distributed import tpu_distributed
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils

P = jax.sharding.PartitionSpec

# Mesh axis name. Also the name the kernel passes to `lax.axis_index`.
_AXIS = "ring"

# Kernel tile. 8x128 is the native VMEM tile; only rows 0..2 carry information.
_ROWS = 8
_COLS = 128

# Row meanings in the kernel output, which is `(1, _ROWS, _COLS)` int32.
_ROW_AXIS_INDEX = 0  # what `lax.axis_index(_AXIS)` resolved to.
_ROW_SENT = 1  # the payload this device put on the wire.
_ROW_RECEIVED = 2  # the payload this device took off the wire.

# Custom-kernel functions are inlined into `@main` during `BuildMlirModule`,
# leaving each Pallas kernel's `@tpu_custom_call` (or un-inlined `call @`) at
# the call site.
_KERNEL_CALL_SITE_RE = re.compile(r"(?:\bcall @|@tpu_custom_call\b)")

# How a case reacts to a permuted DeviceAssignment.
_ORDER_SENSITIVE = "order-sensitive"
_CONTROL = "control (permutation-invariant)"
_MEMBERSHIP_SENSITIVE = "membership-sensitive"
_ROOT_SELECTION = "root-selection"

_FUSED_BROADCAST_SKIP = (
    "TPU has no native broadcast collective (process_group_tpu.cc:882-885); it"
    " is emulated as zero-non-roots + all_reduce(SUM), and torch.compile"
    " traces the zeroing into the graph, so non-root ranks return zeros by"
    " construction. Co-located broadcast device order is untestable here."
)


# Broadcast roots: first, last, middle.
def _broadcast_sources(world_size: int) -> tuple[int, ...]:
  return (0, world_size - 1, world_size // 2)


# ==============================================================================
# The kernel: a one-hop ring DMA addressed through MESH coordinates.
# ==============================================================================


def _build_ring_op(
    mesh: jax.sharding.Mesh, world_size: int, op_name: str
) -> Callable[[torch.Tensor], torch.Tensor]:
  """Registers a mesh-carrying Pallas kernel as a torch custom op.

  Each device copies its own shard one hop along `_AXIS` with
  `make_async_remote_copy` addressed by MESH coordinate, then reports in the
  first three rows of its output: row 0 `lax.axis_index(_AXIS)`, row 1 what it
  sent (its own input, which carries its rank), row 2 what it received.

  Args:
    mesh: The device mesh whose order is under test.
    world_size: Number of devices along `_AXIS`.
    op_name: Namespaced torch custom op name.

  Returns:
    A callable taking the local `(1, _ROWS, _COLS)` int32 shard.
  """

  def body(x_ref, o_ref, scratch, send_sem, recv_sem):
    my_id = jax.lax.axis_index(_AXIS)
    next_id = (my_id + 1) % world_size

    scratch[0] = x_ref[0]
    copy = pltpu.make_async_remote_copy(
        src_ref=scratch.at[0],
        dst_ref=scratch.at[1],
        send_sem=send_sem,
        recv_sem=recv_sem,
        device_id={_AXIS: next_id},
    )
    copy.start()
    copy.wait()

    row = jax.lax.broadcasted_iota(jnp.int32, (_ROWS, _COLS), 0)
    mine = jnp.full((_ROWS, _COLS), 0, jnp.int32) + my_id.astype(jnp.int32)
    o_ref[0] = jnp.where(
        row == _ROW_AXIS_INDEX,
        mine,
        jnp.where(row == _ROW_SENT, x_ref[0], scratch[1]),
    )

  kernel = pl.pallas_call(
      body,
      out_shape=jax.ShapeDtypeStruct((1, _ROWS, _COLS), jnp.int32),
      scratch_shapes=[
          pltpu.VMEM((2, _ROWS, _COLS), jnp.int32),
          pltpu.SemaphoreType.DMA,
          pltpu.SemaphoreType.DMA,
      ],
  )

  def fn(x: jax.Array) -> jax.Array:
    sharded = jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=(P(_AXIS),),
        out_specs=P(_AXIS),
        check_vma=False,
    )
    return sharded(x)

  return pallas.jax_op(
      op_name,
      fn,
      mesh=mesh,
      input_partition_specs=[P(_AXIS)],
      output_partition_specs=[P(_AXIS)],
  )


@dataclasses.dataclass(frozen=True)
class _KernelReport:
  """The three numbers the kernel reports about itself."""

  axis_index: int
  sent: int
  received: int


def _read_kernel_output(y: torch.Tensor) -> _KernelReport:
  flat = y.reshape(-1).cpu()
  return _KernelReport(
      axis_index=int(flat[_ROW_AXIS_INDEX * _COLS]),
      sent=int(flat[_ROW_SENT * _COLS]),
      received=int(flat[_ROW_RECEIVED * _COLS]),
  )


# ==============================================================================
# Placement.
# ==============================================================================


def _mesh_device_ids(mesh: jax.sharding.Mesh) -> list[int]:
  return [int(d.id) for d in mesh.devices.flat]


def _pjrt_device_ids() -> list[int]:
  return [int(v) for v in tpu_distributed.all_global_device_ids()]


def _log_placement(mesh: jax.sharding.Mesh) -> None:
  logging.info(
      "[placement] rank=%d device_id=%d rank->device=%s pjrt devices()=%s"
      " mesh=%s",
      dist.get_rank(),
      int(tpu_distributed.global_device_id()),
      device_mesh.get_device_assignment(),
      _pjrt_device_ids(),
      _mesh_device_ids(mesh),
  )


def _require_non_identity_device_order(mesh: jax.sharding.Mesh) -> None:
  """Stops the run when the rank -> device map is the identity.

  Args:
    mesh: The mesh the kernel will carry.

  Raises:
    RuntimeError: if the rank -> device map is the identity, in which case the
      Mesh installs the default DeviceAssignment and nothing here can detect a
      device-order fault.
  """
  mesh_ids = _mesh_device_ids(mesh)
  pjrt_ids = _pjrt_device_ids()
  if mesh_ids == pjrt_ids:
    raise RuntimeError(
        f"rank->device map is identity ({mesh_ids}); this test cannot detect "
        "device-order faults on this topology. It needs a part where rank "
        "order and device order differ, such as TPU v7x."
    )


# ==============================================================================
# The collectives.
#
# Every payload is int32 and twice as long as it needs to be: even slots carry
# the value under test, odd slots carry `ballast`. In the fused cases ballast is
# a real element of the kernel's output, which is what gives the collective an
# unavoidable data dependency on the kernel; in the solo cases it is a constant.
# Nothing reads the odd slots, so the two runs are directly comparable.
# ==============================================================================


def _interleave(base: torch.Tensor, ballast: torch.Tensor) -> torch.Tensor:
  """Interleaves `base` (even slots) with `ballast` (odd slots)."""
  filler = ballast.to(torch.int32).reshape(1).expand(base.shape)
  return torch.stack((base, filler), dim=1).reshape(-1)


def _even_slots(out: torch.Tensor) -> list[int]:
  return out.cpu().tolist()[0::2]


@dataclasses.dataclass(frozen=True)
class _CollectiveCase:
  """One collective, its input on this rank and the answer it must produce.

  Attributes:
    name: Case name, used in every log line and failure message.
    kind: How the case reacts to a permuted DeviceAssignment. One of
      `_ORDER_SENSITIVE`, `_CONTROL`, `_MEMBERSHIP_SENSITIVE`,
      `_ROOT_SELECTION`.
    shlo_op: The StableHLO op this collective lowers to, which the fused path
      looks for in the compiled module. Empty where the fused path skips the
      case.
    payload: This rank's even-slot input values.
    expected: The even-slot values this rank must observe.
    run: Runs the collective, given the even-slot payload and the ballast.
  """

  name: str
  kind: str
  shlo_op: str
  payload: list[int]
  expected: list[int]
  run: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _build_cases(
    rank: int,
    world_size: int,
    subgroup: dist.ProcessGroup,
    subgroup_members: Sequence[int],
) -> list[_CollectiveCase]:
  """Builds the six collective cases.

  Args:
    rank: This process's rank in the world group.
    world_size: World size.
    subgroup: The half-world group this rank belongs to.
    subgroup_members: The ranks in `subgroup`.

  Returns:
    The cases.
  """

  def all_gather(base: torch.Tensor, ballast: torch.Tensor) -> torch.Tensor:
    out = torch.empty(2 * world_size, dtype=torch.int32, device="tpu")
    dist.all_gather_into_tensor(out, _interleave(base, ballast))
    return out

  def reduce_scatter(base: torch.Tensor, ballast: torch.Tensor) -> torch.Tensor:
    out = torch.empty(2, dtype=torch.int32, device="tpu")
    dist.reduce_scatter_tensor(out, _interleave(base, ballast))
    return out

  def all_to_all(base: torch.Tensor, ballast: torch.Tensor) -> torch.Tensor:
    out = torch.empty(2 * world_size, dtype=torch.int32, device="tpu")
    dist.all_to_all_single(out, _interleave(base, ballast))
    return out

  def all_reduce_world(
      base: torch.Tensor, ballast: torch.Tensor
  ) -> torch.Tensor:
    inp = _interleave(base, ballast)
    dist.all_reduce(inp)
    return inp

  def all_reduce_subgroup(
      base: torch.Tensor, ballast: torch.Tensor
  ) -> torch.Tensor:
    inp = _interleave(base, ballast)
    dist.all_reduce(inp, group=subgroup)
    return inp

  def broadcast(base: torch.Tensor, ballast: torch.Tensor) -> torch.Tensor:
    outs = []
    for i, src in enumerate(_broadcast_sources(world_size)):
      held = _interleave(base[i : i + 1], ballast)
      dist.broadcast(held, src=src)
      outs.append(held)
    return torch.cat(outs)

  srcs = _broadcast_sources(world_size)

  return [
      _CollectiveCase(
          name="all_gather",
          kind=_ORDER_SENSITIVE,
          shlo_op="stablehlo.all_gather",  # csrc/distributed/allgather.cc:65.
          payload=[rank],
          expected=list(range(world_size)),
          run=all_gather,
      ),
      _CollectiveCase(
          name="reduce_scatter",
          kind=_ORDER_SENSITIVE,
          # csrc/distributed/reduce_scatter.cc:69.
          shlo_op="stablehlo.reduce_scatter",
          # Only the slot destined for rank r is non-zero, and only rank r sets
          # it, so the sum that lands on rank r is r and a permutation shows up
          # directly in the value.
          payload=[rank if i == rank else 0 for i in range(world_size)],
          expected=[rank],
          run=reduce_scatter,
      ),
      _CollectiveCase(
          name="all_to_all",
          kind=_ORDER_SENSITIVE,
          shlo_op="stablehlo.all_to_all",  # csrc/distributed/alltoall.cc:59.
          # Chunk j goes to rank j carrying `rank * world + j`, so every
          # returned integer decodes to a (sender, slot) pair.
          payload=[rank * world_size + j for j in range(world_size)],
          expected=[j * world_size + rank for j in range(world_size)],
          run=all_to_all,
      ),
      _CollectiveCase(
          name="all_reduce_world",
          kind=_CONTROL,
          shlo_op="stablehlo.all_reduce",  # csrc/distributed/allreduce.cc:53.
          payload=[rank],
          expected=[world_size * (world_size - 1) // 2],
          run=all_reduce_world,
      ),
      _CollectiveCase(
          name="all_reduce_subgroup",
          kind=_MEMBERSHIP_SENSITIVE,
          shlo_op="stablehlo.all_reduce",  # csrc/distributed/allreduce.cc:53.
          payload=[rank],
          expected=[sum(subgroup_members)],
          run=all_reduce_subgroup,
      ),
      _CollectiveCase(
          name="broadcast",
          kind=_ROOT_SELECTION,
          # No literal: the fused path skips broadcast (`_FUSED_BROADCAST_SKIP`)
          # and the solo path does not read this field.
          shlo_op="",
          # Membership is not covered: the emulation all-reduces over WORLD, so
          # covering it would take a broadcast over a subgroup.
          payload=[1000 + src if rank == src else 0 for src in srcs],
          expected=[1000 + src for src in srcs],
          run=broadcast,
      ),
  ]


def _select_case(case_name: str, rank: int, world_size: int) -> _CollectiveCase:
  """Creates the half-world subgroups and returns the case called `case_name`.

  Args:
    case_name: The `_CollectiveCase.name` to select.
    rank: This process's rank in the world group.
    world_size: World size.

  Returns:
    The selected case.

  Raises:
    ValueError: if `case_name` names no case.
  """
  half = world_size // 2
  low_ranks = list(range(half))
  high_ranks = list(range(half, world_size))

  low = dist.new_group(ranks=low_ranks)
  high = dist.new_group(ranks=high_ranks)
  in_low = rank in low_ranks
  subgroup_members = low_ranks if in_low else high_ranks
  cases = {
      case.name: case
      for case in _build_cases(
          rank,
          world_size,
          subgroup=low if in_low else high,
          subgroup_members=subgroup_members,
      )
  }
  if case_name not in cases:
    raise ValueError(f"unknown case {case_name!r}; have {sorted(cases)}")
  return cases[case_name]


def _run_solo(
    case: _CollectiveCase, base: torch.Tensor, ballast: torch.Tensor
) -> list[int]:
  return _even_slots(case.run(base, ballast))


def _assert_one_module_holds_both(
    modules: Sequence[str], shlo_op: str, what: str
) -> None:
  """Asserts a single compiled module holds the kernel call and the collective.

  Args:
    modules: The StableHLO text of every module the region compiled to.
    shlo_op: The `_CollectiveCase.shlo_op` under test.
    what: Names the case in the failure message.

  Raises:
    AssertionError: if no one module holds both.
  """
  seen = [(bool(_KERNEL_CALL_SITE_RE.search(m)), shlo_op in m) for m in modules]
  if any(kernel and collective for kernel, collective in seen):
    return
  per_module = "\n".join(
      f"  module {i}: kernel_call_site={kernel} {shlo_op}={collective}"
      for i, (kernel, collective) in enumerate(seen)
  )
  raise AssertionError(
      f"{what}: no single compiled module holds both a Pallas kernel call site "
      f"and {shlo_op}, so the collective never ran under the kernel's "
      f"DeviceAssignment. {len(modules)} module(s):\n{per_module}"
  )


def _run_fused(
    case: _CollectiveCase,
    base: torch.Tensor,
    kernel_op: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
) -> list[int]:
  """Runs the collective inside one compiled region with the kernel.

  `fullgraph=True` is load-bearing: a graph break between the kernel and the
  collective raises instead of silently degrading this into a second solo run.
  It is not sufficient, though -- it buys one Dynamo graph, not one compiled
  module -- so the module text is read back and checked.

  Args:
    case: The collective under test.
    base: The even-slot payload, built outside the region so that tracing it
      cannot introduce a graph break.
    kernel_op: The mesh-carrying Pallas op.
    x: The kernel input.

  Returns:
    The even-slot values this rank observed.
  """

  # `spmd_safe` is what actually keeps the two in one module. Without it the
  # splitter gives every collective its own partition
  # (_internal/compile/split_compiler.py:635-643), so the collective compiles
  # away from the kernel and never sees the kernel's DeviceAssignment. The
  # decorator only suppresses that split -- it does not touch DeviceAssignment,
  # meshes or replica groups -- so it cannot manufacture the property under
  # test. Under compile it annotates the traced nodes (see `spmd_safe` in
  # _internal/distributed/spmd_util.py).
  @torch.tpu.distributed.spmd_safe
  def region(kernel_input: torch.Tensor, payload: torch.Tensor):
    kernel_out = kernel_op(kernel_input)
    # A real element of the kernel output, so the collective cannot be
    # scheduled without the kernel.
    ballast = kernel_out.reshape(-1)[0]
    return case.run(payload, ballast)

  # `debug_callback` (_internal/compile/_backend.py:683-711) is how the
  # compiled module text is read back; `serializable=False` keeps it a fresh
  # compile rather than an AOTAutogradCache hit. `sync.computation_mlir` is not
  # usable here: a torch.compile output carries no deferred op, so the
  # traversal it builds is a passthrough (csrc/eager/traversal.cc:116-130).
  debugs = []
  compiled = torch.compile(
      region,
      backend="tpu",
      fullgraph=True,
      dynamic=False,
      options={"serializable": False, "debug_callback": debugs.append},
  )
  observed = _even_slots(compiled(x, base))
  _assert_one_module_holds_both(
      [mlir for debug in debugs for mlir in debug.stablehlo_forward_text],
      case.shlo_op,
      f"fused {case.name}, rank {dist.get_rank()}",
  )
  return observed


# ==============================================================================
# Workers.
# ==============================================================================


def _in_process_group(body: Callable[[], None]) -> None:
  """Runs `body` inside a `tpu_dist` process group.

  The failure path deliberately does NOT barrier: a rank that failed an
  assertion must not wait for peers that may have passed, or a clean failure
  becomes a slice-wide hang.

  Args:
    body: The worker logic.
  """
  dist.init_process_group(backend="tpu_dist")
  try:
    body()
  except BaseException:
    dist.destroy_process_group()
    raise
  dist.barrier()
  dist.destroy_process_group()


def _rank_ordered_kernel_op(
    op_name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
  """Builds the ring op over the rank-ordered mesh, after checking placement."""
  mesh = torch.tpu.pallas.get_pallas_mesh(axis_names=(_AXIS,))
  _log_placement(mesh)
  _require_non_identity_device_order(mesh)
  return _build_ring_op(mesh, dist.get_world_size(), op_name)


def run_kernel_report_check(check: str) -> None:
  """Checks one field of the kernel's self-report against this rank.

  Args:
    check: One of `axis_index`, `sent`, `received`.
  """

  def body() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    op = _rank_ordered_kernel_op("pallas_mesh_order::ring")

    x = torch.full((1, _ROWS, _COLS), rank, dtype=torch.int32, device="tpu")
    report = _read_kernel_output(op(x))
    logging.info("[kernel] rank=%d %s", rank, report)

    expected = {
        "axis_index": rank,
        "sent": rank,
        "received": (rank - 1) % world_size,
    }[check]
    observed = getattr(report, check)
    if observed != expected:
      # The competing hypothesis is that the kernel's identity follows the
      # device enumeration, so report that index alongside the rank.
      device_order_index = _pjrt_device_ids().index(
          int(tpu_distributed.global_device_id())
      )
      raise AssertionError(
          f"kernel {check}: rank {rank} expected {expected}, observed "
          f"{observed}. Full report: {report}. This device's index in "
          f"pjrt_client->devices() is {device_order_index}."
      )

  _in_process_group(body)


def run_collective_solo(case_name: str) -> None:
  """Runs one collective in a traversal of its own.

  Args:
    case_name: The `_CollectiveCase.name` to run.
  """

  def body() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    op = _rank_ordered_kernel_op("pallas_mesh_order::ring_solo")

    x = torch.full((1, _ROWS, _COLS), rank, dtype=torch.int32, device="tpu")
    # Materialized here so the collective below gets a traversal to itself.
    logging.info("[kernel] rank=%d %s", rank, _read_kernel_output(op(x)))

    case = _select_case(case_name, rank, world_size)
    payload = torch.tensor(case.payload, dtype=torch.int32, device="tpu")
    ballast = torch.zeros((), dtype=torch.int32, device="tpu")
    observed = _run_solo(case, payload, ballast)
    if observed != case.expected:
      raise AssertionError(
          f"solo {case.name} ({case.kind}): rank {rank} expected "
          f"{case.expected}, observed {observed}"
      )

  _in_process_group(body)


def run_collective_fused(case_name: str) -> None:
  """Runs one collective in the same compiled module as the kernel.

  Args:
    case_name: The `_CollectiveCase.name` to run.
  """

  def body() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    op = _rank_ordered_kernel_op("pallas_mesh_order::ring_fused")

    x = torch.full((1, _ROWS, _COLS), rank, dtype=torch.int32, device="tpu")
    case = _select_case(case_name, rank, world_size)
    payload = torch.tensor(case.payload, dtype=torch.int32, device="tpu")
    observed = _run_fused(case, payload, op, x)
    if observed != case.expected:
      raise AssertionError(
          f"fused {case.name} ({case.kind}): rank {rank} expected "
          f"{case.expected}, observed {observed}. The solo case of the same "
          "name is the baseline; if it is green and this is not, the "
          "collective changed answer once it shared the kernel's module."
      )

  _in_process_group(body)


def run_two_mesh_probe(probe_id: str) -> None:
  """Compiles two Pallas kernels into one `torch.compile` region and inspects it.

  Args:
    probe_id: One of `two_same_mesh_kernels_share_one_traversal`,
      `two_same_mesh_kernels_compile_cleanly`, or
      `two_conflicting_mesh_kernels_are_rejected`.
  """

  def body() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    forward_mesh = torch.tpu.pallas.get_pallas_mesh(axis_names=(_AXIS,))
    _log_placement(forward_mesh)
    _require_non_identity_device_order(forward_mesh)

    if probe_id == "two_conflicting_mesh_kernels_are_rejected":
      second_mesh = jax.sharding.Mesh(
          forward_mesh.devices[::-1], axis_names=forward_mesh.axis_names
      )
      if _mesh_device_ids(second_mesh) == _mesh_device_ids(forward_mesh):
        raise RuntimeError(
            "rank->device map is a palindrome"
            f" ({_mesh_device_ids(forward_mesh)}), so the reversed mesh equals"
            " the forward one and there is no conflict for the compiler to"
            " resolve."
        )
    else:
      second_mesh = forward_mesh
    logging.info(
        "[two-mesh] rank=%d probe_id=%s mesh_a=%s mesh_b=%s",
        rank,
        probe_id,
        _mesh_device_ids(forward_mesh),
        _mesh_device_ids(second_mesh),
    )

    first = _build_ring_op(
        forward_mesh, world_size, "pallas_mesh_order::ring_a"
    )
    second = _build_ring_op(
        second_mesh, world_size, "pallas_mesh_order::ring_b"
    )
    x = torch.full((1, _ROWS, _COLS), rank, dtype=torch.int32, device="tpu")

    @torch.tpu.distributed.spmd_safe
    def region(t: torch.Tensor) -> torch.Tensor:
      return second(first(t))

    debugs = []
    compiled = torch.compile(
        region,
        backend="tpu",
        fullgraph=True,
        dynamic=False,
        options={"serializable": False, "debug_callback": debugs.append},
    )

    if probe_id == "two_same_mesh_kernels_share_one_traversal":
      _ = compiled(x)
      modules = [m for d in debugs for m in d.stablehlo_forward_text]
      if not any(len(_KERNEL_CALL_SITE_RE.findall(m)) == 2 for m in modules):
        counts = [len(_KERNEL_CALL_SITE_RE.findall(m)) for m in modules]
        raise AssertionError(
            f"{probe_id} (rank {rank}): expected at least one compiled module "
            f"with 2 Pallas kernel call sites, observed counts={counts} across "
            f"{len(modules)} module(s)."
        )
    elif probe_id == "two_same_mesh_kernels_compile_cleanly":
      _ = compiled(x).cpu()
      torch.accelerator.synchronize()
    elif probe_id == "two_conflicting_mesh_kernels_are_rejected":
      with absltest.TestCase().assertRaisesRegex(  # ABSLTEST_OK=helper only ASSERT_RAISES_OK=Requires multi-device hardware to test mesh conflict rejection in traversal
          (RuntimeError, torch._dynamo.exc.BackendCompilerFailed),
          r"expected Pallas kernels and scans in the same program to have the"
          r" same device assignment",
      ):
        _ = compiled(x)
    else:
      raise ValueError(f"unknown probe_id {probe_id!r}")

  # No success barrier: on the conflicting path compilation fails before any
  # device program executes.
  dist.init_process_group(backend="tpu_dist")
  try:
    body()
  finally:
    dist.destroy_process_group()


_KERNEL_CHECKS = (
    ("axis_index_equals_rank", "axis_index"),
    ("reports_its_own_shard", "sent"),
    ("mesh_addressed_dma_lands_on_rank_predecessor", "received"),
)

_COLLECTIVE_CASES = (
    ("all_gather_assembles_in_rank_order", "all_gather"),
    ("reduce_scatter_delivers_to_addressed_rank", "reduce_scatter"),
    ("all_to_all_matches_both_edge_endpoints", "all_to_all"),
    ("all_reduce_world_sums_over_all_ranks", "all_reduce_world"),
    ("all_reduce_subgroup_sums_over_its_half", "all_reduce_subgroup"),
    ("broadcast_takes_its_value_from_each_addressed_root", "broadcast"),
)

_TWO_MESH_PROBES = (
    (
        "same_mesh_kernels_share_one_traversal",
        "two_same_mesh_kernels_share_one_traversal",
    ),
    (
        "same_mesh_kernels_compile_cleanly",
        "two_same_mesh_kernels_compile_cleanly",
    ),
    (
        "conflicting_mesh_kernels_are_rejected",
        "two_conflicting_mesh_kernels_are_rejected",
    ),
)


class PallasMeshCollectiveDeviceOrderTest(
    seed_test_utils.MultiProcessRepeatableTest
):
  _world_size = 8

  def _dist_run(self, fn: Callable[..., None], **kwargs) -> None:
    distributed_utils.dist_run(
        nproc_per_node=self._world_size,
        fn=fn,
        **kwargs,
    )

  @parameterized.named_parameters(*_KERNEL_CHECKS)
  def test_kernel(self, check: str):
    self._dist_run(run_kernel_report_check, check=check)

  @parameterized.named_parameters(*_COLLECTIVE_CASES)
  def test_solo(self, case_name: str):
    self._dist_run(run_collective_solo, case_name=case_name)

  @parameterized.named_parameters(*_COLLECTIVE_CASES)
  def test_fused(self, case_name: str):
    if case_name == "broadcast":
      self.skipTest(_FUSED_BROADCAST_SKIP)
    self._dist_run(run_collective_fused, case_name=case_name)

  @parameterized.named_parameters(*_TWO_MESH_PROBES)
  def test_two(self, probe_id: str):
    self._dist_run(run_two_mesh_probe, probe_id=probe_id)


if __name__ == "__main__":  # We are in the parent process.
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
