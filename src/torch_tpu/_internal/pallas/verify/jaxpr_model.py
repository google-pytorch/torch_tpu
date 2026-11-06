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

"""Extraction of event-level concurrency models from Pallas TPU jaxprs.

`extract_kernel_models` traces a TorchTPU `@jax_op` or a raw JAX function to a
closed Jaxpr without initializing TPU backends, finds every `pallas_call` and
`mpmd_map` (`pl.core_map`, including inside `shard_map` for multi-chip kernels),
and unrolls it per mesh device and grid point into a `KernelVerificationModel`:

* Implicit pipeline `BlockSpec` input copies (`HBM -> VMEM`) and output copies
  (`VMEM -> HBM`) at the boundaries of each sequential grid step (plus nested
  `pltpu.emit_pipeline` steps), with symbolic evaluation of `index_map` Jaxprs
  to check out-of-bounds block slices and compute full-grid output write
  coverage (`unwritten_output`).
* Explicit kernel body operations (`get`, `swap`, `addupdate`, `masked_load`,
  `masked_swap`, `dma_start`, `dma_wait`, `semaphore_signal`, `semaphore_wait`,
  `get_barrier_semaphore`, `run_scoped`, `emit_pipeline`, `dot_general`) across
  nested `cond`, `while`, and `scan` control-flow primitives up to
  `max_unrolled_iterations`, tracking scalar expressions to resolve
  hyper-rectangular `HyperRectSlice` regions and remote device/core IDs.
* Static structural and hardware checks (`out_of_bounds_access`,
  `invalid_device_id`, `hardware_constraint_violation`, `unwritten_output`,
  `low_precision_accumulation`, and `verification_incomplete`), exposed via
  `static_model_findings` and `check_kernel_static_models`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import itertools
import math
import traceback
from typing import Any

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal.pallas import pallas as pallas_lib
from torch_tpu._internal.pallas.verify import findings
from torch_tpu._internal.pallas.verify import fx_verifier

_Category = findings.FindingCategory
_Severity = findings.Severity

# Default bound on unrolled grid points, loop iterations, and scan steps.
DEFAULT_MAX_UNROLLED_ITERATIONS = 32
# Grid points whose writes are tracked for the output coverage and bounds
# checks; only the first `max_unrolled_iterations` of them are model checked.
_MAX_COVERAGE_GRID_POINTS = 4096
# Bound on the cells examined when computing uncovered output regions.
_MAX_COVERAGE_CELLS = 1 << 20
# Hard cap on total equation steps walked per kernel model.
_MAX_TOTAL_EQUATION_STEPS = 200_000
# Primitives whose sub-jaxprs are walked like inlined function calls.
_CALL_PRIMITIVES = frozenset({
    "checkpoint",
    "closed_call",
    "core_call",
    "custom_jvp_call",
    "custom_vjp_call",
    "custom_vjp_call_jaxpr",
    "jit",
    "pjit",
    "remat",
})
# Primitives that signal or wait on semaphores.
_SYNC_PRIMITIVES = frozenset({
    "dma_start",
    "dma_wait",
    "semaphore_signal",
    "semaphore_wait",
})


@dataclasses.dataclass(frozen=True)
class SliceDim:
  """Half-open interval `[start, start + size)` along one buffer axis."""

  start: int
  size: int
  exact: bool = True

  @property
  def stop(self) -> int:
    return self.start + self.size

  def overlaps(self, other: SliceDim) -> bool:
    if self.size <= 0 or other.size <= 0:
      return False
    return self.start < other.stop and other.start < self.stop

  def __str__(self) -> str:
    if self.size == 1:
      return f"{self.start}"
    return f"{self.start}:{self.stop}"


@dataclasses.dataclass(frozen=True)
class HyperRectSlice:
  """Axis-aligned hyper-rectangular region inside a root buffer."""

  dims: tuple[SliceDim, ...]
  opaque: bool = False
  covers_full_buffer: bool = dataclasses.field(default=False, compare=False)

  @classmethod
  def full(cls, shape: Sequence[int]) -> HyperRectSlice:
    return cls(
        tuple(SliceDim(0, int(s)) for s in shape),
        covers_full_buffer=True,
    )

  def intersects(self, other: HyperRectSlice) -> bool:
    if len(self.dims) != len(other.dims):
      # Different ranks (e.g. scalar vs. array views): assume overlap.
      return True
    return all(a.overlaps(b) for a, b in zip(self.dims, other.dims))

  def definitely_intersects(self, other: HyperRectSlice) -> bool:
    """Whether `self` and `other` overlap without relying on widened axes."""
    if len(self.dims) != len(other.dims):
      return False
    if any(d.size <= 0 for d in self.dims) or any(
        d.size <= 0 for d in other.dims
    ):
      return False
    if (self.covers_full_buffer and not other.opaque) or (
        other.covers_full_buffer and not self.opaque
    ):
      return self.intersects(other)
    if self.opaque or other.opaque:
      return False
    for a, b in zip(self.dims, other.dims):
      if a.exact and b.exact:
        if not a.overlaps(b):
          return False
      elif a.exact and not b.exact:
        if not (a.start <= b.start and b.stop <= a.stop):
          return False
      elif b.exact and not a.exact:
        if not (b.start <= a.start and a.stop <= b.stop):
          return False
      else:
        return False
    return True

  def __str__(self) -> str:
    if not self.dims:
      return "[...]"
    return "[" + ", ".join(str(d) for d in self.dims) + "]"


@dataclasses.dataclass(frozen=True)
class BufferKey:
  """Physical memory allocation identity."""

  device_id: int
  core_id: int
  memory_space: str
  buffer_id: str
  # Core or parallel-iteration coordinates that own a private (VMEM, SMEM)
  # buffer; empty for memory shared across cores of a chip (HBM, VMEM_SHARED,
  # or collective_axes run_scoped allocations).
  owner_coords: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class SemaphoreKey:
  """Physical semaphore element identity.

  Iterations of `parallel` grid axes are modeled as if each ran on its own
  core, so per-core semaphores carry the parallel grid coordinates of the
  iteration that uses them.
  """

  device_id: int
  core_id: int
  sem_type: str  # "dma", "regular", or "barrier"
  sem_id: str
  sem_index: tuple[int, ...] = ()
  parallel_coords: tuple[int, ...] = ()

  def short_str(self) -> str:
    idx_str = (
        f"[{','.join(str(i) for i in self.sem_index)}]"
        if self.sem_index
        else ""
    )
    par_str = f", par={self.parallel_coords}" if self.parallel_coords else ""
    return (
        f"{self.sem_id}{idx_str}({self.sem_type}@dev{self.device_id}{par_str})"
    )


@dataclasses.dataclass(frozen=True)
class ActorKey:
  """Sequential execution stream `(device_id, core_id, parallel_grid_coords)`."""

  device_id: int
  core_id: int
  parallel_coords: tuple[int, ...] = ()

  def short_str(self) -> str:
    if self.parallel_coords:
      return f"tc(par={self.parallel_coords})"
    return "tc"


@dataclasses.dataclass(frozen=True)
class MemoryAccessEvent:
  """Synchronous load or store executed on an `ActorKey`."""

  event_id: int
  actor: ActorKey
  grid_point: tuple[int, ...]
  buffer: BufferKey
  region: HyperRectSlice
  is_write: bool
  is_implicit_pipeline: bool = False
  source_location: str | None = None
  speculative: bool = False


@dataclasses.dataclass(frozen=True)
class DmaStartEvent:
  """Asynchronous local or cross-chip DMA launch."""

  event_id: int
  dma_id: int
  actor: ActorKey
  grid_point: tuple[int, ...]
  src_buffer: BufferKey
  src_region: HyperRectSlice
  dst_buffer: BufferKey
  dst_region: HyperRectSlice
  byte_count: int
  dst_sem: SemaphoreKey | None
  src_sem: SemaphoreKey | None
  is_remote: bool
  source_location: str | None = None


@dataclasses.dataclass(frozen=True)
class DmaWaitEvent:
  """Wait on a DMA completion semaphore for `byte_count` bytes."""

  event_id: int
  actor: ActorKey
  grid_point: tuple[int, ...]
  sem: SemaphoreKey
  byte_count: int
  is_wait_send: bool
  source_location: str | None = None


@dataclasses.dataclass(frozen=True)
class SemaphoreSignalEvent:
  """Explicit `pl.semaphore_signal` increment on local or remote semaphore."""

  event_id: int
  actor: ActorKey
  grid_point: tuple[int, ...]
  target_sem: SemaphoreKey
  amount: int
  is_remote: bool
  source_location: str | None = None


@dataclasses.dataclass(frozen=True)
class SemaphoreWaitEvent:
  """Explicit `pl.semaphore_wait` on a local semaphore."""

  event_id: int
  actor: ActorKey
  grid_point: tuple[int, ...]
  sem: SemaphoreKey
  amount: int
  decrement: bool
  source_location: str | None = None


@dataclasses.dataclass(frozen=True)
class StaticIssue:
  """A problem found while extracting the model, before any solving."""

  category: findings.FindingCategory
  severity: findings.Severity
  message: str
  source_location: str | None = None


@dataclasses.dataclass(frozen=True)
class KernelVerificationModel:
  """Extracted concurrency and memory-region model of one `pallas_call`.

  Attributes:
    kernel_name: Name of the `pallas_call` (or of the op that calls it).
    num_devices: Number of devices the kernel runs on.
    grid: The kernel grid.
    dimension_semantics: "parallel" or "arbitrary" per grid axis.
    collective_id: `collective_id` from the compiler params, if any.
    actor_event_orders: Event ids per actor in program order.
    memory_events: Synchronous loads and stores.
    dma_starts: DMA launches.
    dma_waits: DMA semaphore waits.
    sem_signals: Semaphore signals.
    sem_waits: Semaphore waits.
    static_issues: Problems found during extraction.
    incomplete_reasons: Why the model is bounded or approximate, if it is.
    sync_reliable: False if semaphore or remote-device identities could not be
      resolved, or if semaphore operations sit in code whose execution depends
      on runtime data, in which case deadlock and race results are not
      meaningful.
    sync_truncated: True if only a prefix of the semaphore operations is modeled
      (bounded unrolling), so the semaphore values at kernel exit are unknown
      and waits may depend on signals of unchecked iterations.
    uses_remote_operations: Whether the kernel issues DMAs or semaphore signals
      to other devices.
    truncated_sem_keys: Semaphores with operations in iterations that are not
      modeled, whose values at kernel exit are therefore unknown.
    unwalked_sync: True if a loop, equation cap, or grid-point cap truncated
      unwalked semaphore operations inside a grid point or beyond the coverage
      cap.
  """

  kernel_name: str
  num_devices: int
  grid: tuple[int, ...]
  dimension_semantics: tuple[str, ...]
  collective_id: int | None
  actor_event_orders: Mapping[ActorKey, tuple[int, ...]]
  memory_events: tuple[MemoryAccessEvent, ...]
  dma_starts: tuple[DmaStartEvent, ...]
  dma_waits: tuple[DmaWaitEvent, ...]
  sem_signals: tuple[SemaphoreSignalEvent, ...]
  sem_waits: tuple[SemaphoreWaitEvent, ...]
  static_issues: tuple[StaticIssue, ...] = ()
  incomplete_reasons: tuple[str, ...] = ()
  sync_reliable: bool = True
  sync_truncated: bool = False
  uses_remote_operations: bool = False
  truncated_sem_keys: frozenset[SemaphoreKey] = frozenset()
  unwalked_sync: bool = False
  core_axis_names: tuple[str, ...] = ()
  signaled_sem_keys: frozenset[SemaphoreKey] = frozenset()


@dataclasses.dataclass(frozen=True)
class RefOperand:
  """A ref operand of a DMA primitive and the transforms applied to it."""

  ref: Any
  transforms: tuple[Any, ...] = ()


@dataclasses.dataclass(frozen=True)
class DmaOperands:
  """Decoded operands of a `dma_start` or `dma_wait` equation."""

  src: RefOperand
  dst: RefOperand
  dst_sem: RefOperand | None
  src_sem: RefOperand | None
  device_id: Any


def _unwrap_ref_and_transforms(ref_or_view: Any) -> tuple[Any, tuple[Any, ...]]:
  """Unwraps nested `TransformedRef`s into `(root, transforms)`."""
  transforms: list[Any] = []
  cur = ref_or_view
  while hasattr(cur, "ref") and hasattr(cur, "transforms"):
    transforms = list(cur.transforms) + transforms
    cur = cur.ref
  return cur, tuple(transforms)


def _ref_operand(ref: Any, transforms: Any = ()) -> RefOperand | None:
  if ref is None:
    return None
  root, inner = _unwrap_ref_and_transforms(ref)
  return RefOperand(root, inner + tuple(transforms or ()))


def decode_dma_operands(operands: Sequence[Any]) -> DmaOperands:
  """Decodes unflattened DMA operands across JAX versions.

  JAX up to 0.9 flattens `(src, src_transforms, dst, dst_transforms, dst_sem,
  dst_sem_transforms, src_sem, src_sem_transforms, device_id)`; newer releases
  flatten `(src, dst, dst_sem, src_sem, device_id)` of `TransformedRef`s.

  Args:
    operands: The result of `eqn.params["tree"].unflatten(eqn.invars)`.

  Returns:
    The operands with their transforms.

  Raises:
    ValueError: If the operand layout is not recognized.
  """
  if len(operands) == 9:
    src, src_tr, dst, dst_tr, dst_sem, dst_sem_tr, src_sem, src_sem_tr, dev = (
        operands
    )
  elif len(operands) == 5:
    src, dst, dst_sem, src_sem, dev = operands
    src_tr = dst_tr = dst_sem_tr = src_sem_tr = ()
  else:
    raise ValueError(f"unsupported dma operand layout of {len(operands)} items")
  src_op = _ref_operand(src, src_tr)
  dst_op = _ref_operand(dst, dst_tr)
  if src_op is None or dst_op is None:
    raise ValueError("dma operands are missing a source or destination ref")
  return DmaOperands(
      src=src_op,
      dst=dst_op,
      dst_sem=_ref_operand(dst_sem, dst_sem_tr),
      src_sem=_ref_operand(src_sem, src_sem_tr),
      device_id=dev,
  )


@dataclasses.dataclass(frozen=True)
class _RefView:
  """A `Ref` or view of it, tracked back to its root buffer or semaphore."""

  buffer_id: str
  memory_space: str
  root_shape: tuple[int, ...]
  dtype: Any
  dims: tuple[SliceDim, ...]
  active_axes: tuple[int, ...]
  logical_shape: tuple[int, ...] | None = None
  # None for data buffers, else "dma", "regular", or "barrier".
  sem_type: str | None = None
  # Set once a transform that changes the shape (reshape, bitcast) was applied;
  # later indexing is then not interpreted.
  opaque: bool = False
  covers_full_buffer: bool = True
  shared: bool = False

  @property
  def view_shape(self) -> tuple[int, ...]:
    if self.logical_shape is not None:
      return self.logical_shape
    return tuple(self.dims[ax].size for ax in self.active_axes)

  @property
  def region(self) -> HyperRectSlice:
    return HyperRectSlice(
        self.dims,
        opaque=self.opaque,
        covers_full_buffer=self.covers_full_buffer,
    )

  @property
  def sem_index(self) -> tuple[int, ...] | None:
    """Element index of a 0D semaphore view, or None if not statically resolved."""
    if self.opaque or self.view_shape != ():
      return None
    if any(d.size != 1 or not d.exact for d in self.dims):
      return None
    return tuple(d.start for d in self.dims)

  @property
  def byte_count(self) -> int:
    itemsize = int(np.dtype(self.dtype).itemsize)
    return max(1, int(math.prod(self.view_shape)) * itemsize)


def _full_view(
    buffer_id: str,
    memory_space: str,
    shape: Sequence[int],
    dtype: Any,
    sem_type: str | None = None,
    *,
    shared: bool = False,
) -> _RefView:
  shape = tuple(int(s) for s in shape)
  return _RefView(
      buffer_id=buffer_id,
      memory_space=memory_space,
      root_shape=shape,
      dtype=np.int32 if sem_type is not None else dtype,
      dims=tuple(SliceDim(0, s) for s in shape),
      active_axes=tuple(range(len(shape))),
      logical_shape=shape,
      sem_type=sem_type,
      covers_full_buffer=True,
      shared=shared or memory_space in ("hbm", "vmem_shared"),
  )


@dataclasses.dataclass(frozen=True)
class _MeshContext:
  axis_names: tuple[str, ...] = ()
  mesh_shape: tuple[int, ...] = ()

  @property
  def num_devices(self) -> int:
    return int(math.prod(self.mesh_shape)) if self.mesh_shape else 1

  def iter_devices(self) -> list[tuple[int, tuple[int, ...]]]:
    if not self.axis_names:
      return [(0, ())]
    coords = itertools.product(*(range(s) for s in self.mesh_shape))
    return list(enumerate(coords))

  def coord_dict(self, coords: tuple[int, ...]) -> dict[Any, int]:
    return dict(zip(self.axis_names, coords))


def _is_jax_var(x: Any) -> bool:
  return type(x).__name__ == "Var"


def _is_drop_var(x: Any) -> bool:
  return type(x).__name__ == "DropVar"


def _unwrap_closed_jaxpr(sub: Any) -> Any:
  return sub.jaxpr if hasattr(sub, "jaxpr") else sub


def _summarize_source_info(eqn: Any) -> str | None:
  source_info = getattr(eqn, "source_info", None)
  tb = getattr(source_info, "traceback", None)
  if tb is None or not hasattr(tb, "as_python_traceback"):
    return None
  py_tb = tb.as_python_traceback()
  if py_tb is None:
    return None
  for frame in reversed(traceback.extract_tb(py_tb)):
    if (
        "/jax/_src/" not in frame.filename
        and "/jax/experimental/" not in frame.filename
    ):
      return f"{frame.filename}:{frame.lineno} ({frame.name})"
  return None


def _normalize_memory_space(ms: Any) -> str:
  # Compare values rather than members: `pl.MemorySpace` gained `DEFAULT`
  # after JAX 0.9.
  value = getattr(ms, "value", ms)
  if value is None or value == "default":
    return "vmem"
  if value == "any":
    return "hbm"
  if isinstance(value, str):
    return value
  return str(ms).lower()


def _detect_semaphore_type(aval: Any) -> str | None:
  dtype = getattr(aval, "dtype", None)
  ms = getattr(aval, "memory_space", None)
  dtype_str = str(dtype).lower()
  dtype_cls_name = type(dtype).__name__
  if dtype_cls_name == "DMASemaphore" or dtype_str.startswith("dma_sem"):
    return "dma"
  if dtype_cls_name == "BarrierSemaphore" or dtype_str.startswith(
      "barrier_sem"
  ):
    return "barrier"
  if (
      dtype_cls_name in ("Semaphore", "RegularSemaphore")
      or ms == pltpu.MemorySpace.SEMAPHORE
      or dtype_str in ("semaphore", "regular_semaphore")
      or dtype_str.startswith("sem<")
  ):
    return "regular"
  return None


def _eval_atom(
    atom: Any, scalar_env: Mapping[Any, int | bool]
) -> int | bool | None:
  """Returns the concrete integer or Boolean value of an atom, if known."""
  if isinstance(atom, (bool, int)):
    return atom
  if isinstance(atom, (np.generic, np.ndarray)) and np.shape(atom) == ():
    if np.issubdtype(atom.dtype, np.bool_):
      return bool(atom.item())
    if np.issubdtype(atom.dtype, np.integer):
      return int(atom.item())
    return None
  if type(atom).__name__ == "Literal" and hasattr(atom, "val"):
    return _eval_atom(atom.val, scalar_env)
  if isinstance(atom, jax.Array) and atom.shape == ():
    if jnp.issubdtype(atom.dtype, jnp.bool_):
      return bool(atom.item())
    if jnp.issubdtype(atom.dtype, jnp.integer):
      return int(atom.item())
    return None
  if _is_jax_var(atom):
    return scalar_env.get(atom)
  return None


def _int_binary(name: str, a: int, b: int) -> int | None:
  """Integer semantics of `lax` binary ops on scalars (C-style division)."""
  if name in ("add", "add_any"):
    return a + b
  if name == "sub":
    return a - b
  if name == "mul":
    return a * b
  if name in ("div", "rem"):
    if b == 0:
      return None
    quotient = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
    return quotient if name == "div" else a - quotient * b
  if name == "min":
    return min(a, b)
  if name == "max":
    return max(a, b)
  if name == "shift_left":
    return a << b if b >= 0 else None
  if name in ("shift_right_arithmetic", "shift_right_logical"):
    return a >> b if b >= 0 else None
  return None


_COMPARISONS: dict[str, Callable[[int, int], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}
_PASSTHROUGH = frozenset({
    "broadcast_in_dim",
    "convert_element_type",
    "copy",
    "copy_p",
    "multiple_of",
    "reshape",
    "squeeze",
})


def _eval_scalar_op(
    name: str, vals: Sequence[int | bool], is_bool: bool, params: Any
) -> int | bool | None:
  """Evaluates a scalar `lax` primitive on concrete operands."""
  if name in _COMPARISONS and len(vals) == 2:
    return _COMPARISONS[name](int(vals[0]), int(vals[1]))
  if name in _PASSTHROUGH:
    return bool(vals[0]) if is_bool else int(vals[0])
  if name in ("and", "or", "xor"):
    if is_bool:
      a, b = bool(vals[0]), bool(vals[1])
      return {"and": a and b, "or": a or b, "xor": a != b}[name]
    a, b = int(vals[0]), int(vals[1])
    return {"and": a & b, "or": a | b, "xor": a ^ b}[name]
  if name == "not":
    return (not bool(vals[0])) if is_bool else ~int(vals[0])
  if name == "neg":
    return -int(vals[0])
  if name == "abs":
    return abs(int(vals[0]))
  if name == "sign":
    v = int(vals[0])
    return (v > 0) - (v < 0)
  if name == "integer_pow":
    return int(vals[0]) ** int(params["y"])
  if name == "clamp" and len(vals) == 3:
    return min(max(int(vals[1]), int(vals[0])), int(vals[2]))
  if name == "select_n":
    which = int(vals[0])
    choices = vals[1:]
    return choices[which] if 0 <= which < len(choices) else None
  if len(vals) == 2:
    return _int_binary(name, int(vals[0]), int(vals[1]))
  return None


def _sym_atom(
    atom: Any,
    scalar_env: Mapping[Any, int | bool],
    sym_env: Mapping[Any, Any],
) -> Any:
  val = _eval_atom(atom, scalar_env)
  if val is not None:
    return ("const", int(val))
  if _is_jax_var(atom):
    return sym_env.get(atom, ("var", id(atom)))
  return ("atom", id(atom))


def _negate_sym_pred(expr: Any) -> Any:
  if isinstance(expr, tuple) and len(expr) == 2 and expr[0] == "not":
    return expr[1]
  return ("not", expr)


def _record_symbolic_scalar_eqn(
    eqn: Any,
    scalar_env: Mapping[Any, int | bool],
    sym_env: dict[Any, Any] | None,
) -> None:
  """Tracks canonical symbolic boolean/comparison expressions for `cond`."""
  if sym_env is None or len(eqn.outvars) != 1:
    return
  out_var = eqn.outvars[0]
  if _is_drop_var(out_var):
    return
  name = eqn.primitive.name
  out_aval = getattr(out_var, "aval", None)
  is_bool = isinstance(out_aval, jax.core.ShapedArray) and (
      out_aval.dtype == np.bool_
  )
  if name in _PASSTHROUGH and len(eqn.invars) == 1:
    sym_env[out_var] = _sym_atom(eqn.invars[0], scalar_env, sym_env)
    return
  if name == "not" and len(eqn.invars) == 1 and is_bool:
    sym_env[out_var] = _negate_sym_pred(
        _sym_atom(eqn.invars[0], scalar_env, sym_env)
    )
    return
  if name in _COMPARISONS and len(eqn.invars) == 2:
    lhs = _sym_atom(eqn.invars[0], scalar_env, sym_env)
    rhs = _sym_atom(eqn.invars[1], scalar_env, sym_env)
    in0_aval = getattr(eqn.invars[0], "aval", None)
    in0_bool = isinstance(in0_aval, jax.core.ShapedArray) and (
        in0_aval.dtype == np.bool_
    )
    if in0_bool and rhs == ("const", 0) and name in ("eq", "ne"):
      sym_env[out_var] = _negate_sym_pred(lhs) if name == "eq" else lhs
      return
    if in0_bool and rhs == ("const", 1) and name in ("eq", "ne"):
      sym_env[out_var] = lhs if name == "eq" else _negate_sym_pred(lhs)
      return
    if name == "lt":
      sym_env[out_var] = ("lt", lhs, rhs)
    elif name == "ge":
      sym_env[out_var] = _negate_sym_pred(("lt", lhs, rhs))
    elif name == "gt":
      sym_env[out_var] = ("lt", rhs, lhs)
    elif name == "le":
      sym_env[out_var] = _negate_sym_pred(("lt", rhs, lhs))
    elif name == "eq":
      pair = (lhs, rhs) if str(lhs) <= str(rhs) else (rhs, lhs)
      sym_env[out_var] = ("eq", pair[0], pair[1])
    elif name == "ne":
      pair = (lhs, rhs) if str(lhs) <= str(rhs) else (rhs, lhs)
      sym_env[out_var] = _negate_sym_pred(("eq", pair[0], pair[1]))


def _try_eval_scalar_eqn(
    eqn: Any,
    scalar_env: dict[Any, int | bool],
    *,
    grid_coords: tuple[int, ...],
    grid_sizes: Sequence[int | None],
    axis_env: Mapping[Any, int],
    sym_env: dict[Any, Any] | None = None,
) -> bool:
  """Evaluates a scalar integer/Boolean equation into `scalar_env`."""
  name = eqn.primitive.name
  if name in ("program_id", "num_programs"):
    axis = int(eqn.params["axis"])
    source = grid_coords if name == "program_id" else grid_sizes
    if 0 <= axis < len(source):
      val = source[axis]
      if val is not None:
        scalar_env[eqn.outvars[0]] = val
        return True
    return False
  if name == "axis_index":
    ax_name = eqn.params["axis_name"]
    if ax_name in axis_env:
      scalar_env[eqn.outvars[0]] = axis_env[ax_name]
      return True
    return False
  if name in _CALL_PRIMITIVES:
    return _try_eval_scalar_call(
        eqn,
        scalar_env,
        grid_coords=grid_coords,
        grid_sizes=grid_sizes,
        axis_env=axis_env,
    )
  if len(eqn.outvars) != 1:
    return False
  out_aval = getattr(eqn.outvars[0], "aval", None)
  if not isinstance(out_aval, jax.core.ShapedArray) or out_aval.shape != ():
    return False
  if not (
      out_aval.dtype == np.bool_ or np.issubdtype(out_aval.dtype, np.integer)
  ):
    return False
  for v in eqn.invars:
    in_aval = getattr(v, "aval", None)
    if not isinstance(in_aval, jax.core.ShapedArray) or in_aval.shape != ():
      return False
    if not (
        in_aval.dtype == np.bool_ or np.issubdtype(in_aval.dtype, np.integer)
    ):
      return False
  vals = [_eval_atom(v, scalar_env) for v in eqn.invars]
  is_bool = out_aval.dtype == np.bool_
  # Short-circuit boolean and selection ops when one operand determines the
  # result even if the other operand depends on runtime data.
  if name == "and" and is_bool and any(v is False or v == 0 for v in vals):
    scalar_env[eqn.outvars[0]] = False
    return True
  if name == "or" and is_bool and any(v is True or v == 1 for v in vals):
    scalar_env[eqn.outvars[0]] = True
    return True
  if (
      name == "mul"
      and not is_bool
      and any(v == 0 for v in vals if v is not None)
  ):
    scalar_env[eqn.outvars[0]] = 0
    return True
  if name == "select_n" and vals and vals[0] is not None:
    which = int(vals[0])
    choices = vals[1:]
    if 0 <= which < len(choices):
      chosen = choices[which]
      if chosen is not None:
        scalar_env[eqn.outvars[0]] = bool(chosen) if is_bool else int(chosen)
        return True
  concrete = [v for v in vals if v is not None]
  if not vals or len(concrete) != len(vals):
    _record_symbolic_scalar_eqn(eqn, scalar_env, sym_env)
    return False
  res = _eval_scalar_op(name, concrete, is_bool, eqn.params)
  if res is None:
    return False
  scalar_env[eqn.outvars[0]] = res
  return True


def _try_eval_scalar_call(
    eqn: Any,
    scalar_env: dict[Any, int | bool],
    *,
    grid_coords: tuple[int, ...],
    grid_sizes: Sequence[int | None],
    axis_env: Mapping[Any, int],
) -> bool:
  """Evaluates a nested call, such as `jnp.floor_divide`, on scalars.

  Only calls with known scalar operands whose equations are all scalar
  arithmetic are evaluated, so calls that access refs are still walked.

  Args:
    eqn: The call equation.
    scalar_env: Known scalar values; receives the call's results.
    grid_coords: Grid point being walked.
    grid_sizes: Grid sizes (`None` for runtime-sized axes).
    axis_env: Device coordinates by mesh axis name.

  Returns:
    Whether every result of the call was evaluated.
  """
  vals = [_eval_atom(v, scalar_env) for v in eqn.invars]
  known = [v for v in vals if v is not None]
  if not eqn.outvars or len(known) != len(vals):
    return False
  for param in eqn.params.values():
    inner = _unwrap_closed_jaxpr(param)
    if not hasattr(inner, "eqns") or len(inner.invars) != len(known):
      continue
    inner_env: dict[Any, int | bool] = dict(zip(inner.invars, known))
    for var, const in zip(inner.constvars, getattr(param, "consts", ())):
      value = _eval_atom(const, {})
      if value is not None:
        inner_env[var] = value
    for inner_eqn in inner.eqns:
      if not _try_eval_scalar_eqn(
          inner_eqn,
          inner_env,
          grid_coords=grid_coords,
          grid_sizes=grid_sizes,
          axis_env=axis_env,
      ):
        return False
    outs = [_eval_atom(v, inner_env) for v in inner.outvars]
    if len(outs) != len(eqn.outvars) or any(o is None for o in outs):
      return False
    for outer, value in zip(eqn.outvars, outs):
      if value is not None and not _is_drop_var(outer):
        scalar_env[outer] = value
    return True
  return False


def _apply_transforms(
    base: _RefView,
    transforms: Sequence[Any],
    scalar_env: Mapping[Any, int | bool],
) -> tuple[_RefView, str | None]:
  """Composes indexing transforms onto `base`.

  Returns:
    The view and, if a statically known index is out of bounds, a description
    of the violation.
  """
  dims = list(base.dims)
  active_axes = list(base.active_axes)
  logical_shape = list(
      base.logical_shape
      if base.logical_shape is not None
      else (dims[ax].size for ax in active_axes)
  )
  memory_space = base.memory_space
  opaque = base.opaque
  covers_full_buffer = base.covers_full_buffer
  violation: str | None = None
  for tr in transforms:
    tr_name = type(tr).__name__
    if tr_name == "MemorySpaceCastTransform" and hasattr(tr, "memory_space"):
      memory_space = _normalize_memory_space(tr.memory_space)
      continue
    if tr_name == "TransposeTransform" and hasattr(tr, "perm"):
      perm = tuple(int(p) for p in tr.perm)
      if not opaque and len(perm) == len(active_axes):
        active_axes = [active_axes[p] for p in perm]
      if len(perm) == len(logical_shape):
        logical_shape = [logical_shape[p] for p in perm]
      continue
    if tr_name != "NDIndexer" or not hasattr(tr, "indices"):
      opaque = True  # Reshapes and bitcasts change the indexed shape.
      tr_shape = getattr(tr, "shape", None)
      if tr_shape is not None:
        logical_shape = [int(s) for s in tr_shape]
      continue
    if opaque:
      covers_full_buffer = False
      idx_shape = getattr(tr, "shape", None)
      if idx_shape is not None:
        logical_shape = [int(s) for s in idx_shape]
      continue
    new_active: list[int] = []
    new_logical: list[int] = []
    int_indexer_inserted = False
    int_indexer_shape = getattr(tr, "int_indexer_shape", None)
    num_indexed = min(len(active_axes), len(tr.indices))
    for view_pos, (ax_pos, idx_entry) in enumerate(
        zip(active_axes[:num_indexed], tr.indices[:num_indexed])
    ):
      parent = dims[ax_pos]
      axis_size = (
          logical_shape[view_pos]
          if view_pos < len(logical_shape)
          else parent.size
      )
      if isinstance(idx_entry, pl.Slice):
        start = _eval_atom(idx_entry.start, scalar_env)
        size = _eval_atom(idx_entry.size, scalar_env)
        raw_stride = getattr(idx_entry, "stride", None)
        stride_val = (
            1 if raw_stride is None else _eval_atom(raw_stride, scalar_env)
        )
        stride = 1 if stride_val is None else int(stride_val)
        if stride <= 0:
          violation = violation or (
              f"slice stride {stride} along axis {ax_pos} must be positive"
          )
          dims[ax_pos] = SliceDim(parent.start, 0, exact=False)
          covers_full_buffer = False
        elif size is not None and int(size) < 0:
          violation = violation or (
              f"slice size {int(size)} along axis {ax_pos} is negative"
          )
          dims[ax_pos] = SliceDim(parent.start, 0, exact=False)
          covers_full_buffer = False
        elif size is not None and int(size) == 0:
          if start is not None and not 0 <= int(start) <= axis_size:
            violation = violation or (
                f"slice {int(start)}:{int(start)} is outside axis {ax_pos} of"
                f" size {axis_size}"
            )
          dims[ax_pos] = SliceDim(
              parent.start + (int(start) if start is not None else 0),
              0,
              exact=parent.exact and start is not None,
          )
          covers_full_buffer = False
        elif start is not None and size is not None:
          span = (int(size) - 1) * stride + 1
          if int(start) < 0 or int(start) + span > axis_size:
            violation = violation or (
                f"slice {int(start)}:{int(start) + span} is outside axis"
                f" {ax_pos} of size {axis_size}"
            )
            dims[ax_pos] = SliceDim(parent.start, 0, exact=False)
          else:
            dims[ax_pos] = SliceDim(
                parent.start + int(start),
                span,
                exact=parent.exact and stride == 1 and stride_val is not None,
            )
          if int(start) != 0 or int(size) != axis_size or stride != 1:
            covers_full_buffer = False
        else:
          if size is not None:
            span = (int(size) - 1) * stride + 1
            if span > axis_size:
              violation = violation or (
                  f"slice of span {span} exceeds axis {ax_pos} of size"
                  f" {axis_size}"
              )
          if start is not None and not 0 <= int(start) < axis_size:
            violation = violation or (
                f"slice start {int(start)} is outside axis {ax_pos} of size"
                f" {axis_size}"
            )
          dims[ax_pos] = SliceDim(parent.start, parent.size, exact=False)
          covers_full_buffer = False
        new_active.append(ax_pos)
        new_logical.append(max(0, int(size)) if size is not None else axis_size)
        continue
      idx_aval = getattr(idx_entry, "aval", None)
      if isinstance(idx_aval, jax.core.ShapedArray) and idx_aval.shape:
        dims[ax_pos] = SliceDim(parent.start, parent.size, exact=False)
        covers_full_buffer = False
        new_active.append(ax_pos)  # Gather: the whole axis may be read.
        if not int_indexer_inserted:
          int_indexer_inserted = True
          bcast_shape = (
              int_indexer_shape
              if int_indexer_shape is not None
              else idx_aval.shape
          )
          new_logical.extend(int(s) for s in bcast_shape)
        continue
      k = _eval_atom(idx_entry, scalar_env)
      if k is not None:
        idx = int(k)
        if not 0 <= idx < axis_size:
          violation = violation or (
              f"index {idx} is outside axis {ax_pos} of size {axis_size}"
          )
          dims[ax_pos] = SliceDim(parent.start, 0, exact=False)
        else:
          dims[ax_pos] = SliceDim(parent.start + idx, 1, exact=parent.exact)
        if axis_size != 1 or idx != 0:
          covers_full_buffer = False
      else:
        dims[ax_pos] = SliceDim(parent.start, parent.size, exact=False)
        covers_full_buffer = False
    if num_indexed < len(active_axes):
      new_active.extend(active_axes[num_indexed:])
      new_logical.extend(logical_shape[num_indexed:])
    active_axes = new_active
    logical_shape = new_logical
  view = dataclasses.replace(
      base,
      memory_space=memory_space,
      dims=tuple(dims),
      active_axes=tuple(active_axes),
      logical_shape=tuple(logical_shape),
      opaque=opaque,
      covers_full_buffer=covers_full_buffer,
  )
  return view, violation


def _extract_mosaic_compiler_params(compiler_params: Any) -> Any:
  if compiler_params is None:
    return None
  if isinstance(compiler_params, Mapping):
    return compiler_params.get("mosaic_tpu") or compiler_params.get("mosaic")
  return compiler_params


def _kernel_name(params: Mapping[str, Any]) -> str:
  for key in ("name", "name_and_src_info"):
    value = params.get(key)
    name = getattr(value, "name", value)
    if isinstance(name, str) and name:
      return name
  return "pallas_call"


class _KernelModelBuilder:
  """Accumulates events and static issues across devices and grid points."""

  def __init__(
      self,
      *,
      kernel_name: str,
      num_devices: int,
      grid: tuple[int, ...],
      dimension_semantics: tuple[str, ...],
      collective_id: int | None,
      max_unrolled_iterations: int,
      tpu_device_kind: str = fx_verifier.DEFAULT_TPU_DEVICE_KIND,
      core_axis_names: tuple[str, ...] = (),
  ):
    self.kernel_name = kernel_name
    self.num_devices = num_devices
    self.grid = grid
    self.dimension_semantics = dimension_semantics
    self.collective_id = collective_id
    self.max_unrolled_iterations = max_unrolled_iterations
    self.tpu_device_kind = tpu_device_kind
    self.core_axis_names = core_axis_names

    self._next_event_id = 0
    self._next_dma_id = 0
    self.step_count = 0
    self.actor_events: dict[ActorKey, list[int]] = {}
    self.memory_events: list[MemoryAccessEvent] = []
    self.dma_starts: list[DmaStartEvent] = []
    self.dma_waits: list[DmaWaitEvent] = []
    self.sem_signals: list[SemaphoreSignalEvent] = []
    self.sem_waits: list[SemaphoreWaitEvent] = []
    self.static_issues: list[StaticIssue] = []
    self._issue_keys: set[tuple[findings.FindingCategory, str]] = set()
    self._dedup_keys: set[tuple[Any, ...]] = set()
    self.incomplete_reasons: list[str] = []
    self.coverage_known = True
    self.sync_reliable = True
    self.sync_truncated = False
    self.unwalked_sync = False
    self.uses_remote_operations = False
    # Semaphores whose operations are only partly modeled.
    self.truncated_sem_keys: set[SemaphoreKey] = set()
    # Semaphores signaled at any walked grid point (including coverage-only).
    self.signaled_sem_keys: set[SemaphoreKey] = set()
    self._sem_key_scopes: list[set[SemaphoreKey]] = []
    self._has_sync: dict[int, bool] = {}
    # (device id, output buffer id) -> regions written by stores or DMAs.
    self.output_writes: dict[tuple[int, str], list[HyperRectSlice]] = {}
    # (actor, operand index) -> list of (grid_coords, block_region).
    self.output_block_visits: dict[
        tuple[ActorKey, int], list[tuple[tuple[int, ...], HyperRectSlice]]
    ] = {}
    # (device id, operand index, block_region) ->
    #   list of (actor, grid_coords, tile_writes).
    self.parallel_output_blocks: dict[
        tuple[int, int, HyperRectSlice],
        list[
            tuple[
                ActorKey,
                tuple[int, ...],
                list[HyperRectSlice],
            ]
        ],
    ] = {}
    # (device id, operand index) -> visited block regions for aliased VMEM outputs.
    self.aliased_pipelined_blocks: dict[
        tuple[int, int], list[HyperRectSlice]
    ] = {}
    # Active output block and accumulated tile writes on each (actor, op_index)
    # across consecutive reduction steps targeting the same output block.
    self.active_output_tile_block: dict[
        tuple[ActorKey, int], HyperRectSlice
    ] = {}
    self.active_output_tile_writes: dict[
        tuple[ActorKey, int], list[HyperRectSlice]
    ] = {}
    # (actor, canonical buffer id) written in an earlier grid or loop step.
    self.prior_step_written_buffers: set[tuple[ActorKey, str]] = set()
    self._scope_counters: dict[tuple[Any, ...], int] = {}
    self._source_locations: dict[int, str | None] = {}

  def next_scope_index(self, key: tuple[Any, ...]) -> int:
    """Returns 0, 1, ... for successive calls with the same `key`."""
    index = self._scope_counters.get(key, -1) + 1
    self._scope_counters[key] = index
    return index

  def source_location(self, eqn: Any) -> str | None:
    """Returns the user source location of `eqn`, cached across grid points."""
    key = id(eqn)
    if key not in self._source_locations:
      self._source_locations[key] = _summarize_source_info(eqn)
    return self._source_locations[key]

  def new_event_id(self, actor: ActorKey) -> int:
    eid = self._next_event_id
    self._next_event_id += 1
    self.actor_events.setdefault(actor, []).append(eid)
    return eid

  def insert_event_id(self, actor: ActorKey, index: int) -> int:
    eid = self._next_event_id
    self._next_event_id += 1
    self.actor_events.setdefault(actor, []).insert(index, eid)
    return eid

  def new_dma_id(self) -> int:
    did = self._next_dma_id
    self._next_dma_id += 1
    return did

  def record_issue(
      self,
      category: findings.FindingCategory,
      message: str,
      loc: str | None = None,
      severity: findings.Severity = _Severity.ERROR,
      *,
      dedup_key: tuple[Any, ...] | None = None,
  ) -> None:
    if dedup_key is not None:
      if dedup_key in self._dedup_keys:
        return
      self._dedup_keys.add(dedup_key)
    if (category, message) not in self._issue_keys:
      self._issue_keys.add((category, message))
      self.static_issues.append(StaticIssue(category, severity, message, loc))

  def note_incomplete(self, reason: str, *, affects_coverage: bool) -> None:
    if reason not in self.incomplete_reasons:
      self.incomplete_reasons.append(reason)
    if affects_coverage:
      self.coverage_known = False

  def note_unresolved_sync(
      self, reason: str, *, affects_coverage: bool = False
  ) -> None:
    self.sync_reliable = False
    self.note_incomplete(reason, affects_coverage=affects_coverage)

  def note_truncated_sync(self, reason: str, *, affects_coverage: bool) -> None:
    """Records that only a prefix of the semaphore operations is modeled."""
    self.sync_truncated = True
    if affects_coverage:
      self.unwalked_sync = True
    self.note_incomplete(reason, affects_coverage=affects_coverage)

  def has_sync(self, jaxpr: Any) -> bool:
    """Returns whether `jaxpr` or a nested jaxpr uses semaphores."""
    key = id(jaxpr)
    if key not in self._has_sync:
      self._has_sync[key] = any(
          eqn.primitive.name in _SYNC_PRIMITIVES
          or any(self.has_sync(sub) for sub in _sub_jaxprs(eqn))
          for eqn in jaxpr.eqns
      )
    return self._has_sync[key]

  def saw_semaphore(
      self, key: SemaphoreKey, *, recorded: bool, is_signal: bool = False
  ) -> None:
    """Notes a semaphore operation, which may not be recorded as an event."""
    if is_signal:
      self.signaled_sem_keys.add(key)
    if not recorded:
      self.truncated_sem_keys.add(key)
    for scope in self._sem_key_scopes:
      scope.add(key)

  def push_semaphore_scope(self) -> None:
    """Starts collecting the semaphores that the walked code uses."""
    self._sem_key_scopes.append(set())

  def pop_semaphore_scope(self, *, truncated: bool) -> None:
    """Stops collecting; a truncated walk only partly models them."""
    keys = self._sem_key_scopes.pop()
    if truncated:
      self.truncated_sem_keys |= keys

  def record_output_write(
      self, device_id: int, buffer_id: str, region: HyperRectSlice
  ) -> None:
    if not (
        buffer_id.startswith("arg_out_")
        or buffer_id.startswith("pipelined_out_")
    ):
      return
    if region.opaque or not all(d.exact for d in region.dims):
      self.note_incomplete(
          "an output write uses runtime-dependent or strided indices, so"
          " unwritten-output coverage is not checked",
          affects_coverage=True,
      )
      return
    self.output_writes.setdefault((device_id, buffer_id), []).append(region)

  def build(self) -> KernelVerificationModel:
    return KernelVerificationModel(
        kernel_name=self.kernel_name,
        num_devices=self.num_devices,
        grid=self.grid,
        dimension_semantics=self.dimension_semantics,
        collective_id=self.collective_id,
        actor_event_orders={k: tuple(v) for k, v in self.actor_events.items()},
        memory_events=tuple(self.memory_events),
        dma_starts=tuple(self.dma_starts),
        dma_waits=tuple(self.dma_waits),
        sem_signals=tuple(self.sem_signals),
        sem_waits=tuple(self.sem_waits),
        static_issues=tuple(self.static_issues),
        incomplete_reasons=tuple(self.incomplete_reasons),
        sync_reliable=self.sync_reliable,
        sync_truncated=self.sync_truncated,
        uses_remote_operations=self.uses_remote_operations,
        truncated_sem_keys=frozenset(self.truncated_sem_keys),
        unwalked_sync=self.unwalked_sync,
        core_axis_names=self.core_axis_names,
        signaled_sem_keys=frozenset(self.signaled_sem_keys),
    )


@dataclasses.dataclass
class _Env:
  """Refs, concrete scalars, and symbolic expressions bound to jaxpr vars."""

  refs: dict[Any, _RefView]
  scalars: dict[Any, int | bool]
  sym_exprs: dict[Any, Any] = dataclasses.field(default_factory=dict)

  def child(self) -> _Env:
    return _Env(dict(self.refs), dict(self.scalars), dict(self.sym_exprs))

  def bind(self, outer: Any, inner: Any, outer_env: _Env) -> None:
    """Binds `inner` to the ref or scalar value of `outer` in `outer_env`."""
    if _is_jax_var(outer) and outer in outer_env.refs:
      self.refs[inner] = outer_env.refs[outer]
    val = _eval_atom(outer, outer_env.scalars)
    if val is not None:
      self.scalars[inner] = val
    elif _is_jax_var(outer) and outer in outer_env.sym_exprs:
      self.sym_exprs[inner] = outer_env.sym_exprs[outer]


@dataclasses.dataclass
class _WalkContext:
  """Position of a jaxpr walk: device, grid point, actor, and scope."""

  builder: _KernelModelBuilder
  actor: ActorKey
  device_id: int
  device_coords: tuple[int, ...]
  mesh_ctx: _MeshContext
  grid_coords: tuple[int, ...]
  grid_sizes: tuple[int | None, ...]
  scope_path: str
  touched_refs: set[str]
  # Buffer id -> regions stored by the vector unit at this grid point.
  written: dict[str, list[HyperRectSlice]]
  # Whether events are recorded for model checking (only the first grid points
  # are; the rest only feed the coverage and bounds checks).
  record_events: bool = True
  extra_axis_env: Mapping[Any, int] = dataclasses.field(default_factory=dict)
  core_axis_names: tuple[str, ...] = ()
  remote_written: dict[tuple[int, str], list[HyperRectSlice]] = (
      dataclasses.field(default_factory=dict)
  )
  speculative: bool = False
  # Unaliased output buffer/tile id -> (op.describe(), op.index).
  unaliased_output_buffers: dict[str, tuple[str, int]] = dataclasses.field(
      default_factory=dict
  )
  # Buffers read in the current step/iteration (for cross-step 16-bit accum).
  read_in_step: set[str] = dataclasses.field(default_factory=set)
  # Complementary runtime-predicate branch writes in the current scope:
  # (base_pred, truth_value) -> {buf_id: [regions]}.
  guarded_written: dict[tuple[Any, bool], dict[str, list[HyperRectSlice]]] = (
      dataclasses.field(default_factory=dict)
  )
  guarded_output_writes: dict[
      tuple[Any, bool], dict[tuple[int, str], list[HyperRectSlice]]
  ] = dataclasses.field(default_factory=dict)

  @property
  def axis_env(self) -> dict[Any, int]:
    env = self.mesh_ctx.coord_dict(self.device_coords)
    env.update(self.extra_axis_env)
    return env

  def scoped(self, suffix: str) -> _WalkContext:
    return dataclasses.replace(
        self,
        scope_path=f"{self.scope_path}.{suffix}",
        guarded_written={},
        guarded_output_writes={},
    )

  def grid_tag(self) -> str:
    return "g" + "_".join(str(c) for c in self.grid_coords)

  def arb_grid_tag(self) -> str:
    arb = [
        str(c)
        for c, sem in zip(self.grid_coords, self.builder.dimension_semantics)
        if sem != "parallel"
    ]
    return "g" + "_".join(arb)


def _canonical_accum_buffer_id(view: _RefView) -> str:
  buf_id = view.buffer_id
  if buf_id.startswith("tile_"):
    parts = buf_id.split("_", 2)
    if len(parts) >= 2:
      return f"tile_{parts[1]}"
  return buf_id


def _resolve_view(
    ctx: _WalkContext,
    env: _Env,
    ref_or_view: Any,
    extra_transforms: Sequence[Any],
    loc: str | None,
) -> _RefView | None:
  """Resolves a ref operand to a view and records out-of-bounds indexing."""
  root, pre = _unwrap_ref_and_transforms(ref_or_view)
  if not _is_jax_var(root) or root not in env.refs:
    return None
  view, violation = _apply_transforms(
      env.refs[root], (*pre, *extra_transforms), env.scalars
  )
  if violation is not None:
    suffix = (
        " (inside a runtime-dependent branch or loop)"
        if ctx.speculative
        else ""
    )
    ctx.builder.record_issue(
        _Category.OUT_OF_BOUNDS_ACCESS,
        f"access to {view.memory_space}:{view.buffer_id}: {violation}{suffix}",
        loc,
        severity=_Severity.WARNING if ctx.speculative else _Severity.ERROR,
        dedup_key=(
            "ref_oob",
            view.memory_space,
            _canonical_accum_buffer_id(view),
            violation,
            loc,
            ctx.speculative,
        ),
    )
  return view


def _buffer_key(
    ctx: _WalkContext,
    view: _RefView,
    device_id: int,
    target_parallel_coords: tuple[int, ...] | None = None,
) -> BufferKey:
  buf_id = view.buffer_id
  if (
      target_parallel_coords is not None
      and target_parallel_coords != ctx.actor.parallel_coords
  ):
    par_iter = iter(target_parallel_coords)
    target_grid = [
        next(par_iter) if sem == "parallel" else c
        for c, sem in zip(ctx.grid_coords, ctx.builder.dimension_semantics)
    ]
    target_tag = "g" + "_".join(str(c) for c in target_grid)
    cur_tag = ctx.grid_tag()
    if buf_id.startswith(f"{cur_tag}."):
      buf_id = target_tag + buf_id[len(cur_tag) :]
    elif buf_id.endswith(f"_{cur_tag}"):
      buf_id = buf_id[: -len(cur_tag)] + target_tag
  is_shared = view.shared or view.memory_space in ("hbm", "vmem_shared")
  owner_coords = (
      ()
      if is_shared
      else (
          ctx.actor.parallel_coords
          if target_parallel_coords is None
          else target_parallel_coords
      )
  )
  return BufferKey(
      device_id,
      0 if is_shared else ctx.actor.core_id,
      view.memory_space,
      buf_id,
      owner_coords=owner_coords,
  )


def _sem_key(
    ctx: _WalkContext,
    view: _RefView,
    device_id: int,
    default_type: str,
    *,
    parallel_coords: tuple[int, ...] | None = None,
    is_signal: bool = False,
) -> SemaphoreKey | None:
  index = view.sem_index
  if index is None or view.opaque:
    ctx.builder.note_unresolved_sync(
        f"the element of semaphore array {view.buffer_id} could not be"
        " determined statically"
    )
    return None
  sem_type = view.sem_type or default_type
  target_par = (
      ctx.actor.parallel_coords if parallel_coords is None else parallel_coords
  )
  is_shared_sem = (
      view.shared
      or sem_type == "barrier"
      or view.buffer_id.startswith(("scratch_", "arg_", "barrier_sem_"))
  )
  key = SemaphoreKey(
      device_id=device_id,
      core_id=0 if is_shared_sem else ctx.actor.core_id,
      sem_type=sem_type,
      sem_id=view.buffer_id,
      sem_index=index,
      # Each core has its own local semaphore counter (including per-core
      # barrier semaphores); a remote signal targets `target_par`.
      parallel_coords=target_par,
  )
  ctx.builder.saw_semaphore(
      key, recorded=ctx.record_events, is_signal=is_signal
  )
  return key


def _check_uninitialized_output_read(
    ctx: _WalkContext, view: _RefView, loc: str | None
) -> None:
  """Flags reads from an unaliased output buffer/tile before it is written."""
  if ctx.speculative or not ctx.builder.coverage_known:
    return
  out_info = ctx.unaliased_output_buffers.get(view.buffer_id)
  if out_info is None:
    return
  op_desc, op_idx = out_info
  prior_writes = list(
      ctx.builder.active_output_tile_writes.get((ctx.actor, op_idx), ())
  ) + list(ctx.written.get(view.buffer_id, ()))
  if not prior_writes:
    ctx.builder.record_issue(
        _Category.UNWRITTEN_OUTPUT,
        f"{op_desc} is read before being written at grid point"
        f" {ctx.grid_coords} (initialize the output tile on the first step or"
        " alias it via input_output_aliases)",
        loc,
        dedup_key=("uninit_out_read", op_idx, loc),
    )
    return
  if view.region.opaque or not all(d.exact for d in view.region.dims):
    return
  shifted_writes: list[HyperRectSlice] = []
  for w in prior_writes:
    if w.opaque or len(w.dims) != len(view.region.dims):
      return
    if not all(d.exact for d in w.dims):
      return
    shifted_writes.append(
        HyperRectSlice(
            tuple(
                SliceDim(wd.start - vd.start, wd.size)
                for wd, vd in zip(w.dims, view.region.dims)
            )
        )
    )
  sub_shape = tuple(d.size for d in view.region.dims)
  if _uncovered_region(sub_shape, shifted_writes) is not None:
    ctx.builder.record_issue(
        _Category.UNWRITTEN_OUTPUT,
        f"{op_desc} region {view.region} is read before being completely"
        f" written at grid point {ctx.grid_coords}",
        loc,
        dedup_key=("uninit_out_read", op_idx, loc),
    )


def _check_low_precision_ref_accumulation(
    ctx: _WalkContext,
    view: _RefView,
    *,
    is_additive_update: bool,
    loc: str | None,
) -> None:
  """Warns when cross-step accumulation writes into a <32-bit float Ref."""
  try:
    dt = jnp.dtype(view.dtype)
  except TypeError:
    return
  if not (jnp.issubdtype(dt, jnp.floating) and dt.itemsize < 4):
    return
  canon_id = _canonical_accum_buffer_id(view)
  if is_additive_update or (
      canon_id in ctx.read_in_step
      and (ctx.actor, canon_id) in ctx.builder.prior_step_written_buffers
  ):
    ctx.builder.record_issue(
        _Category.LOW_PRECISION_ACCUMULATION,
        f"accumulating across steps into {dt.name} buffer {canon_id} loses"
        f" precision; accumulate in float32 and cast to {dt.name} on the final"
        " write",
        loc,
        severity=_Severity.WARNING,
        dedup_key=("low_prec_ref_accum", canon_id, dt.name),
    )


def _record_access(
    ctx: _WalkContext,
    view: _RefView,
    *,
    is_write: bool,
    loc: str | None,
    is_additive_update: bool = False,
) -> None:
  if view.memory_space == "hbm":
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        f"synchronous {'store to' if is_write else 'load from'} HBM buffer"
        f" {view.buffer_id} is not supported on TPU; use a pipelined"
        " BlockSpec or pltpu.async_copy / pltpu.make_async_copy",
        loc,
        dedup_key=("hbm_sync_access", view.buffer_id, is_write),
    )
  ctx.touched_refs.add(view.buffer_id)
  canon_id = _canonical_accum_buffer_id(view)
  if not is_write:
    _check_uninitialized_output_read(ctx, view, loc)
    ctx.read_in_step.add(canon_id)
  if is_write:
    _check_low_precision_ref_accumulation(
        ctx, view, is_additive_update=is_additive_update, loc=loc
    )
    ctx.written.setdefault(view.buffer_id, []).append(view.region)
    ctx.builder.record_output_write(ctx.device_id, view.buffer_id, view.region)
  if not ctx.record_events:
    return
  builder = ctx.builder
  builder.memory_events.append(
      MemoryAccessEvent(
          event_id=builder.new_event_id(ctx.actor),
          actor=ctx.actor,
          grid_point=ctx.grid_coords,
          buffer=_buffer_key(ctx, view, ctx.device_id),
          region=view.region,
          is_write=is_write,
          source_location=loc,
          speculative=ctx.speculative,
      )
  )


def _resolve_axis_mapping(
    ctx: _WalkContext,
    env: _Env,
    mapping_expr: Mapping[Any, Any],
    *,
    allow_outer_axes: bool,
    core_coords: dict[Any, int | bool | None],
    dev_coords: dict[Any, int | bool | None],
    loc: str | None,
    affects_coverage: bool,
) -> tuple[bool, bool, bool]:
  """Resolves an axis-name dictionary into `core_coords` and `dev_coords`.

  Returns:
    `(ok, has_outer_axis, has_core_axis)`.
  """
  mesh = ctx.mesh_ctx
  core_axes = ctx.core_axis_names
  known_axes = (*mesh.axis_names, *core_axes) if allow_outer_axes else core_axes
  has_outer = False
  has_core = False
  for key, value_expr in mapping_expr.items():
    axes = key if isinstance(key, tuple) else (key,)
    if not axes:
      continue
    for ax in axes:
      if ax in core_axes:
        has_core = True
      elif allow_outer_axes and ax in mesh.axis_names:
        has_outer = True
      else:
        ctx.builder.record_issue(
            _Category.INVALID_DEVICE_ID,
            f"remote operation targets unknown axis {ax!r} (known axes:"
            f" {known_axes})",
            loc,
        )
        ctx.builder.note_unresolved_sync(
            f"a remote operation targeted unknown axis {ax!r}",
            affects_coverage=affects_coverage,
        )
        return False, has_outer, has_core
    val = _eval_atom(value_expr, env.scalars)
    if len(axes) == 1:
      ax = axes[0]
      if ax in core_axes:
        core_coords[ax] = val
      else:
        dev_coords[ax] = val
      continue
    sub_sizes = [
        ctx.grid_sizes[core_axes.index(ax)]
        if ax in core_axes
        else mesh.mesh_shape[mesh.axis_names.index(ax)]
        for ax in axes
    ]
    sub_size_ints = [int(s) for s in sub_sizes if s is not None]
    if val is None or len(sub_size_ints) != len(sub_sizes):
      for ax in axes:
        if ax in core_axes:
          core_coords[ax] = None
        else:
          dev_coords[ax] = None
      continue
    total_sub = int(math.prod(sub_size_ints)) if sub_size_ints else 1
    flat_v = int(val)
    if not 0 <= flat_v < total_sub:
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          f"remote operation targets index {flat_v} along joint axes"
          f" {axes!r}, which only has {total_sub} core(s)/device(s) (wrap the"
          f" index with lax.rem(idx + {total_sub}, {total_sub}))",
          loc,
      )
      ctx.builder.note_unresolved_sync(
          "a remote operation targets a device or core outside the mesh",
          affects_coverage=affects_coverage,
      )
      return False, has_outer, has_core
    sub_coords = (
        np.unravel_index(flat_v, tuple(sub_size_ints)) if sub_size_ints else ()
    )
    for ax, c_val in zip(axes, sub_coords):
      if ax in core_axes:
        core_coords[ax] = int(c_val)
      else:
        dev_coords[ax] = int(c_val)
  return True, has_outer, has_core


def _resolve_target_location(
    ctx: _WalkContext,
    env: _Env,
    device_id_expr: Any,
    loc: str | None,
    *,
    core_index_expr: Any = None,
    device_id_type: Any = None,
    affects_coverage: bool = False,
) -> tuple[int, tuple[int, ...], bool] | None:
  """Resolves a Pallas `device_id` (and optional `core_index`) to `(device_id, parallel_coords, is_remote)`.

  Returns:
    `(target_device_id, target_parallel_coords, is_remote)`, or None if the
    target is invalid or unknown (both are recorded on the builder).
  """
  if device_id_expr is None and core_index_expr is None:
    return ctx.device_id, ctx.actor.parallel_coords, False

  mesh = ctx.mesh_ctx
  core_axes = ctx.core_axis_names
  id_type_str = (
      getattr(device_id_type, "value", str(device_id_type)).lower()
      if device_id_type is not None
      else "mesh"
  )
  if "logical" in id_type_str:
    id_type_str = "logical"
  else:
    id_type_str = "mesh"

  dev_coords: dict[Any, int | bool | None] = dict(
      zip(mesh.axis_names, ctx.device_coords)
  )
  core_coords: dict[Any, int | bool | None] = dict(
      zip(core_axes, ctx.grid_coords)
  )
  has_outer_axis = False
  has_core_axis = False

  if isinstance(device_id_expr, Mapping):
    if id_type_str == "logical":
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          "device_id is an axis-name dictionary, which is incompatible with"
          " device_id_type=pltpu.DeviceIdType.LOGICAL (use"
          " pltpu.DeviceIdType.MESH)",
          loc,
      )
      ctx.builder.note_unresolved_sync(
          "a remote operation used a dictionary device_id with LOGICAL mode",
          affects_coverage=affects_coverage,
      )
      return None
    ok, has_outer_axis, has_core_axis = _resolve_axis_mapping(
        ctx,
        env,
        device_id_expr,
        allow_outer_axes=True,
        core_coords=core_coords,
        dev_coords=dev_coords,
        loc=loc,
        affects_coverage=affects_coverage,
    )
    if not ok:
      return None

  if core_index_expr is not None:
    if not core_axes:
      ctx.builder.note_unresolved_sync(
          "a core_index was specified without an enclosing core mesh",
          affects_coverage=affects_coverage,
      )
      return None
    has_core_axis = True
    if isinstance(core_index_expr, Mapping):
      ok, _, _ = _resolve_axis_mapping(
          ctx,
          env,
          core_index_expr,
          allow_outer_axes=False,
          core_coords=core_coords,
          dev_coords=dev_coords,
          loc=loc,
          affects_coverage=affects_coverage,
      )
      if not ok:
        return None
    elif isinstance(core_index_expr, tuple):
      if len(core_index_expr) != len(core_axes):
        ctx.builder.record_issue(
            _Category.INVALID_DEVICE_ID,
            f"core_index tuple has {len(core_index_expr)} coordinate(s), but"
            f" core mesh {core_axes} has {len(core_axes)} axis/axes",
            loc,
        )
        ctx.builder.note_unresolved_sync(
            "a remote core_index tuple had the wrong rank",
            affects_coverage=affects_coverage,
        )
        return None
      for axis, value_expr in zip(core_axes, core_index_expr):
        core_coords[axis] = _eval_atom(value_expr, env.scalars)
    else:
      if len(core_axes) > 1:
        ctx.builder.record_issue(
            _Category.INVALID_DEVICE_ID,
            f"scalar core_index is ambiguous on {len(core_axes)}D core mesh"
            f" {core_axes}; pass a dict of axis names or a"
            f" {len(core_axes)}-tuple",
            loc,
        )
        ctx.builder.note_unresolved_sync(
            "a scalar core_index was used on a multi-axis core mesh",
            affects_coverage=affects_coverage,
        )
        return None
      core_coords[core_axes[0]] = _eval_atom(core_index_expr, env.scalars)

  target_parallel_coords = ctx.actor.parallel_coords
  if has_core_axis:
    c_values = [core_coords.get(a) for a in core_axes]
    c_ints = [int(v) for v in c_values if v is not None]
    if len(c_ints) != len(c_values) or len(c_ints) != len(ctx.grid_sizes):
      ctx.builder.note_unresolved_sync(
          "a remote core index could not be determined statically",
          affects_coverage=affects_coverage,
      )
      return None
    for name, value, size in zip(core_axes, c_ints, ctx.grid_sizes):
      if size is None:
        ctx.builder.note_unresolved_sync(
            "a remote core mesh dimension could not be determined statically",
            affects_coverage=affects_coverage,
        )
        return None
      if not 0 <= value < size:
        ctx.builder.record_issue(
            _Category.INVALID_DEVICE_ID,
            f"remote operation targets index {value} along {name!r}, which"
            f" only has {size} core(s)/device(s) (wrap the index with"
            f" lax.rem(idx + {size}, {size}))",
            loc,
        )
        ctx.builder.note_unresolved_sync(
            "a remote operation targets a core outside the mesh",
            affects_coverage=affects_coverage,
        )
        return None
    target_parallel_coords = tuple(
        c_ints[ax]
        for ax, sem in enumerate(ctx.builder.dimension_semantics)
        if sem == "parallel"
    )

  if device_id_expr is None or (
      isinstance(device_id_expr, Mapping) and not has_outer_axis
  ):
    is_remote = target_parallel_coords != ctx.actor.parallel_coords
    return ctx.device_id, target_parallel_coords, is_remote

  ctx.builder.uses_remote_operations = True
  if isinstance(device_id_expr, Mapping):
    names: Sequence[str] = mesh.axis_names
    sizes: Sequence[int] = mesh.mesh_shape
    values = [dev_coords.get(a) for a in mesh.axis_names]
  elif isinstance(device_id_expr, tuple):
    if id_type_str == "logical":
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          "device_id is a coordinate tuple, which is incompatible with"
          " device_id_type=pltpu.DeviceIdType.LOGICAL",
          loc,
      )
      ctx.builder.note_unresolved_sync(
          "a tuple device_id was used with LOGICAL mode",
          affects_coverage=affects_coverage,
      )
      return None
    if len(device_id_expr) != len(mesh.mesh_shape):
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          f"remote operation device_id tuple has {len(device_id_expr)}"
          f" coordinate(s), but mesh {mesh.axis_names} has rank"
          f" {len(mesh.mesh_shape)}",
          loc,
      )
      ctx.builder.note_unresolved_sync(
          "a remote device_id tuple had the wrong rank",
          affects_coverage=affects_coverage,
      )
      return None
    names, sizes = mesh.axis_names, mesh.mesh_shape
    values = [_eval_atom(e, env.scalars) for e in device_id_expr]
  else:
    if id_type_str == "mesh" and len(mesh.mesh_shape) > 1:
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          "scalar device_id with DeviceIdType.MESH is invalid on"
          f" {len(mesh.mesh_shape)}D mesh {mesh.axis_names}; pass a dict of"
          f" axis names, a {len(mesh.mesh_shape)}-tuple of coordinates, or"
          " device_id_type=pltpu.DeviceIdType.LOGICAL",
          loc,
      )
      ctx.builder.note_unresolved_sync(
          "a scalar device_id was used in MESH mode on a multi-axis mesh",
          affects_coverage=affects_coverage,
      )
      return None
    names, sizes = ("logical device id",), (mesh.num_devices,)
    values = [_eval_atom(device_id_expr, env.scalars)]
  ints = [int(v) for v in values if v is not None]
  if len(ints) != len(values) or len(ints) != len(sizes):
    ctx.builder.note_unresolved_sync(
        "a remote device id could not be determined statically",
        affects_coverage=affects_coverage,
    )
    return None
  for name, value, size in zip(names, ints, sizes):
    if not 0 <= value < size:
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          f"remote operation targets index {value} along {name!r}, which"
          f" only has {size} device(s) (wrap the index with"
          f" lax.rem(idx + {size}, {size}))",
          loc,
      )
      ctx.builder.note_unresolved_sync(
          "a remote operation targets a device outside the mesh",
          affects_coverage=affects_coverage,
      )
      return None
  if not sizes:
    target_dev = ctx.device_id
  elif len(ints) == 1 and len(mesh.mesh_shape) != 1:
    target_dev = ints[0]  # A logical id on a multi-axis mesh.
  else:
    target_dev = int(np.ravel_multi_index(tuple(ints), tuple(sizes)))
  is_remote = (
      target_dev != ctx.device_id
      or target_parallel_coords != ctx.actor.parallel_coords
  )
  return target_dev, target_parallel_coords, is_remote


_UNRESOLVED_ACCESS = (
    "a ref access could not be resolved statically, so it is not checked"
)
_UNRESOLVED_SYNC = "a DMA or semaphore operand could not be resolved statically"


def _handle_get(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  view = None
  if "tree" in eqn.params:
    transforms = jax.tree_util.tree_unflatten(
        eqn.params["tree"], eqn.invars[1:]
    )
    view = _resolve_view(ctx, env, eqn.invars[0], transforms, loc)
  if view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_ACCESS, affects_coverage=False)
    return
  _record_access(ctx, view, is_write=False, loc=loc)
  if (
      len(eqn.outvars) == 1
      and not _is_drop_var(eqn.outvars[0])
      and view.sem_index is not None
  ):
    env.sym_exprs[eqn.outvars[0]] = (
        "load",
        view.buffer_id,
        view.sem_index,
        len(ctx.written.get(view.buffer_id, ())),
    )


def _handle_swap(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  view = None
  if "tree" in eqn.params:
    transforms = jax.tree_util.tree_unflatten(
        eqn.params["tree"], eqn.invars[2:]
    )
    view = _resolve_view(ctx, env, eqn.invars[0], transforms, loc)
  if view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_ACCESS, affects_coverage=True)
    return
  is_add = eqn.primitive.name == "addupdate"
  if is_add:
    _record_access(ctx, view, is_write=False, loc=loc)
  _record_access(ctx, view, is_write=True, loc=loc, is_additive_update=is_add)


def _handle_masked(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  is_write = eqn.primitive.name == "masked_swap"
  view = None
  if "args_tree" in eqn.params:
    ref_arg, transforms, *_ = eqn.params["args_tree"].unflatten(eqn.invars)
    view = _resolve_view(ctx, env, ref_arg, transforms, loc)
  if view is None:
    ctx.builder.note_incomplete(_UNRESOLVED_ACCESS, affects_coverage=is_write)
    return
  # Masked stores are treated as writing their whole region, which can only
  # hide unwritten-output findings, never create false ones.
  _record_access(ctx, view, is_write=is_write, loc=loc)


def _check_dma_sem_view(
    ctx: _WalkContext,
    view: _RefView,
    loc: str | None,
    *,
    allow_done_count_regular: bool = False,
    is_smem_dma: bool = False,
) -> bool:
  if view.sem_type is None:
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        f"asynchronous copy semaphore operand {view.buffer_id} is a data"
        " buffer ref, not a semaphore",
        loc,
    )
    ctx.builder.note_unresolved_sync(
        "an asynchronous copy used a data buffer ref as a semaphore"
    )
    return False
  if not view.opaque and view.view_shape != ():
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        f"asynchronous copy semaphore {view.buffer_id} has shape"
        f" {view.view_shape}; DMA semaphore operands must be 0D scalar"
        " semaphore refs (index the semaphore array to a single element)",
        loc,
    )
    ctx.builder.note_unresolved_sync(
        "an asynchronous copy used a non-0D semaphore ref"
    )
    return False
  if view.sem_type == "dma":
    return True
  if view.sem_type == "regular" and allow_done_count_regular:
    if is_smem_dma:
      ctx.builder.record_issue(
          _Category.HARDWARE_CONSTRAINT_VIOLATION,
          f"asynchronous SMEM copy uses regular semaphore {view.buffer_id};"
          " SMEM DMA copies require pltpu.SemaphoreType.DMA",
          loc,
      )
      ctx.builder.note_unresolved_sync(
          "an asynchronous SMEM copy used a regular semaphore"
      )
      return False
    if findings.supports_dma_done_count_mode(ctx.builder.tpu_device_kind):
      return True
  ctx.builder.record_issue(
      _Category.HARDWARE_CONSTRAINT_VIOLATION,
      f"asynchronous copy uses {view.sem_type} semaphore"
      f" {view.buffer_id}; DMA copies require pltpu.SemaphoreType.DMA",
      loc,
  )
  ctx.builder.note_unresolved_sync(
      "an asynchronous copy used a non-DMA semaphore"
  )
  return False


def _check_non_dma_sem_view(
    ctx: _WalkContext, view: _RefView, op_desc: str, loc: str | None
) -> bool:
  if view.sem_type is None:
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        f"explicit semaphore {op_desc} targets data buffer {view.buffer_id},"
        " not a semaphore",
        loc,
    )
    ctx.builder.note_unresolved_sync(
        f"explicit semaphore {op_desc} targeted a data buffer ref"
    )
    return False
  if not view.opaque and view.view_shape != ():
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        f"explicit semaphore {op_desc} targets semaphore {view.buffer_id} of"
        f" shape {view.view_shape}; semaphore operations require a 0D scalar"
        " semaphore ref (index the semaphore array to a single element)",
        loc,
    )
    ctx.builder.note_unresolved_sync(
        f"explicit semaphore {op_desc} targeted a non-0D semaphore ref"
    )
    return False
  if view.sem_type == "dma":
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        f"explicit semaphore {op_desc} targets DMA semaphore"
        f" {view.buffer_id}; use pltpu.SemaphoreType.REGULAR for explicit"
        " signal/wait or wait via the async copy descriptor",
        loc,
    )
    ctx.builder.note_unresolved_sync(
        f"explicit semaphore {op_desc} targeted a DMA semaphore"
    )
    return False
  return True


def _handle_dma_start(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  ops = decode_dma_operands(eqn.params["tree"].unflatten(eqn.invars))
  src = _resolve_view(ctx, env, ops.src.ref, ops.src.transforms, loc)
  dst = _resolve_view(ctx, env, ops.dst.ref, ops.dst.transforms, loc)
  if src is None or dst is None:
    affects_cov = dst is None or (
        dst.buffer_id.startswith("arg_out_")
        or dst.buffer_id.startswith("tile_")
    )
    ctx.builder.note_unresolved_sync(
        _UNRESOLVED_SYNC, affects_coverage=affects_cov
    )
    return
  if not src.opaque and not dst.opaque:
    if src.view_shape != dst.view_shape:
      ctx.builder.record_issue(
          _Category.HARDWARE_CONSTRAINT_VIOLATION,
          f"asynchronous copy from {src.memory_space}:{src.buffer_id} (shape"
          f" {src.view_shape}) to {dst.memory_space}:{dst.buffer_id} (shape"
          f" {dst.view_shape}) has mismatched shapes",
          loc,
      )
    if np.dtype(src.dtype) != np.dtype(dst.dtype):
      ctx.builder.record_issue(
          _Category.HARDWARE_CONSTRAINT_VIOLATION,
          f"asynchronous copy from {src.memory_space}:{src.buffer_id} (dtype"
          f" {np.dtype(src.dtype)}) to {dst.memory_space}:{dst.buffer_id}"
          f" (dtype {np.dtype(dst.dtype)}) has mismatched dtypes",
          loc,
      )
  resolved = _resolve_target_location(
      ctx,
      env,
      ops.device_id,
      loc,
      device_id_type=eqn.params.get("device_id_type"),
      affects_coverage=(
          dst.buffer_id.startswith("arg_out_")
          or dst.buffer_id.startswith("tile_")
      ),
  )
  if resolved is None:
    return
  target, target_par, is_remote = resolved
  if ops.device_id is None and ops.src_sem is not None:
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        "local asynchronous copy cannot specify a source/send semaphore"
        " (src_sem is only valid on remote copies with device_id)",
        loc,
    )
  elif is_remote and ops.src_sem is None:
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        "remote asynchronous copy to another device or core requires a send"
        " semaphore (send_sem / src_sem)",
        loc,
    )
  ctx.touched_refs.add(src.buffer_id)
  is_add = bool(eqn.params.get("add", False))
  if (
      is_add
      and target == ctx.device_id
      and target_par == ctx.actor.parallel_coords
  ):
    _check_uninitialized_output_read(ctx, dst, loc)
    _check_low_precision_ref_accumulation(
        ctx, dst, is_additive_update=True, loc=loc
    )
  ctx.builder.record_output_write(target, dst.buffer_id, dst.region)
  if target == ctx.device_id and target_par == ctx.actor.parallel_coords:
    ctx.written.setdefault(dst.buffer_id, []).append(dst.region)
  else:
    ctx.remote_written.setdefault((target, dst.buffer_id), []).append(
        dst.region
    )
  is_smem_dma = src.memory_space == "smem" or dst.memory_space == "smem"
  dst_sem = _operand_sem(
      ctx,
      env,
      ops.dst_sem,
      target,
      loc,
      allow_done_count_regular=True,
      is_smem_dma=is_smem_dma,
      parallel_coords=target_par,
  )
  src_sem = _operand_sem(
      ctx,
      env,
      ops.src_sem,
      ctx.device_id,
      loc,
      allow_done_count_regular=True,
      is_smem_dma=is_smem_dma,
  )
  if (
      dst_sem is not None
      and src_sem is not None
      and dst_sem.sem_type != src_sem.sem_type
  ):
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        "remote asynchronous copy uses mismatched semaphore types: send_sem"
        f" {src_sem.sem_id} is {src_sem.sem_type} while recv_sem"
        f" {dst_sem.sem_id} is {dst_sem.sem_type}",
        loc,
    )
    ctx.builder.note_unresolved_sync(
        "a remote asynchronous copy used mismatched send/recv semaphore types"
    )
    return
  if not ctx.record_events:
    return
  builder = ctx.builder
  builder.dma_starts.append(
      DmaStartEvent(
          event_id=builder.new_event_id(ctx.actor),
          dma_id=builder.new_dma_id(),
          actor=ctx.actor,
          grid_point=ctx.grid_coords,
          src_buffer=_buffer_key(ctx, src, ctx.device_id),
          src_region=src.region,
          dst_buffer=_buffer_key(ctx, dst, target, target_par),
          dst_region=dst.region,
          byte_count=min(src.byte_count, dst.byte_count),
          dst_sem=dst_sem,
          src_sem=src_sem,
          is_remote=is_remote,
          source_location=loc,
      )
  )


def _operand_sem(
    ctx: _WalkContext,
    env: _Env,
    operand: RefOperand | None,
    device_id: int,
    loc: str | None,
    *,
    allow_done_count_regular: bool = False,
    is_smem_dma: bool = False,
    parallel_coords: tuple[int, ...] | None = None,
) -> SemaphoreKey | None:
  if operand is None:
    return None
  view = _resolve_view(ctx, env, operand.ref, operand.transforms, loc)
  if view is None:
    ctx.builder.note_unresolved_sync(_UNRESOLVED_SYNC)
    return None
  if not _check_dma_sem_view(
      ctx,
      view,
      loc,
      allow_done_count_regular=allow_done_count_regular,
      is_smem_dma=is_smem_dma,
  ):
    return None
  return _sem_key(
      ctx,
      view,
      device_id,
      "dma",
      parallel_coords=parallel_coords,
      is_signal=True,
  )


def _handle_dma_wait(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  ops = decode_dma_operands(eqn.params["tree"].unflatten(eqn.invars))
  # `wait_send` swaps the operands, so the waited semaphore is always in the
  # destination-semaphore slot and is always local.
  src = _resolve_view(ctx, env, ops.src.ref, ops.src.transforms, loc)
  dst = _resolve_view(ctx, env, ops.dst.ref, ops.dst.transforms, loc)
  sem_view = (
      None
      if ops.dst_sem is None
      else _resolve_view(ctx, env, ops.dst_sem.ref, ops.dst_sem.transforms, loc)
  )
  if src is None or dst is None or sem_view is None:
    ctx.builder.note_unresolved_sync(_UNRESOLVED_SYNC)
    return
  is_smem_dma = src.memory_space == "smem" or dst.memory_space == "smem"
  # JAX Pallas requires `pl.semaphore_wait` (not `dma_wait`) when waiting on a
  # done-count REGULAR semaphore used by `dma_start`.
  if not _check_dma_sem_view(
      ctx,
      sem_view,
      loc,
      allow_done_count_regular=False,
      is_smem_dma=is_smem_dma,
  ):
    return
  sem = _sem_key(ctx, sem_view, ctx.device_id, "dma")
  if sem is None or not ctx.record_events:
    return
  builder = ctx.builder
  builder.dma_waits.append(
      DmaWaitEvent(
          event_id=builder.new_event_id(ctx.actor),
          actor=ctx.actor,
          grid_point=ctx.grid_coords,
          sem=sem,
          byte_count=min(src.byte_count, dst.byte_count),
          is_wait_send=bool(eqn.params.get("is_wait_send", False)),
          source_location=loc,
      )
  )


def _handle_barrier(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  tag = eqn.params.get("tag")
  sem_id = (
      f"barrier_sem_{tag if tag is not None else ctx.builder.collective_id}"
  )
  env.refs[eqn.outvars[0]] = _full_view(
      sem_id, "semaphore_mem", (), np.int32, sem_type="barrier"
  )


def _semaphore_amount(
    ctx: _WalkContext,
    env: _Env,
    amount_expr: Any,
    what: str,
    loc: str | None = None,
) -> int | None:
  amount = _eval_atom(amount_expr, env.scalars)
  if amount is None:
    ctx.builder.note_unresolved_sync(
        f"a semaphore {what} amount could not be determined statically"
    )
    return None
  val = int(amount)
  if val < 0:
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        f"semaphore {what} amount {val} must be non-negative on TPU",
        loc,
    )
    ctx.builder.note_unresolved_sync(
        f"a semaphore {what} used a negative amount ({val})"
    )
    return None
  return val


def _handle_sem_signal(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  unflattened = eqn.params["args_tree"].unflatten(eqn.invars)
  sem_arg, sem_tr, inc_arg, dev_id_arg, *rest = unflattened
  core_index_arg = rest[0] if rest else None
  view = _resolve_view(ctx, env, sem_arg, sem_tr, loc)
  if view is None:
    ctx.builder.note_unresolved_sync(_UNRESOLVED_SYNC)
    return
  if not _check_non_dma_sem_view(ctx, view, "signal", loc):
    return
  amount = _semaphore_amount(ctx, env, inc_arg, "signal", loc)
  resolved = _resolve_target_location(
      ctx,
      env,
      dev_id_arg,
      loc,
      core_index_expr=core_index_arg,
      device_id_type=eqn.params.get("device_id_type"),
  )
  if amount is None or resolved is None:
    return
  target, target_par, is_remote = resolved
  sem = _sem_key(
      ctx,
      view,
      target,
      "regular",
      parallel_coords=target_par,
      is_signal=True,
  )
  if sem is None or not ctx.record_events:
    return
  builder = ctx.builder
  builder.sem_signals.append(
      SemaphoreSignalEvent(
          event_id=builder.new_event_id(ctx.actor),
          actor=ctx.actor,
          grid_point=ctx.grid_coords,
          target_sem=sem,
          amount=amount,
          is_remote=is_remote,
          source_location=loc,
      )
  )


def _handle_sem_wait(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  sem_arg, sem_tr, val_arg, *rest = eqn.params["args_tree"].unflatten(
      eqn.invars
  )
  decrement_arg = rest[0] if rest else True
  view = _resolve_view(ctx, env, sem_arg, sem_tr, loc)
  if view is None:
    ctx.builder.note_unresolved_sync(_UNRESOLVED_SYNC)
    return
  if not _check_non_dma_sem_view(ctx, view, "wait", loc):
    return
  decrement = _eval_atom(decrement_arg, env.scalars)
  if decrement is False:
    ctx.builder.record_issue(
        _Category.HARDWARE_CONSTRAINT_VIOLATION,
        "pl.semaphore_wait with decrement=False is not supported on TPU",
        loc,
    )
    ctx.builder.note_unresolved_sync(
        "pl.semaphore_wait used unsupported decrement=False"
    )
    return
  amount = _semaphore_amount(ctx, env, val_arg, "wait", loc)
  sem = _sem_key(ctx, view, ctx.device_id, "regular")
  if amount is None or sem is None or not ctx.record_events:
    return
  builder = ctx.builder
  builder.sem_waits.append(
      SemaphoreWaitEvent(
          event_id=builder.new_event_id(ctx.actor),
          actor=ctx.actor,
          grid_point=ctx.grid_coords,
          sem=sem,
          amount=amount,
          decrement=True if decrement is None else bool(decrement),
          source_location=loc,
      )
  )


def _handle_run_scoped(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  sub_jaxpr = _unwrap_closed_jaxpr(eqn.params["jaxpr"])
  ref_transforms = eqn.params.get("ref_transforms", ())
  shared = bool(eqn.params.get("collective_axes", ()))
  nested = env.child()
  for outer_v, const_v in zip(eqn.invars, sub_jaxpr.constvars):
    nested.bind(outer_v, const_v, env)
  scope = ctx.scoped(f"s{_next_scope_index(ctx)}")
  for alloc_idx, alloc_var in enumerate(sub_jaxpr.invars):
    aval = alloc_var.aval
    sem_type = _detect_semaphore_type(aval)
    if shared:
      prefix = scope.scope_path
    elif sem_type is not None:
      prefix = f"{ctx.arb_grid_tag()}{scope.scope_path}"
    else:
      prefix = f"{ctx.grid_tag()}{scope.scope_path}"
    base = _full_view(
        f"{prefix}.b{alloc_idx}",
        _normalize_memory_space(getattr(aval, "memory_space", None)),
        getattr(aval, "shape", ()),
        getattr(aval, "dtype", np.float32),
        sem_type=sem_type,
        shared=shared,
    )
    init_tr = (
        ref_transforms[alloc_idx] if alloc_idx < len(ref_transforms) else ()
    )
    view, violation = _apply_transforms(base, init_tr, nested.scalars)
    if violation is not None:
      ctx.builder.record_issue(
          _Category.OUT_OF_BOUNDS_ACCESS,
          f"access to {view.memory_space}:{view.buffer_id}: {violation}",
          loc,
      )
    nested.refs[alloc_var] = view
  _walk(sub_jaxpr, scope, nested)
  _bind_outputs(sub_jaxpr.outvars, eqn.outvars, nested, env)


def _next_scope_index(ctx: _WalkContext) -> int:
  """Returns a scope index unique within the current scope path."""
  return ctx.builder.next_scope_index(
      (ctx.device_id, ctx.grid_coords, ctx.scope_path)
  )


def _bind_outputs(
    inner_outs: Sequence[Any],
    outer_outs: Sequence[Any],
    inner: _Env,
    outer: _Env,
) -> None:
  for inner_out, outer_out in zip(inner_outs, outer_outs):
    val = _eval_atom(inner_out, inner.scalars)
    if val is not None and not _is_drop_var(outer_out):
      outer.scalars[outer_out] = val
    elif (
        _is_jax_var(inner_out)
        and inner_out in inner.sym_exprs
        and not _is_drop_var(outer_out)
    ):
      outer.sym_exprs[outer_out] = inner.sym_exprs[inner_out]


def _record_step_written_buffers(ctx: _WalkContext) -> None:
  for buf_id, regions in ctx.written.items():
    if not regions:
      continue
    canon_id = buf_id
    if buf_id.startswith("tile_"):
      parts = buf_id.split("_", 2)
      if len(parts) >= 2:
        canon_id = f"tile_{parts[1]}"
    ctx.builder.prior_step_written_buffers.add((ctx.actor, canon_id))


def _scan_layout(params: Mapping[str, Any]) -> tuple[int, int]:
  """Returns `(num_consts, num_carry)` of a scan equation."""
  if "num_consts" in params:  # Older JAX releases.
    return int(params["num_consts"]), int(params["num_carry"])
  consts, carry, _ = params["ft_in"].unpack()
  return len(consts), len(carry)


def _note_truncated_loop(ctx: _WalkContext, body: Any, reason: str) -> None:
  """Records a loop whose later iterations are not walked."""
  if ctx.builder.has_sync(body):
    ctx.builder.note_truncated_sync(reason, affects_coverage=True)
  else:
    ctx.builder.note_incomplete(reason, affects_coverage=True)


def _handle_scan(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  body = _unwrap_closed_jaxpr(eqn.params["jaxpr"])
  num_consts, num_carry = _scan_layout(eqn.params)
  length = int(eqn.params["length"])
  limit = ctx.builder.max_unrolled_iterations
  truncated = length > limit
  if truncated:
    _note_truncated_loop(
        ctx,
        body,
        f"only the first {limit} of {length} loop iterations are checked",
    )
  consts = eqn.invars[:num_consts]
  init_carry = list(eqn.invars[num_consts : num_consts + num_carry])
  carry = list(init_carry)
  carry_env = env
  ctx.builder.push_semaphore_scope()
  try:
    for step in range(min(length, limit)):
      nested = env.child()
      for outer_v, inner_v in zip(consts, body.invars[:num_consts]):
        nested.bind(outer_v, inner_v, env)
      for outer_v, inner_v in zip(carry, body.invars[num_consts:]):
        nested.bind(outer_v, inner_v, carry_env)
      step_ctx = dataclasses.replace(
          ctx.scoped(f"lp{step}"), read_in_step=set()
      )
      _walk(body, step_ctx, nested)
      _record_step_written_buffers(step_ctx)
      carry = list(body.outvars[:num_carry])
      carry_env = nested
  finally:
    ctx.builder.pop_semaphore_scope(truncated=truncated)
  if not truncated:
    _bind_outputs(carry, eqn.outvars[:num_carry], carry_env, env)
  else:
    for init_v, final_v, out_v in zip(
        init_carry, carry, eqn.outvars[:num_carry]
    ):
      init_val = _eval_atom(init_v, env.scalars)
      final_val = _eval_atom(final_v, carry_env.scalars)
      if (
          init_val is not None
          and init_val == final_val
          and not _is_drop_var(out_v)
      ):
        env.scalars[out_v] = init_val


def _eval_predicate(jaxpr: Any, ctx: _WalkContext, env: _Env) -> bool | None:
  scalars = dict(env.scalars)
  for eqn in jaxpr.eqns:
    _try_eval_scalar_eqn(
        eqn,
        scalars,
        grid_coords=ctx.grid_coords,
        grid_sizes=ctx.grid_sizes,
        axis_env=ctx.axis_env,
    )
  value = _eval_atom(jaxpr.outvars[0], scalars)
  return None if value is None else bool(value)


def _handle_while(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  cond = _unwrap_closed_jaxpr(eqn.params["cond_jaxpr"])
  body = _unwrap_closed_jaxpr(eqn.params["body_jaxpr"])
  cond_nconsts = int(eqn.params["cond_nconsts"])
  body_nconsts = int(eqn.params["body_nconsts"])
  cond_consts = eqn.invars[:cond_nconsts]
  body_consts = eqn.invars[cond_nconsts : cond_nconsts + body_nconsts]
  init_carry = list(eqn.invars[cond_nconsts + body_nconsts :])
  carry = list(init_carry)
  carry_env = env
  limit = ctx.builder.max_unrolled_iterations
  truncated = False
  exited = False
  ctx.builder.push_semaphore_scope()
  try:
    for step in range(limit + 1):
      cond_env = env.child()
      for outer_v, inner_v in zip(cond_consts, cond.invars[:cond_nconsts]):
        cond_env.bind(outer_v, inner_v, env)
      for outer_v, inner_v in zip(carry, cond.invars[cond_nconsts:]):
        cond_env.bind(outer_v, inner_v, carry_env)
      keep_going = _eval_predicate(cond, ctx, cond_env)
      if keep_going is False:
        exited = True
        break
      if step == limit:
        truncated = True
        _note_truncated_loop(
            ctx,
            body,
            f"only the first {limit} iterations of a while loop are checked",
        )
        break
      if keep_going is None:
        reason = (
            "a while loop's trip count depends on runtime data, so its body"
            " is checked once"
        )
        if ctx.builder.has_sync(body):
          # One iteration is neither a prefix nor all of the semaphore
          # traffic.
          ctx.builder.note_unresolved_sync(reason, affects_coverage=True)
        else:
          ctx.builder.note_incomplete(reason, affects_coverage=True)
      nested = env.child()
      for outer_v, inner_v in zip(body_consts, body.invars[:body_nconsts]):
        nested.bind(outer_v, inner_v, env)
      for outer_v, inner_v in zip(carry, body.invars[body_nconsts:]):
        nested.bind(outer_v, inner_v, carry_env)
      step_ctx = dataclasses.replace(
          ctx.scoped(f"wh{step}"),
          speculative=ctx.speculative or (keep_going is None),
          read_in_step=set(),
      )
      _walk(body, step_ctx, nested)
      _record_step_written_buffers(step_ctx)
      carry = list(body.outvars)
      carry_env = nested
      if keep_going is None:
        break
  finally:
    ctx.builder.pop_semaphore_scope(truncated=truncated)
  if exited:
    _bind_outputs(carry, eqn.outvars, carry_env, env)
  else:
    for init_v, final_v, out_v in zip(init_carry, carry, eqn.outvars):
      init_val = _eval_atom(init_v, env.scalars)
      final_val = _eval_atom(final_v, carry_env.scalars)
      if (
          init_val is not None
          and init_val == final_val
          and not _is_drop_var(out_v)
      ):
        env.scalars[out_v] = init_val


def _intersect_branch_regions(
    branch_regions: Sequence[Sequence[HyperRectSlice]],
) -> list[HyperRectSlice]:
  """Returns exact sub-regions covered along every branch in `branch_regions`."""
  if not branch_regions or any(not br for br in branch_regions):
    return []
  exact_branches = [
      [
          r
          for r in br
          if not r.opaque and all(d.exact and d.size > 0 for d in r.dims)
      ]
      for br in branch_regions
  ]
  if any(not br for br in exact_branches):
    return []
  rank = len(exact_branches[0][0].dims)
  if any(len(r.dims) != rank for br in exact_branches for r in br):
    return [
        r
        for r in branch_regions[0]
        if all(r in other for other in branch_regions[1:])
    ]
  if rank == 0:
    return [HyperRectSlice(())]
  cuts: list[list[int]] = []
  for axis in range(rank):
    points = {
        bound
        for br in exact_branches
        for r in br
        for bound in (r.dims[axis].start, r.dims[axis].stop)
    }
    cuts.append(sorted(points))
  cell_shape = tuple(max(0, len(c) - 1) for c in cuts)
  if any(s == 0 for s in cell_shape):
    return []
  if math.prod(cell_shape) > _MAX_COVERAGE_CELLS:
    return [
        r
        for r in branch_regions[0]
        if all(r in other for other in branch_regions[1:])
    ]
  common_mask = np.ones(cell_shape, dtype=bool)
  for br in exact_branches:
    br_mask = np.zeros(cell_shape, dtype=bool)
    for r in br:
      slices = [
          slice(
              int(np.searchsorted(cuts[a], r.dims[a].start)),
              int(np.searchsorted(cuts[a], r.dims[a].stop)),
          )
          for a in range(rank)
      ]
      br_mask[tuple(slices)] = True
    common_mask &= br_mask
    if not bool(common_mask.any()):
      return []
  if bool(common_mask.all()):
    return [HyperRectSlice(tuple(SliceDim(c[0], c[-1] - c[0]) for c in cuts))]
  common: list[HyperRectSlice] = []
  for cell in np.argwhere(common_mask):
    common.append(
        HyperRectSlice(
            tuple(
                SliceDim(cuts[a][int(i)], cuts[a][int(i) + 1] - cuts[a][int(i)])
                for a, i in enumerate(cell)
            )
        )
    )
  return common


def _canonical_cond_predicate(
    pred_atom: Any, env: _Env
) -> tuple[Any, bool] | None:
  """Returns `(base_pred, polarity)` for a 2-branch `cond` selector atom."""
  if not _is_jax_var(pred_atom) or pred_atom not in env.sym_exprs:
    return None
  sym = env.sym_exprs[pred_atom]
  polarity = True
  while isinstance(sym, tuple) and len(sym) == 2 and sym[0] == "not":
    polarity = not polarity
    sym = sym[1]
  return sym, polarity


def _handle_cond(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  branches = eqn.params["branches"]
  index = _eval_atom(eqn.invars[0], env.scalars)
  if index is not None and 0 <= int(index) < len(branches):
    selected = [int(index)]
  else:
    selected = list(range(len(branches)))
    reason = (
        "a branch condition depends on runtime data, so all branches are"
        " checked in sequence"
    )
    if any(ctx.builder.has_sync(_unwrap_closed_jaxpr(b)) for b in branches):
      # Semaphore operations of branches that exclude each other would be
      # counted together.
      ctx.builder.note_unresolved_sync(reason)
    else:
      ctx.builder.note_incomplete(reason, affects_coverage=False)
  if len(selected) == 1:
    br_idx = selected[0]
    branch = _unwrap_closed_jaxpr(branches[br_idx])
    nested = env.child()
    for outer_v, inner_v in zip(eqn.invars[1:], branch.invars):
      nested.bind(outer_v, inner_v, env)
    _walk(branch, ctx.scoped(f"br{br_idx}"), nested)
    _bind_outputs(branch.outvars, eqn.outvars, nested, env)
    return

  base_written = {k: list(v) for k, v in ctx.written.items()}
  base_output_writes = {
      k: list(v) for k, v in ctx.builder.output_writes.items()
  }
  base_remote_written = {k: list(v) for k, v in ctx.remote_written.items()}
  branch_added_written: list[dict[str, list[HyperRectSlice]]] = []
  branch_added_outputs: list[dict[tuple[int, str], list[HyperRectSlice]]] = []
  branch_added_remote: list[dict[tuple[int, str], list[HyperRectSlice]]] = []
  for br_idx in selected:
    ctx.written.clear()
    ctx.written.update({k: list(v) for k, v in base_written.items()})
    ctx.builder.output_writes.clear()
    ctx.builder.output_writes.update(
        {k: list(v) for k, v in base_output_writes.items()}
    )
    ctx.remote_written.clear()
    ctx.remote_written.update(
        {k: list(v) for k, v in base_remote_written.items()}
    )
    branch = _unwrap_closed_jaxpr(branches[br_idx])
    nested = env.child()
    for outer_v, inner_v in zip(eqn.invars[1:], branch.invars):
      nested.bind(outer_v, inner_v, env)
    br_ctx = dataclasses.replace(ctx.scoped(f"br{br_idx}"), speculative=True)
    _walk(branch, br_ctx, nested)
    branch_added_written.append({
        k: v[len(base_written.get(k, ())) :]
        for k, v in ctx.written.items()
        if len(v) > len(base_written.get(k, ()))
    })
    branch_added_outputs.append({
        k: v[len(base_output_writes.get(k, ())) :]
        for k, v in ctx.builder.output_writes.items()
        if len(v) > len(base_output_writes.get(k, ()))
    })
    branch_added_remote.append({
        k: v[len(base_remote_written.get(k, ())) :]
        for k, v in ctx.remote_written.items()
        if len(v) > len(base_remote_written.get(k, ()))
    })

  # Only regions written along every branch are guaranteed to be initialized
  # after the conditional completes.
  ctx.written.clear()
  ctx.written.update({k: list(v) for k, v in base_written.items()})
  if branch_added_written:
    for buf_id in branch_added_written[0]:
      common = _intersect_branch_regions(
          [br.get(buf_id, ()) for br in branch_added_written]
      )
      if common:
        ctx.written.setdefault(buf_id, []).extend(common)

  ctx.builder.output_writes.clear()
  ctx.builder.output_writes.update(
      {k: list(v) for k, v in base_output_writes.items()}
  )
  if branch_added_outputs:
    for key in branch_added_outputs[0]:
      common = _intersect_branch_regions(
          [br.get(key, ()) for br in branch_added_outputs]
      )
      if common:
        ctx.builder.output_writes.setdefault(key, []).extend(common)

  ctx.remote_written.clear()
  ctx.remote_written.update(
      {k: list(v) for k, v in base_remote_written.items()}
  )
  if branch_added_remote:
    for key in branch_added_remote[0]:
      common = _intersect_branch_regions(
          [br.get(key, ()) for br in branch_added_remote]
      )
      if common:
        ctx.remote_written.setdefault(key, []).extend(common)

  # Track complementary 2-branch predicates (such as `@pl.when(p)` followed by
  # `@pl.when(~p)`) so writes covering both truth values of `p` are combined.
  if len(branches) == 2:
    canon = _canonical_cond_predicate(eqn.invars[0], env)
    if canon is not None:
      base_pred, polarity = canon
      # In `lax.cond`, branch 0 is `index == 0` (False) and branch 1 is True.
      br_truths = (not polarity, polarity)
      for br_idx, truth in enumerate(br_truths):
        g_written = ctx.guarded_written.setdefault((base_pred, truth), {})
        for buf_id, regs in branch_added_written[br_idx].items():
          g_written.setdefault(buf_id, []).extend(regs)
        g_out = ctx.guarded_output_writes.setdefault((base_pred, truth), {})
        for key, regs in branch_added_outputs[br_idx].items():
          g_out.setdefault(key, []).extend(regs)
      false_w = ctx.guarded_written.get((base_pred, False), {})
      true_w = ctx.guarded_written.get((base_pred, True), {})
      for buf_id in set(false_w).intersection(true_w):
        common = _intersect_branch_regions([false_w[buf_id], true_w[buf_id]])
        if common:
          ctx.written.setdefault(buf_id, []).extend(common)
      false_out = ctx.guarded_output_writes.get((base_pred, False), {})
      true_out = ctx.guarded_output_writes.get((base_pred, True), {})
      for key in set(false_out).intersection(true_out):
        common = _intersect_branch_regions([false_out[key], true_out[key]])
        if common:
          ctx.builder.output_writes.setdefault(key, []).extend(common)


def _block_aval(bm: Any) -> Any:
  """Returns `bm`'s block aval across JAX versions."""
  aval = getattr(bm, "transformed_block_aval", None)
  return bm.block_aval if aval is None else aval


def _handle_emit_pipeline(
    eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None
) -> None:
  """Symbolically unrolls a `pltpu.emit_pipeline` primitive equation."""
  grid_mapping = eqn.params.get("grid_mapping")
  raw_body = eqn.params.get("body_jaxpr")
  args_tree = eqn.params.get("args_tree")
  refs_tree = eqn.params.get("refs_tree")
  if (
      grid_mapping is None
      or raw_body is None
      or args_tree is None
      or refs_tree is None
  ):
    _handle_unknown(eqn, ctx, env, loc)
    return
  body_jaxpr = _unwrap_closed_jaxpr(raw_body)
  prim_args = args_tree.unflatten(eqn.invars)
  outer_refs_pytree = refs_tree.unflatten(prim_args.refs_flat)
  flat_outer_refs = jax.tree.leaves(
      outer_refs_pytree,
      is_leaf=lambda x: hasattr(x, "ref") and hasattr(x, "transforms"),
  )
  pipe_grid: list[int] = []
  dyn_specs = list(getattr(prim_args, "dynamic_grid_spec", ()) or ())
  dyn_iter = iter(dyn_specs)
  for d in grid_mapping.grid:
    if isinstance(d, (int, np.integer)):
      pipe_grid.append(int(d))
    else:
      dyn_atom = next(dyn_iter, None)
      dyn_val = (
          _eval_atom(dyn_atom, env.scalars) if dyn_atom is not None else None
      )
      if dyn_val is not None and int(dyn_val) > 0:
        pipe_grid.append(int(dyn_val))
      else:
        pipe_grid.append(1)
        _note_truncated_loop(
            ctx,
            body_jaxpr,
            "an emit_pipeline grid axis is only sized at run time, so one"
            " step of it is checked",
        )
  pipe_grid_t = tuple(pipe_grid)
  num_pipe_points = math.prod(pipe_grid_t) if pipe_grid_t else 1
  limit = ctx.builder.max_unrolled_iterations
  if num_pipe_points > limit:
    reason = (
        f"only the first {limit} of {num_pipe_points} emit_pipeline steps are"
        " model checked"
    )
    if ctx.builder.has_sync(body_jaxpr):
      ctx.builder.note_truncated_sync(reason, affects_coverage=False)
    else:
      ctx.builder.note_incomplete(reason, affects_coverage=False)
  if num_pipe_points > _MAX_COVERAGE_GRID_POINTS:
    _note_truncated_loop(
        ctx,
        body_jaxpr,
        f"only the first {_MAX_COVERAGE_GRID_POINTS} of {num_pipe_points}"
        " emit_pipeline steps are checked for coverage and bounds",
    )

  const_env: dict[Any, int | bool] = dict(env.scalars)
  all_imap_consts = list(getattr(prim_args, "all_index_map_consts", ()) or ())
  imap_offset = 0
  bm_const_envs: list[dict[Any, int | bool]] = []
  for bm in grid_mapping.block_mappings:
    n_c = len(bm.index_map_jaxpr.jaxpr.constvars)
    bm_c_env = dict(const_env)
    for c_var, outer_c in zip(
        bm.index_map_jaxpr.jaxpr.constvars,
        all_imap_consts[imap_offset : imap_offset + n_c],
    ):
      c_val = _eval_atom(outer_c, env.scalars)
      if c_val is not None:
        bm_c_env[c_var] = c_val
    imap_offset += n_c
    bm_const_envs.append(bm_c_env)

  num_g = len(pipe_grid_t)
  header_len = 2 * num_g + 1
  block_invars = body_jaxpr.invars[header_len:]
  num_inputs = grid_mapping.num_inputs
  scope_base = ctx.scoped(f"ep{_next_scope_index(ctx)}")
  pipe_points = list(
      itertools.islice(
          itertools.product(*(range(s) for s in pipe_grid_t)),
          _MAX_COVERAGE_GRID_POINTS,
      )
  )
  for step_idx, pipe_coords in enumerate(pipe_points):
    nested = env.child()
    for outer_c, const_v in zip(prim_args.body_consts, body_jaxpr.constvars):
      nested.bind(outer_c, const_v, env)
    for ax_i, coord in enumerate(pipe_coords):
      if ax_i < num_g:
        nested.scalars[body_jaxpr.invars[ax_i]] = coord
    if num_g < len(body_jaxpr.invars):
      nested.scalars[body_jaxpr.invars[num_g]] = step_idx
    for ax_i, g_size in enumerate(pipe_grid_t):
      if num_g + 1 + ax_i < len(body_jaxpr.invars):
        nested.scalars[body_jaxpr.invars[num_g + 1 + ax_i]] = g_size

    step_unaliased = dict(ctx.unaliased_output_buffers)
    step_tiles: list[tuple[int, _RefView, str, _BlockSlice | None, bool]] = []
    for b_idx, (b_var, bm, bm_c_env) in enumerate(
        zip(block_invars, grid_mapping.block_mappings, bm_const_envs)
    ):
      outer_ref = (
          flat_outer_refs[b_idx] if b_idx < len(flat_outer_refs) else None
      )
      outer_view = (
          _resolve_view(ctx, env, outer_ref, (), loc)
          if outer_ref is not None
          else None
      )
      if outer_view is None:
        continue
      block = _block_slice(
          bm, pipe_coords, axis_env=ctx.axis_env, const_env=bm_c_env
      )
      is_output = b_idx >= num_inputs
      desc = (
          f"emit_pipeline {'output' if is_output else 'input'}"
          f" {b_idx - num_inputs if is_output else b_idx}"
      )
      if block is not None and block.violation is not None:
        ctx.builder.record_issue(
            _Category.OUT_OF_BOUNDS_ACCESS,
            f"the BlockSpec index map of {desc} at pipeline step"
            f" {pipe_coords}: {block.violation}",
            loc,
            dedup_key=("ep_block_oob", id(eqn), b_idx),
        )
      if block is None:
        ctx.builder.note_incomplete(
            f"the block index map of {desc} depends on runtime data, so its"
            " pipelined accesses are not checked",
            affects_coverage=is_output,
        )
      block_aval = _block_aval(bm)
      tile_shape = tuple(int(s) for s in block_aval.shape)
      dtype = getattr(block_aval, "dtype", outer_view.dtype)
      tile_id = f"{scope_base.scope_path}.s{step_idx}.t{b_idx}"
      nested.refs[b_var] = _full_view(tile_id, "vmem", tile_shape, dtype)
      if is_output and outer_view.buffer_id in ctx.unaliased_output_buffers:
        _, out_op_idx = ctx.unaliased_output_buffers[outer_view.buffer_id]
        step_unaliased[tile_id] = (desc, out_op_idx)
        tile_key = (ctx.actor, out_op_idx)
        if block is not None:
          if ctx.builder.active_output_tile_block.get(tile_key) != block.region:
            ctx.builder.active_output_tile_block[tile_key] = block.region
            ctx.builder.active_output_tile_writes[tile_key] = []
      step_tiles.append((b_idx, outer_view, tile_id, block, is_output))

    step_ctx = dataclasses.replace(
        scope_base.scoped(f"st{step_idx}"),
        grid_coords=pipe_coords,
        grid_sizes=pipe_grid_t,
        record_events=ctx.record_events and (step_idx < limit),
        read_in_step=set(),
        unaliased_output_buffers=step_unaliased,
    )
    body_actor_pos = len(ctx.builder.actor_events.get(ctx.actor, ()))
    body_mem_pos = len(ctx.builder.memory_events)
    _walk(body_jaxpr, step_ctx, nested)
    _record_step_written_buffers(step_ctx)
    for _, outer_view, tile_id, block, is_output in step_tiles:
      if block is None:
        continue
      if tile_id in step_ctx.touched_refs:
        ctx.touched_refs.add(outer_view.buffer_id)
      block_mapped_dims = list(outer_view.dims)
      for ax_pos, d in zip(outer_view.active_axes, block.region.dims):
        block_mapped_dims[ax_pos] = SliceDim(
            outer_view.dims[ax_pos].start + d.start,
            d.size,
            exact=outer_view.dims[ax_pos].exact and d.exact,
        )
      block_mapped = HyperRectSlice(
          tuple(block_mapped_dims),
          opaque=outer_view.opaque or block.region.opaque,
      )
      if not is_output:
        if step_ctx.record_events and tile_id in step_ctx.touched_refs:
          eid = ctx.builder.insert_event_id(ctx.actor, body_actor_pos)
          body_actor_pos += 1
          ctx.builder.memory_events.insert(
              body_mem_pos,
              MemoryAccessEvent(
                  event_id=eid,
                  actor=ctx.actor,
                  grid_point=step_ctx.grid_coords,
                  buffer=_buffer_key(step_ctx, outer_view, step_ctx.device_id),
                  region=block_mapped,
                  is_write=False,
                  source_location=loc,
                  is_implicit_pipeline=True,
                  speculative=step_ctx.speculative,
              ),
          )
          body_mem_pos += 1
        continue
      out_info = step_unaliased.get(tile_id)
      for region in step_ctx.written.get(tile_id, ()):
        if out_info is not None:
          ctx.builder.active_output_tile_writes.setdefault(
              (ctx.actor, out_info[1]), []
          ).append(region)
        arr_reg = _tile_region_to_array(region, block)
        if arr_reg is None:
          ctx.builder.note_incomplete(
              "an emit_pipeline output block layout is not understood",
              affects_coverage=True,
          )
          continue
        mapped_dims = list(outer_view.dims)
        for ax_pos, d in zip(outer_view.active_axes, arr_reg.dims):
          mapped_dims[ax_pos] = SliceDim(
              outer_view.dims[ax_pos].start + d.start,
              d.size,
              exact=outer_view.dims[ax_pos].exact and d.exact,
          )
        mapped = HyperRectSlice(
            tuple(mapped_dims),
            opaque=outer_view.opaque or arr_reg.opaque,
        )
        ctx.written.setdefault(outer_view.buffer_id, []).append(mapped)
        ctx.builder.record_output_write(
            ctx.device_id, outer_view.buffer_id, mapped
        )
      if step_ctx.record_events and tile_id in step_ctx.written:
        ctx.builder.memory_events.append(
            MemoryAccessEvent(
                event_id=ctx.builder.new_event_id(ctx.actor),
                actor=ctx.actor,
                grid_point=step_ctx.grid_coords,
                buffer=_buffer_key(step_ctx, outer_view, step_ctx.device_id),
                region=block_mapped,
                is_write=True,
                source_location=loc,
                is_implicit_pipeline=True,
                speculative=step_ctx.speculative,
            )
        )


def _sub_jaxprs(eqn: Any) -> Iterator[Any]:
  for value in eqn.params.values():
    candidates = value if isinstance(value, (tuple, list)) else (value,)
    for candidate in candidates:
      inner = _unwrap_closed_jaxpr(candidate)
      if hasattr(inner, "eqns") and hasattr(inner, "invars"):
        yield inner


def _handle_call(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del loc
  for inner in _sub_jaxprs(eqn):
    if len(inner.invars) != len(eqn.invars):
      continue
    nested = env.child()
    for outer_v, inner_v in zip(eqn.invars, inner.invars):
      nested.bind(outer_v, inner_v, env)
    _walk(inner, ctx, nested)
    _bind_outputs(inner.outvars, eqn.outvars, nested, env)
    return
  _handle_unknown(eqn, ctx, env, None)


def _handle_unknown(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  del env, loc
  subs = list(_sub_jaxprs(eqn))
  if not subs:
    return
  reason = (
      f"`{eqn.primitive.name}` is not interpreted, so accesses inside it are"
      " not checked"
  )
  if any(ctx.builder.has_sync(sub) for sub in subs):
    # Its semaphore operations would be missing from the model.
    ctx.builder.note_unresolved_sync(reason, affects_coverage=True)
  else:
    ctx.builder.note_incomplete(reason, affects_coverage=True)


_HANDLERS: dict[str, Callable[[Any, _WalkContext, _Env, str | None], None]] = {
    "addupdate": _handle_swap,
    "cond": _handle_cond,
    "dma_start": _handle_dma_start,
    "dma_wait": _handle_dma_wait,
    "emit_pipeline": _handle_emit_pipeline,
    "get": _handle_get,
    "get_barrier_semaphore": _handle_barrier,
    "masked_load": _handle_masked,
    "masked_swap": _handle_masked,
    "run_scoped": _handle_run_scoped,
    "scan": _handle_scan,
    "semaphore_signal": _handle_sem_signal,
    "semaphore_wait": _handle_sem_wait,
    "swap": _handle_swap,
    "while": _handle_while,
}


def _walk(jaxpr: Any, ctx: _WalkContext, env: _Env) -> None:
  """Symbolically executes `jaxpr` for one device and grid point."""
  for eqn in jaxpr.eqns:
    ctx.builder.step_count += 1
    if ctx.builder.step_count > _MAX_TOTAL_EQUATION_STEPS:
      reason = f"stopped walking after {_MAX_TOTAL_EQUATION_STEPS} equations"
      if ctx.builder.has_sync(jaxpr):
        ctx.builder.note_truncated_sync(reason, affects_coverage=True)
      else:
        ctx.builder.note_incomplete(reason, affects_coverage=True)
      return
    if _try_eval_scalar_eqn(
        eqn,
        env.scalars,
        grid_coords=ctx.grid_coords,
        grid_sizes=ctx.grid_sizes,
        axis_env=ctx.axis_env,
        sym_env=env.sym_exprs,
    ):
      continue
    name = eqn.primitive.name
    handler = _HANDLERS.get(name)
    if handler is None:
      handler = _handle_call if name in _CALL_PRIMITIVES else _handle_unknown
    handler(eqn, ctx, env, ctx.builder.source_location(eqn))


def _check_static_hardware_rules(
    jaxpr: Any, collective_id: int | None, builder: _KernelModelBuilder
) -> None:
  """Recursively checks hardware usage rules that need no unrolling."""
  for eqn in jaxpr.eqns:
    loc = _summarize_source_info(eqn)
    name = eqn.primitive.name
    if name == "dot_general":
      out_dtype = getattr(eqn.outvars[0].aval, "dtype", None)
      if (
          out_dtype is not None
          and jnp.issubdtype(out_dtype, jnp.floating)
          and jnp.dtype(out_dtype).itemsize < 4
      ):
        dt_name = jnp.dtype(out_dtype).name
        builder.record_issue(
            _Category.LOW_PRECISION_ACCUMULATION,
            f"dot_general produces a {dt_name} result; accumulating"
            " low-precision matmul results (for example across grid steps)"
            " loses precision, so pass preferred_element_type=jnp.float32",
            loc,
            severity=_Severity.WARNING,
        )
    elif name == "get_barrier_semaphore":
      if collective_id is None and eqn.params.get("tag") is None:
        builder.record_issue(
            _Category.HARDWARE_CONSTRAINT_VIOLATION,
            "get_barrier_semaphore needs `collective_id` in"
            " pltpu.CompilerParams",
            loc,
        )
    elif name == "run_scoped":
      for var in _unwrap_closed_jaxpr(eqn.params["jaxpr"]).invars:
        ms = getattr(var.aval, "memory_space", None)
        if ms is not None and _normalize_memory_space(ms) == "hbm":
          builder.record_issue(
              _Category.HARDWARE_CONSTRAINT_VIOLATION,
              "pl.run_scoped cannot allocate HBM buffers on TPU",
              loc,
          )
    for sub in _sub_jaxprs(eqn):
      _check_static_hardware_rules(sub, collective_id, builder)


@dataclasses.dataclass(frozen=True)
class _BlockSlice:
  """Region of the full array that a block covers at one grid point."""

  region: HyperRectSlice
  active_axes: tuple[int, ...]
  violation: str | None = None


def _uses_any(jaxpr: Any, variables: Sequence[Any]) -> bool:
  targets = set(variables)
  if any(v in targets for v in jaxpr.outvars if _is_jax_var(v)):
    return True
  return any(
      v in targets for eqn in jaxpr.eqns for v in eqn.invars if _is_jax_var(v)
  )


def _eval_index_map(
    bm: Any,
    grid_coords: tuple[int, ...],
    *,
    axis_env: Mapping[Any, int] | None = None,
    const_env: Mapping[Any, int | bool] | None = None,
) -> list[Any] | None:
  """Evaluates a block index map; None if it depends on runtime data.

  The map is evaluated symbolically rather than executed, which would
  initialize JAX backends.

  Args:
    bm: The block mapping.
    grid_coords: The grid point.
    axis_env: Optional mesh/core axis coordinates for `lax.axis_index`.
    const_env: Optional pre-bound constants for `index_map_jaxpr.constvars`.

  Returns:
    The block indices, or None if they are not statically known.
  """
  closed = bm.index_map_jaxpr
  jaxpr = closed.jaxpr
  grid_vars = jaxpr.invars[: len(grid_coords)]
  extra_vars = jaxpr.invars[len(grid_coords) :]
  if extra_vars and _uses_any(jaxpr, extra_vars):
    return None  # Depends on scalar-prefetch operands.
  scalars: dict[Any, int | bool] = dict(zip(grid_vars, grid_coords))
  if const_env:
    scalars.update(const_env)
  for var, const in zip(jaxpr.constvars, closed.consts):
    value = _eval_atom(const, scalars)
    if value is not None:
      scalars[var] = value
  eff_axis_env = dict(axis_env or {})
  for eqn in jaxpr.eqns:
    _try_eval_scalar_eqn(
        eqn,
        scalars,
        grid_coords=grid_coords,
        grid_sizes=(),
        axis_env=eff_axis_env,
    )
  outs = [_eval_atom(v, scalars) for v in jaxpr.outvars]
  if any(o is None for o in outs):
    return None
  return bm.index_map_out_tree.unflatten(outs)


def _block_slice(
    bm: Any,
    grid_coords: tuple[int, ...],
    *,
    axis_env: Mapping[Any, int] | None = None,
    const_env: Mapping[Any, int | bool] | None = None,
) -> _BlockSlice | None:
  """Returns the array region of a block at a grid point, if computable."""
  indices = _eval_index_map(
      bm, grid_coords, axis_env=axis_env, const_env=const_env
  )
  if indices is None:
    return None
  array_shape = tuple(int(s) for s in bm.array_aval.shape)
  dims: list[SliceDim] = []
  active: list[int] = []
  violation: str | None = None
  for axis, (value, block_dim) in enumerate(zip(indices, bm.block_shape)):
    size = array_shape[axis]
    if isinstance(block_dim, pl.BoundedSlice) and isinstance(value, pl.Slice):
      if not isinstance(
          value.start, (int, np.integer, np.ndarray)
      ) or not isinstance(value.size, (int, np.integer, np.ndarray)):
        return None
      start = int(np.asarray(value.start).item())
      length = int(np.asarray(value.size).item())
      max_block = int(block_dim.block_size)
      if start < 0 or length < 0 or start + length > size:
        violation = violation or (
            f"slice {start}:{start + length} is outside axis {axis} of size"
            f" {size}"
        )
      elif length > max_block:
        violation = violation or (
            f"BoundedSlice length {length} exceeds maximum block_size"
            f" {max_block} along axis {axis}"
        )
      dims.append(SliceDim(start, max(length, 0)))
      active.append(axis)
      continue
    if not isinstance(value, (int, np.integer, np.ndarray)) or (
        isinstance(value, np.ndarray) and value.ndim != 0
    ):
      return None
    index = int(np.asarray(value).item())
    if isinstance(block_dim, pl.Squeezed) or block_dim is None:
      if not 0 <= index < size:
        violation = violation or (
            f"index {index} is outside axis {axis} of size {size}"
        )
      dims.append(SliceDim(index, 1))
      continue
    if isinstance(block_dim, pl.Element):
      block_size = int(block_dim.block_size)
      pad_lo, pad_hi = getattr(block_dim, "padding", None) or (0, 0)
      unpadded = max(block_size - int(pad_lo) - int(pad_hi), 0)
      if index < 0 or index + unpadded > size:
        violation = violation or (
            f"element slice {index}:{index + unpadded} is outside axis"
            f" {axis} of size {size}"
        )
      dims.append(SliceDim(index, unpadded))
      active.append(axis)
      continue
    block = int(getattr(block_dim, "block_size", block_dim))
    if size == 0 or index < 0 or index * block >= size:
      violation = violation or (
          f"block index {index} along axis {axis} starts at element"
          f" {index * block}, past the array size {size}"
      )
    dims.append(SliceDim(index * block, block))
    active.append(axis)
  return _BlockSlice(HyperRectSlice(tuple(dims)), tuple(active), violation)


def _tile_region_to_array(
    tile_region: HyperRectSlice, block: _BlockSlice
) -> HyperRectSlice | None:
  """Maps a region of a VMEM tile to the array region it is written back to."""
  if len(tile_region.dims) != len(block.active_axes):
    return None
  dims = list(block.region.dims)
  for tile_dim, axis in zip(tile_region.dims, block.active_axes):
    dims[axis] = SliceDim(
        dims[axis].start + tile_dim.start,
        tile_dim.size,
        exact=dims[axis].exact and tile_dim.exact,
    )
  return HyperRectSlice(
      tuple(dims), opaque=tile_region.opaque or block.region.opaque
  )


def _uncovered_region(
    shape: Sequence[int], regions: Sequence[HyperRectSlice]
) -> HyperRectSlice | None:
  """Returns the bounding box of the part of `shape` no region covers.

  Returns None when everything is covered or the check would be too costly.
  """
  rank = len(shape)
  usable = [r for r in regions if len(r.dims) == rank]
  if rank == 0:
    return None if usable else HyperRectSlice(())
  if any(s == 0 for s in shape):
    return None
  if not usable:
    return HyperRectSlice(tuple(SliceDim(0, int(s)) for s in shape))
  cuts: list[np.ndarray] = []
  for axis, size in enumerate(shape):
    points = {0, int(size)}
    for region in usable:
      points.add(min(max(region.dims[axis].start, 0), int(size)))
      points.add(min(max(region.dims[axis].stop, 0), int(size)))
    cuts.append(np.asarray(sorted(points), dtype=np.int64))
  cell_shape = tuple(len(c) - 1 for c in cuts)
  if any(n <= 0 for n in cell_shape):
    return None
  if math.prod(cell_shape) > _MAX_COVERAGE_CELLS:
    return None
  covered = np.zeros(cell_shape, dtype=bool)
  for region in usable:
    slices = []
    valid = True
    for axis, c in enumerate(cuts):
      lo = min(max(region.dims[axis].start, 0), int(shape[axis]))
      hi = min(max(region.dims[axis].stop, 0), int(shape[axis]))
      if hi <= lo:
        valid = False
        break
      i0 = int(np.searchsorted(c, lo, side="left"))
      i1 = int(np.searchsorted(c, hi, side="right")) - 1
      if i1 <= i0:
        valid = False
        break
      slices.append(slice(i0, i1))
    if valid:
      covered[tuple(slices)] = True
  if covered.all():
    return None
  uncovered = ~covered
  dims: list[SliceDim] = []
  for axis, c in enumerate(cuts):
    other_axes = tuple(i for i in range(rank) if i != axis)
    axis_uncov = uncovered.any(axis=other_axes) if other_axes else uncovered
    idx = np.flatnonzero(axis_uncov)
    lo = int(c[int(idx[0])])
    hi = int(c[int(idx[-1]) + 1])
    dims.append(SliceDim(lo, hi - lo))
  return HyperRectSlice(tuple(dims))


@dataclasses.dataclass(frozen=True)
class _Operand:
  """A pipelined or directly accessed `pallas_call` or `core_map` operand."""

  index: int  # Position among all block operands.
  io_index: int  # Position among the inputs or among the outputs.
  var: Any
  block_mapping: Any
  buffer_id: str
  is_output: bool
  memory_space: str
  array_shape: tuple[int, ...] = ()

  def describe(self) -> str:
    return f"{'output' if self.is_output else 'input'} {self.io_index}"

  def full_shape(self) -> tuple[int, ...]:
    if self.array_shape:
      return self.array_shape
    if self.block_mapping is not None:
      return tuple(int(s) for s in self.block_mapping.array_aval.shape)
    return ()


def _operands(eqn: Any, kernel_jaxpr: Any, grid_mapping: Any) -> list[_Operand]:
  io_aliases = dict(eqn.params.get("input_output_aliases", ()))
  num_prefetch = int(getattr(grid_mapping, "num_index_operands", 0))
  out_to_in = {out_i: in_i for in_i, out_i in io_aliases.items()}
  block_vars = kernel_jaxpr.invars[grid_mapping.slice_block_ops]
  num_inputs = grid_mapping.num_inputs
  operands = []
  for b_idx, (b_var, bm) in enumerate(
      zip(block_vars, grid_mapping.block_mappings)
  ):
    is_output = b_idx >= num_inputs
    io_index = b_idx - num_inputs if is_output else b_idx
    if not is_output:
      buffer_id = f"arg_in_{b_idx}"
    elif io_index in out_to_in:
      raw_in = out_to_in[io_index]
      block_in = raw_in - num_prefetch
      buffer_id = (
          f"arg_in_{block_in}" if block_in >= 0 else f"prefetch_{raw_in}"
      )
    else:
      buffer_id = f"arg_out_{io_index}"
    memory_space = _normalize_memory_space(
        getattr(_block_aval(bm), "memory_space", None)
    )
    operands.append(
        _Operand(
            b_idx,
            io_index,
            b_var,
            bm,
            buffer_id,
            is_output,
            memory_space,
            tuple(int(s) for s in bm.array_aval.shape),
        )
    )
  return operands


def _bind_operands(
    ctx: _WalkContext,
    env: _Env,
    operands: Sequence[_Operand],
    pipelined: bool,
) -> list[tuple[_Operand, str, _BlockSlice | None]]:
  """Binds operand refs at a grid point; returns pipelined tiles."""
  tiles = []
  for op in operands:
    bm = op.block_mapping
    block_aval = _block_aval(bm)
    dtype = getattr(block_aval, "dtype", np.float32)
    array_shape = op.full_shape()
    block = (
        _block_slice(bm, ctx.grid_coords, axis_env=ctx.axis_env)
        if pipelined
        else None
    )
    if block is not None and block.violation is not None:
      ctx.builder.record_issue(
          _Category.OUT_OF_BOUNDS_ACCESS,
          f"the BlockSpec index map of {op.describe()} at grid point"
          f" {ctx.grid_coords}: {block.violation}",
          dedup_key=("block_oob", op.index),
      )
    if op.memory_space == "hbm" or not pipelined:
      env.refs[op.var] = _full_view(
          op.buffer_id, op.memory_space, array_shape, dtype
      )
      if op.is_output and op.buffer_id.startswith("arg_out_"):
        ctx.unaliased_output_buffers[op.buffer_id] = (op.describe(), op.index)
      continue
    tile_shape = tuple(int(s) for s in block_aval.shape)
    tile_id = f"tile_{op.index}_{ctx.grid_tag()}"
    env.refs[op.var] = _full_view(tile_id, op.memory_space, tile_shape, dtype)
    if op.is_output and op.buffer_id.startswith("arg_out_"):
      ctx.unaliased_output_buffers[tile_id] = (op.describe(), op.index)
      tile_key = (ctx.actor, op.index)
      if block is not None:
        if ctx.builder.active_output_tile_block.get(tile_key) != block.region:
          ctx.builder.active_output_tile_block[tile_key] = block.region
          ctx.builder.active_output_tile_writes[tile_key] = []
      else:
        ctx.builder.active_output_tile_block.pop(tile_key, None)
        ctx.builder.active_output_tile_writes.pop(tile_key, None)
    if block is None:
      ctx.builder.note_incomplete(
          f"the block index map of {op.describe()} depends on runtime data,"
          " so its pipelined accesses are not checked",
          affects_coverage=op.is_output,
      )
    tiles.append((op, tile_id, block))
  return tiles


def _emit_pipeline_events(
    ctx: _WalkContext,
    tiles: Sequence[tuple[_Operand, str, _BlockSlice | None]],
    body_actor_pos: int,
    body_mem_pos: int,
) -> None:
  """Adds the implicit HBM loads/stores and output writes of tiles."""
  builder = ctx.builder
  _record_step_written_buffers(ctx)
  for op, tile_id, block in tiles:
    if block is None:
      continue
    if op.is_output:
      tile_writes = list(ctx.written.get(tile_id, ()))
      if ctx.grid_sizes:
        builder.output_block_visits.setdefault(
            (ctx.actor, op.index), []
        ).append((ctx.grid_coords, block.region))
        builder.parallel_output_blocks.setdefault(
            (ctx.device_id, op.index, block.region), []
        ).append((ctx.actor, ctx.grid_coords, tile_writes))
      if op.buffer_id.startswith("arg_out_"):
        builder.active_output_tile_writes.setdefault(
            (ctx.actor, op.index), []
        ).extend(tile_writes)
      elif op.buffer_id.startswith("arg_in_"):
        builder.aliased_pipelined_blocks.setdefault(
            (ctx.device_id, op.index), []
        ).append(block.region)
      for region in tile_writes:
        array_region = _tile_region_to_array(region, block)
        if array_region is None:
          builder.note_incomplete(
              f"the block layout of {op.describe()} is not understood",
              affects_coverage=True,
          )
        else:
          builder.record_output_write(ctx.device_id, op.buffer_id, array_region)
          if op.buffer_id.startswith("arg_in_"):
            builder.record_output_write(
                ctx.device_id, f"pipelined_out_{op.io_index}", array_region
            )
      for (target_dev, w_tile_id), regions in ctx.remote_written.items():
        if w_tile_id != tile_id:
          continue
        for region in regions:
          array_region = _tile_region_to_array(region, block)
          if array_region is None:
            builder.note_incomplete(
                f"the block layout of {op.describe()} is not understood",
                affects_coverage=True,
            )
          else:
            builder.record_output_write(target_dev, op.buffer_id, array_region)
            if op.buffer_id.startswith("arg_in_"):
              builder.record_output_write(
                  target_dev, f"pipelined_out_{op.io_index}", array_region
              )
    if not ctx.record_events:
      continue
    buffer = BufferKey(ctx.device_id, ctx.actor.core_id, "hbm", op.buffer_id)
    if not op.is_output and tile_id in ctx.touched_refs:
      eid = builder.insert_event_id(ctx.actor, body_actor_pos)
      body_actor_pos += 1
      builder.memory_events.insert(
          body_mem_pos,
          MemoryAccessEvent(
              eid,
              ctx.actor,
              ctx.grid_coords,
              buffer,
              block.region,
              False,
              is_implicit_pipeline=True,
          ),
      )
      body_mem_pos += 1
    elif op.is_output and tile_id in ctx.written:
      builder.memory_events.append(
          MemoryAccessEvent(
              builder.new_event_id(ctx.actor),
              ctx.actor,
              ctx.grid_coords,
              buffer,
              block.region,
              True,
              is_implicit_pipeline=True,
          ),
      )


def _check_output_coverage(
    builder: _KernelModelBuilder,
    operands: Sequence[_Operand],
    mesh_ctx: _MeshContext,
) -> None:
  """Records non-consecutive output block revisits and unwritten output regions."""
  op_by_index = {op.index: op for op in operands}
  for (_, op_idx), visits in builder.output_block_visits.items():
    op = op_by_index.get(op_idx)
    if op is None:
      continue
    seen: set[HyperRectSlice] = set()
    last: HyperRectSlice | None = None
    for g_coords, region in sorted(visits, key=lambda item: item[0]):
      if last is not None and region != last and region in seen:
        builder.record_issue(
            _Category.HARDWARE_CONSTRAINT_VIOLATION,
            f"the BlockSpec of {op.describe()} at grid point {g_coords}"
            f" revisits output block {region} after visiting {last}; the TPU"
            " output pipeline requires iterations targeting the same output"
            " block to be consecutive",
            dedup_key=("non_consec_revisit", op_idx),
        )
        break
      last = region
      seen.add(region)

  if not builder.coverage_known:
    return

  for (
      device_id,
      op_idx,
      region,
  ), entries in builder.parallel_output_blocks.items():
    op = op_by_index.get(op_idx)
    if op is None:
      continue
    tile_shape = (
        tuple(int(s) for s in _block_aval(op.block_mapping).shape)
        if op.block_mapping is not None
        else ()
    )
    by_actor: dict[ActorKey, tuple[tuple[int, ...], list[HyperRectSlice]]] = {}
    for act, g, regs in entries:
      prev = by_actor.get(act)
      if prev is None:
        by_actor[act] = (g, list(regs))
      else:
        prev[1].extend(regs)
    if len(by_actor) < 2:
      continue
    writers = [g for _, (g, regs) in by_actor.items() if regs]
    incomplete_actors: list[tuple[ActorKey, tuple[int, ...]]] = []
    for act, (g, regs) in by_actor.items():
      if not any(
          other_act != act and o_regs
          for other_act, (_, o_regs) in by_actor.items()
      ):
        continue
      if not regs or (
          tile_shape and _uncovered_region(tile_shape, regs) is not None
      ):
        incomplete_actors.append((act, g))
    if writers and incomplete_actors:
      where = f" on device {device_id}" if mesh_ctx.num_devices > 1 else ""
      builder.record_issue(
          _Category.UNWRITTEN_OUTPUT,
          f"{op.describe()} block {region} is mapped by multiple parallel grid"
          f" points ({writers[0]} and {incomplete_actors[0][1]}){where}, and"
          f" grid point {incomplete_actors[0][1]} leaves its VMEM tile"
          " unwritten (or partially unwritten) so its output pipeline"
          " writeback clobbers the block with uninitialized memory",
      )

  for op in operands:
    if not op.buffer_id.startswith("arg_out_"):
      continue
    shape = op.full_shape()
    for device_id, _ in mesh_ctx.iter_devices():
      regions = builder.output_writes.get((device_id, op.buffer_id), [])
      missing = _uncovered_region(shape, regions)
      if missing is None:
        continue
      where = f" on device {device_id}" if mesh_ctx.num_devices > 1 else ""
      builder.record_issue(
          _Category.UNWRITTEN_OUTPUT,
          f"{op.describe()} (shape {shape}) region {missing} is never"
          f" written{where}, so it returns uninitialized memory",
      )

  for (
      device_id,
      op_idx,
  ), visited_blocks in builder.aliased_pipelined_blocks.items():
    op = op_by_index.get(op_idx)
    if op is None:
      continue
    pipe_key = (device_id, f"pipelined_out_{op.io_index}")
    writes = builder.output_writes.get(pipe_key, [])
    seen_blocks: set[HyperRectSlice] = set()
    for block_reg in visited_blocks:
      if block_reg in seen_blocks:
        continue
      seen_blocks.add(block_reg)
      block_shape = tuple(
          max(min(d.stop, full_s) - max(d.start, 0), 0)
          for d, full_s in zip(block_reg.dims, op.full_shape())
      )
      if any(s == 0 for s in block_shape):
        continue
      rel_writes: list[HyperRectSlice] = []
      for w in writes:
        if len(w.dims) != len(block_reg.dims):
          continue
        rel_dims = tuple(
            SliceDim(
                wd.start - max(bd.start, 0),
                wd.size,
                exact=wd.exact and bd.exact,
            )
            for wd, bd in zip(w.dims, block_reg.dims)
        )
        rel_writes.append(HyperRectSlice(rel_dims, opaque=w.opaque))
      missing_rel = _uncovered_region(block_shape, rel_writes)
      if missing_rel is not None:
        missing_abs = HyperRectSlice(
            tuple(
                SliceDim(max(bd.start, 0) + md.start, md.size)
                for bd, md in zip(block_reg.dims, missing_rel.dims)
            )
        )
        where = f" on device {device_id}" if mesh_ctx.num_devices > 1 else ""
        builder.record_issue(
            _Category.UNWRITTEN_OUTPUT,
            f"{op.describe()} (aliased to input) pipelined VMEM block"
            f" {block_reg} leaves region {missing_abs} unwritten{where};"
            " output pipeline writeback overwrites the entire block from VMEM",
            dedup_key=("aliased_pipe_unwritten", device_id, op_idx),
        )
        break


def _grid_and_semantics(
    grid_mapping: Any, mosaic_params: Any, builder_note: Callable[[str], None]
) -> tuple[tuple[int, ...], tuple[int | None, ...], tuple[str, ...]]:
  grid: list[int] = []
  grid_sizes: list[int | None] = []
  for d in grid_mapping.grid:
    if isinstance(d, (int, np.integer)):
      grid.append(int(d))
      grid_sizes.append(int(d))
    else:
      grid.append(1)
      grid_sizes.append(None)
      builder_note(
          "a grid axis is only sized at run time, so one step of it is checked"
      )
  raw_sems = getattr(mosaic_params, "dimension_semantics", None) or ()
  semantics = []
  for axis in range(len(grid)):
    sem = raw_sems[axis] if axis < len(raw_sems) else "arbitrary"
    sem_val = getattr(sem, "value", str(sem))
    is_parallel = sem_val in ("parallel", "core_parallel", "subcore_parallel")
    semantics.append("parallel" if is_parallel else "arbitrary")
  return tuple(grid), tuple(grid_sizes), tuple(semantics)


def _select_recorded_grid_points(
    points: Sequence[tuple[int, ...]],
    semantics: Sequence[str],
    max_recorded: int,
) -> set[tuple[int, ...]]:
  """Selects up to `max_recorded` grid points balanced across `parallel` axes."""
  if len(points) <= max_recorded:
    return set(points)
  by_parallel: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
  for pt in points:
    par_key = tuple(
        pt[ax] for ax, sem in enumerate(semantics) if sem == "parallel"
    )
    by_parallel.setdefault(par_key, []).append(pt)
  if len(by_parallel) <= 1:
    return set(points[:max_recorded])
  selected: set[tuple[int, ...]] = set()
  groups = list(by_parallel.values())
  step = 0
  while len(selected) < max_recorded:
    added = False
    for group in groups:
      if step < len(group):
        selected.add(group[step])
        added = True
        if len(selected) >= max_recorded:
          break
    if not added:
      break
    step += 1
  return selected


def _extract_pallas_call_model(
    eqn: Any,
    mesh_ctx: _MeshContext,
    *,
    max_unrolled_iterations: int,
    tpu_device_kind: str = fx_verifier.DEFAULT_TPU_DEVICE_KIND,
) -> KernelVerificationModel:
  """Extracts a `KernelVerificationModel` from one `pallas_call` equation."""
  kernel_jaxpr = _unwrap_closed_jaxpr(eqn.params["jaxpr"])
  grid_mapping: Any = eqn.params["grid_mapping"]
  mosaic_params = _extract_mosaic_compiler_params(
      eqn.params.get("compiler_params")
  )
  collective_id = getattr(mosaic_params, "collective_id", None)
  notes: list[str] = []
  grid, grid_sizes, semantics = _grid_and_semantics(
      grid_mapping, mosaic_params, notes.append
  )
  grid_names = tuple(getattr(grid_mapping, "grid_names", None) or ())
  builder = _KernelModelBuilder(
      kernel_name=_kernel_name(eqn.params),
      num_devices=mesh_ctx.num_devices,
      grid=grid,
      dimension_semantics=semantics,
      collective_id=collective_id,
      max_unrolled_iterations=max_unrolled_iterations,
      tpu_device_kind=tpu_device_kind,
  )
  kernel_has_sync = builder.has_sync(kernel_jaxpr)
  for note in notes:
    if kernel_has_sync:
      builder.note_truncated_sync(note, affects_coverage=True)
    else:
      builder.note_incomplete(note, affects_coverage=True)
  _check_static_hardware_rules(kernel_jaxpr, collective_id, builder)
  for idx, var in enumerate(
      kernel_jaxpr.invars[grid_mapping.slice_scratch_ops]
  ):
    ms = getattr(var.aval, "memory_space", None)
    if ms is not None and _normalize_memory_space(ms) == "hbm":
      builder.record_issue(
          _Category.HARDWARE_CONSTRAINT_VIOLATION,
          f"scratch_shapes[{idx}] cannot allocate HBM buffers on TPU",
      )

  operands = _operands(eqn, kernel_jaxpr, grid_mapping)
  num_points = math.prod(grid) if grid else 1
  if num_points > max_unrolled_iterations:
    reason = (
        f"only the first {max_unrolled_iterations} of {num_points} grid"
        " points are model checked for races and deadlocks"
    )
    if kernel_has_sync:
      builder.note_truncated_sync(reason, affects_coverage=False)
    else:
      builder.note_incomplete(reason, affects_coverage=False)
  if num_points > _MAX_COVERAGE_GRID_POINTS:
    cov_reason = (
        f"only the first {_MAX_COVERAGE_GRID_POINTS} of {num_points} grid"
        " points are checked for out-of-bounds and unwritten outputs"
    )
    if kernel_has_sync:
      builder.note_truncated_sync(cov_reason, affects_coverage=True)
    else:
      builder.note_incomplete(cov_reason, affects_coverage=True)
  points = list(
      itertools.islice(
          itertools.product(*(range(s) for s in grid)),
          _MAX_COVERAGE_GRID_POINTS,
      )
  )
  recorded_points = _select_recorded_grid_points(
      points, semantics, max_unrolled_iterations
  )
  recorded_list = [p for p in points if p in recorded_points]
  coverage_only_list = [p for p in points if p not in recorded_points]
  for point_group, is_recorded in (
      (recorded_list, True),
      (coverage_only_list, False),
  ):
    if not point_group:
      continue
    for dev_id, dev_coords in mesh_ctx.iter_devices():
      for g_coords in point_group:
        _walk_grid_point(
            eqn,
            kernel_jaxpr,
            grid_mapping,
            operands,
            _WalkContext(
                builder=builder,
                actor=ActorKey(
                    dev_id,
                    0,
                    tuple(
                        g_coords[ax]
                        for ax, sem in enumerate(semantics)
                        if sem == "parallel"
                    ),
                ),
                device_id=dev_id,
                device_coords=dev_coords,
                mesh_ctx=mesh_ctx,
                grid_coords=g_coords,
                grid_sizes=grid_sizes,
                scope_path="",
                touched_refs=set(),
                written={},
                record_events=is_recorded,
                extra_axis_env=(
                    dict(zip(grid_names, g_coords)) if grid_names else {}
                ),
            ),
        )
  _check_output_coverage(builder, operands, mesh_ctx)
  return builder.build()


def _walk_grid_point(
    eqn: Any,
    kernel_jaxpr: Any,
    grid_mapping: Any,
    operands: Sequence[_Operand],
    ctx: _WalkContext,
) -> None:
  """Walks the kernel body for one device and grid point."""
  env = _Env({}, {})
  for outer_v, const_v in zip(
      eqn.invars[: len(kernel_jaxpr.constvars)], kernel_jaxpr.constvars
  ):
    value = _eval_atom(outer_v, {})
    if value is not None:
      env.scalars[const_v] = value
  for idx, var in enumerate(kernel_jaxpr.invars[grid_mapping.slice_index_ops]):
    env.refs[var] = _full_view(
        f"prefetch_{idx}",
        "smem",
        getattr(var.aval, "shape", ()),
        getattr(var.aval, "dtype", np.int32),
    )
  for idx, var in enumerate(
      kernel_jaxpr.invars[grid_mapping.slice_scratch_ops]
  ):
    aval = var.aval
    env.refs[var] = _full_view(
        f"scratch_{idx}",
        _normalize_memory_space(getattr(aval, "memory_space", None)),
        getattr(var.aval, "shape", ()),
        getattr(var.aval, "dtype", np.float32),
        sem_type=_detect_semaphore_type(aval),
    )
  tiles = _bind_operands(ctx, env, operands, pipelined=True)
  body_actor_pos = len(ctx.builder.actor_events.get(ctx.actor, ()))
  body_mem_pos = len(ctx.builder.memory_events)
  _walk(kernel_jaxpr, ctx, env)
  _emit_pipeline_events(ctx, tiles, body_actor_pos, body_mem_pos)


def _find_outer_empty_ref_outputs(jaxpr: Any) -> set[Any]:
  """Returns variables in `jaxpr` created by `empty_ref` that flow to `outvars`."""
  empty_refs: set[Any] = set()
  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "empty_ref":
      for v in eqn.outvars:
        if _is_jax_var(v):
          empty_refs.add(v)
  if not empty_refs:
    return set()
  out_vars = {v for v in jaxpr.outvars if _is_jax_var(v)}
  flowing: set[Any] = set(empty_refs & out_vars)
  for eqn in jaxpr.eqns:
    if eqn.primitive.name in ("get", "freeze") and eqn.invars:
      ref_v = eqn.invars[0]
      if ref_v in empty_refs and any(ov in out_vars for ov in eqn.outvars):
        flowing.add(ref_v)
  return flowing


def _extract_mpmd_map_models(
    eqn: Any,
    mesh_ctx: _MeshContext,
    *,
    max_unrolled_iterations: int,
    tpu_device_kind: str = fx_verifier.DEFAULT_TPU_DEVICE_KIND,
    outer_output_vars: frozenset[Any] | set[Any] = frozenset(),
) -> list[KernelVerificationModel]:
  """Extracts `KernelVerificationModel`s from a `pl.core_map` (`mpmd_map`) equation."""
  meshes = eqn.params.get("meshes", ())
  jaxprs = eqn.params.get("jaxprs", ())
  mosaic_params = _extract_mosaic_compiler_params(
      eqn.params.get("compiler_params")
  )
  collective_id = getattr(mosaic_params, "collective_id", None)
  io_aliases = dict(eqn.params.get("input_output_aliases", ()))
  out_to_in = {out_i: in_i for in_i, out_i in io_aliases.items()}
  num_in = len(eqn.invars)
  out_avals = eqn.params.get("out_avals", eqn.outvars)
  num_out = len(out_avals)
  outer_out_indices: dict[int, int] = {}
  next_outer_out = num_out
  for idx, invar in enumerate(eqn.invars):
    if _is_jax_var(invar) and invar in outer_output_vars:
      outer_out_indices[idx] = next_outer_out
      next_outer_out += 1
  shared_builder: _KernelModelBuilder | None = None
  first_operands: list[_Operand] = []
  for mesh_idx, (pallas_mesh, raw_jaxpr) in enumerate(zip(meshes, jaxprs)):
    kernel_jaxpr = _unwrap_closed_jaxpr(raw_jaxpr)
    mesh_shape_dict = dict(getattr(pallas_mesh, "shape", {}) or {})
    axis_names = tuple(str(k) for k in mesh_shape_dict.keys())
    grid = tuple(int(s) for s in mesh_shape_dict.values())
    raw_sems = getattr(pallas_mesh, "dimension_semantics", None) or ()
    semantics = tuple(
        "parallel"
        if getattr(
            raw_sems[ax] if ax < len(raw_sems) else "parallel",
            "value",
            str(raw_sems[ax] if ax < len(raw_sems) else "parallel"),
        )
        in ("parallel", "core_parallel", "subcore_parallel")
        else "arbitrary"
        for ax in range(len(grid))
    )
    if shared_builder is None:
      shared_builder = _KernelModelBuilder(
          kernel_name=_kernel_name(eqn.params),
          num_devices=mesh_ctx.num_devices,
          grid=grid,
          dimension_semantics=semantics,
          collective_id=collective_id,
          max_unrolled_iterations=max_unrolled_iterations,
          tpu_device_kind=tpu_device_kind,
          core_axis_names=axis_names,
      )
    elif axis_names and not shared_builder.core_axis_names:
      shared_builder.core_axis_names = axis_names
    builder = shared_builder
    kernel_has_sync = builder.has_sync(kernel_jaxpr)
    _check_static_hardware_rules(kernel_jaxpr, collective_id, builder)
    num_points = math.prod(grid) if grid else 1
    if num_points > max_unrolled_iterations:
      reason = (
          f"only the first {max_unrolled_iterations} of {num_points} grid"
          " points are model checked for races and deadlocks"
      )
      if kernel_has_sync:
        builder.note_truncated_sync(reason, affects_coverage=False)
      else:
        builder.note_incomplete(reason, affects_coverage=False)
    if num_points > _MAX_COVERAGE_GRID_POINTS:
      cov_reason = (
          f"only the first {_MAX_COVERAGE_GRID_POINTS} of {num_points} grid"
          " points are checked for out-of-bounds and unwritten outputs"
      )
      if kernel_has_sync:
        builder.note_truncated_sync(cov_reason, affects_coverage=True)
      else:
        builder.note_incomplete(cov_reason, affects_coverage=True)
    mpmd_operands: list[_Operand] = []
    for idx, out_i in outer_out_indices.items():
      if idx >= len(kernel_jaxpr.invars):
        continue
      var = kernel_jaxpr.invars[idx]
      aval = getattr(var, "aval", None)
      mpmd_operands.append(
          _Operand(
              index=idx,
              io_index=out_i,
              var=var,
              block_mapping=None,
              buffer_id=f"arg_out_{out_i}",
              is_output=True,
              memory_space=_normalize_memory_space(
                  getattr(aval, "memory_space", None)
              ),
              array_shape=tuple(int(s) for s in getattr(aval, "shape", ())),
          )
      )
    for idx in range(num_in, min(len(kernel_jaxpr.invars), num_in + num_out)):
      var = kernel_jaxpr.invars[idx]
      aval = getattr(var, "aval", None)
      out_i = idx - num_in
      buf_id = (
          f"arg_in_{out_to_in[out_i]}"
          if out_i in out_to_in
          else f"arg_out_{out_i}"
      )
      mpmd_operands.append(
          _Operand(
              index=idx,
              io_index=out_i,
              var=var,
              block_mapping=None,
              buffer_id=buf_id,
              is_output=True,
              memory_space=_normalize_memory_space(
                  getattr(aval, "memory_space", None)
              ),
              array_shape=tuple(int(s) for s in getattr(aval, "shape", ())),
          )
      )
    if mesh_idx == 0:
      first_operands = mpmd_operands
    points = list(
        itertools.islice(
            itertools.product(*(range(s) for s in grid)),
            _MAX_COVERAGE_GRID_POINTS,
        )
    )
    recorded_points = _select_recorded_grid_points(
        points, semantics, max_unrolled_iterations
    )
    recorded_list = [p for p in points if p in recorded_points]
    coverage_only_list = [p for p in points if p not in recorded_points]
    for point_group, is_recorded in (
        (recorded_list, True),
        (coverage_only_list, False),
    ):
      if not point_group:
        continue
      for dev_id, dev_coords in mesh_ctx.iter_devices():
        for g_coords in point_group:
          env = _Env({}, {})
          unaliased_out: dict[str, tuple[str, int]] = {}
          for idx, var in enumerate(kernel_jaxpr.invars):
            aval = getattr(var, "aval", None)
            if idx < num_in:
              val = _eval_atom(eqn.invars[idx], {})
              if val is not None:
                env.scalars[var] = val
              if idx in outer_out_indices:
                out_i = outer_out_indices[idx]
                buf_id = f"arg_out_{out_i}"
                unaliased_out[buf_id] = (f"output {out_i}", idx)
              else:
                buf_id = f"arg_in_{idx}"
              env.refs[var] = _full_view(
                  buf_id,
                  _normalize_memory_space(getattr(aval, "memory_space", None)),
                  getattr(aval, "shape", ()),
                  getattr(aval, "dtype", np.float32),
                  sem_type=_detect_semaphore_type(aval),
              )
            elif idx < num_in + num_out:
              out_i = idx - num_in
              buf_id = (
                  f"arg_in_{out_to_in[out_i]}"
                  if out_i in out_to_in
                  else f"arg_out_{out_i}"
              )
              if buf_id.startswith("arg_out_"):
                unaliased_out[buf_id] = (f"output {out_i}", idx)
              env.refs[var] = _full_view(
                  buf_id,
                  _normalize_memory_space(getattr(aval, "memory_space", None)),
                  getattr(aval, "shape", ()),
                  getattr(aval, "dtype", np.float32),
                  sem_type=_detect_semaphore_type(aval),
              )
            else:
              scr_i = idx - (num_in + num_out)
              env.refs[var] = _full_view(
                  f"scratch_{scr_i}",
                  _normalize_memory_space(getattr(aval, "memory_space", None)),
                  getattr(aval, "shape", ()),
                  getattr(aval, "dtype", np.float32),
                  sem_type=_detect_semaphore_type(aval),
              )
          ctx = _WalkContext(
              builder=builder,
              actor=ActorKey(
                  dev_id,
                  mesh_idx,
                  tuple(
                      g_coords[ax]
                      for ax, sem in enumerate(semantics)
                      if sem == "parallel"
                  ),
              ),
              device_id=dev_id,
              device_coords=dev_coords,
              mesh_ctx=mesh_ctx,
              grid_coords=g_coords,
              grid_sizes=grid,
              scope_path=f"m{mesh_idx}" if len(meshes) > 1 else "",
              touched_refs=set(),
              written={},
              record_events=is_recorded,
              extra_axis_env=dict(zip(axis_names, g_coords)),
              core_axis_names=axis_names,
              unaliased_output_buffers=unaliased_out,
          )
          _walk(kernel_jaxpr, ctx, env)
          _record_step_written_buffers(ctx)

  if shared_builder is None:
    return []
  _check_output_coverage(shared_builder, first_operands, mesh_ctx)
  return [shared_builder.build()]


def _collect_models(
    jaxpr: Any,
    mesh_ctx: _MeshContext,
    out_models: list[KernelVerificationModel],
    *,
    max_unrolled_iterations: int,
    tpu_device_kind: str = fx_verifier.DEFAULT_TPU_DEVICE_KIND,
) -> None:
  """Recursively locates `shard_map`, `pallas_call`, and `mpmd_map` equations."""
  outer_empty_outputs = _find_outer_empty_ref_outputs(jaxpr)
  for eqn in jaxpr.eqns:
    name = eqn.primitive.name
    if name == "pallas_call":
      out_models.append(
          _extract_pallas_call_model(
              eqn,
              mesh_ctx,
              max_unrolled_iterations=max_unrolled_iterations,
              tpu_device_kind=tpu_device_kind,
          )
      )
      continue
    if name == "mpmd_map":
      out_models.extend(
          _extract_mpmd_map_models(
              eqn,
              mesh_ctx,
              max_unrolled_iterations=max_unrolled_iterations,
              tpu_device_kind=tpu_device_kind,
              outer_output_vars=outer_empty_outputs,
          )
      )
      continue
    sub_ctx = mesh_ctx
    if name == "shard_map":
      mesh = eqn.params["mesh"]
      sub_ctx = _MeshContext(
          axis_names=tuple(mesh.axis_names),
          mesh_shape=tuple(int(s) for s in dict(mesh.shape).values()),
      )
    for sub in _sub_jaxprs(eqn):
      _collect_models(
          sub,
          sub_ctx,
          out_models,
          max_unrolled_iterations=max_unrolled_iterations,
          tpu_device_kind=tpu_device_kind,
      )


def _append_placeholder_slot(
    val: Any,
    dyn_placeholders: list[Any],
    *,
    is_static: bool = False,
    mesh: Any = None,
    partition_spec: Any = None,
) -> tuple[tuple[bool, Any], bool]:
  """Appends a dynamic placeholder or static value slot; returns `(slot, has_named_sharding)`."""
  if is_static or not isinstance(
      val, (torch.Tensor, jax.Array, jax.ShapeDtypeStruct)
  ):
    return (True, val), False
  slot = (False, len(dyn_placeholders))
  if isinstance(val, torch.Tensor):
    dyn_placeholders.append(
        pallas_lib.jax_placeholder(
            val, mesh=mesh, partition_spec=partition_spec
        )
    )
    return slot, False
  sharding = getattr(val, "sharding", None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    dyn_placeholders.append(
        jax.ShapeDtypeStruct(val.shape, val.dtype, sharding=sharding)
    )
    return slot, True
  dyn_placeholders.append(jax.ShapeDtypeStruct(val.shape, val.dtype))
  return slot, False


def _trace_op_to_closed_jaxpr(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any],
    *,
    tpu_device_kind: str = fx_verifier.DEFAULT_TPU_DEVICE_KIND,
) -> Any:
  """Traces a TorchTPU `jax_op` or raw JAX callable to a `ClosedJaxpr`."""
  jax_callable = fx_verifier.extract_jax_callable(op_or_fn)
  if jax_callable is not None:
    # Tracing needs no devices, so trace with the op's own (possibly
    # compile-only) mesh like TorchTPU does when it lowers the op. Unlike
    # building a CPU mesh, this does not initialize any JAX backend, which on
    # a TPU host would try to open the TPU.
    raw_fn, active_mesh = fx_verifier.unwrap_raw_jax_fn(jax_callable)
    # pylint: disable=protected-access
    norm_args, norm_kwargs = jax_callable._normalize_invocation_args(
        example_args, dict(example_kwargs)
    )
    # pylint: enable=protected-access
    in_specs = jax_callable.input_partition_specs
    dyn_placeholders: list[Any] = []
    arg_slots: list[tuple[bool, Any]] = []
    for idx, val in enumerate(norm_args):
      spec = (
          in_specs[idx]
          if in_specs is not None and idx < len(in_specs)
          else None
      )
      slot, _ = _append_placeholder_slot(
          val,
          dyn_placeholders,
          is_static=fx_verifier.is_static_positional_arg(jax_callable, idx),
          mesh=active_mesh,
          partition_spec=spec,
      )
      arg_slots.append(slot)

    kw_slots: dict[str, tuple[bool, Any]] = {}
    for k, val in norm_kwargs.items():
      slot, _ = _append_placeholder_slot(
          val, dyn_placeholders, mesh=active_mesh
      )
      kw_slots[k] = slot

    def _make_target(fn_to_call):
      def _trace_target(*dyn_args):
        rebuilt_args = [
            slot_val if is_static else dyn_args[slot_val]
            for is_static, slot_val in arg_slots
        ]
        rebuilt_kwargs = {
            k: slot_val if is_static else dyn_args[slot_val]
            for k, (is_static, slot_val) in kw_slots.items()
        }
        return fn_to_call(*rebuilt_args, **rebuilt_kwargs)

      return _trace_target

    try:
      return jax.make_jaxpr(_make_target(raw_fn))(*dyn_placeholders)
    except ValueError as exc:
      if "Unsupported TPU device kind" not in str(exc):
        raise
      if active_mesh is None:
        with fx_verifier.override_abstract_mesh(
            fx_verifier.abstract_tpu_mesh(tpu_device_kind)
        ):
          return jax.make_jaxpr(_make_target(raw_fn))(*dyn_placeholders)
      if (
          isinstance(active_mesh, jax.sharding.AbstractMesh)
          and active_mesh.abstract_device is None
      ):
        tpu_mesh = active_mesh.update(
            abstract_device=fx_verifier.abstract_tpu_device(tpu_device_kind)
        )
        rebound_fn = fx_verifier.rebind_function_meshes(
            raw_fn, tpu_mesh, clone_closures=True
        )
        rebound_phs = fx_verifier.replace_mock_meshes_in_pytree(
            dyn_placeholders, tpu_mesh
        )
        with fx_verifier.override_abstract_mesh(tpu_mesh):
          return jax.make_jaxpr(_make_target(rebound_fn))(*rebound_phs)
      raise

  # Raw JAX function path.
  raw_fn, _ = fx_verifier.unwrap_raw_jax_fn(op_or_fn)
  dyn_phs: list[Any] = []
  pos_map: list[tuple[bool, Any]] = []
  has_named_sharding = False
  for val in example_args:
    slot, named = _append_placeholder_slot(val, dyn_phs)
    pos_map.append(slot)
    has_named_sharding = has_named_sharding or named

  kw_map: dict[str, tuple[bool, Any]] = {}
  for k, val in example_kwargs.items():
    slot, named = _append_placeholder_slot(val, dyn_phs)
    kw_map[k] = slot
    has_named_sharding = has_named_sharding or named

  trace_ctx: Any = contextlib.nullcontext()
  if not has_named_sharding:
    fn_mesh = fx_verifier.find_function_mesh(raw_fn)
    if isinstance(fn_mesh, jax.sharding.AbstractMesh) and any(
        at == jax.sharding.AxisType.Explicit
        for at in getattr(fn_mesh, "axis_types", ())
    ):
      auto_mesh = fn_mesh.update(
          axis_types=(jax.sharding.AxisType.Auto,) * len(fn_mesh.axis_names)
      )
      raw_fn = fx_verifier.rebind_function_meshes(
          raw_fn, auto_mesh, clone_closures=True
      )
      if not jax.sharding.get_abstract_mesh().empty:
        trace_ctx = fx_verifier.override_abstract_mesh(auto_mesh)

  def _raw_target(*dyn_args):
    args = [v if is_st else dyn_args[v] for is_st, v in pos_map]
    kwargs = {
        k: v if is_st else dyn_args[v] for k, (is_st, v) in kw_map.items()
    }
    return raw_fn(*args, **kwargs)

  try:
    with trace_ctx:
      return jax.make_jaxpr(_raw_target)(*dyn_phs)
  except ValueError as exc:
    if "Unsupported TPU device kind" not in str(exc):
      raise
    with fx_verifier.override_abstract_mesh(
        fx_verifier.abstract_tpu_mesh(tpu_device_kind)
    ):
      return jax.make_jaxpr(_raw_target)(*dyn_phs)


def display_name(op_or_fn: Any) -> str:
  """Returns the name used for `op_or_fn` in findings."""
  return fx_verifier.get_op_display_name(op_or_fn)


def extract_kernel_models(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    max_unrolled_iterations: int = DEFAULT_MAX_UNROLLED_ITERATIONS,
    tpu_device_kind: str = fx_verifier.DEFAULT_TPU_DEVICE_KIND,
) -> tuple[KernelVerificationModel, ...]:
  """Traces a TorchTPU `jax_op` or JAX callable and extracts Pallas models."""
  tpu_device_kind = fx_verifier.infer_tpu_device_kind(op_or_fn, tpu_device_kind)
  closed = _trace_op_to_closed_jaxpr(
      op_or_fn,
      example_args,
      dict(example_kwargs or {}),
      tpu_device_kind=tpu_device_kind,
  )
  models: list[KernelVerificationModel] = []
  _collect_models(
      closed.jaxpr,
      _MeshContext(),
      models,
      max_unrolled_iterations=max_unrolled_iterations,
      tpu_device_kind=tpu_device_kind,
  )
  name = display_name(op_or_fn)
  return tuple(
      dataclasses.replace(
          model,
          kernel_name=name
          if len(models) == 1
          else f"{name} ({model.kernel_name} #{idx})",
      )
      for idx, model in enumerate(models)
  )


def static_model_findings(
    model: KernelVerificationModel,
) -> list[findings.VerificationFinding]:
  """Converts static issues and incompleteness notes on `model` into findings."""
  op_name = model.kernel_name
  detected = [
      findings.VerificationFinding(
          category=issue.category,
          severity=issue.severity,
          phase=findings.Phase.MODEL_CHECK,
          op_name=op_name,
          message=issue.message,
          source_location=issue.source_location,
      )
      for issue in model.static_issues
  ]
  if model.incomplete_reasons:
    detected.append(
        findings.VerificationFinding(
            category=_Category.VERIFICATION_INCOMPLETE,
            severity=_Severity.INFO,
            phase=findings.Phase.MODEL_CHECK,
            op_name=op_name,
            message=(
                "the model is bounded or approximate, so findings may be"
                " missed: "
                + "; ".join(model.incomplete_reasons)
            ),
        )
    )
  return detected


def _classify_trace_exception(
    exc: Exception,
) -> tuple[_Category, _Severity]:
  """Classifies JAX/Pallas trace-time validation exceptions into finding categories."""
  msg = str(exc)
  lower = msg.lower()
  if "out of bound slice" in lower or "index out of bounds" in lower:
    return _Category.OUT_OF_BOUNDS_ACCESS, _Severity.ERROR
  if any(
      pat in lower
      for pat in (
          "cannot signal on a non-regular semaphore",
          "cannot signal on a non-0d semaphore",
          "cannot wait on a non-regular semaphore",
          "cannot wait on a non-0d semaphore",
          "must signal semaphores of the following types",
          "must wait on semaphores of the following types",
          "dma_wait requires a dma semaphore",
          "dma_start requires a dma semaphore",
          "only supports 0d semaphores",
          "must have the same shape",
          "must have the same dtype",
          "block_shape",
          "index_map",
          "program_id",
          "num_programs",
      )
  ):
    return _Category.HARDWARE_CONSTRAINT_VIOLATION, _Severity.ERROR
  if "axis_name" in lower or "device_id" in lower or "core_index" in lower:
    return _Category.INVALID_DEVICE_ID, _Severity.ERROR
  return _Category.VERIFICATION_INCOMPLETE, _Severity.WARNING


def check_kernel_static_models(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    max_unrolled_iterations: int = DEFAULT_MAX_UNROLLED_ITERATIONS,
    tpu_device_kind: str = fx_verifier.DEFAULT_TPU_DEVICE_KIND,
) -> tuple[
    findings.VerificationReport,
    tuple[KernelVerificationModel, ...] | None,
]:
  """Traces `op_or_fn`, extracts `KernelVerificationModel`s, and reports static issues."""
  tpu_device_kind = fx_verifier.infer_tpu_device_kind(op_or_fn, tpu_device_kind)
  try:
    closed = _trace_op_to_closed_jaxpr(
        op_or_fn,
        example_args,
        dict(example_kwargs or {}),
        tpu_device_kind=tpu_device_kind,
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    cat, sev = _classify_trace_exception(exc)
    prefix = (
        "could not trace the kernel for model checking"
        if cat == _Category.VERIFICATION_INCOMPLETE
        else "kernel failed Pallas static validation during tracing"
    )
    report = findings.report_of(
        findings.Phase.MODEL_CHECK,
        [
            findings.VerificationFinding(
                category=cat,
                severity=sev,
                phase=findings.Phase.MODEL_CHECK,
                op_name=display_name(op_or_fn),
                message=f"{prefix} ({type(exc).__name__}: {exc})",
            )
        ],
    )
    return report, None
  models_list: list[KernelVerificationModel] = []
  _collect_models(
      closed.jaxpr,
      _MeshContext(),
      models_list,
      max_unrolled_iterations=max_unrolled_iterations,
      tpu_device_kind=tpu_device_kind,
  )
  name = display_name(op_or_fn)
  models = tuple(
      dataclasses.replace(
          model,
          kernel_name=name
          if len(models_list) == 1
          else f"{name} ({model.kernel_name} #{idx})",
      )
      for idx, model in enumerate(models_list)
  )
  if not models:
    report = findings.report_of(
        findings.Phase.MODEL_CHECK,
        [
            findings.VerificationFinding(
                category=_Category.VERIFICATION_INCOMPLETE,
                severity=_Severity.INFO,
                phase=findings.Phase.MODEL_CHECK,
                op_name=name,
                message=(
                    "no pallas_call was found, so there was nothing to model"
                    " check"
                ),
            )
        ],
    )
    return report, ()
  detected: list[findings.VerificationFinding] = []
  for model in models:
    detected.extend(static_model_findings(model))
  return findings.report_of(findings.Phase.MODEL_CHECK, detected), models
