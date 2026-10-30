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

`extract_kernel_models` traces a TorchTPU `@jax_op` or a JAX function, finds
every `pallas_call` (inside `shard_map` for multi-chip kernels), and unrolls it
per device and grid point into a `KernelVerificationModel`: ordered memory
accesses on hyper-rectangular buffer regions, DMA starts and waits, and
semaphore signals and waits. Problems that are visible without solving
anything (out-of-bounds slices, invalid remote device ids, output regions that
are never written, unsupported hardware patterns) are recorded on the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
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

# Grid points whose writes are tracked for the output coverage and bounds
# checks; only the first `max_unrolled_iterations` of them are model checked.
_MAX_COVERAGE_GRID_POINTS = 4096
# Bound on the cells examined when computing uncovered output regions.
_MAX_COVERAGE_CELLS = 1 << 20
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

  @classmethod
  def full(cls, shape: Sequence[int]) -> HyperRectSlice:
    return cls(tuple(SliceDim(0, int(s)) for s in shape))

  def intersects(self, other: HyperRectSlice) -> bool:
    if len(self.dims) != len(other.dims):
      # Different ranks (e.g. scalar vs. array views): assume overlap.
      return True
    return all(a.overlaps(b) for a, b in zip(self.dims, other.dims))

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
  # None for data buffers, else "dma", "regular", or "barrier".
  sem_type: str | None = None
  # Set once a transform that changes the shape (reshape, bitcast) was applied;
  # later indexing is then not interpreted.
  opaque: bool = False

  @property
  def view_shape(self) -> tuple[int, ...]:
    return tuple(self.dims[ax].size for ax in self.active_axes)

  @property
  def region(self) -> HyperRectSlice:
    return HyperRectSlice(self.dims)

  @property
  def sem_index(self) -> tuple[int, ...] | None:
    """Element index of a semaphore view, or None if it is not one element."""
    if any(d.size != 1 for d in self.dims):
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
) -> _RefView:
  shape = tuple(int(s) for s in shape)
  return _RefView(
      buffer_id=buffer_id,
      memory_space=memory_space,
      root_shape=shape,
      dtype=np.int32 if sem_type is not None else dtype,
      dims=tuple(SliceDim(0, s) for s in shape),
      active_axes=tuple(range(len(shape))),
      sem_type=sem_type,
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
  if dtype_cls_name == "DMASemaphore" or "dma_sem" in dtype_str:
    return "dma"
  if dtype_cls_name == "BarrierSemaphore" or "barrier_sem" in dtype_str:
    return "barrier"
  if (
      dtype_cls_name == "Semaphore"
      or ms == pltpu.MemorySpace.SEMAPHORE
      or "sem" in dtype_str
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


def _try_eval_scalar_eqn(
    eqn: Any,
    scalar_env: dict[Any, int | bool],
    *,
    grid_coords: tuple[int, ...],
    grid_sizes: tuple[int, ...],
    axis_env: Mapping[Any, int],
) -> bool:
  """Evaluates a scalar integer/Boolean equation into `scalar_env`."""
  name = eqn.primitive.name
  if name in ("program_id", "num_programs"):
    axis = int(eqn.params["axis"])
    source = grid_coords if name == "program_id" else grid_sizes
    if 0 <= axis < len(source):
      scalar_env[eqn.outvars[0]] = source[axis]
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
  vals = [_eval_atom(v, scalar_env) for v in eqn.invars]
  concrete = [v for v in vals if v is not None]
  if not vals or len(concrete) != len(vals):
    return False
  is_bool = out_aval.dtype == np.bool_
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
    grid_sizes: tuple[int, ...],
    axis_env: Mapping[Any, int],
) -> bool:
  """Evaluates a nested call, such as `jnp.floor_divide`, on scalars.

  Only calls with known scalar operands whose equations are all scalar
  arithmetic are evaluated, so calls that access refs are still walked.

  Args:
    eqn: The call equation.
    scalar_env: Known scalar values; receives the call's results.
    grid_coords: Grid point being walked.
    grid_sizes: Grid sizes.
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
  memory_space = base.memory_space
  opaque = base.opaque
  violation: str | None = None
  for tr in transforms:
    tr_name = type(tr).__name__
    if tr_name == "MemorySpaceCastTransform" and hasattr(tr, "memory_space"):
      memory_space = _normalize_memory_space(tr.memory_space)
      continue
    if tr_name != "NDIndexer" or not hasattr(tr, "indices"):
      opaque = True  # Reshapes and bitcasts change the indexed shape.
    if opaque:
      continue
    new_active: list[int] = []
    for ax_pos, idx_entry in zip(active_axes, tr.indices):
      parent = dims[ax_pos]
      if isinstance(idx_entry, pl.Slice):
        start = _eval_atom(idx_entry.start, scalar_env)
        size = _eval_atom(idx_entry.size, scalar_env)
        stride = int(getattr(idx_entry, "stride", 1) or 1)
        if start is not None and size is not None:
          span = (int(size) - 1) * stride + 1 if int(size) > 0 else 0
          if int(start) < 0 or int(start) + span > parent.size:
            violation = violation or (
                f"slice {int(start)}:{int(start) + span} is outside axis"
                f" {ax_pos} of size {parent.size}"
            )
          dims[ax_pos] = SliceDim(parent.start + int(start), span)
        new_active.append(ax_pos)
        continue
      idx_aval = getattr(idx_entry, "aval", None)
      if isinstance(idx_aval, jax.core.ShapedArray) and idx_aval.shape:
        new_active.append(ax_pos)  # Gather: the whole axis may be read.
        continue
      k = _eval_atom(idx_entry, scalar_env)
      if k is not None:
        idx = int(k) + (parent.size if int(k) < 0 else 0)
        if not 0 <= idx < parent.size:
          violation = violation or (
              f"index {int(k)} is outside axis {ax_pos} of size {parent.size}"
          )
        dims[ax_pos] = SliceDim(parent.start + idx, 1)
    active_axes = new_active
  view = dataclasses.replace(
      base,
      memory_space=memory_space,
      dims=tuple(dims),
      active_axes=tuple(active_axes),
      opaque=opaque,
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
  ):
    self.kernel_name = kernel_name
    self.num_devices = num_devices
    self.grid = grid
    self.dimension_semantics = dimension_semantics
    self.collective_id = collective_id
    self.max_unrolled_iterations = max_unrolled_iterations

    self._next_event_id = 0
    self._next_dma_id = 0
    self.actor_events: dict[ActorKey, list[int]] = {}
    self.memory_events: list[MemoryAccessEvent] = []
    self.dma_starts: list[DmaStartEvent] = []
    self.dma_waits: list[DmaWaitEvent] = []
    self.sem_signals: list[SemaphoreSignalEvent] = []
    self.sem_waits: list[SemaphoreWaitEvent] = []
    self.static_issues: list[StaticIssue] = []
    self._issue_keys: set[tuple[findings.FindingCategory, str]] = set()
    self.incomplete_reasons: list[str] = []
    self.coverage_known = True
    self.sync_reliable = True
    self.sync_truncated = False
    self.uses_remote_operations = False
    # Semaphores whose operations are only partly modeled.
    self.truncated_sem_keys: set[SemaphoreKey] = set()
    self._sem_key_scopes: list[set[SemaphoreKey]] = []
    self._has_sync: dict[int, bool] = {}
    # (device id, output buffer id) -> regions written by stores or DMAs.
    self.output_writes: dict[tuple[int, str], list[HyperRectSlice]] = {}
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
  ) -> None:
    if (category, message) not in self._issue_keys:
      self._issue_keys.add((category, message))
      self.static_issues.append(StaticIssue(category, severity, message, loc))

  def note_incomplete(self, reason: str, *, affects_coverage: bool) -> None:
    if reason not in self.incomplete_reasons:
      self.incomplete_reasons.append(reason)
    if affects_coverage:
      self.coverage_known = False

  def note_unresolved_sync(self, reason: str) -> None:
    self.sync_reliable = False
    self.note_incomplete(reason, affects_coverage=True)

  def note_truncated_sync(self, reason: str, *, affects_coverage: bool) -> None:
    """Records that only a prefix of the semaphore operations is modeled."""
    self.sync_truncated = True
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

  def saw_semaphore(self, key: SemaphoreKey, *, recorded: bool) -> None:
    """Notes a semaphore operation, which may not be recorded as an event."""
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
    if buffer_id.startswith("arg_out_"):
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
    )


@dataclasses.dataclass
class _Env:
  """Refs and concrete scalars bound to jaxpr variables."""

  refs: dict[Any, _RefView]
  scalars: dict[Any, int | bool]

  def child(self) -> _Env:
    return _Env(dict(self.refs), dict(self.scalars))

  def bind(self, outer: Any, inner: Any, outer_env: _Env) -> None:
    """Binds `inner` to the ref or scalar value of `outer` in `outer_env`."""
    if _is_jax_var(outer) and outer in outer_env.refs:
      self.refs[inner] = outer_env.refs[outer]
    val = _eval_atom(outer, outer_env.scalars)
    if val is not None:
      self.scalars[inner] = val


@dataclasses.dataclass
class _WalkContext:
  """Position of a jaxpr walk: device, grid point, actor, and scope."""

  builder: _KernelModelBuilder
  actor: ActorKey
  device_id: int
  device_coords: tuple[int, ...]
  mesh_ctx: _MeshContext
  grid_coords: tuple[int, ...]
  grid_sizes: tuple[int, ...]
  scope_path: str
  touched_refs: set[str]
  # Buffer id -> regions stored by the vector unit at this grid point.
  written: dict[str, list[HyperRectSlice]]
  # Whether events are recorded for model checking (only the first grid points
  # are; the rest only feed the coverage and bounds checks).
  record_events: bool = True

  @property
  def axis_env(self) -> dict[Any, int]:
    return self.mesh_ctx.coord_dict(self.device_coords)

  def scoped(self, suffix: str) -> _WalkContext:
    return dataclasses.replace(self, scope_path=f"{self.scope_path}.{suffix}")

  def grid_tag(self) -> str:
    return "g" + "_".join(str(c) for c in self.grid_coords)


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
    ctx.builder.record_issue(
        _Category.OUT_OF_BOUNDS_ACCESS,
        f"access to {view.memory_space}:{view.buffer_id}: {violation}",
        loc,
    )
  return view


def _buffer_key(ctx: _WalkContext, view: _RefView, device_id: int) -> BufferKey:
  return BufferKey(
      device_id, ctx.actor.core_id, view.memory_space, view.buffer_id
  )


def _sem_key(
    ctx: _WalkContext, view: _RefView, device_id: int, default_type: str
) -> SemaphoreKey | None:
  index = view.sem_index
  if index is None or view.opaque:
    ctx.builder.note_unresolved_sync(
        f"the element of semaphore array {view.buffer_id} could not be"
        " determined statically"
    )
    return None
  sem_type = view.sem_type or default_type
  key = SemaphoreKey(
      device_id=device_id,
      core_id=ctx.actor.core_id,
      sem_type=sem_type,
      sem_id=view.buffer_id,
      sem_index=index,
      # A remote signal is assumed to target the same parallel iteration on
      # the other device; barrier semaphores are shared by the whole core.
      parallel_coords=(
          () if sem_type == "barrier" else ctx.actor.parallel_coords
      ),
  )
  ctx.builder.saw_semaphore(key, recorded=ctx.record_events)
  return key


def _record_access(
    ctx: _WalkContext,
    view: _RefView,
    *,
    is_write: bool,
    loc: str | None,
) -> None:
  ctx.touched_refs.add(view.buffer_id)
  if is_write:
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
      )
  )


def _resolve_target_device(
    ctx: _WalkContext, env: _Env, device_id_expr: Any, loc: str | None
) -> int | None:
  """Resolves a Pallas `device_id` to a flat device index.

  Returns:
    The device index, or None if it is invalid or unknown (both are recorded
    on the builder).
  """
  if device_id_expr is None:
    return ctx.device_id
  ctx.builder.uses_remote_operations = True
  mesh = ctx.mesh_ctx
  if isinstance(device_id_expr, Mapping):
    coords: dict[Any, int | bool | None] = dict(
        zip(mesh.axis_names, ctx.device_coords)
    )
    for key, value_expr in device_id_expr.items():
      axis = key[0] if isinstance(key, tuple) and len(key) == 1 else key
      coords[axis] = _eval_atom(value_expr, env.scalars)
    names: Sequence[str] = mesh.axis_names
    sizes: Sequence[int] = mesh.mesh_shape
    values = [coords.get(a) for a in mesh.axis_names]
  elif isinstance(device_id_expr, tuple):
    names, sizes = mesh.axis_names, mesh.mesh_shape
    values = [_eval_atom(e, env.scalars) for e in device_id_expr]
  else:
    names, sizes = ("logical device id",), (mesh.num_devices,)
    values = [_eval_atom(device_id_expr, env.scalars)]
  ints = [int(v) for v in values if v is not None]
  if len(ints) != len(values) or len(ints) != len(sizes):
    ctx.builder.note_unresolved_sync(
        "a remote device id could not be determined statically"
    )
    return None
  for name, value, size in zip(names, ints, sizes):
    if not 0 <= value < size:
      ctx.builder.record_issue(
          _Category.INVALID_DEVICE_ID,
          f"remote operation targets index {value} along {name!r}, which"
          f" only has {size} device(s) (wrap the index with lax.rem)",
          loc,
      )
      ctx.builder.note_unresolved_sync(
          "a remote operation targets a device outside the mesh"
      )
      return None
  if not sizes:
    return ctx.device_id
  if len(ints) == 1 and len(mesh.mesh_shape) != 1:
    return ints[0]  # A logical id on a multi-axis mesh.
  return int(np.ravel_multi_index(tuple(ints), tuple(sizes)))


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
  if eqn.primitive.name == "addupdate":
    _record_access(ctx, view, is_write=False, loc=loc)
  _record_access(ctx, view, is_write=True, loc=loc)


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


def _handle_dma_start(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  ops = decode_dma_operands(eqn.params["tree"].unflatten(eqn.invars))
  src = _resolve_view(ctx, env, ops.src.ref, ops.src.transforms, loc)
  dst = _resolve_view(ctx, env, ops.dst.ref, ops.dst.transforms, loc)
  if src is None or dst is None:
    ctx.builder.note_unresolved_sync(_UNRESOLVED_SYNC)
    return
  target = _resolve_target_device(ctx, env, ops.device_id, loc)
  if target is None:
    return
  ctx.touched_refs.add(src.buffer_id)
  ctx.builder.record_output_write(target, dst.buffer_id, dst.region)
  if target == ctx.device_id:
    ctx.written.setdefault(dst.buffer_id, []).append(dst.region)
  dst_sem = _operand_sem(ctx, env, ops.dst_sem, target, loc)
  src_sem = _operand_sem(ctx, env, ops.src_sem, ctx.device_id, loc)
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
          dst_buffer=_buffer_key(ctx, dst, target),
          dst_region=dst.region,
          byte_count=src.byte_count,
          dst_sem=dst_sem,
          src_sem=src_sem,
          is_remote=ops.device_id is not None,
          source_location=loc,
      )
  )


def _operand_sem(
    ctx: _WalkContext,
    env: _Env,
    operand: RefOperand | None,
    device_id: int,
    loc: str | None,
) -> SemaphoreKey | None:
  if operand is None:
    return None
  view = _resolve_view(ctx, env, operand.ref, operand.transforms, loc)
  if view is None:
    ctx.builder.note_unresolved_sync(_UNRESOLVED_SYNC)
    return None
  return _sem_key(ctx, view, device_id, "dma")


def _handle_dma_wait(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  ops = decode_dma_operands(eqn.params["tree"].unflatten(eqn.invars))
  # `wait_send` swaps the operands, so the waited semaphore is always in the
  # destination-semaphore slot and is always local.
  src = _resolve_view(ctx, env, ops.src.ref, ops.src.transforms, loc)
  sem_view = (
      None
      if ops.dst_sem is None
      else _resolve_view(ctx, env, ops.dst_sem.ref, ops.dst_sem.transforms, loc)
  )
  if src is None or sem_view is None:
    ctx.builder.note_unresolved_sync(_UNRESOLVED_SYNC)
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
          byte_count=src.byte_count,
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
    ctx: _WalkContext, env: _Env, amount_expr: Any, what: str
) -> int | None:
  amount = _eval_atom(amount_expr, env.scalars)
  if amount is None:
    ctx.builder.note_unresolved_sync(
        f"a semaphore {what} amount could not be determined statically"
    )
    return None
  return int(amount)


def _handle_sem_signal(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  sem_arg, sem_tr, inc_arg, dev_id_arg, *_ = eqn.params["args_tree"].unflatten(
      eqn.invars
  )
  view = _resolve_view(ctx, env, sem_arg, sem_tr, loc)
  if view is None:
    ctx.builder.note_unresolved_sync(_UNRESOLVED_SYNC)
    return
  amount = _semaphore_amount(ctx, env, inc_arg, "signal")
  target = _resolve_target_device(ctx, env, dev_id_arg, loc)
  if amount is None or target is None:
    return
  sem = _sem_key(ctx, view, target, "regular")
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
          is_remote=dev_id_arg is not None,
          source_location=loc,
      )
  )


def _handle_sem_wait(eqn: Any, ctx: _WalkContext, env: _Env, loc: str | None):
  sem_arg, sem_tr, val_arg, decrement_arg = eqn.params["args_tree"].unflatten(
      eqn.invars
  )
  view = _resolve_view(ctx, env, sem_arg, sem_tr, loc)
  if view is None:
    ctx.builder.note_unresolved_sync(_UNRESOLVED_SYNC)
    return
  amount = _semaphore_amount(ctx, env, val_arg, "wait")
  sem = _sem_key(ctx, view, ctx.device_id, "regular")
  if amount is None or sem is None or not ctx.record_events:
    return
  decrement = _eval_atom(decrement_arg, env.scalars)
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
  del loc
  sub_jaxpr = _unwrap_closed_jaxpr(eqn.params["jaxpr"])
  ref_transforms = eqn.params.get("ref_transforms", ())
  shared = bool(eqn.params.get("collective_axes", ()))
  nested = env.child()
  for outer_v, const_v in zip(eqn.invars, sub_jaxpr.constvars):
    nested.bind(outer_v, const_v, env)
  scope = ctx.scoped(f"s{_next_scope_index(ctx)}")
  prefix = scope.scope_path if shared else f"{ctx.grid_tag()}{scope.scope_path}"
  for alloc_idx, alloc_var in enumerate(sub_jaxpr.invars):
    aval = alloc_var.aval
    base = _full_view(
        f"{prefix}.b{alloc_idx}",
        _normalize_memory_space(getattr(aval, "memory_space", None)),
        getattr(aval, "shape", ()),
        getattr(aval, "dtype", np.float32),
        sem_type=_detect_semaphore_type(aval),
    )
    init_tr = (
        ref_transforms[alloc_idx] if alloc_idx < len(ref_transforms) else ()
    )
    nested.refs[alloc_var], _ = _apply_transforms(base, init_tr, nested.scalars)
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
  carry = list(eqn.invars[num_consts : num_consts + num_carry])
  carry_env = env
  ctx.builder.push_semaphore_scope()
  try:
    for step in range(min(length, limit)):
      nested = env.child()
      for outer_v, inner_v in zip(consts, body.invars[:num_consts]):
        nested.bind(outer_v, inner_v, env)
      for outer_v, inner_v in zip(carry, body.invars[num_consts:]):
        nested.bind(outer_v, inner_v, carry_env)
      _walk(body, ctx.scoped(f"lp{step}"), nested)
      carry = list(body.outvars[:num_carry])
      carry_env = nested
  finally:
    ctx.builder.pop_semaphore_scope(truncated=truncated)
  _bind_outputs(carry, eqn.outvars[:num_carry], carry_env, env)


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
  carry = list(eqn.invars[cond_nconsts + body_nconsts :])
  carry_env = env
  limit = ctx.builder.max_unrolled_iterations
  truncated = False
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
          ctx.builder.note_unresolved_sync(reason)
        else:
          ctx.builder.note_incomplete(reason, affects_coverage=True)
      nested = env.child()
      for outer_v, inner_v in zip(body_consts, body.invars[:body_nconsts]):
        nested.bind(outer_v, inner_v, env)
      for outer_v, inner_v in zip(carry, body.invars[body_nconsts:]):
        nested.bind(outer_v, inner_v, carry_env)
      _walk(body, ctx.scoped(f"wh{step}"), nested)
      carry = list(body.outvars)
      carry_env = nested
      if keep_going is None:
        break
  finally:
    ctx.builder.pop_semaphore_scope(truncated=truncated)
  _bind_outputs(carry, eqn.outvars, carry_env, env)


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
  for br_idx in selected:
    branch = _unwrap_closed_jaxpr(branches[br_idx])
    nested = env.child()
    for outer_v, inner_v in zip(eqn.invars[1:], branch.invars):
      nested.bind(outer_v, inner_v, env)
    _walk(branch, ctx.scoped(f"br{br_idx}"), nested)
    if len(selected) == 1:
      _bind_outputs(branch.outvars, eqn.outvars, nested, env)


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
    ctx.builder.note_unresolved_sync(reason)
  else:
    ctx.builder.note_incomplete(reason, affects_coverage=True)


_HANDLERS: dict[str, Callable[[Any, _WalkContext, _Env, str | None], None]] = {
    "addupdate": _handle_swap,
    "cond": _handle_cond,
    "dma_start": _handle_dma_start,
    "dma_wait": _handle_dma_wait,
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
    if _try_eval_scalar_eqn(
        eqn,
        env.scalars,
        grid_coords=ctx.grid_coords,
        grid_sizes=ctx.grid_sizes,
        axis_env=ctx.axis_env,
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
      in_dtypes = {getattr(v.aval, "dtype", None) for v in eqn.invars[:2]}
      out_dtype = getattr(eqn.outvars[0].aval, "dtype", None)
      if jnp.dtype(jnp.bfloat16) in in_dtypes and out_dtype == jnp.bfloat16:
        builder.record_issue(
            _Category.LOW_PRECISION_ACCUMULATION,
            "dot_general on bfloat16 operands produces a bfloat16 result;"
            " accumulating such results (for example across grid steps)"
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
        if getattr(var.aval, "memory_space", None) == pltpu.MemorySpace.HBM:
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


def _eval_index_map(bm: Any, grid_coords: tuple[int, ...]) -> list[Any] | None:
  """Evaluates a block index map; None if it depends on runtime data.

  The map is evaluated symbolically rather than executed, which would
  initialize JAX backends.

  Args:
    bm: The block mapping.
    grid_coords: The grid point.

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
  for var, const in zip(jaxpr.constvars, closed.consts):
    value = _eval_atom(const, {})
    if value is not None:
      scalars[var] = value
  for eqn in jaxpr.eqns:
    _try_eval_scalar_eqn(
        eqn, scalars, grid_coords=(), grid_sizes=(), axis_env={}
    )
  outs = [_eval_atom(v, scalars) for v in jaxpr.outvars]
  if any(o is None for o in outs):
    return None
  return bm.index_map_out_tree.unflatten(outs)


def _block_slice(bm: Any, grid_coords: tuple[int, ...]) -> _BlockSlice | None:
  """Returns the array region of a block at a grid point, if computable."""
  indices = _eval_index_map(bm, grid_coords)
  if indices is None:
    return None
  array_shape = tuple(int(s) for s in bm.array_aval.shape)
  dims: list[SliceDim] = []
  active: list[int] = []
  violation: str | None = None
  for axis, (value, block_dim) in enumerate(zip(indices, bm.block_shape)):
    size = array_shape[axis]
    if isinstance(block_dim, pl.BoundedSlice) and isinstance(value, pl.Slice):
      start = int(np.asarray(value.start).item())
      length = int(np.asarray(value.size).item())
      if start < 0 or start + length > size:
        violation = violation or (
            f"slice {start}:{start + length} is outside axis {axis} of size"
            f" {size}"
        )
      dims.append(SliceDim(start, length))
      active.append(axis)
      continue
    index = int(np.asarray(value).item())
    if isinstance(block_dim, pl.Squeezed) or block_dim is None:
      if not 0 <= index < size:
        violation = violation or (
            f"index {index} is outside axis {axis} of size {size}"
        )
      dims.append(SliceDim(index, 1))
      continue
    if isinstance(block_dim, pl.Element):
      dims.append(SliceDim(index, int(block_dim.block_size)))
      active.append(axis)
      continue
    block = int(getattr(block_dim, "block_size", block_dim))
    if index < 0 or index * block >= max(size, 1):
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
    dims[axis] = SliceDim(dims[axis].start + tile_dim.start, tile_dim.size)
  return HyperRectSlice(tuple(dims))


def _uncovered_region(
    shape: Sequence[int], regions: Sequence[HyperRectSlice]
) -> HyperRectSlice | None:
  """Returns the bounding box of the part of `shape` no region covers.

  Returns None when everything is covered or the check would be too costly.
  """
  rank = len(shape)
  usable = [r for r in regions if len(r.dims) == rank]
  if not rank or any(s == 0 for s in shape):
    return None
  cuts: list[list[int]] = []
  for axis, size in enumerate(shape):
    points = {0, size}
    for region in usable:
      points.add(min(max(region.dims[axis].start, 0), size))
      points.add(min(max(region.dims[axis].stop, 0), size))
    cuts.append(sorted(points))
  if math.prod(len(c) - 1 for c in cuts) > _MAX_COVERAGE_CELLS:
    return None
  lows = list(shape)
  highs = [0] * rank
  for cell in itertools.product(*(range(len(c) - 1) for c in cuts)):
    bounds = [(cuts[a][i], cuts[a][i + 1]) for a, i in enumerate(cell)]
    covered = any(
        all(
            r.dims[a].start <= lo and hi <= r.dims[a].stop
            for a, (lo, hi) in enumerate(bounds)
        )
        for r in usable
    )
    if not covered:
      for a, (lo, hi) in enumerate(bounds):
        lows[a] = min(lows[a], lo)
        highs[a] = max(highs[a], hi)
  if highs[0] <= lows[0]:
    return None
  return HyperRectSlice(
      tuple(SliceDim(lo, hi - lo) for lo, hi in zip(lows, highs))
  )


@dataclasses.dataclass(frozen=True)
class _Operand:
  """A pipelined or directly accessed `pallas_call` operand."""

  index: int  # Position among all block operands.
  io_index: int  # Position among the inputs or among the outputs.
  var: Any
  block_mapping: Any
  buffer_id: str
  is_output: bool
  memory_space: str

  def describe(self) -> str:
    return f"{'output' if self.is_output else 'input'} {self.io_index}"


def _operands(eqn: Any, kernel_jaxpr: Any, grid_mapping: Any) -> list[_Operand]:
  io_aliases = dict(eqn.params.get("input_output_aliases", ()))
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
      buffer_id = f"arg_in_{out_to_in[io_index]}"
    else:
      buffer_id = f"arg_out_{io_index}"
    memory_space = _normalize_memory_space(
        getattr(bm.transformed_block_aval, "memory_space", None)
    )
    operands.append(
        _Operand(b_idx, io_index, b_var, bm, buffer_id, is_output, memory_space)
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
    dtype = getattr(bm.transformed_block_aval, "dtype", np.float32)
    array_shape = tuple(int(s) for s in bm.array_aval.shape)
    block = _block_slice(bm, ctx.grid_coords) if pipelined else None
    if block is not None and block.violation is not None:
      ctx.builder.record_issue(
          _Category.OUT_OF_BOUNDS_ACCESS,
          f"the BlockSpec index map of {op.describe()} at grid point"
          f" {ctx.grid_coords}: {block.violation}",
      )
    if op.memory_space == "hbm" or not pipelined:
      env.refs[op.var] = _full_view(
          op.buffer_id, op.memory_space, array_shape, dtype
      )
      continue
    tile_shape = tuple(int(s) for s in bm.transformed_block_aval.shape)
    tile_id = f"tile_{op.index}_{ctx.grid_tag()}"
    env.refs[op.var] = _full_view(tile_id, op.memory_space, tile_shape, dtype)
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
  for op, tile_id, block in tiles:
    if block is None:
      continue
    if op.is_output:
      for region in ctx.written.get(tile_id, ()):
        array_region = _tile_region_to_array(region, block)
        if array_region is None:
          builder.note_incomplete(
              f"the block layout of {op.describe()} is not understood",
              affects_coverage=True,
          )
        else:
          builder.record_output_write(ctx.device_id, op.buffer_id, array_region)
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
          )
      )


def _check_output_coverage(
    builder: _KernelModelBuilder,
    operands: Sequence[_Operand],
    mesh_ctx: _MeshContext,
) -> None:
  """Records outputs whose elements are never written."""
  if not builder.coverage_known:
    return
  for op in operands:
    if not op.buffer_id.startswith("arg_out_"):
      continue  # Outputs aliased to inputs start out initialized.
    shape = tuple(int(s) for s in op.block_mapping.array_aval.shape)
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


def _grid_and_semantics(
    grid_mapping: Any, mosaic_params: Any, builder_note: Callable[[str], None]
) -> tuple[tuple[int, ...], tuple[str, ...]]:
  grid: list[int] = []
  for d in grid_mapping.grid:
    if isinstance(d, (int, np.integer)):
      grid.append(int(d))
    else:
      grid.append(1)
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
  return tuple(grid), tuple(semantics)


def _extract_pallas_call_model(
    eqn: Any,
    mesh_ctx: _MeshContext,
    *,
    max_unrolled_iterations: int,
) -> KernelVerificationModel:
  """Extracts a `KernelVerificationModel` from one `pallas_call` equation."""
  kernel_jaxpr = _unwrap_closed_jaxpr(eqn.params["jaxpr"])
  grid_mapping: Any = eqn.params["grid_mapping"]
  mosaic_params = _extract_mosaic_compiler_params(
      eqn.params.get("compiler_params")
  )
  collective_id = getattr(mosaic_params, "collective_id", None)
  notes: list[str] = []
  grid, semantics = _grid_and_semantics(
      grid_mapping, mosaic_params, notes.append
  )
  builder = _KernelModelBuilder(
      kernel_name=_kernel_name(eqn.params),
      num_devices=mesh_ctx.num_devices,
      grid=grid,
      dimension_semantics=semantics,
      collective_id=collective_id,
      max_unrolled_iterations=max_unrolled_iterations,
  )
  kernel_has_sync = builder.has_sync(kernel_jaxpr)
  for note in notes:
    if kernel_has_sync:
      builder.note_truncated_sync(note, affects_coverage=True)
    else:
      builder.note_incomplete(note, affects_coverage=True)
  _check_static_hardware_rules(kernel_jaxpr, collective_id, builder)

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
    builder.note_incomplete(
        f"only the first {_MAX_COVERAGE_GRID_POINTS} of {num_points} grid"
        " points are checked for out-of-bounds and unwritten outputs",
        affects_coverage=True,
    )
  points = list(
      itertools.islice(
          itertools.product(*(range(s) for s in grid)),
          _MAX_COVERAGE_GRID_POINTS,
      )
  )
  for dev_id, dev_coords in mesh_ctx.iter_devices():
    for point_idx, g_coords in enumerate(points):
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
              grid_sizes=grid,
              scope_path="",
              touched_refs=set(),
              written={},
              record_events=point_idx < max_unrolled_iterations,
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
        getattr(aval, "shape", ()),
        getattr(aval, "dtype", np.float32),
        sem_type=_detect_semaphore_type(aval),
    )
  tiles = _bind_operands(ctx, env, operands, pipelined=bool(ctx.grid_sizes))
  body_actor_pos = len(ctx.builder.actor_events.get(ctx.actor, ()))
  body_mem_pos = len(ctx.builder.memory_events)
  _walk(kernel_jaxpr, ctx, env)
  _emit_pipeline_events(ctx, tiles, body_actor_pos, body_mem_pos)


def _collect_models(
    jaxpr: Any,
    mesh_ctx: _MeshContext,
    out_models: list[KernelVerificationModel],
    *,
    max_unrolled_iterations: int,
) -> None:
  """Recursively locates `shard_map` and `pallas_call` equations."""
  for eqn in jaxpr.eqns:
    name = eqn.primitive.name
    if name == "pallas_call":
      out_models.append(
          _extract_pallas_call_model(
              eqn, mesh_ctx, max_unrolled_iterations=max_unrolled_iterations
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
      )


def _trace_op_to_closed_jaxpr(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any],
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
    spec_pos = 0
    for idx, val in enumerate(norm_args):
      if idx in jax_callable.static_argnums or not isinstance(
          val, (torch.Tensor, jax.Array)
      ):
        arg_slots.append((True, val))
      else:
        spec = (
            in_specs[spec_pos]
            if in_specs is not None and spec_pos < len(in_specs)
            else None
        )
        spec_pos += 1
        ph = (
            pallas_lib.jax_placeholder(
                val, mesh=active_mesh, partition_spec=spec
            )
            if isinstance(val, torch.Tensor)
            else jax.ShapeDtypeStruct(val.shape, val.dtype)
        )
        arg_slots.append((False, len(dyn_placeholders)))
        dyn_placeholders.append(ph)

    kw_slots: dict[str, tuple[bool, Any]] = {}
    for k, val in norm_kwargs.items():
      if isinstance(val, torch.Tensor):
        ph = pallas_lib.jax_placeholder(val, mesh=active_mesh)
        kw_slots[k] = (False, len(dyn_placeholders))
        dyn_placeholders.append(ph)
      elif isinstance(val, jax.Array):
        kw_slots[k] = (False, len(dyn_placeholders))
        dyn_placeholders.append(jax.ShapeDtypeStruct(val.shape, val.dtype))
      else:
        kw_slots[k] = (True, val)

    def _trace_target(*dyn_args):
      rebuilt_args = [
          slot_val if is_static else dyn_args[slot_val]
          for is_static, slot_val in arg_slots
      ]
      rebuilt_kwargs = {
          k: slot_val if is_static else dyn_args[slot_val]
          for k, (is_static, slot_val) in kw_slots.items()
      }
      return raw_fn(*rebuilt_args, **rebuilt_kwargs)

    return jax.make_jaxpr(_trace_target)(*dyn_placeholders)

  # Raw JAX function path.
  raw_fn: Callable[..., Any] = op_or_fn
  dyn_phs: list[Any] = []
  pos_map: list[tuple[bool, Any]] = []
  for val in example_args:
    if isinstance(val, torch.Tensor):
      pos_map.append((False, len(dyn_phs)))
      dyn_phs.append(pallas_lib.jax_placeholder(val))
    elif isinstance(val, (jax.Array, jax.ShapeDtypeStruct)):
      pos_map.append((False, len(dyn_phs)))
      dyn_phs.append(jax.ShapeDtypeStruct(val.shape, val.dtype))
    else:
      pos_map.append((True, val))

  def _raw_target(*dyn_args):
    args = [v if is_st else dyn_args[v] for is_st, v in pos_map]
    return raw_fn(*args, **example_kwargs)

  return jax.make_jaxpr(_raw_target)(*dyn_phs)


def display_name(op_or_fn: Any) -> str:
  """Returns the name used for `op_or_fn` in findings."""
  return fx_verifier.get_op_display_name(op_or_fn)


def extract_kernel_models(
    op_or_fn: Any,
    example_args: Sequence[Any],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    max_unrolled_iterations: int = 32,
) -> tuple[KernelVerificationModel, ...]:
  """Traces a TorchTPU `jax_op` or JAX callable and extracts Pallas models."""
  closed = _trace_op_to_closed_jaxpr(
      op_or_fn, example_args, dict(example_kwargs or {})
  )
  models: list[KernelVerificationModel] = []
  _collect_models(
      closed.jaxpr,
      _MeshContext(),
      models,
      max_unrolled_iterations=max_unrolled_iterations,
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
